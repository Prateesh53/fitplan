# FitPlan: an AI training team with shared memory
# ITS 58100 - Agentic AI | Fall 2026 | Project 2
# Team: Prateesh Kuruba, Charan Kumar Vemuri
#
# Runs in Google Colab, Jupyter, VS Code, or as a plain script (python project2_fitplan.py).
# Provide a DeepSeek key as an environment variable or Colab secret named DEEPSEEK_API_KEY.
# app.py - the FitPlan app only (no test cases). Run: python app.py, then open http://127.0.0.1:7860


# ==========================================================================
# Cell 2 - Imports
# ==========================================================================
import os
import re
import io
import json
import time
import uuid
import html
import sqlite3
import hashlib
import secrets
import contextlib
from datetime import datetime, date
from typing import TypedDict, Annotated, Literal

import pandas as pd

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.store.sqlite import SqliteStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

from langchain_deepseek import ChatDeepSeek


# ==========================================================================
# Cell 3 - Load the API key
# ==========================================================================
# > The key is never written in the code: environment variable, then Colab
# > secret, then a hidden prompt.
key = os.environ.get("DEEPSEEK_API_KEY")
if not key:
    try:
        from google.colab import userdata
        key = userdata.get("DEEPSEEK_API_KEY")
    except Exception:
        key = None
if not key and os.environ.get("SPACE_ID"):
    raise RuntimeError("Add DEEPSEEK_API_KEY as a secret in the Space settings (Settings > Variables and secrets).")
if not key:
    from getpass import getpass
    key = getpass("Paste a DeepSeek API key (input is hidden, not saved): ")
os.environ["DEEPSEEK_API_KEY"] = key
print("API key loaded:", bool(os.environ.get("DEEPSEEK_API_KEY")))


# ==========================================================================
# Cell 4 - Model check
# ==========================================================================
# > Both agents use the same model at temperature 0. A failed check stops the run.
MODEL_NAME = "deepseek-v4-flash"
llm = ChatDeepSeek(model=MODEL_NAME, temperature=0)
try:
    llm.invoke("Reply with the single word: ready")
    print(f"Model ready: {MODEL_NAME}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    raise RuntimeError("Model check failed - fix the API key or model name, then run again.") from e


# ==========================================================================
# Cell 5 - The data folder and synthetic training data
# ==========================================================================
# > NEW: everything FitPlan saves lives in ONE folder, fitplan_data/, so it is easy
# > to show where each file is:
# >   fitplan_users.db        accounts and profiles
# >   fitplan_memory.db       private memory + the shared workout library (LangGraph store)
# >   fitplan_checkpoints.db  conversations in progress (LangGraph checkpointer)
# >   workout_log.csv, daily_metrics.csv, exercise_library.csv  training data
# > In Colab the folder is on Google Drive (MyDrive/FitPlan_data), so rerunning the
# > notebook never loses accounts, profiles, memory or shared workouts.
# > Set FITPLAN_DATA to choose another folder (for example on Hugging Face later).
# > There are NO demo accounts. The tests (Cells 18-21) create two temporary TEST members
# > with invented training data and remove them again in Cell 22.
ON_SPACE = bool(os.environ.get("SPACE_ID"))        # Hugging Face sets SPACE_ID inside a Space


def pick_data_dir():
    if os.environ.get("FITPLAN_DATA"):
        return os.path.abspath(os.environ["FITPLAN_DATA"])
    if ON_SPACE:
        return "/data/fitplan_data" if os.path.isdir("/data") else os.path.abspath("fitplan_data")
    try:
        from google.colab import drive          # only exists inside Colab
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        return "/content/drive/MyDrive/FitPlan_data"
    except ImportError:
        return os.path.abspath("fitplan_data")
    except Exception as e:
        print("Google Drive not mounted, using a local folder (data is lost when Colab stops):", e)
        return os.path.abspath("fitplan_data")


DATA_DIR = pick_data_dir()
os.makedirs(DATA_DIR, exist_ok=True)
LOG_PATH = os.path.join(DATA_DIR, "workout_log.csv")
METRICS_PATH = os.path.join(DATA_DIR, "daily_metrics.csv")
LIB_PATH = os.path.join(DATA_DIR, "exercise_library.csv")
USERS_DB = os.path.join(DATA_DIR, "fitplan_users.db")
MEMORY_DB = os.path.join(DATA_DIR, "fitplan_memory.db")
CHECKPOINT_DB = os.path.join(DATA_DIR, "fitplan_checkpoints.db")

# --- Hugging Face: keep a copy of the data folder in a PRIVATE dataset --------------------
# > A free Space loses its files when it restarts. If FITPLAN_BACKUP_REPO is set (for example
# > "your-name/fitplan-data") and the Space has an HF_TOKEN secret with write access, FitPlan
# > restores the files from that dataset when it starts and uploads a safe copy every 2 minutes
# > (only when something changed). Colab and your own computer don't need this.
import shutil
import tempfile
import threading
import atexit

BACKUP_REPO = os.environ.get("FITPLAN_BACKUP_REPO", "").strip()
BACKUP_TEST_DIR = os.environ.get("FITPLAN_BACKUP_TEST_DIR")       # tests only: a local folder instead of Hugging Face
BACKUP_EVERY = int(os.environ.get("FITPLAN_BACKUP_SECONDS", "120"))
BACKUP_FILES = ["fitplan_users.db", "fitplan_memory.db", "fitplan_checkpoints.db", "workout_log.csv", "daily_metrics.csv"]
BACKUP_ON = bool(BACKUP_REPO or BACKUP_TEST_DIR)
_backup_state = {"sig": None, "last": "never", "lock": threading.Lock()}


def _hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def restore_backup():
    """Copy the saved files into the data folder when the app starts (only into an empty folder)."""
    if not BACKUP_ON:
        return "no backup set up"
    if os.path.exists(USERS_DB):
        return "data folder already has files, kept them"
    if BACKUP_TEST_DIR:
        src = BACKUP_TEST_DIR
    else:
        from huggingface_hub import snapshot_download
        _hf_api().create_repo(BACKUP_REPO, repo_type="dataset", private=True, exist_ok=True)
        src = snapshot_download(BACKUP_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
                                local_dir=tempfile.mkdtemp())
    found = [f for f in BACKUP_FILES if os.path.exists(os.path.join(src, f))]
    for f in found:
        shutil.copy2(os.path.join(src, f), os.path.join(DATA_DIR, f))
    return f"restored {len(found)} files" if found else "backup is empty (first start)"


def backup_now(reason="scheduled"):
    """Take a consistent copy of each database (SQLite backup API) and upload it if anything changed."""
    if not BACKUP_ON:
        return "no backup set up"
    with _backup_state["lock"]:
        stage = tempfile.mkdtemp()
        try:
            for f in BACKUP_FILES:
                src = os.path.join(DATA_DIR, f)
                if not os.path.exists(src):
                    continue
                if f.endswith(".db"):
                    a, b_ = sqlite3.connect(src), sqlite3.connect(os.path.join(stage, f))
                    a.backup(b_)
                    b_.close()
                    a.close()
                else:
                    shutil.copy2(src, os.path.join(stage, f))
            sig = {f: hashlib.sha256(open(os.path.join(stage, f), "rb").read()).hexdigest() for f in os.listdir(stage)}
            if sig == _backup_state["sig"]:
                return "unchanged"
            if BACKUP_TEST_DIR:
                os.makedirs(BACKUP_TEST_DIR, exist_ok=True)
                for f in os.listdir(stage):
                    shutil.copy2(os.path.join(stage, f), os.path.join(BACKUP_TEST_DIR, f))
            else:
                _hf_api().upload_folder(repo_id=BACKUP_REPO, repo_type="dataset", folder_path=stage,
                                        commit_message=f"FitPlan data backup ({reason})")
            _backup_state["sig"], _backup_state["last"] = sig, datetime.now().isoformat(timespec="seconds")
            return "saved"
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def _backup_loop():
    while True:
        time.sleep(BACKUP_EVERY)
        try:
            backup_now()
        except Exception as e:
            print(f"  [backup] failed: {type(e).__name__}: {str(e)[:120]}")


if BACKUP_ON:
    try:
        print("Backup:", restore_backup(), "| copying to", BACKUP_REPO or BACKUP_TEST_DIR, f"every {BACKUP_EVERY} s")
    except Exception as e:
        print(f"Backup restore failed ({type(e).__name__}: {str(e)[:150]}). Check HF_TOKEN and FITPLAN_BACKUP_REPO.")
    threading.Thread(target=_backup_loop, daemon=True).start()
    atexit.register(lambda: backup_now("app closing"))
    if ON_SPACE:                     # a Space is stopped with SIGTERM: save once more before closing
        import signal

        def _save_and_exit(*_):
            try:
                backup_now("app closing")
            finally:
                os._exit(0)
        signal.signal(signal.SIGTERM, _save_and_exit)

DATA_START = pd.Timestamp("2026-10-26")   # Monday of week 1 (4 weeks of data)
WEEKS = 4
DAY = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

# Each lift: (session, day, sets, target reps, weight per week in kg, RPE per week)
PROGRAM_A = {
    "barbell bench press":     ("Push", "Mon", 3, 6,  [80, 82.5, 85, 85],      [7.5, 8, 8.5, 9.5]),
    "overhead press":          ("Push", "Mon", 3, 8,  [50, 50, 50, 50],        [9, 9, 9.5, 9.5]),
    "incline dumbbell press":  ("Push", "Thu", 3, 10, [30, 30, 32, 32],        [8, 8, 8.5, 8]),
    "triceps pushdown":        ("Push", "Thu", 3, 12, [35, 37.5, 37.5, 40],    [8, 8, 8, 8]),
    "barbell row":             ("Pull", "Tue", 3, 8,  [70, 72.5, 75, 77.5],    [7.5, 8, 8, 8.5]),
    "lat pulldown":            ("Pull", "Tue", 3, 10, [60, 62.5, 65, 65],      [8, 8, 8.5, 8.5]),
    "seated cable row":        ("Pull", "Fri", 3, 10, [55, 57.5, 60, 60],      [8, 8, 8, 8]),
    "dumbbell curl":           ("Pull", "Fri", 3, 12, [14, 14, 16, 16],        [8, 8.5, 9, 8.5]),
    "barbell back squat":      ("Legs", "Wed", 3, 6,  [100, 105, 107.5, 110],  [7.5, 8, 8, 8.5]),
    "romanian deadlift":       ("Legs", "Wed", 3, 8,  [90, 92.5, 95, 97.5],    [7.5, 8, 8, 8]),
    "leg press":               ("Legs", "Sat", 3, 12, [160, 170, 180, 180],    [8, 8, 8.5, 8.5]),
    "seated leg curl":         ("Legs", "Sat", 3, 12, [45, 47.5, 50, 50],      [8, 8, 8, 8.5]),
}
MISSED_A = {(3, "Fri")}           # week 4: missed Friday's pull session
PROGRAM_B = {
    "goblet squat":          ("Full body", "Mon", 3, 10, [16, 18, 20, 20],     [7, 7.5, 8, 8]),
    "dumbbell bench press":  ("Full body", "Wed", 3, 10, [14, 16, 16, 18],     [7.5, 8, 8, 8.5]),
    "lat pulldown":          ("Full body", "Fri", 3, 10, [35, 37.5, 40, 40],   [7.5, 8, 8, 8]),
}
TEST_IDS = ["test-member-A", "test-member-B"]     # temporary test members, removed in Cell 22

rows = []
for athlete, program, missed in [("test-member-A", PROGRAM_A, MISSED_A), ("test-member-B", PROGRAM_B, set())]:
    for w in range(WEEKS):
        for ex, (session, day, sets, reps, kg, rpe) in program.items():
            day_str = (DATA_START + pd.Timedelta(weeks=w, days=DAY[day])).date().isoformat()
            done = (w, day) not in missed
            for s in range(1, sets + 1):
                got = reps - 2 if (rpe[w] >= 9.5 and s == sets) else reps
                rows.append({"date": day_str, "athlete_id": athlete, "session": session,
                             "exercise": ex, "set_no": s, "target_reps": reps,
                             "reps": got if done else 0, "weight_kg": kg[w],
                             "rpe": rpe[w] if done else None, "completed": done})


def write_keeping_users(new_df, path):
    """Rewrite the test members' rows but keep rows logged by real app users."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        new_df = pd.concat([new_df, old[~old["athlete_id"].isin(TEST_IDS)]], ignore_index=True)
    new_df.to_csv(path, index=False)


# Recovery data. Member A wears a watch (HRV, resting HR); the last 3 days show short
# sleep, so today's readiness is not green. Member B has no watch (sleep + soreness only).
metrics = []
for d in range(WEEKS * 7):
    day_str = (DATA_START + pd.Timedelta(days=d)).date().isoformat()
    late = d >= WEEKS * 7 - 3
    metrics.append({"date": day_str, "athlete_id": "test-member-A",
                    "sleep_hours": 6.2 if late else round(7.4 + 0.3 * ((d % 3) - 1), 1),
                    "resting_hr": 58 if late else 55 + (d % 2),
                    "hrv_ms": 60 if late else 66 + (d % 4) - 2,
                    "soreness_1_5": 3 if late else 2})
    metrics.append({"date": day_str, "athlete_id": "test-member-B",
                    "sleep_hours": round(7.6 + 0.2 * ((d % 2) - 0.5), 1), "resting_hr": None,
                    "hrv_ms": None, "soreness_1_5": 2})


def write_test_history():
    """(Re)write the test members' invented workouts and check-ins; real members' rows are kept."""
    write_keeping_users(pd.DataFrame(rows), LOG_PATH)
    write_keeping_users(pd.DataFrame(metrics), METRICS_PATH)


write_test_history()

library = [
    ("barbell bench press", "chest", "push", "barbell", "intermediate"),
    ("incline dumbbell press", "chest", "push", "dumbbell", "beginner"),
    ("dumbbell bench press", "chest", "push", "dumbbell", "beginner"),
    ("machine chest press", "chest", "push", "machine", "beginner"),
    ("cable fly", "chest", "push", "cable", "beginner"),
    ("push-up", "chest", "push", "bodyweight", "beginner"),
    ("overhead press", "shoulders", "push", "barbell", "intermediate"),
    ("seated dumbbell shoulder press", "shoulders", "push", "dumbbell", "beginner"),
    ("cable lateral raise", "shoulders", "push", "cable", "beginner"),
    ("dumbbell lateral raise", "shoulders", "push", "dumbbell", "beginner"),
    ("triceps pushdown", "triceps", "push", "cable", "beginner"),
    ("overhead triceps extension", "triceps", "push", "dumbbell", "beginner"),
    ("dips", "triceps", "push", "bodyweight", "intermediate"),
    ("barbell row", "back", "pull", "barbell", "intermediate"),
    ("lat pulldown", "back", "pull", "machine", "beginner"),
    ("pull-up", "back", "pull", "bodyweight", "intermediate"),
    ("seated cable row", "back", "pull", "cable", "beginner"),
    ("one-arm dumbbell row", "back", "pull", "dumbbell", "beginner"),
    ("chest-supported row", "back", "pull", "machine", "beginner"),
    ("face pull", "rear delts", "pull", "cable", "beginner"),
    ("dumbbell curl", "biceps", "pull", "dumbbell", "beginner"),
    ("cable curl", "biceps", "pull", "cable", "beginner"),
    ("hammer curl", "biceps", "pull", "dumbbell", "beginner"),
    ("barbell back squat", "quads", "legs", "barbell", "intermediate"),
    ("hack squat", "quads", "legs", "machine", "beginner"),
    ("leg press", "quads", "legs", "machine", "beginner"),
    ("goblet squat", "quads", "legs", "dumbbell", "beginner"),
    ("bulgarian split squat", "quads", "legs", "dumbbell", "intermediate"),
    ("split squat", "quads", "legs", "bodyweight", "beginner"),
    ("leg extension", "quads", "legs", "machine", "beginner"),
    ("romanian deadlift", "hamstrings", "legs", "barbell", "intermediate"),
    ("dumbbell romanian deadlift", "hamstrings", "legs", "dumbbell", "beginner"),
    ("seated leg curl", "hamstrings", "legs", "machine", "beginner"),
    ("hip thrust", "glutes", "legs", "barbell", "intermediate"),
    ("glute bridge", "glutes", "legs", "bodyweight", "beginner"),
    ("standing calf raise", "calves", "legs", "machine", "beginner"),
    ("plank", "core", "core", "bodyweight", "beginner"),
    ("cable crunch", "core", "core", "cable", "beginner"),
    ("hanging knee raise", "core", "core", "bodyweight", "intermediate"),
    ("incline treadmill walk", "conditioning", "cardio", "machine", "beginner"),
    ("mobility flow", "full body", "recovery", "bodyweight", "beginner"),
]
pd.DataFrame(library, columns=["exercise", "muscle_group", "pattern", "equipment", "level"]).to_csv(LIB_PATH, index=False)

print("Data folder:", DATA_DIR)
print(f"  workout_log.csv: {len(rows)} logged sets for {TEST_IDS}")
print(f"  daily_metrics.csv: {len(metrics)} daily check-ins")
print(f"  exercise_library.csv: {len(library)} exercises")


# ==========================================================================
# Cell 6 - Tools 1 and 2: training history and recovery
# ==========================================================================
# > Unchanged from Project 1. The Progress Checker agent uses both.
def to_py(obj):
    """Turn numpy numbers into plain Python ones so they can be saved."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def e1rm(weight, reps):
    """Epley estimate of a one-rep max."""
    return round(weight * (1 + reps / 30), 1) if reps > 0 else 0.0


def _with_weeks(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    return df


@tool
def get_training_history(athlete_id: str) -> dict:
    """Summarise an athlete's last training week from the gym log."""
    log = pd.read_csv(LOG_PATH)
    df = log[log["athlete_id"] == athlete_id]
    if df.empty:
        return {"athlete_id": athlete_id, "has_history": False}
    df = _with_weeks(df)
    last = df["week_start"].max()
    lw = df[df["week_start"] == last]
    done = df[df["completed"]]
    vol = (done["reps"] * done["weight_kg"]).groupby(done["week_start"]).sum()
    prev_avg = vol[vol.index < last].mean() if (vol.index < last).any() else 0
    sessions = lw.groupby(["date", "session"])["completed"].first().reset_index()
    lifts = {}
    for ex, g in done.groupby("exercise"):
        by_week = (g.assign(e=[e1rm(w, r) for w, r in zip(g["weight_kg"], g["reps"])])
                   .groupby("week_start")["e"].max())
        lifts[ex] = {"e1rm_by_week": [float(x) for x in by_week.values],
                     "last_week_top_kg": float(g[g["week_start"] == g["week_start"].max()]["weight_kg"].max())}
    return to_py({
        "athlete_id": athlete_id, "has_history": True,
        "weeks_of_data": int(df["week_start"].nunique()),
        "last_week_start": last.date().isoformat(),
        "sessions_planned": int(len(sessions)),
        "sessions_completed": int(sessions["completed"].sum()),
        "missed_sessions": [f"{d:%a} {s}" for d, s, c in sessions.itertuples(index=False) if not c],
        "volume_last_week_kg": round(float(vol.get(last, 0)), 0),
        "volume_avg_previous_weeks_kg": round(float(prev_avg), 0),
        "volume_change_pct": round(100 * (vol.get(last, 0) - prev_avg) / prev_avg, 1) if prev_avg else None,
        "lifts": lifts,
    })


def _num(v):
    try:
        v = float(v)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def readiness_score(day, hrv_b, rhr_b):
    """Readiness heuristic: HRV 0.35, sleep 0.25, resting HR 0.20, soreness 0.20 (no watch: sleep + soreness)."""
    clamp = lambda x: max(0.0, min(100.0, x))
    parts = [(0.25, clamp(100 * _num(day["sleep_hours"]) / 8)),
             (0.20, clamp(100 * (5 - _num(day["soreness_1_5"])) / 4))]
    hrv, rhr = _num(day.get("hrv_ms")), _num(day.get("resting_hr"))
    if hrv is not None and hrv_b:
        parts.append((0.35, clamp(50 + 250 * (hrv - hrv_b) / hrv_b)))
    if rhr is not None and rhr_b:
        parts.append((0.20, clamp(50 - 500 * (rhr - rhr_b) / rhr_b)))
    return round(sum(w * x for w, x in parts) / sum(w for w, _ in parts))


@tool
def get_recovery_status(athlete_id: str) -> dict:
    """Today's readiness (0-100) from sleep, resting heart rate, HRV and soreness."""
    m = pd.read_csv(METRICS_PATH)
    m = m[m["athlete_id"] == athlete_id]
    if m.empty:
        return {"athlete_id": athlete_id, "has_data": False}
    m = m.sort_values("date")
    base = m.iloc[:-3] if len(m) > 3 else m
    today = m.iloc[-1]
    hrv_b, rhr_b = _num(base["hrv_ms"].mean()), _num(base["resting_hr"].mean())
    score = readiness_score(today, hrv_b, rhr_b)
    zone = "green" if score >= 67 else "yellow" if score >= 34 else "red"
    return to_py({
        "athlete_id": athlete_id, "has_data": True, "date": today["date"],
        "readiness": score, "zone": zone,
        "sleep_hours": _num(today["sleep_hours"]), "hrv_ms": _num(today["hrv_ms"]),
        "resting_hr": _num(today["resting_hr"]), "soreness_1_5": _num(today["soreness_1_5"]),
        "guidance": {"green": "train as planned",
                     "yellow": "train, but cut the first session's volume by about 20% and avoid max attempts",
                     "red": "replace the first session with recovery or mobility work"}[zone],
    })


print("Tool check (test-member-A):", {k: v for k, v in get_training_history.invoke({"athlete_id": "test-member-A"}).items() if k != "lifts"})
print("Recovery test-member-A:", get_recovery_status.invoke({"athlete_id": "test-member-A"})["readiness"],
      "| test-member-B (no watch):", get_recovery_status.invoke({"athlete_id": "test-member-B"})["readiness"])


# ==========================================================================
# Cell 7 - Tools 3 and 4: progression targets and exercise search
# ==========================================================================
# > Unchanged from Project 1. The Plan Coach chooses when to call them.
LOWER = {"quads", "hamstrings", "glutes", "calves"}


@tool
def get_progression_targets(athlete_id: str) -> list:
    """Next week's weight for each lift the athlete logged, with the reason."""
    log = pd.read_csv(LOG_PATH)
    df = log[(log["athlete_id"] == athlete_id) & (log["completed"])]
    if df.empty:
        return []
    df = _with_weeks(df)
    lib = pd.read_csv(LIB_PATH).set_index("exercise")
    out = []
    for ex, g in df.groupby("exercise"):
        last = g[g["week_start"] == g["week_start"].max()]
        kg, target = float(last["weight_kg"].max()), int(last["target_reps"].max())
        hit_all = bool((last["reps"] >= last["target_reps"]).all())
        rpe = float(last["rpe"].mean())
        weekly = (g.assign(e=[e1rm(w, r) for w, r in zip(g["weight_kg"], g["reps"])])
                  .groupby("week_start")["e"].max())
        stalled = len(weekly) >= 3 and weekly.iloc[-1] <= weekly.iloc[-3]
        lower = ex in lib.index and lib.loc[ex, "muscle_group"] in LOWER
        dumbbell = ex in lib.index and lib.loc[ex, "equipment"] == "dumbbell"
        step = 2.0 if dumbbell else 5.0 if lower else 2.5
        if stalled:
            nxt, why = round(kg * 0.9 / 2.5) * 2.5, "stalled 3 weeks -> deload 10%"
        elif hit_all and rpe <= 8.5:
            nxt, why = kg + step, f"all reps at RPE {rpe:.1f} -> +{step} kg"
        else:
            nxt, why = kg, f"missed reps or RPE {rpe:.1f} -> hold"
        out.append({"exercise": ex, "last_kg": kg, "target_reps": target, "next_kg": nxt, "reason": why})
    return to_py(out)


def _equipment_allowed(equipment: str) -> list:
    text = (equipment or "").lower()
    if "gym" in text or "barbell" in text or "machine" in text:
        return ["barbell", "dumbbell", "machine", "cable", "bodyweight"]
    if "dumbbell" in text:
        return ["dumbbell", "bodyweight"]
    return ["bodyweight"]


@tool
def search_exercises(equipment: str = "full gym", pattern: str = "any", muscle_group: str = "any") -> list:
    """Find exercises from the library that the athlete can do.

    equipment: what they have, e.g. 'full gym', 'dumbbells', 'bodyweight'.
    pattern: 'push', 'pull', 'legs', 'core', 'cardio', 'recovery' or 'any'.
    muscle_group: e.g. 'chest', 'back', 'quads' or 'any'. Only exercises returned may appear in a plan.
    """
    lib = pd.read_csv(LIB_PATH)
    out = lib[lib["equipment"].isin(_equipment_allowed(equipment))]
    if pattern and pattern != "any":
        out = out[out["pattern"] == pattern]
    if muscle_group and muscle_group != "any":
        out = out[out["muscle_group"] == muscle_group]
    return out[["exercise", "muscle_group", "pattern", "equipment", "level"]].to_dict("records")


for t in get_progression_targets.invoke({"athlete_id": "test-member-A"})[:4]:
    print(f"   {t['exercise']:<24}{t['last_kg']:>7} kg -> {t['next_kg']:>7} kg   {t['reason']}")


# ==========================================================================
# Cell 8 - Accounts and profiles (SQLite: fitplan_users.db)
# ==========================================================================
# > Passwords are never stored: only a salted PBKDF2-SHA256 hash. NEW profile fields:
# > exercise preferences, units, fitness watch, and permission for the AI team to
# > suggest sharing workouts. NEW: last sign-in time and sign-in count (for the members list
# > and the admin page). The two TEST members get a random password nobody knows.
MIN_AGE = 16
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PROFILE_TO_MEMORY = ["goal", "training_days", "split", "equipment", "session_minutes",
                     "experience_level", "exercise_preferences"]


def db():
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    return conn


with db() as conn:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL,
        name TEXT, salt TEXT NOT NULL, pw_hash TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS profiles (
        user_id TEXT PRIMARY KEY REFERENCES users(user_id),
        display_name TEXT, dob TEXT, sex TEXT, units TEXT, height_cm REAL, weight_kg REAL,
        goal TEXT, experience_level TEXT, training_days TEXT, session_minutes INTEGER,
        equipment TEXT, split TEXT, exercise_preferences TEXT, has_watch INTEGER,
        consent_llm INTEGER, share_suggestions INTEGER, updated_at TEXT);
    CREATE TABLE IF NOT EXISTS password_resets (
        token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL, used_at TEXT);
    CREATE TABLE IF NOT EXISTS emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT, to_email TEXT, kind TEXT, subject TEXT,
        status TEXT, created_at TEXT);
    """)
    have = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    for col, kind in [("last_login", "TEXT"), ("login_count", "INTEGER DEFAULT 0")]:
        if col not in have:                 # older database files get the new columns added
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {kind}")


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return salt, digest.hex()


def password_problems(password, confirm=None):
    """Plain-language rules shown live under the password box."""
    p = password or ""
    return {"length": len(p) >= 8,
            "mix": bool(re.search(r"[A-Za-z]", p) and re.search(r"\d", p)),
            "match": confirm is not None and p != "" and p == confirm}


def create_account(name, email, password, confirm, agree=True):
    """Returns (ok, message, user_id). The username is the part of the email before @."""
    name, email = (name or "").strip(), (email or "").strip().lower()
    if not name:
        return False, "Please tell us your name.", None
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return False, "Please enter a valid email address, like you@example.com.", None
    rules = password_problems(password, confirm)
    if not (rules["length"] and rules["mix"]):
        return False, "Your password needs at least 8 characters, with a letter and a number.", None
    if not rules["match"]:
        return False, "The two passwords don't match yet.", None
    if not agree:
        return False, "Please tick the box to agree before continuing.", None
    username = re.sub(r"[^a-z0-9_.]", "", email.split("@")[0])[:30] or "member"
    with db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            return False, "That email already has an account. Try signing in.", None
        base, n = username, 1
        while conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            n += 1
            username = f"{base}{n}"
        user_id = f"user-{secrets.token_hex(3)}"
        salt, pw_hash = hash_password(password)
        conn.execute("INSERT INTO users (user_id, username, email, name, salt, pw_hash, created_at) VALUES (?,?,?,?,?,?,?)",
                     (user_id, username, email, name, salt, pw_hash, datetime.now().isoformat(timespec="seconds")))
    return True, f"Account created for {email}. Sign in to continue.", user_id


def login(username_or_email, password):
    """Returns (ok, message, user_id)."""
    key_ = (username_or_email or "").strip()
    if not key_ or not password:
        return False, "Please enter your email or username and your password.", None
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=? OR email=?", (key_.lower(), key_.lower())).fetchone()
    if not row or hash_password(password, row["salt"])[1] != row["pw_hash"]:
        return False, "That email or password doesn't match. Check it and try again.", None
    with db() as conn:
        conn.execute("UPDATE users SET last_login=?, login_count=COALESCE(login_count, 0) + 1 WHERE user_id=?",
                      (datetime.now().isoformat(timespec="seconds"), row["user_id"]))
    return True, f"Welcome back, {row['name'] or row['username']}.", row["user_id"]


def calc_age(dob):
    try:
        d = datetime.strptime(str(dob or "").strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    t = date.today()
    return t.year - d.year - ((t.month, t.day) < (d.month, d.day))


def to_metric(units, height, weight):
    """The form accepts cm/kg or ft/lb; the database always stores cm and kg."""
    h, w = float(height), float(weight)
    if units == "ft and lb":
        return round(h * 30.48, 1), round(w * 0.4536, 1)
    return h, w


def save_profile(user_id, name, dob, sex, units, height, weight, goal, level, days, minutes,
                 equipment, split, prefs, watch, consent, share_ok):
    """Validate and save the profile form. Returns (ok, message)."""
    age = calc_age(dob)
    if not (name or "").strip():
        return False, "Please enter a profile name."
    if age is None or age > 100:
        return False, "Please enter your date of birth."
    if age < MIN_AGE:
        return False, f"FitPlan is for members aged {MIN_AGE} and over."
    try:
        h, w = to_metric(units, height, weight)
    except (TypeError, ValueError):
        return False, "Please enter your height and weight as numbers."
    if not (120 <= h <= 230) or not (30 <= w <= 300):
        return False, "Height should be 120-230 cm (4-7.5 ft) and weight 30-300 kg (66-660 lb)."
    if not goal:
        return False, "Please pick your main goal."
    if not level:
        return False, "Please tell us how experienced you are."
    if not days:
        return False, "Please pick at least one training day."
    days_txt = ", ".join(d for d in DAYS if d in days)
    with db() as conn:
        conn.execute("""INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET display_name=excluded.display_name, dob=excluded.dob,
            sex=excluded.sex, units=excluded.units, height_cm=excluded.height_cm, weight_kg=excluded.weight_kg,
            goal=excluded.goal, experience_level=excluded.experience_level, training_days=excluded.training_days,
            session_minutes=excluded.session_minutes, equipment=excluded.equipment, split=excluded.split,
            exercise_preferences=excluded.exercise_preferences, has_watch=excluded.has_watch,
            consent_llm=excluded.consent_llm, share_suggestions=excluded.share_suggestions,
            updated_at=excluded.updated_at""",
            (user_id, name.strip(), str(dob)[:10], sex, units, h, w, goal, level, days_txt, int(minutes),
             equipment, split, (prefs or "").strip(), int(watch == "Yes"), int(bool(consent)),
             int(bool(share_ok)), datetime.now().isoformat(timespec="seconds")))
    return True, "Profile saved."


def get_profile(user_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user(user_id):
    with db() as conn:
        row = conn.execute("SELECT user_id, username, email, name, created_at FROM users WHERE user_id=?",
                           (user_id,)).fetchone()
    return dict(row) if row else None


def create_test_member(user_id, name, *profile):
    """A temporary member used only by the tests. Random password: nobody can sign in as it."""
    with db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone():
            return
        salt, pw_hash = hash_password(secrets.token_hex(16))
        conn.execute("INSERT INTO users (user_id, username, email, name, salt, pw_hash, created_at) VALUES (?,?,?,?,?,?,?)",
                     (user_id, user_id, f"{user_id}@fitplan.test", name, salt, pw_hash,
                      datetime.now().isoformat(timespec="seconds")))
    save_profile(user_id, name, *profile)


def create_test_members():
    write_test_history()
    create_test_member("test-member-A", "Test member A", "2000-05-14", "Male", "cm and kg", 178, 80,
                       "Build muscle", "Intermediate", DAYS[:6], 75, "Full gym", "Push, pull, legs",
                       "", "Yes", True, True)
    create_test_member("test-member-B", "Test member B", "2001-08-02", "Male", "cm and kg", 175, 72,
                       "Get stronger", "Beginner", ["Monday", "Wednesday", "Friday"], 60, "Full gym",
                       "Full body", "", "No", False, True)


def is_test_member(uid):
    return str(uid or "").startswith("test-member-")


print("Accounts database:", USERS_DB)
with db() as conn:
    print("Members saved so far:", conn.execute("SELECT COUNT(*) FROM users WHERE user_id NOT LIKE 'test-member-%'").fetchone()[0])
print("Validation:", create_account("", "x", "a", "b")[1], "|", create_account("A", "a@b.co", "abcdefgh", "abcdefgh")[1])


# ==========================================================================
# Cell 8b - Emails: "account created" and "reset your password" (Gmail)
# ==========================================================================
# > NEW. FitPlan sends real emails through Gmail. Create a Gmail App password
# > (Google Account > Security > 2-Step Verification > App passwords) and add two
# > Colab secrets: FITPLAN_EMAIL (the Gmail address) and FITPLAN_EMAIL_PASSWORD (the
# > 16-character app password). Nothing is written in the code.
# > Reset links: random token, only its SHA-256 hash is stored, valid 30 minutes,
# > works once, at most 3 requests per hour per email. Every email is logged
# > (who, what kind, sent or failed) in fitplan_users.db, table "emails".
import smtplib
import threading
from email.message import EmailMessage

RESET_MINUTES = 30
MAX_RESETS_PER_HOUR = 3


def secret(name):
    v = os.environ.get(name)
    if not v:
        try:
            from google.colab import userdata
            v = userdata.get(name)
        except Exception:
            v = None
    return (v or "").strip()


EMAIL_FROM = secret("FITPLAN_EMAIL")
EMAIL_PASSWORD = secret("FITPLAN_EMAIL_PASSWORD").replace(" ", "")
BREVO_KEY = secret("BREVO_API_KEY")                             # Hugging Face: Gmail is blocked there, Brevo works over HTTPS
EMAIL_TEST_DIR = os.environ.get("FITPLAN_EMAIL_TEST_DIR")      # tests only: save emails as files instead of sending
EMAIL_VIA = ("test files" if EMAIL_TEST_DIR else "Brevo" if (BREVO_KEY and EMAIL_FROM) else
             "Gmail" if (EMAIL_FROM and EMAIL_PASSWORD) else "")
EMAIL_READY = bool(EMAIL_VIA)
APP_URL = os.environ.get("FITPLAN_APP_URL", "").rstrip("/") or \
    (f"https://{os.environ['SPACE_HOST']}" if os.environ.get("SPACE_HOST") else "")     # the Space's own address


def log_email(to, kind, subject, status):
    with db() as conn:
        conn.execute("INSERT INTO emails (to_email, kind, subject, status, created_at) VALUES (?,?,?,?,?)",
                     (to, kind, subject, status, datetime.now().isoformat(timespec="seconds")))


def email_page(title, lines, button=None):
    """A small, plain HTML email that also reads well as text."""
    body = "".join(f"<p style='margin:0 0 12px'>{l}</p>" for l in lines)
    btn = (f"<p style='margin:20px 0'><a href='{button[1]}' style='background:#2563EB;color:#fff;padding:12px 20px;"
           f"border-radius:8px;text-decoration:none;font-weight:600'>{button[0]}</a></p>") if button else ""
    return (f"<div style='font-family:Arial,sans-serif;max-width:520px;color:#0F172A;font-size:15px;line-height:1.5'>"
            f"<p style='font-size:20px;font-weight:700;color:#2563EB;margin:0 0 16px'>FitPlan</p>"
            f"<p style='font-size:18px;font-weight:700;margin:0 0 12px'>{title}</p>{body}{btn}"
            f"<p style='color:#5B6B82;font-size:12px;margin-top:24px'>FitPlan, ITS 58100 class project. "
            f"You get this email because this address was used on FitPlan.</p></div>")


def send_email(to, kind, subject, text, html_body):
    """Send one email. Returns (ok, status). Never raises, so a mail problem never breaks the app."""
    if not EMAIL_READY:
        log_email(to, kind, subject, "not sent: email is not set up")
        return False, "not set up"
    msg = EmailMessage()
    msg["Subject"], msg["To"] = subject, to
    msg["From"] = f"FitPlan <{EMAIL_FROM or 'fitplan@test.local'}>"
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    try:
        if EMAIL_TEST_DIR:
            os.makedirs(EMAIL_TEST_DIR, exist_ok=True)
            with open(os.path.join(EMAIL_TEST_DIR, f"{time.time_ns()}-{kind}.eml"), "wb") as f:
                f.write(bytes(msg))
        elif EMAIL_VIA == "Brevo":
            import urllib.request
            req = urllib.request.Request("https://api.brevo.com/v3/smtp/email", method="POST",
                                         headers={"api-key": BREVO_KEY, "content-type": "application/json",
                                                  "accept": "application/json"},
                                         data=json.dumps({"sender": {"name": "FitPlan", "email": EMAIL_FROM},
                                                          "to": [{"email": to}], "subject": subject,
                                                          "htmlContent": html_body, "textContent": text}).encode())
            urllib.request.urlopen(req, timeout=20).read()
        else:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
                s.login(EMAIL_FROM, EMAIL_PASSWORD)
                s.send_message(msg)
        log_email(to, kind, subject, "sent")
        return True, "sent"
    except Exception as e:
        log_email(to, kind, subject, f"failed: {type(e).__name__}")
        print(f"  [email] could not send '{kind}' email: {type(e).__name__}")
        return False, "failed"


def send_in_background(*args):
    threading.Thread(target=send_email, args=args, daemon=True).start()


def welcome_email(name, email):
    when = datetime.now().strftime("%d %B %Y at %H:%M")
    text = (f"Hi {name},\n\nYour FitPlan account was created with this email ({email}) on {when}.\n"
            "If this wasn't you, use 'Forgot password?' on the sign-in page to secure the account.\n\nFitPlan")
    page = email_page(f"Welcome to FitPlan, {html.escape(name)}",
                      [f"Your FitPlan account was created with this email (<b>{html.escape(email)}</b>) on {when}.",
                       "Next, sign in and set up your profile so your AI team can plan your training.",
                       "If this wasn't you, use <b>Forgot password?</b> on the sign-in page."])
    send_in_background(email, "account_created", "Your FitPlan account was created", text, page)


def base_url(request=None):
    """The address members open: FITPLAN_APP_URL, else the public share link, else the address in the browser."""
    if APP_URL:
        return APP_URL
    share = globals().get("demo") and getattr(globals()["demo"], "share_url", None)
    if share:
        return share.rstrip("/")
    if request is not None:
        h = {k.lower(): v for k, v in dict(request.headers).items()}
        host = h.get("x-forwarded-host") or h.get("host")
        if host:
            return f"{h.get('x-forwarded-proto', 'http').split(',')[0]}://{host}"
    return f"http://127.0.0.1:{os.environ.get('FITPLAN_PORT', '7860')}"


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def request_password_reset(email, request=None):
    """Always returns the same message, so nobody can find out which emails have accounts."""
    email = (email or "").strip().lower()
    same = "If an account uses that email, a reset link is on its way. It works for 30 minutes."
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return False, "Please enter the email you signed up with."
    if not EMAIL_READY:
        return False, "Email isn't set up on this FitPlan yet, so a reset link can't be sent. Ask the person running it."
    with db() as conn:
        row = conn.execute("SELECT user_id, name FROM users WHERE email=?", (email,)).fetchone()
        if not row or is_test_member(row["user_id"]):
            return True, same
        hour_ago = (datetime.now() - pd.Timedelta(hours=1)).isoformat(timespec="seconds")
        recent = conn.execute("SELECT COUNT(*) FROM password_resets WHERE user_id=? AND created_at>?",
                              (row["user_id"], hour_ago)).fetchone()[0]
        if recent >= MAX_RESETS_PER_HOUR:
            return True, same
        token = secrets.token_urlsafe(32)
        now = datetime.now()
        conn.execute("INSERT INTO password_resets VALUES (?,?,?,?,NULL)",
                     (token_hash(token), row["user_id"], now.isoformat(timespec="seconds"),
                      (now + pd.Timedelta(minutes=RESET_MINUTES)).isoformat(timespec="seconds")))
    link = f"{base_url(request)}/?reset={token}"
    text = (f"Hi {row['name']},\n\nSomeone asked to reset the password for your FitPlan account.\n"
            f"Open this link within {RESET_MINUTES} minutes to choose a new password:\n{link}\n\n"
            "If you didn't ask for this, ignore this email. Your password stays the same.\n\nFitPlan")
    page = email_page("Reset your FitPlan password",
                      [f"Hi {html.escape(row['name'] or '')}, someone asked to reset the password for your FitPlan account.",
                       f"The button works once, for {RESET_MINUTES} minutes."],
                      (f"Choose a new password", link)) + \
        (f"<p style='font-family:Arial;font-size:12px;color:#5B6B82'>Button not working? Copy this link:<br>{html.escape(link)}</p>"
         "<p style='font-family:Arial;font-size:12px;color:#5B6B82'>If you didn't ask for this, ignore this email.</p>")
    send_in_background(email, "password_reset", "Reset your FitPlan password", text, page)
    return True, same


def check_reset_token(token):
    """Returns (user_id, message). user_id is None when the link can't be used."""
    token = (token or "").strip()
    if "reset=" in token:                       # a whole link was pasted
        token = token.split("reset=", 1)[1].split("&")[0]
    if not token:
        return None, "That reset link is missing its code."
    with db() as conn:
        row = conn.execute("SELECT * FROM password_resets WHERE token_hash=?", (token_hash(token),)).fetchone()
    if not row:
        return None, "That reset link isn't valid. Ask for a new one."
    if row["used_at"]:
        return None, "That reset link was already used. Ask for a new one."
    if row["expires_at"] < datetime.now().isoformat(timespec="seconds"):
        return None, "That reset link has expired. Ask for a new one."
    return row["user_id"], ""


def reset_password(token, password, confirm):
    """Set a new password with a valid link. Returns (ok, message)."""
    uid, err = check_reset_token(token)
    if not uid:
        return False, err
    rules = password_problems(password, confirm)
    if not (rules["length"] and rules["mix"]):
        return False, "Your password needs at least 8 characters, with a letter and a number."
    if not rules["match"]:
        return False, "The two passwords don't match yet."
    salt, pw_hash = hash_password(password)
    now = datetime.now().isoformat(timespec="seconds")
    with db() as conn:
        conn.execute("UPDATE users SET salt=?, pw_hash=? WHERE user_id=?", (salt, pw_hash, uid))
        conn.execute("UPDATE password_resets SET used_at=? WHERE user_id=? AND used_at IS NULL", (now, uid))
        user = conn.execute("SELECT name, email FROM users WHERE user_id=?", (uid,)).fetchone()
    send_in_background(user["email"], "password_changed", "Your FitPlan password was changed",
                       f"Hi {user['name']},\n\nYour FitPlan password was changed on {now.replace('T', ' at ')}.\n"
                       "If this wasn't you, use 'Forgot password?' straight away.\n\nFitPlan",
                       email_page("Your password was changed",
                                  [f"Your FitPlan password was changed on {now.replace('T', ' at ')}.",
                                   "If this wasn't you, use <b>Forgot password?</b> on the sign-in page straight away."]))
    return True, "Your password is changed. Sign in with your new password."


print("Email:", "test mode (emails saved as files)" if EMAIL_TEST_DIR else
      f"ready via {EMAIL_VIA} (sending from {EMAIL_FROM})" if EMAIL_READY else
      "off (optional: add FITPLAN_EMAIL + FITPLAN_EMAIL_PASSWORD to send sign-up and reset emails)")


# ==========================================================================
# Cell 9 - Memory that survives a restart: SQLite store + MemoryGateway with roles
# ==========================================================================
# > NEW. Long-term memory moves from RAM to fitplan_memory.db (LangGraph SqliteStore).
# >   ("athletes", <member id>, "profile")  private memory, one namespace per member
# >   ("shared", "library")                 community workouts, shared by everyone
# >   ("audit", "log")                      every access decision (allowed or denied)
# > Every read and write goes through the MemoryGateway. An "actor" is WHO is asking:
# > the member, or one of the two agents acting FOR the signed-in member.
mem_conn = sqlite3.connect(MEMORY_DB, check_same_thread=False, isolation_level=None)
store = SqliteStore(mem_conn)
store.setup()

ALLOWED_KEYS = {"goal", "training_days", "split", "equipment", "session_minutes",
                "experience_level", "exercise_preferences", "plan_format"}
HEALTH_PATTERN = re.compile(
    r"\b(injur\w*|pain\w*|hurt\w*|aches?|aching|sprain\w*|strain(ed)?|tendinitis|"
    r"surgery|medication\w*|diagnos\w*|asthma|diabet\w*|blood pressure|pregnan\w*|illness)\b", re.IGNORECASE)

# Role-based access: what each kind of actor may do. "own" = only for the signed-in member.
ROLE_RIGHTS = {
    "member":           {"read_private": "own", "write_private": "own", "read_library": True,
                         "propose_share": "own", "decide_share": "own", "update_record": "own"},
    "progress_analyst": {"read_private": "own", "write_private": False, "read_library": True,
                         "propose_share": "own", "decide_share": False, "update_record": False},
    "planner":          {"read_private": "own", "write_private": False, "read_library": True,
                         "propose_share": False, "decide_share": False, "update_record": False},
}
REVIEW_DAYS = 90


class AccessDenied(Exception):
    pass


class MemoryGateway:
    """The only code path that touches long-term memory. Checks the role first, logs every decision."""

    def __init__(self, store):
        self.store = store

    # ---------- helpers ----------
    @staticmethod
    def ns(owner):
        return ("athletes", owner, "profile")

    LIB = ("shared", "library")

    def log(self, actor, action, target, allowed, note=""):
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "role": actor[0], "acting_for": actor[1],
                 "action": action, "target": target, "allowed": allowed, "note": note}
        self.store.put(("audit", "log"), f"{time.time_ns()}", entry)
        print(f"  [gateway] {'ALLOW' if allowed else 'DENIED'} {action} {target} by {actor[0]} for {actor[1]}"
              + (f" - {note}" if note else ""))

    def check(self, actor, right, owner, target):
        role, acting_for = actor
        rule = ROLE_RIGHTS.get(role, {}).get(right, False)
        ok = rule is True or (rule == "own" and acting_for == owner)
        if not ok:
            self.log(actor, right, target, False, "not allowed for this role or member")
            raise AccessDenied(f"{role} acting for {acting_for} may not {right} on {target}")
        return True

    # ---------- private memory ----------
    def read_profile(self, actor, owner):
        self.check(actor, "read_private", owner, f"{owner}/profile")
        return {i.key: i.value for i in self.store.search(self.ns(owner), limit=50)}

    def write_profile(self, actor, owner, key, value, source="member statement"):
        self.check(actor, "write_private", owner, f"{owner}/profile/{key}")
        old = self.store.get(self.ns(owner), key)
        if old and old.value["value"] == value:
            return "unchanged"
        self.store.put(self.ns(owner), key, {"value": value, "previous_value": old.value["value"] if old else None,
                                             "updated_at": datetime.now().isoformat(timespec="seconds"),
                                             "source": source})
        return "corrected" if old else "created"

    # ---------- shared library ----------
    def _all_records(self):
        return [i.value for i in self.store.search(self.LIB, limit=500)]

    def library_all(self, actor):
        """Everything the actor may see: approved records for all, plus the actor's own pending/old ones."""
        self.check(actor, "read_library", None, "shared/library")
        return [r for r in self._all_records() if r["status"] == "approved" or r["author"] == actor[1]]

    def library_search(self, actor, equipment="any", max_minutes=None):
        """What agents may use in a plan: approved AND current versions only."""
        self.check(actor, "read_library", None, "shared/library")
        self.review_library()                     # flag out-of-date records before anyone uses them
        allowed = _equipment_allowed(equipment) if equipment and equipment != "any" else None
        out = []
        for r in self._all_records():
            if r["status"] != "approved" or r.get("superseded_by"):
                continue
            if allowed and r["equipment"] not in allowed:
                continue
            if max_minutes and r["minutes"] > max_minutes:
                continue
            out.append(r)
        self.log(actor, "read_library", "shared/library", True, f"{len(out)} approved, current records")
        return out

    def library_propose(self, actor, author, rec):
        """An agent (or the member) proposes a record. It is saved as PENDING, never published."""
        self.check(actor, "propose_share", author, "shared/library")
        rid = rec.get("record_id") or f"lib-{secrets.token_hex(2)}"
        now = datetime.now()
        value = {"record_id": rid, "version": 1, "title": rec["title"], "minutes": int(rec["minutes"]),
                 "equipment": rec["equipment"], "level": rec.get("level", "beginner"),
                 "exercises": list(rec["exercises"]), "author": author, "proposed_by": actor[0],
                 "status": "pending", "superseded_by": None, "why": rec.get("why", ""),
                 "created_at": now.isoformat(timespec="seconds"),
                 "review_after": (now + pd.Timedelta(days=REVIEW_DAYS)).date().isoformat(),
                 "history": [f"{now:%d %b}: proposed by {actor[0].replace('_', ' ')}"]}
        self.store.put(self.LIB, f"{rid}-v1", value)
        self.log(actor, "propose_share", f"{rid} v1", True, "saved as pending, author must approve")
        return rid

    def _get(self, rid, version=None):
        recs = [r for r in self._all_records() if r["record_id"] == rid]
        if version:
            recs = [r for r in recs if r["version"] == version]
        return max(recs, key=lambda r: r["version"]) if recs else None

    def library_decide(self, actor, rid, approve):
        """Only the author decides whether a pending record is published."""
        rec = self._get(rid)
        if not rec:
            return "missing"
        self.check(actor, "decide_share", rec["author"], f"{rid} v{rec['version']}")
        if approve:
            rec["status"] = "approved"
            rec["history"].append(f"{datetime.now():%d %b}: approved by the author")
            self.store.put(self.LIB, f"{rid}-v{rec['version']}", rec)
            self.log(actor, "decide_share", f"{rid} v{rec['version']}", True, "approved -> published")
            return "approved"
        self.store.delete(self.LIB, f"{rid}-v{rec['version']}")
        self.log(actor, "decide_share", f"{rid} v{rec['version']}", True, "rejected -> deleted, nothing shared")
        return "rejected"

    def library_update(self, actor, rid, changes, note):
        """Only the author corrects a record. A new version is written; the old one is marked superseded."""
        old = self._get(rid)
        self.check(actor, "update_record", old["author"] if old else None, f"{rid}")
        new = {**old, **changes, "version": old["version"] + 1, "status": "approved", "superseded_by": None,
               "review_after": (datetime.now() + pd.Timedelta(days=REVIEW_DAYS)).date().isoformat(),
               "history": old["history"] + [f"{datetime.now():%d %b}: corrected by the author - {note}"]}
        old["superseded_by"] = new["version"]
        old["status"] = "superseded"
        old["history"] = old["history"] + [f"{datetime.now():%d %b}: replaced by version {new['version']}"]
        self.store.put(self.LIB, f"{rid}-v{old['version']}", old)
        self.store.put(self.LIB, f"{rid}-v{new['version']}", new)
        self.log(actor, "update_record", f"{rid} v{old['version']} -> v{new['version']}", True, note)
        return new["version"]

    def review_library(self):
        """Code check (no LLM): mark approved records outdated when they use an exercise that is no longer
        in the library, clash with the author's current preferences, or are past their review date."""
        lib = set(pd.read_csv(LIB_PATH)["exercise"])
        flagged = []
        for r in self._all_records():
            if r["status"] != "approved" or r.get("superseded_by"):
                continue
            reason = ""
            missing = [e for e in r["exercises"] if e.split(" x")[0].strip() not in lib]
            prefs = (self.store.get(self.ns(r["author"]), "exercise_preferences") or None)
            avoid = (prefs.value["value"].lower() if prefs else "")
            clash = [e for e in r["exercises"] if any(w in avoid for w in [f"no {e.split(' x')[0].strip()}",
                                                                           f"no {e.split(' x')[0].strip()}s"])]
            if missing:
                reason = f"uses exercises no longer in the library: {', '.join(missing)}"
            elif clash:
                reason = f"clashes with the author's current preferences ({avoid})"
            elif r.get("review_after") and r["review_after"] < date.today().isoformat():
                reason = "older than the 90-day review date"
            if reason:
                r["status"] = "outdated"
                r["history"].append(f"{datetime.now():%d %b}: marked out of date - {reason}")
                self.store.put(self.LIB, f"{r['record_id']}-v{r['version']}", r)
                self.log(("member", "system"), "review_library", f"{r['record_id']} v{r['version']}", True,
                         f"outdated: {reason}")
                flagged.append((r["record_id"], reason))
        return flagged


gateway = MemoryGateway(store)


def check_memory_candidate(key, value):
    """Policy check in code: is this safe and useful to keep long term?"""
    if key not in ALLOWED_KEYS:
        return False, f"'{key}' is not a profile field we keep"
    if not value or len(str(value).strip()) < 2:
        return False, "empty value"
    if HEALTH_PATTERN.search(str(value)):
        return False, "health detail - never stored"
    return True, "ok"


def sync_profile(uid):
    """Copy profile-form fields that memory does not have yet (chat corrections are never overwritten)."""
    me = ("member", uid)
    mems = gateway.read_profile(me, uid)
    prof = get_profile(uid) or {}
    for k in PROFILE_TO_MEMORY:
        if k not in mems and prof.get(k) not in (None, ""):
            gateway.write_profile(me, uid, k, str(prof[k]), source="profile form")


def member_details(uid):
    """What the agents may see from the profile form: body details only with consent; never name or email."""
    prof = get_profile(uid)
    if not prof:
        return {}
    if not prof["consent_llm"]:
        return {"note": "member did not allow body details to be shared with the AI"}
    return {"age": calc_age(prof["dob"]), "sex": prof["sex"], "height_cm": prof["height_cm"], "weight_kg": prof["weight_kg"]}


@tool
def search_library(equipment: str = "full gym", max_minutes: int = 0) -> list:
    """Community workouts other members chose to share (approved, current versions only).

    equipment: what the member has, e.g. 'full gym', 'dumbbells', 'bodyweight'.
    max_minutes: longest workout that fits (0 = any length).
    """
    actor = CURRENT_ACTOR.get("planner") or ("planner", "unknown")
    recs = gateway.library_search(actor, equipment, max_minutes or None)
    return [{k: r[k] for k in ["record_id", "version", "title", "minutes", "equipment", "level", "exercises"]} | {"shared_by": r["author"]}
            for r in recs]


CURRENT_ACTOR = {}      # set by the planner node: the tool runs FOR the signed-in member

print("Memory store:", MEMORY_DB)
try:
    gateway.read_profile(("planner", "test-member-B"), "test-member-A")
except AccessDenied as e:
    print("Access check works:", e)


# ==========================================================================
# Cell 10 - Workflow state
# ==========================================================================
# > One conversation's state, saved per thread_id by the SQLite checkpointer.
class PlanState(TypedDict, total=False):
    session_user: str
    athlete_id: str
    request: str
    member: dict
    memories: dict
    understanding: dict
    memory_log: list
    history: dict
    recovery: dict
    route: str
    progress_report: dict          # NEW: the Progress Checker's hand-off to the Plan Coach
    share_rid: str                 # NEW: a community workout the Progress Checker proposed
    messages: Annotated[list, add_messages]
    plan: str
    plan_issues: list
    plan_fixes: int
    approved: bool
    feedback: str
    revisions: int
    status: str
    share_status: str
    reply: str


# ==========================================================================
# Cell 11 - Nodes 1-3: load memory, understand the request, update memory
# ==========================================================================
def load_memory(state: PlanState) -> PlanState:
    uid = state["athlete_id"]
    me = ("member", state["session_user"])
    if state["session_user"] == uid:
        sync_profile(uid)
    mems = gateway.read_profile(me, uid)
    print(f"  [load_memory] {len(mems)} private records for {uid}")
    return {"memories": mems, "member": member_details(uid), "plan": "", "status": "", "reply": "",
            "plan_issues": [], "plan_fixes": 0, "progress_report": {}, "share_rid": "", "share_status": "",
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}


UNDERSTAND_PROMPT = """
You read one message from a gym member to their training coach.

Profile already stored: {profile}
Message: {request}

Return JSON only, no code fence:
{{"wants_plan": true, "pain_or_injury": false, "this_week_notes": "",
  "profile_updates": [{{"key": "...", "value": "..."}}]}}

Rules:
- wants_plan: true if they ask for a training plan, a change to one, or how they are progressing.
- pain_or_injury: true if they mention pain, injury, illness or a medical condition.
- this_week_notes: constraints for next week only. Empty if none.
- profile_updates: lasting facts the member stated about themselves, using ONLY these keys:
  goal, training_days, split, equipment, session_minutes, experience_level, exercise_preferences, plan_format.
  If they correct a stored fact, give the new full value. Never put health details here.
"""


def parse_json(text):
    raw = (text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}


def understand_request(state: PlanState) -> PlanState:
    profile = {k: v["value"] for k, v in state.get("memories", {}).items()}
    u = parse_json(llm.invoke(UNDERSTAND_PROMPT.format(profile=json.dumps(profile), request=state["request"])).content)
    u = {"wants_plan": bool(u.get("wants_plan", True)), "pain_or_injury": bool(u.get("pain_or_injury", False)),
         "this_week_notes": str(u.get("this_week_notes", "") or ""), "profile_updates": u.get("profile_updates", []) or []}
    print(f"  [understand_request] wants_plan={u['wants_plan']} pain={u['pain_or_injury']} updates={len(u['profile_updates'])}")
    return {"understanding": u}


def update_memory(state: PlanState) -> PlanState:
    me, uid, log = ("member", state["session_user"]), state["athlete_id"], []
    for upd in state["understanding"]["profile_updates"]:
        k, v = str(upd.get("key", "")).strip(), str(upd.get("value", "")).strip()
        ok, why = check_memory_candidate(k, v)
        res = gateway.write_profile(me, uid, k, v) if ok else f"refused ({why})"
        log.append({"key": k, "value": v, "result": res})
        print(f"  [update_memory] {k} = '{v}' -> {res}")
    if state["understanding"]["pain_or_injury"]:
        print("  [update_memory] health detail mentioned -> not stored (policy)")
    return {"memory_log": log, "memories": gateway.read_profile(me, uid)}


# ==========================================================================
# Cell 12 - Route 1: what kind of request is this?
# ==========================================================================
# > pain -> safety reply | no plan wanted -> acknowledge | no history -> starter week |
# > otherwise -> the Progress Checker. Decided BEFORE any data is read.
def route_request(state: PlanState) -> Literal["safety_response", "acknowledge", "starter_setup", "progress_analyst"]:
    u = state["understanding"]
    if u["pain_or_injury"]:
        route = "safety_response"
    elif not u["wants_plan"]:
        route = "acknowledge"
    else:
        h = get_training_history.invoke({"athlete_id": state["athlete_id"]})
        route = "progress_analyst" if h["has_history"] else "starter_setup"
    print(f"  [route 1] -> {route}")
    return route


def safety_response(state: PlanState) -> PlanState:
    msg = ("You mentioned pain or an injury, so I won't build a training plan around it. Please get it checked "
           "by a physiotherapist or doctor first, then tell me what you are cleared to do and I'll plan from there.")
    print("  [safety_response] fixed safety reply")
    return {"status": "safety_stop", "route": "safety_response", "reply": msg}


def acknowledge(state: PlanState) -> PlanState:
    print("  [acknowledge] profile updated, no plan requested")
    return {"status": "profile_updated", "route": "acknowledge",
            "reply": "Got it, I've updated your profile. Ask me whenever you want next week's plan."}


def starter_setup(state: PlanState) -> PlanState:
    print("  [starter_setup] no history -> beginner starter week (no progress report)")
    u = dict(state["understanding"])
    u["this_week_notes"] = (u.get("this_week_notes", "") + " No training history: build a beginner starter week "
                            "with light weights, 2-3 sets, technique focus.").strip()
    report = {"summary": "No workouts logged yet, so this is a gentle starter week.", "findings": [],
              "adherence": "-", "readiness": None, "zone": None}
    return {"understanding": u, "route": "starter_setup", "progress_report": report,
            "history": {"has_history": False}, "recovery": get_recovery_status.invoke({"athlete_id": state["athlete_id"]})}


# ==========================================================================
# Cell 13 - AGENT 1: the Progress Checker (progress_analyst)
# ==========================================================================
# > Its job: understand how training went. It reads three tools, then the LLM writes a
# > structured ProgressReport. That report is the HAND-OFF to the Plan Coach.
# > It may also propose ONE of the member's steady sessions for the community library;
# > the proposal is saved as pending and the member decides later (approve_share).
ANALYST_PROMPT = """You are the Progress Checker in a two-agent gym coaching team.
Your job is ONLY to analyse how the member's training went. You do not write plans.

Data from the tools:
{data}

Return JSON only, no code fence:
{{"summary": "one friendly sentence",
  "findings": [{{"id": "R1", "topic": "short name", "finding": "what the data shows, with numbers",
                "action": "what the Plan Coach should do"}}],
  "share_suggestion": null}}

Rules:
- 4 to 6 findings. Cover attendance, workload, readiness, and every lift whose target is
  "hold" or "deload", plus the lifts ready to go up. Use only numbers present in the data.
- Readiness yellow -> action: lighter first day, no max attempts. Red -> recovery first day.
- A missed session -> action: do not add it on top of next week.
- share_suggestion: if sharing_allowed is true and steady_session is not null, return
  {{"title": "a clear name", "why": "one sentence"}}; otherwise null.
- Never give medical advice."""


def steady_session(uid):
    """A session the member completed every week (code finds it; the agent decides whether to suggest it)."""
    log = pd.read_csv(LOG_PATH)
    df = log[log["athlete_id"] == uid]
    if df.empty:
        return None
    df = _with_weeks(df)
    best = None
    for (sess, dow), g in df.groupby([df["session"], df["date"].dt.weekday]):
        weeks = g.groupby("week_start")["completed"].all()
        if len(weeks) >= 3 and weeks.all():
            rpe = g["rpe"].mean()
            if best is None or rpe < best[0]:
                best = (rpe, sess, g)
    if not best:
        return None
    _, sess, g = best
    lib = pd.read_csv(LIB_PATH).set_index("exercise")
    exs = g.drop_duplicates("exercise")
    items = [f"{r.exercise} x {int(g[g.exercise == r.exercise]['set_no'].max())}x{int(r.target_reps)}" for r in exs.itertuples()]
    eq = sorted({lib.loc[e, "equipment"] for e in exs["exercise"] if e in lib.index})
    equipment = "dumbbell" if eq and set(eq) <= {"dumbbell", "bodyweight"} else ("bodyweight" if eq == ["bodyweight"] else "machine")
    return {"session": sess, "exercises": items, "minutes": 7 * len(items) + 5, "equipment": equipment}


def progress_analyst(state: PlanState) -> PlanState:
    uid = state["athlete_id"]
    actor = ("progress_analyst", state["session_user"])
    history = get_training_history.invoke({"athlete_id": uid})
    recovery = get_recovery_status.invoke({"athlete_id": uid})
    targets = get_progression_targets.invoke({"athlete_id": uid})
    print(f"  [progress_analyst] tools: get_training_history, get_recovery_status, get_progression_targets "
          f"({len(targets)} lifts)")
    prof = get_profile(uid) or {}
    steady = steady_session(uid)
    existing = {r["title"].lower() for r in gateway.library_all(actor)}
    data = {"profile": {k: v["value"] for k, v in state.get("memories", {}).items()},
            "last_week": {k: history.get(k) for k in ["sessions_planned", "sessions_completed", "missed_sessions",
                                                      "volume_last_week_kg", "volume_change_pct"]},
            "readiness": {k: recovery.get(k) for k in ["readiness", "zone", "guidance"]},
            "targets": targets, "sharing_allowed": bool(prof.get("share_suggestions")),
            "steady_session": steady}
    rep = parse_json(llm.invoke(ANALYST_PROMPT.format(data=json.dumps(data))).content)
    findings = [f for f in rep.get("findings", []) if isinstance(f, dict) and f.get("finding")]
    if not findings:          # invalid model output -> safe report built in code, noted in the trace
        print("  [progress_analyst] model output not usable -> code-built report")
        findings = [{"id": f"R{i + 1}", "topic": t["exercise"], "finding": t["reason"], "action": f"use {t['next_kg']} kg"}
                    for i, t in enumerate(targets[:5])]
    report = {"summary": rep.get("summary") or "Here is how last week went.",
              "adherence": f"{history['sessions_completed']}/{history['sessions_planned']}",
              "volume_change_pct": history.get("volume_change_pct"), "readiness": recovery.get("readiness"),
              "zone": recovery.get("zone"), "findings": findings[:6], "version": 1,
              "written_by": "progress_analyst", "written_at": datetime.now().isoformat(timespec="seconds")}
    share_rid = ""
    sug = rep.get("share_suggestion")
    if sug and steady and data["sharing_allowed"] and str(sug.get("title", "")).lower() not in existing:
        share_rid = gateway.library_propose(actor, uid, {"title": sug.get("title") or f"{steady['session']} session",
                                                         "minutes": steady["minutes"], "equipment": steady["equipment"],
                                                         "level": prof.get("experience_level", "beginner").lower(),
                                                         "exercises": steady["exercises"], "why": sug.get("why", "")})
    print(f"  [handoff] progress_analyst -> planner : ProgressReport v1 ({len(report['findings'])} findings)")
    return {"progress_report": report, "history": history, "recovery": recovery, "route": "progress_analyst",
            "share_rid": share_rid}


# ==========================================================================
# Cell 14 - AGENT 2: the Plan Coach (planner) + code plan check
# ==========================================================================
# > Its job: turn the ProgressReport into next week's plan. It must use every finding,
# > calls its own tools (progression targets, exercise search, community library),
# > and the plan check in code makes sure it followed the tools and the profile.
PLANNER_SYSTEM = """You are the Plan Coach in a two-agent gym coaching team.
The Progress Checker has already analysed the member's training and handed you a ProgressReport.
Write next week's training plan (Monday to Sunday).

Use your tools:
- get_progression_targets for the weight of every lift the member already does.
- search_exercises for any new or replacement exercise; use only exercises it returns.
- search_library for community workouts other members shared; you may add ONE that fits
  as a finisher or a whole session, naming it as "Library: <title> v<version>".

Rules:
- Act on EVERY finding in the ProgressReport and cite its id (R1, R2...) in the table.
- Train only on the member's training days and follow their split. Other days are rest days.
- Respect exercise_preferences. Keep each session within session_minutes (about 6-8 minutes per exercise).
- Never give medical advice or comment on body weight.

Format:
1. One line: "Readiness today: <score>/100 (<zone>) - <what you changed>".
2. A markdown table: Day | Session | Exercises (sets x reps @ kg) | Report
3. One line starting "Why this plan:" that mentions the goal and at least one number."""

plan_tools = [get_progression_targets, search_exercises, search_library]
planner_llm = llm.bind_tools(plan_tools)
MAX_TOOL_ROUNDS = 4
MAX_PLAN_FIXES = 1
DAY_NAMES = [d.lower() for d in DAYS]


def parse_days(text):
    t = (text or "").lower()
    found = [d for d in DAY_NAMES if re.search(rf"\b{d[:3]}[a-z]*\b", t)]
    hit = re.search(r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\s*(?:-|to|through)\s*(mon|tue|wed|thu|fri|sat|sun)", t)
    if hit and len(found) == 2:
        a, b = [d[:3] for d in DAY_NAMES].index(hit.group(1)), [d[:3] for d in DAY_NAMES].index(hit.group(2))
        found = DAY_NAMES[a:b + 1]
    return set(found)


def plan_rows(plan_md):
    rows = []
    for line in (plan_md or "").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [re.sub(r"<br\s*/?>", "; ", c, flags=re.I).strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and not set(cells[0]) <= set("-: ") and cells[0].lower().strip("* ") != "day":
            rows.append(cells + [""] * (4 - len(cells)))
    return rows


def check_plan(plan_md, targets, profile):
    issues, rows = [], plan_rows(plan_md)
    if not rows:
        return ["the plan has no Day | Session | Exercises table"]
    names = sorted((t["exercise"] for t in targets), key=len, reverse=True)
    limit = {t["exercise"]: t["next_kg"] for t in targets}
    allowed = parse_days(profile.get("training_days", ""))
    mins = re.findall(r"\d+", str(profile.get("session_minutes", "")))
    max_ex = int(mins[0]) // 6 if mins else None
    for day, session, exercises, *_ in rows:
        rest = re.search(r"\b(rest|off)\b", f"{session} {exercises}".lower()) or exercises.strip() in ("", "-", "—")
        d = next((x for x in DAY_NAMES if day.lower().strip("* ").startswith(x[:3])), None)
        if allowed and d and not rest and d not in allowed:
            issues.append(f"{d.title()} has a session but is not one of the training days")
        if rest:
            continue
        n_ex = len(re.findall(r"\d+\s*[x×]\s*\d+", exercises))
        if max_ex and n_ex > max_ex:
            issues.append(f"{day.strip('* ')}: {n_ex} exercises will not fit in {mins[0]} minutes (max {max_ex})")
        for item in re.split(r"\s*[;•·\n]\s*", exercises):
            low = item.lower()
            name = next((n for n in names if n in low), None)
            kgs = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*kg", low)]
            if name and kgs and max(kgs) > limit[name] + 0.01:
                issues.append(f"{day.strip('* ')}: {name} at {max(kgs):g} kg is above the target of {limit[name]:g} kg")
    return issues


def quiet_targets(uid):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_progression_targets.invoke({"athlete_id": uid})


def planner(state: PlanState) -> PlanState:
    CURRENT_ACTOR["planner"] = ("planner", state["session_user"])
    msgs, new = state.get("messages", []), []
    if not msgs:
        profile = {k: v["value"] for k, v in state.get("memories", {}).items()}
        brief = (f"Member id: {state['athlete_id']}\n"
                 f"Long-term profile (memory): {json.dumps(profile)}\n"
                 f"Member details (shared with consent): {json.dumps(state.get('member') or {})}\n"
                 f"ProgressReport from the Progress Checker: {json.dumps(state.get('progress_report') or {})}\n"
                 f"Notes for next week only: {state['understanding']['this_week_notes'] or 'none'}\n"
                 f"Member's message: {state['request']}")
        new = [SystemMessage(PLANNER_SYSTEM), HumanMessage(brief)]
        msgs = new
        print("  [planner] received ProgressReport with "
              f"{len((state.get('progress_report') or {}).get('findings', []))} findings")
    rounds = sum(1 for m in msgs if isinstance(m, AIMessage) and m.tool_calls)
    reply = (planner_llm if rounds < MAX_TOOL_ROUNDS else llm).invoke(msgs)
    if reply.tool_calls:
        print(f"  [planner] calls tool(s): {[c['name'] for c in reply.tool_calls]}")
        return {"messages": new + [reply]}
    print("  [planner] plan drafted")
    profile = {k: v["value"] for k, v in state.get("memories", {}).items()}
    issues = check_plan(reply.content, quiet_targets(state["athlete_id"]), profile)
    fixes = state.get("plan_fixes", 0)
    if not issues:
        print("  [plan_check] passed: loads, training days and session length")
        return {"messages": new + [reply], "plan": reply.content, "plan_issues": []}
    for i in issues:
        print(f"  [plan_check] {i}")
    if fixes < MAX_PLAN_FIXES:
        fix = HumanMessage("Your plan breaks these rules:\n- " + "\n- ".join(issues) + "\nFix only these and return the full plan.")
        return {"messages": new + [reply, fix], "plan": reply.content, "plan_issues": issues, "plan_fixes": fixes + 1}
    return {"messages": new + [reply], "plan": reply.content, "plan_issues": issues}


def route_after_planner(state: PlanState) -> Literal["tools", "planner", "human_approval"]:
    """ROUTE 2: call tools, fix a failed plan check, or ask the member."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    if isinstance(last, HumanMessage):
        return "planner"
    return "human_approval"


# ==========================================================================
# Cell 15 - Human approval, revise, save, and the share approval
# ==========================================================================
CALENDAR = {}            # SIMULATION of the member's calendar
MAX_REVISIONS = 2


def human_approval(state: PlanState) -> PlanState:
    ans = interrupt({"kind": "plan", "question": "Save this plan to your calendar?", "athlete_id": state["athlete_id"],
                     "plan": state["plan"], "plan_check": state.get("plan_issues") or "passed",
                     "report": state.get("progress_report") or {}})
    approved, feedback = bool(ans.get("approved")), str(ans.get("feedback", "") or "")
    print(f"  [human_approval] approved={approved}" + (f" feedback='{feedback}'" if feedback else ""))
    return {"approved": approved, "feedback": feedback}


def route_after_approval(state: PlanState) -> Literal["save_plan", "revise", "not_saved"]:
    """ROUTE 3: the member's decision."""
    if state["approved"]:
        return "save_plan"
    if state.get("feedback") and state.get("revisions", 0) < MAX_REVISIONS:
        return "revise"
    return "not_saved"


def revise(state: PlanState) -> PlanState:
    n = state.get("revisions", 0) + 1
    print(f"  [revise] revision {n}: feedback to the Plan Coach (the ProgressReport is reused, not recomputed)")
    return {"revisions": n, "plan_fixes": 0, "plan_issues": [],
            "messages": [HumanMessage(f"The member rejected this plan. Feedback: {state['feedback']}. "
                                      "Revise it, keep the same format and still act on the ProgressReport.")]}


def save_plan(state: PlanState) -> PlanState:
    CALENDAR[state["athlete_id"]] = {"saved_at": datetime.now().isoformat(timespec="seconds"), "plan": state["plan"]}
    store.put(("calendar", state["athlete_id"]), "next_week", CALENDAR[state["athlete_id"]])
    print(f"  [save_plan] SIMULATION: plan written to {state['athlete_id']}'s calendar")
    return {"status": "saved"}


def not_saved(state: PlanState) -> PlanState:
    print("  [not_saved] nothing written to the calendar")
    return {"status": "not_saved"}


def route_after_plan(state: PlanState) -> Literal["approve_share", "finalize"]:
    return "approve_share" if state.get("share_rid") else "finalize"


def approve_share(state: PlanState) -> PlanState:
    """SECOND human approval: nothing reaches the community library without the author's yes."""
    rec = gateway._get(state["share_rid"])
    ans = interrupt({"kind": "share", "question": "Share this workout with the community?", "record": rec,
                     "shared_fields": ["title", "minutes", "equipment", "exercises (sets x reps, no weights)"],
                     "private_fields": ["goal", "body details", "readiness", "weights you lifted"]})
    res = gateway.library_decide(("member", state["session_user"]), state["share_rid"], bool(ans.get("approved")))
    print(f"  [approve_share] {res}")
    return {"share_status": res}


def finalize(state: PlanState) -> PlanState:
    print(f"  [finalize] status = {state.get('status')}" + (f", share = {state.get('share_status')}" if state.get("share_rid") else ""))
    return {}


# ==========================================================================
# Cell 16 - Build the graph with the SQLite checkpointer
# ==========================================================================
# > 15 nodes, 4 conditional routes (request type, planner loop, member decision, share needed?). Conversations are checkpointed in fitplan_checkpoints.db.
def build_app(checkpoint_path=CHECKPOINT_DB, the_store=None):
    b = StateGraph(PlanState)
    for name, fn in [("load_memory", load_memory), ("understand_request", understand_request),
                     ("update_memory", update_memory), ("safety_response", safety_response),
                     ("acknowledge", acknowledge), ("starter_setup", starter_setup),
                     ("progress_analyst", progress_analyst), ("planner", planner), ("tools", ToolNode(plan_tools)),
                     ("human_approval", human_approval), ("revise", revise), ("save_plan", save_plan),
                     ("not_saved", not_saved), ("approve_share", approve_share), ("finalize", finalize)]:
        b.add_node(name, fn)
    b.add_edge(START, "load_memory")
    b.add_edge("load_memory", "understand_request")
    b.add_edge("understand_request", "update_memory")
    b.add_conditional_edges("update_memory", route_request,
                            ["safety_response", "acknowledge", "starter_setup", "progress_analyst"])
    b.add_edge("starter_setup", "planner")
    b.add_edge("progress_analyst", "planner")                 # the HAND-OFF
    b.add_conditional_edges("planner", route_after_planner, ["tools", "planner", "human_approval"])
    b.add_edge("tools", "planner")
    b.add_conditional_edges("human_approval", route_after_approval, ["save_plan", "revise", "not_saved"])
    b.add_edge("revise", "planner")
    b.add_conditional_edges("save_plan", route_after_plan, ["approve_share", "finalize"])
    b.add_conditional_edges("not_saved", route_after_plan, ["approve_share", "finalize"])
    b.add_edge("approve_share", "finalize")
    for n in ["safety_response", "acknowledge"]:
        b.add_edge(n, "finalize")
    b.add_edge("finalize", END)
    saver = SqliteSaver(sqlite3.connect(checkpoint_path, check_same_thread=False))
    return b.compile(checkpointer=saver, store=the_store or store)


app = build_app()
print("Graph compiled:", len(app.get_graph().nodes) - 2, "nodes; checkpoints in", CHECKPOINT_DB)


# ==========================================================================
# Cell 17 - Test members and test library records (created for the tests, removed after)
# ==========================================================================
# > There are no demo accounts. The tests create two TEST members with invented training
# > data and four library records made through the SAME gateway rules the app uses:
# > proposed -> approved -> corrected (v2), one flagged out of date, one waiting for approval.
# > remove_test_members() deletes all of it again (Cell 22), so only real members remain.
TEST_LIB = ["test-lib-001", "test-lib-002", "test-lib-003", "test-lib-004"]


def seed_test_library():
    if gateway._get("test-lib-001"):
        return "test records already there"
    A, B = "test-member-A", "test-member-B"
    with contextlib.redirect_stdout(io.StringIO()):
        r1 = gateway.library_propose(("member", B), B, {"record_id": "test-lib-001", "title": "3-day full-body starter (test)",
             "minutes": 40, "equipment": "machine", "level": "beginner",
             "exercises": ["goblet squat x 3x10", "dumbbell bench press x 3x10", "lat pulldown x 3x10", "plank x 3x30s"],
             "why": "member B's own beginner routine"})
        gateway.library_decide(("member", B), r1, True)
        r2 = gateway.library_propose(("progress_analyst", A), A, {"record_id": "test-lib-002", "title": "Dumbbell upper circuit (test)",
             "minutes": 45, "equipment": "dumbbell", "level": "intermediate",
             "exercises": ["dumbbell bench press x 4x10", "one-arm dumbbell row x 4x10",
                           "seated dumbbell shoulder press x 4x15", "hammer curl x 4x12"],
             "why": "member A finished it every week while travelling"})
        gateway.library_decide(("member", A), r2, True)
        gateway.library_update(("member", A), r2, {"exercises": ["dumbbell bench press x 4x10", "one-arm dumbbell row x 4x10",
                               "seated dumbbell shoulder press x 4x10", "hammer curl x 4x12"]},
                               "shoulder press 15 -> 10 reps, rest 60 s between rounds")
        r3 = gateway.library_propose(("member", A), A, {"record_id": "test-lib-003", "title": "Leg day with smith machine squat (test)",
             "minutes": 50, "equipment": "machine", "level": "intermediate",
             "exercises": ["smith machine squat x 4x8", "leg press x 3x12", "seated leg curl x 3x12"],
             "why": "member A's older leg day"})
        gateway.library_decide(("member", A), r3, True)
        gateway.review_library()            # smith machine squat is not in the exercise library -> out of date
        gateway.library_propose(("progress_analyst", A), A, {"record_id": "test-lib-004", "title": "Hotel-room bodyweight session (test)",
             "minutes": 30, "equipment": "bodyweight", "level": "beginner",
             "exercises": ["push-up x 3x12", "split squat x 3x10", "glute bridge x 3x15", "plank x 3x40s"],
             "why": "used twice while travelling, finished both times"})
    return "4 test records (1 corrected to v2, 1 out of date, 1 waiting for approval)"


def remove_test_members():
    """Delete every trace of the test members: accounts, memory, library records, log rows, conversations."""
    ids = TEST_IDS + ["test-member-C"]
    with db() as conn:
        for t in ["password_resets", "profiles", "users"]:
            conn.execute(f"DELETE FROM {t} WHERE user_id LIKE 'test-member-%'")
        conn.execute("DELETE FROM emails WHERE to_email LIKE '%@fitplan.test'")
    for uid in ids:
        for item in store.search(("athletes", uid, "profile"), limit=1000):
            store.delete(("athletes", uid, "profile"), item.key)
        store.delete(("calendar", uid), "next_week")
    for item in store.search(("shared", "library"), limit=1000):
        if is_test_member(item.value.get("author")):
            store.delete(("shared", "library"), item.key)
    for item in store.search(("audit", "log"), limit=100000):
        if is_test_member(item.value.get("acting_for")) or str(item.value.get("target", "")).startswith("test-lib"):
            store.delete(("audit", "log"), item.key)
    with sqlite3.connect(CHECKPOINT_DB) as c:
        try:
            tids = [r[0] for r in c.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE 'test-%'")]
        except sqlite3.OperationalError:
            tids = []
    for t in tids:
        app.checkpointer.delete_thread(t)
    for path in [LOG_PATH, METRICS_PATH]:
        df = pd.read_csv(path)
        df[~df["athlete_id"].astype(str).str.startswith("test-member-")].to_csv(path, index=False)
    return f"removed test members, their records and {len(tids)} test conversations"


# ==========================================================================
# Cell 17b - Restart check helpers
# ==========================================================================
# > memory_counts() reads the files from THIS process; fresh_process_counts() starts a
# > brand-new Python process that opens only the saved files. Used by test P2-T8 and the app.
import sys
import subprocess

RESTART_SCRIPT = r"""
import json, os, sqlite3, sys
from langgraph.store.sqlite import SqliteStore
from langgraph.checkpoint.sqlite import SqliteSaver
mem, ck, tid, me = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
st = SqliteStore(sqlite3.connect(mem, isolation_level=None))
out = {"pid": os.getpid()}
spaces = [ns for ns in st.list_namespaces(prefix=("athletes",), limit=10000) if ns[-1] == "profile"]
out["members"] = len(spaces)
out["private_items"] = sum(len(st.search(ns, limit=1000)) for ns in spaces)
out["me"] = len(st.search(("athletes", me, "profile"), limit=1000))
lib = [i.value for i in st.search(("shared", "library"), limit=1000)]
out["library"] = len(lib)
out["versions"] = sorted(f"{r['record_id']} v{r['version']} {r['status']}" for r in lib)
out["paused"] = None
if tid:
    t = SqliteSaver(sqlite3.connect(ck)).get_tuple({"configurable": {"thread_id": tid}})
    out["paused"] = bool(t and any(w[1] == "__interrupt__" for w in (t.pending_writes or [])))
print(json.dumps(out))
"""
RESTART_KEYS = ["me", "members", "private_items", "library", "versions", "paused"]


def memory_counts(tid, uid):
    """What THIS process sees in the saved files."""
    out = {"pid": os.getpid()}
    spaces = [ns for ns in store.list_namespaces(prefix=("athletes",), limit=10000) if ns[-1] == "profile"]
    out["members"] = len(spaces)
    out["private_items"] = sum(len(store.search(ns, limit=1000)) for ns in spaces)
    out["me"] = len(store.search(("athletes", uid, "profile"), limit=1000))
    lib = [i.value for i in store.search(("shared", "library"), limit=1000)]
    out["library"] = len(lib)
    out["versions"] = sorted(f"{r['record_id']} v{r['version']} {r['status']}" for r in lib)
    out["paused"] = bool(app.get_state({"configurable": {"thread_id": tid}}).next) if tid else None
    return out


def fresh_process_counts(tid, uid):
    """Start a NEW Python process that opens only the saved files and reports what it finds."""
    r = subprocess.run([sys.executable, "-c", RESTART_SCRIPT, MEMORY_DB, CHECKPOINT_DB, tid or "", uid or ""],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])
    return json.loads(r.stdout.strip().splitlines()[-1])


# ==========================================================================
# Cell 30 - Interface: look and feel (Ocean Blue)
# ==========================================================================
# > A light, friendly theme. Plain words on screen; technical names only behind
# > "Behind the scenes" switches for the instructor.
import gradio as gr

THEME = gr.themes.Base(primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    body_background_fill="#F7F9FC", body_background_fill_dark="#F7F9FC",
    body_text_color="#0F172A", body_text_color_dark="#0F172A",
    body_text_color_subdued="#5B6B82", body_text_color_subdued_dark="#5B6B82",
    background_fill_primary="#FFFFFF", background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#F7F9FC", background_fill_secondary_dark="#F7F9FC",
    block_background_fill="#FFFFFF", block_background_fill_dark="#FFFFFF",
    block_border_color="#E2E8F0", block_border_color_dark="#E2E8F0",
    border_color_primary="#CBD5E1", border_color_primary_dark="#CBD5E1",
    block_label_text_color="#0F172A", block_label_text_color_dark="#0F172A",
    block_title_text_color="#0F172A", block_title_text_color_dark="#0F172A",
    input_background_fill="#FFFFFF", input_background_fill_dark="#FFFFFF",
    input_border_color="#CBD5E1", input_border_color_dark="#CBD5E1",
    button_primary_background_fill="#2563EB", button_primary_background_fill_dark="#2563EB",
    button_primary_background_fill_hover="#1D4ED8", button_primary_background_fill_hover_dark="#1D4ED8",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_text_color="#0F172A", button_secondary_text_color_dark="#0F172A",
    button_secondary_border_color="#CBD5E1", button_secondary_border_color_dark="#CBD5E1",
    checkbox_background_color_selected="#2563EB", checkbox_background_color_selected_dark="#2563EB",
    slider_color="#2563EB", slider_color_dark="#2563EB",
    color_accent="#2563EB", color_accent_soft="#EFF6FF", color_accent_soft_dark="#EFF6FF",
)

CSS = """
.gradio-container {max-width: 1280px !important; margin: 0 auto !important}
footer {display: none !important}
.topbar {display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
         background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:12px 18px}
.logo {display:flex; align-items:center; gap:10px; font-size:20px; font-weight:800; color:#0F172A}
.logo .mark {width:36px; height:36px; border-radius:10px; background:#2563EB; color:#fff;
             display:flex; align-items:center; justify-content:center}
.signed {font-size:14px; color:#5B6B82} .signed b {color:#0F172A}
/* sign-in page */
.brand {background:#2563EB !important; border-radius:18px !important; padding:40px 44px !important; color:#fff}
.brand * {color:#fff}
.brand h1 {font-size:34px; line-height:1.15; margin:18px 0 12px; font-weight:800}
.brand p {color:#DCE7FE !important; font-size:16px; line-height:1.55}
.brand ul {list-style:none; padding-left:0} .brand li {margin:8px 0; font-size:15.5px}
.team {background:#1D4ED8; border-radius:14px; padding:16px 18px; margin-top:26px}
.team .t {font-size:12.5px; font-weight:700; letter-spacing:.08em; color:#DCE7FE !important}
.agent {display:flex; align-items:center; gap:12px; margin-top:12px}
.agent .av {width:38px; height:38px; border-radius:50%; background:#fff; display:flex; align-items:center; justify-content:center}
.agent .nm {font-weight:700} .agent .sb {font-size:13.5px; color:#DCE7FE !important}
.agent .ok {margin-left:auto; font-size:13px; font-weight:600}
.agent .ok i {display:inline-block; width:8px; height:8px; border-radius:50%; background:#86EFAC; margin-right:6px}
.formcol {padding:12px 28px !important}
.h2 {font-size:30px; font-weight:800; color:#0F172A; margin:6px 0 2px}
.sub {color:#5B6B82; font-size:15px; margin-bottom:6px}
.hintbox {background:#ECFDF5; border:1px solid #A7E3CF; color:#0F5132; border-radius:10px; padding:11px 14px; font-size:14px}
.okbox {background:#EFF6FF; border:1px solid #BFD3FB; color:#1E3A8A; border-radius:10px; padding:11px 14px; font-size:14px}
.errbox {background:#FEF2F2; border:1px solid #F8C9C9; color:#991B1B; border-radius:10px; padding:11px 14px; font-size:14px}
.warnbox {background:#FFFBEB; border:1px solid #F5D48A; color:#7A4B00; border-radius:12px; padding:13px 16px; font-size:14px; line-height:1.5}
.tip {text-align:center; font-size:13px; color:#5B6B82}
.pw {position:relative}
.pw .eye {position:absolute !important; right:10px; top:38px; z-index:5; min-width:0 !important; width:auto !important;
          height:34px !important; padding:0 10px !important; font-size:13px !important; font-weight:600 !important;
          color:#2563EB !important; background:#FFFFFF !important; border:0 !important; box-shadow:none !important}
.pw input {padding-right:80px !important}
.rules {display:flex; flex-wrap:wrap; gap:6px 16px; font-size:13px; margin-top:-4px}
.rules span.ok {color:#15803D} .rules span.no {color:#5B6B82}
.linkbtn, button.linkbtn {align-self:flex-end; width:auto !important; min-width:0 !important; background:none !important;
          border:0 !important; box-shadow:none !important; color:#2563EB !important; font-weight:600 !important; padding:2px 4px !important}
.big button, button.big {height:48px !important; font-size:15px !important; font-weight:700 !important}
/* profile wizard */
.steps {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:10px; margin:8px 0 4px}
.steps div {display:flex; flex-direction:column; gap:8px; font-size:13px; font-weight:600; color:#5B6B82}
.steps div span {height:6px; border-radius:999px; background:#E2E8F0}
.steps div.on span {background:#2563EB} .steps div.cur {color:#0F172A}
.card {background:#FFFFFF !important; border:1px solid #E2E8F0 !important; border-radius:16px !important; padding:22px !important}
.choice label {border:1px solid #CBD5E1 !important; border-radius:10px !important; padding:10px 14px !important;
               background:#FFFFFF !important; min-height:44px; box-sizing:border-box}
.choice label.selected {border:2px solid #2563EB !important; background:#EFF6FF !important}
.summary {display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:6px 20px; font-size:14px}
.summary span {color:#5B6B82}
.done {display:flex; flex-direction:column; gap:12px}
.done .tick {width:56px; height:56px; border-radius:50%; background:#DCFCE7; color:#16A34A; display:flex;
             align-items:center; justify-content:center; font-size:28px; font-weight:800}
"""

LOGO = ("<div class='logo'><div class='mark'><svg width='22' height='22' viewBox='0 0 24 24' fill='none' "
        "stroke='currentColor' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
        "<path d='M6.5 6.5v11M17.5 6.5v11M3 9.5v5M21 9.5v5M6.5 12h11'/></svg></div>FitPlan</div>")


def esc(t):
    return html.escape(str(t))


def topbar(uid=None):
    who = ""
    if uid:
        prof, acct = get_profile(uid) or {}, get_user(uid) or {}
        who = f"<div class='signed'>Signed in as <b>{esc(prof.get('display_name') or acct.get('name') or uid)}</b></div>"
    return f"<div class='topbar'>{LOGO}{who}</div>"


BRAND = """
<div>
  <h1>Your personal training team, powered by AI.</h1>
  <p>Tell FitPlan what you want to achieve. It looks at your workouts and how rested you are,
  then plans your week with you.</p>
  <ul>
    <li>&#10003;&nbsp; Plans built from what you actually lifted</li>
    <li>&#10003;&nbsp; Lighter days when you need more rest</li>
    <li>&#10003;&nbsp; Nothing is saved until you say yes</li>
  </ul>
  <div class='team'><div class='t'>YOUR AI TEAM</div>
    <div class='agent'><div class='av' style='color:#0D9488'>&#8599;</div><div><div class='nm'>Progress Checker</div>
      <div class='sb'>Reviews your last week</div></div><div class='ok'><i></i>Ready</div></div>
    <div class='agent'><div class='av' style='color:#2563EB'>&#9638;</div><div><div class='nm'>Plan Coach</div>
      <div class='sb'>Builds your next 7 days</div></div><div class='ok'><i></i>Ready</div></div>
  </div>
</div>"""

# Eye button: switches a password box between hidden and visible in the browser
# (no server call, so the password never leaves the page for this).
EYE_JS = """(...args) => {
  const ids = %s;
  let show = false;
  ids.forEach((id) => { const i = document.querySelector('#' + id + ' input');
    if (i) { show = i.type === 'password'; i.type = show ? 'text' : 'password'; } });
  document.querySelectorAll(%s).forEach((b) => { b.innerText = show ? 'Hide' : 'Show';
    b.setAttribute('aria-label', show ? 'Hide password' : 'Show password'); });
  return [];
}"""


def eye_js(input_ids, button_selector):
    return EYE_JS % (json.dumps(input_ids), json.dumps(button_selector))


def rules_html(pw, pw2):
    r = password_problems(pw, pw2)
    item = lambda ok, text: f"<span class='{'ok' if ok else 'no'}'>{'&#10003;' if ok else '&#8226;'} {text}</span>"
    return ("<div class='rules'>" + item(r["length"], "At least 8 characters") +
            item(r["mix"], "A letter and a number") + item(r["match"], "Both passwords match") + "</div>")


def box(kind, text):
    return f"<div class='{kind}' role='{'alert' if kind == 'errbox' else 'status'}'>{text}</div>" if text else ""


def steps_html(n):
    labels = ["1. About you", "2. Your training", "3. Privacy"]
    return "<div class='steps' aria-label='Progress'>" + "".join(
        f"<div class='{'on ' if i + 1 <= n else ''}{'cur' if i + 1 == n else ''}'><span></span>{l}</div>"
        for i, l in enumerate(labels)) + "</div>"


GOALS = [("Build muscle  ·  grow size and shape", "Build muscle"),
         ("Get stronger  ·  lift heavier weights", "Get stronger"),
         ("Lose fat  ·  keep strength while leaning out", "Lose fat"),
         ("General fitness  ·  feel fit and healthy", "General fitness"),
         ("Improve endurance  ·  last longer, recover faster", "Improve endurance")]
LEVELS = [("Beginner  ·  new, or under 6 months", "Beginner"),
          ("Intermediate  ·  6 months to 2 years", "Intermediate"),
          ("Advanced  ·  more than 2 years", "Advanced")]
EQUIPMENT = [("Full gym  ·  barbells, machines, cables", "Full gym"),
             ("Home, dumbbells  ·  dumbbells and a bench", "Home, dumbbells"),
             ("No equipment  ·  bodyweight only", "No equipment")]
SPLITS = ["Let the coach decide", "Full body", "Upper and lower", "Push, pull, legs"]
print("Interface theme ready.")


# ==========================================================================
# Cell 31 - Interface: dashboard pieces (your AI team at work)
# ==========================================================================
# > The dashboard streams the graph step by step (app.stream), so the member SEES the
# > Progress Checker work, the report being handed over, and the Plan Coach work.
DASH_CSS = """
.greet {display:flex; flex-wrap:wrap; gap:14px; align-items:stretch}
.greet .hello {flex:2 1 380px} .greet .hello h1 {margin:0; font-size:28px; font-weight:800}
.greet .hello p {margin:6px 0 0; color:#5B6B82}
.tile {flex:1 1 220px; background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:14px 16px;
       display:flex; align-items:center; gap:14px}
.ring {width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center;
       font-weight:800; font-size:18px; flex-shrink:0; border:6px solid #E2E8F0}
.tile .k {font-weight:700} .tile .s {color:#5B6B82; font-size:13.5px}
.team2 {display:flex; flex-wrap:wrap; gap:12px; align-items:stretch}
.acard {flex:1 1 320px; background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:18px; display:flex; flex-direction:column; gap:10px}
.acard.work {border:2px solid #2563EB}
.acard .head {display:flex; align-items:center; gap:12px}
.acard .ic {width:46px; height:46px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; flex-shrink:0}
.acard .nm {font-weight:800; font-size:17px} .acard .rl {font-size:13.5px; color:#5B6B82}
.acard .st {margin-left:auto; font-size:13px; font-weight:700; border-radius:999px; padding:5px 10px}
.acard .now {background:#F7F9FC; border-radius:10px; padding:11px 13px; font-size:14px; color:#334155}
.st.ready {background:#F1F5F9; color:#475569} .st.working {background:#DBEAFE; color:#1E40AF}
.st.done {background:#DCFCE7; color:#166534} .st.waiting {background:#FEF3C7; color:#92400E}
@keyframes fp-pulse {0%{box-shadow:0 0 0 0 rgba(37,99,235,.45)} 70%{box-shadow:0 0 0 12px rgba(37,99,235,0)} 100%{box-shadow:0 0 0 0 rgba(37,99,235,0)}}
@keyframes fp-blink {0%,80%,100%{opacity:.25} 40%{opacity:1}}
.acard.work .ic {animation:fp-pulse 1.4s infinite}
.dots i {display:inline-block; width:6px; height:6px; border-radius:50%; background:#2563EB; margin-left:3px; animation:fp-blink 1.2s infinite}
.dots i:nth-child(2){animation-delay:.2s} .dots i:nth-child(3){animation-delay:.4s}
.hand {flex:0 1 140px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px; font-size:13px; font-weight:700; text-align:center}
.panel {background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:18px 20px}
.panel h3 {margin:0 0 10px; font-size:18px}
.feed {list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:11px}
.feed li {display:flex; gap:11px; align-items:flex-start}
.feed .m {width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:800; flex-shrink:0}
.feed .t {font-size:14px; font-weight:600} .feed .w {font-size:12.5px; color:#5B6B82}
.feed .x {font-family:ui-monospace,monospace; font-size:12px; color:#1E3A8A; margin-top:2px}
.rep li {margin:8px 0; font-size:14px; line-height:1.45}
.rep .tag {font-size:12.5px; font-weight:700; color:#0F766E; background:#F0FDFA; border-radius:999px; padding:4px 10px}
.days {display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:10px}
.dcard {border:1px solid #E2E8F0; border-radius:12px; padding:12px 14px; background:#fff}
.dcard.rest {background:#F7F9FC; border-color:#EEF2F6}
.dcard .d {display:flex; justify-content:space-between; font-weight:700}
.dcard .d span {font-size:12.5px; color:#1D4ED8} .dcard .ss {font-weight:600; margin-top:4px}
.dcard .ex {font-size:13.5px; color:#5B6B82; line-height:1.55; margin-top:4px}
.why {background:#F0FDFA; border:1px solid #99E2D6; border-radius:12px; padding:12px 14px; font-size:14px; color:#134E4A; margin-top:10px}
.pstatus {font-size:13px; font-weight:700; border-radius:999px; padding:6px 12px}
.pstatus.wait {background:#FEF3C7; color:#92400E} .pstatus.saved {background:#DCFCE7; color:#166534} .pstatus.no {background:#FEE2E2; color:#991B1B}
.sharebox {background:#FFFBEB; border:2px solid #F5D48A; border-radius:16px; padding:18px 20px}
.sharebox .cols {display:flex; flex-wrap:wrap; gap:12px; margin-top:10px}
.sharebox .cols div {flex:1 1 260px; background:#fff; border-radius:12px; padding:12px 14px; font-size:14px; line-height:1.55}
.nav button {font-weight:600 !important}
.nav button.on {color:#2563EB !important; border-color:#2563EB !important; background:#EFF6FF !important}
"""

AGENT_TEXT = {
    "idle": ("Waiting for your request", "Waiting for your request"),
}


def greet_html(uid):
    prof = get_profile(uid) or {}
    r = get_recovery_status.invoke({"athlete_id": uid})
    h = get_training_history.invoke({"athlete_id": uid})
    hour = datetime.now().hour
    hello = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    if r.get("has_data"):
        col = {"green": "#16A34A", "yellow": "#D97706", "red": "#DC2626"}[r["zone"]]
        words = {"green": "Well rested. Train as planned.", "yellow": "A bit tired. Take it easier on day one.",
                 "red": "Tired today. Start with recovery."}[r["zone"]]
        ring = f"<div class='ring' style='border-color:{col}'>{r['readiness']}</div>"
    else:
        ring, words = "<div class='ring'>--</div>", "No check-in yet."
    last = f"{h['sessions_completed']} of {h['sessions_planned']}" if h.get("has_history") else "—"
    saved = "Yes" if store.get(("calendar", uid), "next_week") else "Not yet"
    return (f"<div class='greet'><div class='hello'><h1>{hello}, {esc(prof.get('display_name', ''))}</h1>"
            "<p>Your AI team is ready. Ask for next week's plan and watch them work it out together.</p></div>"
            f"<div class='tile'>{ring}<div><div class='k'>Rest score today</div><div class='s'>{words}</div></div></div>"
            f"<div class='tile'><div><div class='s'>Last week</div><div class='k' style='font-size:20px'>{last}</div>"
            f"<div class='s'>Plan saved this week: <b>{saved}</b></div></div></div></div>")


def team_html(checker="ready", coach="ready", checker_now="Waiting for your request",
              coach_now="Waiting for your request", handed=False):
    lab = {"ready": "Ready", "working": "Working", "done": "Done", "waiting": "Waiting"}
    dots = "<span class='dots'><i></i><i></i><i></i></span>"

    def card(state, name, role, now, icon, bg, fg):
        return (f"<div class='acard {'work' if state == 'working' else ''}'><div class='head'>"
                f"<div class='ic' style='background:{bg};color:{fg}'>{icon}</div>"
                f"<div><div class='nm'>{name}</div><div class='rl'>{role}</div></div>"
                f"<span class='st {state}'>{lab[state]}</span></div>"
                f"<div class='now' role='status'>{esc(now)}{dots if state == 'working' else ''}</div></div>")
    arrow = "#2563EB" if handed else "#CBD5E1"
    return ("<div class='team2'>" +
            card(checker, "Progress Checker", "Looks at how your training went", checker_now, "&#8599;", "#F0FDFA", "#0D9488") +
            f"<div class='hand' style='color:{arrow}'><svg width='44' height='24' viewBox='0 0 44 24' fill='none' "
            f"stroke='{arrow}' stroke-width='2.4' stroke-linecap='round' aria-hidden='true'><path d='M2 12h38M30 3l10 9-10 9'/></svg>"
            f"{'Report handed over' if handed else 'Hand-off'}</div>" +
            card(coach, "Plan Coach", "Turns the report into your week", coach_now, "&#9638;", "#EFF6FF", "#2563EB") +
            "</div>")


def feed_html(feed, tech):
    if not feed:
        return "<div class='panel'><h3>What your AI team did</h3><div class='s' style='color:#5B6B82'>Nothing yet. Ask for a plan and each step appears here.</div></div>"
    items = ""
    for f in feed:
        style = {"done": "background:#DCFCE7;color:#166534", "now": "background:#DBEAFE;color:#1E40AF",
                 "wait": "background:#FEF3C7;color:#92400E", "stop": "background:#FEE2E2;color:#991B1B"}[f["kind"]]
        mark = {"done": "&#10003;", "now": "&#8230;", "wait": "!", "stop": "&#10005;"}[f["kind"]]
        x = f"<div class='x'>{esc(f['tech'])}</div>" if tech and f.get("tech") else ""
        items += (f"<li><span class='m' style='{style}'>{mark}</span><div><div class='t'>{esc(f['text'])}</div>"
                  f"<div class='w'>{esc(f['who'])}</div>{x}</div></li>")
    return f"<div class='panel'><h3>What your AI team did</h3><ol class='feed'>{items}</ol></div>"


def report_html(rep):
    if not rep:
        return ("<div class='panel rep'><h3>Weekly progress report</h3><div style='color:#5B6B82;font-size:14px'>"
                "The Progress Checker writes this report, then hands it to the Plan Coach.</div></div>")
    items = "".join(f"<li><b>{esc(f.get('topic', ''))}:</b> {esc(f.get('finding', ''))} &rarr; <i>{esc(f.get('action', ''))}</i> "
                    f"<span style='color:#5B6B82;font-size:12px'>({esc(f.get('id', ''))})</span></li>"
                    for f in rep.get("findings", []))
    return (f"<div class='panel rep'><div style='display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap'>"
            f"<h3>Weekly progress report</h3><span class='tag'>From Progress Checker &rarr; Plan Coach</span></div>"
            f"<div style='font-size:14px;color:#334155'>{esc(rep.get('summary', ''))}</div><ul style='padding-left:18px'>{items}</ul></div>")


def plan_html(plan_md, status, feedback=""):
    if not plan_md:
        return ""
    rows = plan_rows(plan_md)
    why = next((l.split(":", 1)[-1].strip() for l in plan_md.splitlines() if l.lower().strip("* ").startswith("why this plan")), "")
    ready = next((l.strip("*# ") for l in plan_md.splitlines() if l.lower().strip("*# ").startswith("readiness")), "")
    cards = ""
    for day, sess, ex, rep in rows:
        rest = bool(re.search(r"\brest\b", sess.lower())) or ex.strip() in ("", "-")
        items = "<br>".join(esc(p.strip()) for p in re.split(r"\s*;\s*", ex) if p.strip() and p.strip() != "-")
        cards += (f"<div class='dcard {'rest' if rest else ''}'><div class='d'>{esc(day)}<span>{'' if rest else esc(rep)}</span></div>"
                  f"<div class='ss'>{esc(sess)}</div><div class='ex'>{items or 'Recover, walk, stretch'}</div></div>")
    if not rows:
        cards = f"<div style='white-space:pre-wrap'>{esc(plan_md)}</div>"
    st = {"pending": ("wait", "Waiting for your OK"), "saved": ("saved", "Saved to your calendar"),
          "not_saved": ("no", "Not saved")}.get(status, ("wait", "Waiting for your OK"))
    fb = (f"<div class='okbox' style='margin-bottom:10px'>Updated with your request: &ldquo;{esc(feedback)}&rdquo;</div>"
          if feedback else "")
    return (f"<div class='panel' style='border:2px solid {'#16A34A' if status == 'saved' else '#F5D48A'}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px'>"
            f"<h3 style='margin:0'>Your plan for next week</h3><span class='pstatus {st[0]}'>{st[1]}</span></div>"
            f"<div style='color:#B45309;font-size:13.5px;margin-bottom:10px'>{esc(ready)}</div>{fb}<div class='days'>{cards}</div>"
            + (f"<div class='why'><b>Why this plan:</b> {esc(why)}</div>" if why else "") + "</div>")


def share_html(rec):
    if not rec:
        return ""
    ex = ", ".join(e.replace(" x ", " ") for e in rec["exercises"])
    return ("<div class='sharebox'><div style='font-weight:800;font-size:18px'>Share your &ldquo;" + esc(rec["title"]) +
            "&rdquo; with the community?</div><div style='color:#7A4B00;font-size:14px;margin-top:4px'>Suggested by your "
            "Progress Checker: " + esc(rec.get("why") or "you completed it every week.") + "</div><div class='cols'>"
            f"<div style='border:1px solid #A7E3CF'><b style='color:#166534'>Others will see</b><br>{esc(rec['title'])}, "
            f"about {rec['minutes']} minutes: {esc(ex)}. Shared by you.</div>"
            "<div style='border:1px solid #F3B4B4'><b style='color:#991B1B'>Stays private</b><br>Your goal, body details, "
            "rest scores and the weights you lifted.</div></div></div>")


# Friendly words for each step of the graph (the "Behind the scenes" switch shows the technical name)
STEP_WORDS = {
    "load_memory": ("Loaded what your coach remembers about you", "FitPlan"),
    "understand_request": ("Understood your request", "FitPlan"),
    "update_memory": ("Updated your profile where needed", "FitPlan"),
    "progress_analyst": ("Read your workouts, rest score and every lift, then wrote your progress report", "Progress Checker"),
    "planner": ("Planned your week from the report", "Plan Coach"),
    "tools": ("Looked up your targets and the community library", "Plan Coach"),
    "safety_response": ("Paused planning because you mentioned pain", "Safety"),
    "acknowledge": ("Saved your update. No plan needed", "FitPlan"),
    "starter_setup": ("No workouts yet, so a gentle starter week", "Plan Coach"),
    "revise": ("Reworking the plan with your request", "Plan Coach"),
    "human_approval": ("Recorded your decision", "You"),
    "save_plan": ("Saved the plan to your calendar", "You"),
    "not_saved": ("Nothing was saved", "You"),
    "approve_share": ("Recorded your sharing decision", "You"),
    "finalize": ("All done", "FitPlan"),
}
print("Dashboard pieces ready.")


# ==========================================================================
# Cell 32 - Interface: Community, Privacy, My data and Restart check pages
# ==========================================================================
# > Every page reads the REAL files through the same MemoryGateway the agents use,
# > so what the member sees is exactly what the rules allow.
import sys
import subprocess

STATUS = {"approved": ("Ready to use", "#DCFCE7", "#166534"), "pending": ("Waiting for OK", "#FEF3C7", "#92400E"),
          "superseded": ("Replaced by a newer version", "#E0E7FF", "#3730A3"),
          "outdated": ("Out of date", "#FEE2E2", "#991B1B")}
FILTERS = {"All": None, "Ready to use": "approved", "Waiting for OK": "pending",
           "Older versions": "superseded", "Out of date": "outdated"}
PAGE_CSS = """
.badge {display:inline-block; font-size:12.5px; font-weight:700; border-radius:999px; padding:4px 10px}
.kv {display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:8px; margin:10px 0}
.kv div {background:#F7F9FC; border-radius:10px; padding:9px 12px; font-size:14px}
.kv span {display:block; color:#5B6B82; font-size:12.5px}
.hist li {margin:6px 0; font-size:14px}
.rights {background:#EFF6FF; border-radius:12px; padding:12px 14px; font-size:14px; color:#1E3A8A; margin-top:10px}
.mono {font-family:ui-monospace,monospace; font-size:12.5px}
.techbox {background:#0F172A; color:#CBD5E1; border-radius:12px; padding:12px 14px; font-family:ui-monospace,monospace;
          font-size:12.5px; line-height:1.7; white-space:pre-wrap; margin-top:10px}
.spaces {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:12px}
.space {background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:16px}
.space.mine {border:2px solid #16A34A}
.result {border-radius:14px; padding:16px 18px; display:flex; gap:14px; align-items:flex-start}
.result .i {width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; flex-shrink:0}
.files {display:flex; flex-direction:column; gap:10px}
.file {background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:14px 16px}
.file b {font-size:15px} .file .w {color:#5B6B82; font-size:13.5px} .file .n {font-family:ui-monospace,monospace; font-size:12.5px; color:#1E3A8A}
.folder {background:#EFF6FF; border:1px solid #BFD3FB; border-radius:14px; padding:14px 18px}
.rtable {width:100%; border-collapse:collapse; font-size:14px}
.rtable th, .rtable td {border-bottom:1px solid #EEF2F6; padding:10px 12px; text-align:left; vertical-align:top}
.rtable thead th {background:#F7F9FC; color:#5B6B82; font-size:13px}
.no {color:#B91C1C; font-weight:600} .maybe {color:#B45309; font-weight:600}
"""


def nice_ex(e):
    return e.replace(" x ", " ")


def record_key(r):
    return f"{r['record_id']}-v{r['version']}"


def community_choices(uid, filt="All"):
    want = FILTERS.get(filt)
    recs = sorted(gateway.library_all(("member", uid)), key=lambda r: (r["record_id"], -r["version"]))
    out = []
    for r in recs:
        if want and r["status"] != want:
            continue
        who = "you" if r["author"] == uid else member_name(r["author"])
        out.append((f"{r['title']}  ·  v{r['version']}  ·  {STATUS[r['status']][0]}  ·  by {who}", record_key(r)))
    return out


def record_html(uid, key, tech=False):
    rec = next((r for r in gateway.library_all(("member", uid)) if record_key(r) == key), None)
    if not rec:
        return "<div class='panel' style='color:#5B6B82'>Pick a workout to see its details.</div>"
    label, bg, fg = STATUS[rec["status"]]
    author = member_name(rec["author"])
    mine = rec["author"] == uid
    usable = "Yes" if rec["status"] == "approved" and not rec.get("superseded_by") else "No"
    from_ = {"progress_analyst": "Suggested by the Progress Checker", "member": "Shared by the member"}.get(rec["proposed_by"], rec["proposed_by"])
    rights = ("Only you. Other members and both AI helpers can use it, but never change it." if mine else
              f"Only {author}. You can use it in your plans when it is ready.")
    if rec["status"] == "superseded":
        rights = "Nobody. Older versions are kept for history and cannot be changed."
    hist = "".join(f"<li>{esc(h)}</li>" for h in rec["history"])
    exs = "<br>".join(esc(nice_ex(e)) for e in rec["exercises"])
    tech_ = (f"<div class='techbox'>namespace (\"shared\", \"library\")  key {key}\nstatus={rec['status']}  "
             f"superseded_by={rec.get('superseded_by')}  review_after={rec.get('review_after')}\n"
             f"author={rec['author']}  proposed_by={rec['proposed_by']}\n"
             f"search_library() {'includes' if usable == 'Yes' else 'excludes'} this record</div>") if tech else ""
    return (f"<div class='panel'><span class='badge' style='background:{bg};color:{fg}'>{label}</span>"
            f"<h3 style='margin:10px 0 6px'>{esc(rec['title'])} <span style='color:#5B6B82;font-weight:400;font-size:14px'>version {rec['version']}</span></h3>"
            f"<div style='font-size:14px;line-height:1.6'>{exs}</div>"
            f"<div class='kv'><div><span>Shared by</span>{esc(author)}</div><div><span>Idea came from</span>{esc(from_)}</div>"
            f"<div><span>Can AI plans use it?</span>{usable}</div><div><span>About</span>{rec['minutes']} min · {esc(rec['equipment'])} · {esc(rec['level'])}</div></div>"
            f"<b style='font-size:13px;color:#5B6B82'>HISTORY</b><ul class='hist'>{hist}</ul>"
            f"<div class='rights'><b>Who can change this?</b> {rights}</div>{tech_}</div>")


def pending_for(uid):
    recs = [r for r in gateway.library_all(("member", uid)) if r["author"] == uid and r["status"] == "pending"]
    return recs[0] if recs else None


# ---------------- Privacy ----------------
def member_name(uid):
    p, u = get_profile(uid) or {}, get_user(uid) or {}
    return p.get("display_name") or u.get("name") or "a former member"


def other_members(uid):
    """Every other real member, most recently signed in first (test members are never listed)."""
    with db() as conn:
        rows = conn.execute("SELECT user_id FROM users WHERE user_id != ? AND user_id NOT LIKE 'test-member-%' "
                            "ORDER BY COALESCE(last_login, created_at) DESC", (uid,)).fetchall()
    return [r["user_id"] for r in rows]


def other_member(uid):
    """The member used for the 'try to read someone else's data' check (any other member works)."""
    others = other_members(uid) or [m for m in TEST_IDS if m != uid]
    return others[0] if others else "another-member"


def spaces_html(uid, tech=False):
    """Your own space (unlocked), every other member's space (locked), and the shared library."""
    n = len(other_members(uid))
    yours = ("<div class='space mine'><div style='display:flex;justify-content:space-between'><b>Your private space</b>"
             "<span class='badge' style='background:#DCFCE7;color:#166534'>Yours</span></div>"
             f"<div style='color:#5B6B82;font-size:14px;margin-top:6px'>Signed in as {esc(member_name(uid))}. Goal, training days, "
             "equipment, likes and dislikes, and everything the coach remembers.</div>"
             + (f"<div class='mono' style='color:#1E3A8A;margin-top:6px'>(\"athletes\", \"{uid}\", \"profile\")</div>" if tech else "")
             + "</div>")
    others = ("<div class='space'><div style='display:flex;justify-content:space-between'><b>Other members' spaces</b>"
              "<span class='badge' style='background:#FEE2E2;color:#991B1B'>Locked</span></div>"
              f"<div style='color:#5B6B82;font-size:14px;margin-top:6px'>{n if n else 'No'} other member{'' if n == 1 else 's'} "
              "so far. Each one has a private space that only they and their own AI helpers can open.</div>"
              + ("<div class='mono' style='color:#1E3A8A;margin-top:6px'>(\"athletes\", &lt;other id&gt;, \"profile\")</div>" if tech else "")
              + "</div>")
    ns = "<div class='mono' style='color:#1E3A8A;margin-top:6px'>(\"shared\", \"library\")</div>" if tech else ""
    shared = ("<div class='space'><div style='display:flex;justify-content:space-between'><b>Community workouts</b>"
              "<span class='badge' style='background:#DBEAFE;color:#1E40AF'>Shared</span></div><div style='color:#5B6B82;"
              f"font-size:14px;margin-top:6px'>Everyone can use the ready ones. Only the person who shared one can change it.</div>{ns}</div>")
    return f"<div class='spaces'>{yours}{others}{shared}</div>"


PRIVACY_ACTIONS = ["Show my profile", "Ask the coach for the other member's profile", "Use a community workout",
                   "Plan Coach tries to change a community workout", "Progress Checker suggests sharing my workout"]


def try_action(uid, action):
    other = other_member(uid)
    other_name = member_name(other) if other != "another-member" else "another member"
    OK, NO, WAIT = ("#F0FDF4", "#A7E3CF", "#DCFCE7", "#166534", "&#10003;"), ("#FEF2F2", "#F8C9C9", "#FEE2E2", "#991B1B", "&#10005;"), \
                   ("#FFFBEB", "#F5D48A", "#FEF3C7", "#92400E", "?")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            if action == PRIVACY_ACTIONS[0]:
                n = len(gateway.read_profile(("member", uid), uid))
                style, title, text = OK, "Here is your profile", f"You can always see and edit your own memory ({n} items)."
            elif action == PRIVACY_ACTIONS[1]:
                gateway.read_profile(("planner", uid), other)
                style, title, text = OK, "Shown", ""
            elif action == PRIVACY_ACTIONS[2]:
                n = len(gateway.library_search(("planner", uid), "any"))
                style, title, text = OK, f"{n} workouts ready to use", ("Only ready, current versions are shown. Ones waiting "
                                                                       "for approval, older versions and out-of-date ones are hidden.")
            elif action == PRIVACY_ACTIONS[3]:
                recs = gateway.library_all(("planner", uid))
                target = recs[0]["record_id"] if recs else "lib-none"     # any community workout: the role check refuses first
                gateway.library_update(("planner", uid), target, {}, "planner edit attempt")
                style, title, text = OK, "Changed", ""
            else:
                gateway.check(("progress_analyst", uid), "propose_share", uid, "shared/library")
                gateway.log(("progress_analyst", uid), "propose_share", "shared/library", True, "check only: would be saved as pending")
                style, title, text = WAIT, "Allowed as a suggestion only", ("The Progress Checker can suggest one of your "
                                                                            "workouts. Nothing is shared until you say yes.")
        except AccessDenied:
            if action == PRIVACY_ACTIONS[1]:
                title, text = "Sorry, that's private", (f"Your coach can't show {other_name}'s profile. Each member's data is "
                                                        "only for them, and your AI helpers only work for you.")
            else:
                title, text = "The Plan Coach can't do that", ("AI helpers can use community workouts, but only the person "
                                                              "who shared a workout can change or remove it.")
            style = NO
    bg, border, ibg, ifg, icon = style
    html_ = (f"<div class='result' role='status' style='background:{bg};border:1px solid {border}'>"
             f"<div class='i' style='background:{ibg};color:{ifg}'>{icon}</div><div><b style='font-size:17px'>{title}</b>"
             f"<div style='color:#334155;margin-top:4px'>{text}</div></div></div>")
    return html_, buf.getvalue().strip()


def rights_html():
    rows = [("You", "See and edit", "<span class='no'>Never</span>", "Use any that are ready", "Only your own"),
            ("Progress Checker (AI)", "Reads it while helping you", "<span class='no'>Never</span>", "Use any that are ready",
             "<span class='maybe'>Suggest only, you decide</span>"),
            ("Plan Coach (AI)", "Reads it while helping you", "<span class='no'>Never</span>", "Use any that are ready",
             "<span class='no'>Never</span>"),
            ("Other members", "<span class='no'>Never</span>", "Only their own", "Use any that are ready", "Only their own")]
    body = "".join(f"<tr><th scope='row'>{a}</th><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in rows)
    return ("<div class='panel'><h3>Who can do what</h3><div style='overflow-x:auto'><table class='rtable'><thead><tr><th></th>"
            "<th>Your profile and memory</th><th>Another member's profile</th><th>Community workouts</th>"
            f"<th>Share or change a workout</th></tr></thead><tbody>{body}</tbody></table></div></div>")


def audit_frame(uid, n=12):
    items = sorted(store.search(("audit", "log"), limit=1000), key=lambda i: i.key, reverse=True)
    rows = [i.value for i in items if i.value["acting_for"] in (uid, "system")][:n]
    words = {"member": "You", "progress_analyst": "Progress Checker", "planner": "Plan Coach"}
    return pd.DataFrame([[r["time"][11:], words.get(r["role"], r["role"]), r["action"], r["target"],
                          "Allowed" if r["allowed"] else "Refused"] for r in rows] or [["", "", "", "", ""]],
                        columns=["Time", "Who", "Tried to", "On", "Result"])


# ---------------- My data ----------------
def files_html(tech=False):
    files = [("Accounts and profiles", "Your sign-in details and the profile you filled in", "fitplan_users.db"),
             ("What your AI team remembers", "Your private memory, the community library and the access log", "fitplan_memory.db"),
             ("Conversations in progress", "Lets a paused plan continue after FitPlan restarts", "fitplan_checkpoints.db"),
             ("Workouts and check-ins", "Every set you log, and your sleep and soreness", "workout_log.csv · daily_metrics.csv")]
    out = (f"<div class='folder'><b>FitPlan folder on this device</b><div class='mono' style='color:#1E3A8A;margin-top:4px;"
           f"overflow-wrap:anywhere'>{esc(DATA_DIR if tech else 'FitPlan/fitplan_data/')}</div>"
           "<div style='color:#15803D;font-weight:600;font-size:14px;margin-top:4px'>Saved automatically</div></div><div class='files'>")
    for title, what, name in files:
        size = sum(os.path.getsize(os.path.join(DATA_DIR, n.strip())) for n in name.split("·")
                   if os.path.exists(os.path.join(DATA_DIR, n.strip())))
        out += (f"<div class='file'><b>{title}</b><div class='w'>{what}</div>"
                f"<div class='n'>{name}  ·  {size / 1024:.0f} KB</div></div>")
    return out + "</div>"


DATA_TABLES = ["My account", "My profile", "My private memory", "Community library", "My conversations",
               "Access log", "My workouts", "My check-ins"]


def data_table(uid, which):
    """Rows the signed-in member is allowed to see, straight from the files."""
    me = ("member", uid)
    if which == "My account":
        u = get_user(uid) or {}
        df = pd.DataFrame([[u.get("user_id"), u.get("username"), u.get("email"), "scrambled · cannot be read back", u.get("created_at")]],
                          columns=["Member ID", "Username", "Email", "Password", "Joined"])
        tech = "fitplan_users.db → SELECT user_id, username, email, created_at FROM users WHERE user_id = ?\npassword: PBKDF2-SHA256, 200,000 rounds, random salt"
    elif which == "My profile":
        p = get_profile(uid) or {}
        keep = ["display_name", "goal", "experience_level", "training_days", "session_minutes", "equipment", "split",
                "exercise_preferences", "has_watch", "consent_llm", "share_suggestions"]
        df = pd.DataFrame([[k, p.get(k)] for k in keep], columns=["Field", "Saved value"])
        tech = "fitplan_users.db → SELECT * FROM profiles WHERE user_id = ?"
    elif which == "My private memory":
        m = gateway.read_profile(me, uid)
        df = pd.DataFrame([[k, v["value"], v.get("previous_value") or "—", v["updated_at"], v.get("source", "")] for k, v in m.items()]
                          or [["(nothing yet)", "", "", "", ""]], columns=["Remembers", "Now", "Before", "Updated", "From"])
        tech = f'fitplan_memory.db → SqliteStore namespace ("athletes", "{uid}", "profile")\nread through MemoryGateway (owner check first)'
    elif which == "Community library":
        recs = sorted(gateway.library_all(me), key=lambda r: (r["record_id"], r["version"]))
        df = pd.DataFrame([[r["record_id"], r["title"], r["version"], STATUS[r["status"]][0], r["author"]] for r in recs],
                          columns=["Record", "Workout", "Version", "Status", "Shared by"])
        tech = 'fitplan_memory.db → SqliteStore namespace ("shared", "library")\nfields: version, status, superseded_by, review_after'
    elif which == "My conversations":
        with sqlite3.connect(CHECKPOINT_DB) as c:
            tids = [r[0] for r in c.execute("SELECT DISTINCT thread_id FROM checkpoints")]
        rows = []
        for t in tids:
            st = app.get_state({"configurable": {"thread_id": t}})
            if st.values.get("athlete_id") != uid:
                continue
            rows.append([t, st.values.get("request", "")[:40], st.values.get("status") or "—",
                         "Your approval" if st.next else "—"])
        df = pd.DataFrame(rows or [["(none yet)", "", "", ""]], columns=["Conversation", "You asked", "Ended as", "Waiting for"])
        tech = "fitplan_checkpoints.db → SqliteSaver (LangGraph checkpointer), one thread_id per conversation"
    elif which == "Access log":
        df = audit_frame(uid, 20)
        tech = 'fitplan_memory.db → namespace ("audit", "log"), written by MemoryGateway.log()'
    elif which == "My workouts":
        log = pd.read_csv(LOG_PATH)
        df = log[log["athlete_id"] == uid].tail(12)[["date", "exercise", "set_no", "reps", "weight_kg", "rpe"]]
        tech = f"{LOG_PATH}"
    else:
        mt = pd.read_csv(METRICS_PATH)
        df = mt[mt["athlete_id"] == uid].tail(10)[["date", "sleep_hours", "soreness_1_5", "resting_hr", "hrv_ms"]]
        tech = f"{METRICS_PATH}"
    return df, tech


def restart_table(before, after):
    rows = [("Your profile and memory", "me", lambda d: f"{d['me']} items"),
            ("Every member's private memory", "private_items", lambda d: f"{d['private_items']} items, {d['members']} members"),
            ("Community workouts", "library", lambda d: f"{d['library']} records"),
            ("Workout versions and statuses", "versions", lambda d: f"{len(d['versions'])} entries"),
            ("Your plan waiting for your OK", "paused", lambda d: "Waiting" if d["paused"] else ("None waiting" if d["paused"] is False else "—"))]
    body = ""
    for label, k, fmt in rows:
        same = after is not None and before[k] == after[k] and (k != "private_items" or before["members"] == after["members"])
        a = fmt(after) if after else "—"
        mark = ("<span style='color:#16A34A;font-weight:700'>Yes</span>" if same else
                ("<span class='no'>No</span>" if after else "—"))
        body += f"<tr><th scope='row'>{label}</th><td>{fmt(before)}</td><td>{a}</td><td>{mark}</td></tr>"
    return ("<div class='panel'><h3>Before and after</h3><table class='rtable'><thead><tr><th>What</th>"
            f"<th>Before closing</th><th>After reopening</th><th>Same?</th></tr></thead><tbody>{body}</tbody></table></div>")


print("Pages ready.")


# ==========================================================================
# Cell 33 - Interface: build the app (all screens) and launch it
# ==========================================================================
# > Pressing Enter in any box submits that form (Gradio .submit events).
# > Every screen is a Column; show() switches which one is visible.
VIEWS = ["signin", "signup", "profile", "saved", "home"]


def summary_html(name, dob, goal, level, days, minutes, equipment, split):
    age = calc_age(dob)
    items = [("Name", name or "—"), ("Age", age if age is not None else "—"), ("Goal", goal or "—"),
             ("Level", level or "—"), ("Days", ", ".join(d[:3] for d in DAYS if d in (days or [])) or "—"),
             ("Session", f"{int(minutes)} min"), ("Training at", equipment or "—"), ("Week plan", split or "—")]
    return ("<div class='card' style='padding:16px !important'><div style='font-size:13px;font-weight:700;"
            "letter-spacing:.06em;color:#5B6B82;margin-bottom:8px'>YOUR ANSWERS</div><div class='summary'>" +
            "".join(f"<div><span>{k}:</span> <b>{esc(v)}</b></div>" for k, v in items) + "</div></div>")


with gr.Blocks(title="FitPlan") as demo:
    user = gr.State(None)
    top = gr.HTML(topbar())
    with gr.Row(visible=False, elem_classes="nav") as app_nav:
        nav_dash = gr.Button("Dashboard", variant="primary", scale=0, min_width=130)
        nav_comm = gr.Button("Community", scale=0, min_width=130)
        nav_priv = gr.Button("Privacy", scale=0, min_width=110)
        nav_data = gr.Button("My data", scale=0, min_width=110)
        gr.HTML("")
        logout = gr.Button("Sign out", scale=0, min_width=110)

    # ---------- Sign in ----------
    with gr.Row(visible=True, equal_height=True) as signin_view:
        with gr.Column(scale=1, elem_classes="brand"):
            gr.HTML(BRAND)
        with gr.Column(scale=1, elem_classes="formcol"):
            gr.HTML("<div class='h2'>Welcome back</div><div class='sub'>Sign in to see your plan and your AI team.</div>"
                    )
            si_note = gr.HTML("")
            si_user = gr.Textbox(label="Email or username", placeholder="you@example.com", elem_id="si_user")
            with gr.Group(elem_classes="pw"):
                si_pw = gr.Textbox(label="Password", type="password", placeholder="Your password", elem_id="si_pw")
                si_eye = gr.Button("Show", elem_classes="eye", elem_id="si_eye", size="sm")
            go_forgot = gr.Button("Forgot password?", size="sm", elem_classes="linkbtn")
            si_msg = gr.HTML("")
            si_btn = gr.Button("Sign in", variant="primary", elem_classes="big")
            gr.HTML("<div class='tip'>Tip: press <b>Enter</b> to sign in</div>")
            gr.HTML("<div class='tip' style='margin-top:10px'>New to FitPlan?</div>")
            go_signup = gr.Button("Create a new account", elem_classes="big")

    # ---------- Create account ----------
    with gr.Row(visible=False, equal_height=True) as signup_view:
        with gr.Column(scale=1, elem_classes="brand"):
            gr.HTML(BRAND)
        with gr.Column(scale=1, elem_classes="formcol"):
            gr.HTML("<div class='h2'>Create your account</div><div class='sub'>It takes a minute. Next, you'll tell us about your training.</div>")
            su_name = gr.Textbox(label="Your name", placeholder="e.g. Alex")
            su_email = gr.Textbox(label="Email", placeholder="you@example.com", type="email")
            with gr.Group(elem_classes="pw"):
                su_pw = gr.Textbox(label="Password", type="password", placeholder="Create a password", elem_id="su_pw")
                su_eye = gr.Button("Show", elem_classes="eye su-eye", size="sm")
            su_rules = gr.HTML(rules_html("", None))
            with gr.Group(elem_classes="pw"):
                su_pw2 = gr.Textbox(label="Confirm password", type="password", placeholder="Type it again", elem_id="su_pw2")
                su_eye2 = gr.Button("Show", elem_classes="eye su-eye", size="sm")
            su_agree = gr.Checkbox(label="I understand my data is stored on this device and only used to plan my training.")
            su_msg = gr.HTML("")
            su_btn = gr.Button("Create account", variant="primary", elem_classes="big")
            gr.HTML("<div class='tip'>Tip: press <b>Enter</b> to create your account</div>")
            go_signin = gr.Button("Already have an account? Sign in", size="sm")

    # ---------- Forgot password ----------
    with gr.Row(visible=False, equal_height=True) as forgot_view:
        with gr.Column(scale=1, elem_classes="brand"):
            gr.HTML(BRAND)
        with gr.Column(scale=1, elem_classes="formcol"):
            gr.HTML("<div class='h2'>Forgot your password?</div><div class='sub'>Enter the email you signed up with. "
                    f"We'll email you a link to choose a new password. The link works once, for {RESET_MINUTES} minutes.</div>")
            fp_email = gr.Textbox(label="Email", placeholder="you@example.com", type="email", elem_id="fp_email")
            fp_msg = gr.HTML("")
            fp_btn = gr.Button("Email me a reset link", variant="primary", elem_classes="big")
            gr.HTML("<div class='tip'>Tip: press <b>Enter</b> to send</div>")
            with gr.Accordion("The button in the email didn't open FitPlan?", open=False):
                fp_paste = gr.Textbox(label="Paste the whole link from the email", placeholder="https://...?reset=...")
                fp_paste_btn = gr.Button("Use this link")
            fp_back = gr.Button("Back to sign in", size="sm")

    # ---------- Choose a new password (opened from the email link) ----------
    reset_token = gr.State("")
    with gr.Row(visible=False, equal_height=True) as reset_view:
        with gr.Column(scale=1, elem_classes="brand"):
            gr.HTML(BRAND)
        with gr.Column(scale=1, elem_classes="formcol"):
            rs_head = gr.HTML("<div class='h2'>Choose a new password</div>")
            with gr.Group(elem_classes="pw"):
                rs_pw = gr.Textbox(label="New password", type="password", placeholder="Create a password", elem_id="rs_pw")
                rs_eye = gr.Button("Show", elem_classes="eye rs-eye", size="sm")
            rs_rules = gr.HTML(rules_html("", None))
            with gr.Group(elem_classes="pw"):
                rs_pw2 = gr.Textbox(label="Confirm new password", type="password", placeholder="Type it again", elem_id="rs_pw2")
                rs_eye2 = gr.Button("Show", elem_classes="eye rs-eye", size="sm")
            rs_msg = gr.HTML("")
            rs_btn = gr.Button("Save my new password", variant="primary", elem_classes="big")
            gr.HTML("<div class='tip'>Tip: press <b>Enter</b> to save</div>")
            rs_back = gr.Button("Back to sign in", size="sm")

    # ---------- Profile: three short steps ----------
    with gr.Column(visible=False) as profile_view:
        gr.HTML("<div class='h2'>Set up your profile</div><div class='sub'>Three short steps. Your AI team uses this "
                "to plan your training. You can change it any time.</div>")
        p_steps = gr.HTML(steps_html(1))
        p_msg = gr.HTML("")
        with gr.Column(visible=True, elem_classes="card") as step1:
            gr.HTML("<b style='font-size:20px'>About you</b>")
            with gr.Row():
                p_name = gr.Textbox(label="Profile name", placeholder="What should your coach call you?")
                p_dob = gr.DateTime(label="Date of birth", include_time=False, type="string")
            p_age = gr.HTML("<div class='sub' style='font-size:13px'>Your age is worked out for you.</div>")
            p_sex = gr.Radio(["Male", "Female", "Other", "Prefer not to say"], label="Sex", value="Prefer not to say", elem_classes="choice")
            p_units = gr.Radio(["cm and kg", "ft and lb"], label="Units", value="cm and kg", elem_classes="choice")
            with gr.Row():
                p_h = gr.Number(label="Height (cm)", placeholder="e.g. 178")
                p_w = gr.Number(label="Weight (kg)", placeholder="e.g. 80")
            gr.HTML("<div class='sub' style='font-size:13px'>Only used to suggest safe starting weights. Never shown to other members.</div>")
            with gr.Row():
                gr.HTML("<div class='tip' style='text-align:right;padding-top:14px'>Press Enter to continue</div>")
                p_next1 = gr.Button("Next: your training", variant="primary", elem_classes="big", scale=0, min_width=220)
        with gr.Column(visible=False, elem_classes="card") as step2:
            gr.HTML("<b style='font-size:20px'>Your training</b>")
            p_goal = gr.Radio(GOALS, label="What's your main goal?", elem_classes="choice")
            p_level = gr.Radio(LEVELS, label="How experienced are you?", elem_classes="choice")
            p_days = gr.CheckboxGroup(DAYS, label="Which days can you train?", elem_classes="choice")
            p_min = gr.Slider(20, 120, value=60, step=5, label="How long is each session? (minutes)")
            p_eq = gr.Radio(EQUIPMENT, label="Where do you train?", value="Full gym", elem_classes="choice")
            p_split = gr.Radio(SPLITS, label="How should your week be organised?", value="Let the coach decide", elem_classes="choice")
            p_pref = gr.Textbox(label="Exercises you'd like to avoid or prefer (optional)", placeholder="e.g. No back squats, I prefer the hack squat")
            p_watch = gr.Radio(["Yes", "No"], label="Do you wear a fitness watch?", value="No", elem_classes="choice",
                               info="With a watch, your coach also uses heart rate to judge how rested you are.")
            with gr.Row():
                p_back2 = gr.Button("Back", elem_classes="big", scale=0, min_width=120)
                gr.HTML("<div class='tip' style='text-align:right;padding-top:14px'>Press Enter to continue</div>")
                p_next2 = gr.Button("Next: privacy", variant="primary", elem_classes="big", scale=0, min_width=200)
        with gr.Column(visible=False, elem_classes="card") as step3:
            gr.HTML("<b style='font-size:20px'>Privacy and your AI team</b>")
            p_consent = gr.Checkbox(label="Let the AI use my age, sex, height and weight (helps it choose safe starting weights; your name and email are never sent)")
            p_share = gr.Checkbox(value=True, label="Let my AI team suggest workouts to share with other members (you approve every single one)")
            gr.HTML("<div class='warnbox'><b>Please don't add injuries or medical details.</b> FitPlan doesn't store them. "
                    "If something hurts, check with a doctor or physiotherapist first.</div>")
            p_summary = gr.HTML("")
            with gr.Row():
                p_back3 = gr.Button("Back", elem_classes="big", scale=0, min_width=120)
                p_save = gr.Button("Save my profile", variant="primary", elem_classes="big", scale=0, min_width=220)

    # ---------- Profile saved ----------
    with gr.Column(visible=False, elem_classes="card") as saved_view:
        saved_html = gr.HTML("")
        with gr.Row():
            go_home = gr.Button("Go to my dashboard", variant="primary", elem_classes="big", scale=0, min_width=220)
            go_data = gr.Button("See where my data is saved", elem_classes="big", scale=0, min_width=260)
            edit_profile = gr.Button("Edit my answers", elem_classes="big", scale=0, min_width=180)

    # ---------- Dashboard: your AI team at work ----------
    dash = gr.State({})
    thread = gr.State("")
    with gr.Column(visible=False) as home_view:
        home_html = gr.HTML("")
        with gr.Group():
            with gr.Row():
                ask = gr.Textbox(placeholder="Ask your coach, e.g. Plan my next week", show_label=False, scale=5)
                ask_btn = gr.Button("Plan my week", variant="primary", elem_classes="big", scale=1, min_width=160)
        with gr.Row():
            gr.HTML("<div class='tip' style='text-align:left;padding-top:10px'>Try:</div>")
            ex1 = gr.Button("Plan my next week", size="sm", scale=0, min_width=170)
            ex2 = gr.Button("Make my sessions 45 minutes", size="sm", scale=0, min_width=230)
            ex3 = gr.Button("How am I progressing?", size="sm", scale=0, min_width=190)
            new_chat = gr.Button("Start a new conversation", size="sm", scale=0, min_width=210)
        team = gr.HTML(team_html())
        reply_box = gr.HTML("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                feed_box = gr.HTML(feed_html([], False))
                tech = gr.Checkbox(label="Behind the scenes (show the technical step names)", value=False)
            with gr.Column(scale=1):
                report_box = gr.HTML(report_html({}))
        plan_box = gr.HTML("")
        with gr.Row(visible=False) as decide_row:
            approve_btn = gr.Button("Save to my calendar", variant="primary", elem_classes="big", scale=0, min_width=210)
            changes_btn = gr.Button("Ask for changes", elem_classes="big", scale=0, min_width=170)
            nosave_btn = gr.Button("Don't save", elem_classes="big", scale=0, min_width=140)
        with gr.Row(visible=False) as changes_row:
            fb = gr.Textbox(placeholder="What should change? e.g. Make Saturday my heavier day", show_label=False, scale=5)
            fb_btn = gr.Button("Update my plan", variant="primary", scale=1, min_width=170)
        share_box = gr.HTML("")
        with gr.Row(visible=False) as share_row:
            share_yes = gr.Button("Yes, share it", variant="primary", elem_classes="big", scale=0, min_width=170)
            share_no = gr.Button("Keep it private", elem_classes="big", scale=0, min_width=170)
        other_page = gr.HTML("", visible=False)

    # ---------- Community ----------
    with gr.Column(visible=False) as comm_view:
        gr.HTML("<div class='h2'>Community workouts</div><div class='sub'>Workouts members chose to share. Your AI team can "
                "use them in anyone's plan, but only after the person who made it says yes.</div>")
        c_pending = gr.HTML("")
        with gr.Row(visible=False) as c_pending_row:
            c_yes = gr.Button("Yes, share it", variant="primary", elem_classes="big", scale=0, min_width=170)
            c_no = gr.Button("Keep it private", elem_classes="big", scale=0, min_width=170)
        c_note = gr.HTML("")
        c_filter = gr.Radio(list(FILTERS), value="All", label="Show", elem_classes="choice")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                c_list = gr.Radio([], label="Workouts", elem_classes="choice")
            with gr.Column(scale=1):
                c_detail = gr.HTML("")
                with gr.Group(visible=False) as c_edit_box:
                    c_edit_ex = gr.Textbox(label="Exercises (one per line, e.g. push-up x 3x12)", lines=4)
                    c_edit_note = gr.Textbox(label="What did you change?", placeholder="e.g. fewer reps on the shoulder press")
                    c_edit_save = gr.Button("Save as a new version", variant="primary")
                c_edit_btn = gr.Button("Edit my workout (makes a new version)", visible=False)
        c_tech = gr.Checkbox(label="Behind the scenes (show the stored fields)", value=False)

    # ---------- Privacy ----------
    with gr.Column(visible=False) as priv_view:
        gr.HTML("<div class='h2'>Who can see what</div><div class='sub'>Your profile and memory are private. Your AI helpers "
                "work only for the person who is signed in.</div>")
        pv_spaces = gr.HTML("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                pv_action = gr.Radio(PRIVACY_ACTIONS, label="Try it", elem_classes="choice")
            with gr.Column(scale=1):
                pv_result = gr.HTML("<div class='panel' style='color:#5B6B82'>Pick something to see what FitPlan allows.</div>")
                pv_log = gr.Textbox(label="Behind the scenes: what the access check logged", lines=3, interactive=False, visible=False)
        pv_rights = gr.HTML(rights_html())
        pv_tech = gr.Checkbox(label="Behind the scenes (show memory names and the access log)", value=False)
        pv_audit = gr.Dataframe(label="Access log (latest first)", interactive=False, visible=False)

    # ---------- My data ----------
    with gr.Column(visible=False) as data_view:
        gr.HTML("<div class='h2'>Where your data is saved</div><div class='sub'>Everything stays on this device. "
                "Pick a table to see exactly what is stored. You only ever see your own private rows.</div>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                d_files = gr.HTML("")
                d_restart = gr.Button("Try a restart check", variant="primary")
            with gr.Column(scale=2):
                d_which = gr.Radio(DATA_TABLES, value="My account", label="Show me", elem_classes="choice")
                d_table = gr.Dataframe(interactive=False, wrap=True)
                d_tech_txt = gr.Textbox(label="Behind the scenes: file and query", lines=2, interactive=False, visible=False)
        d_tech = gr.Checkbox(label="Show technical details (full folder path, queries)", value=False)

    # ---------- Restart check ----------
    with gr.Column(visible=False) as restart_view:
        gr.HTML("<div class='h2'>Close FitPlan and open it again: nothing is lost</div><div class='sub'>This starts a brand-new "
                "program that opens only the saved files, then compares what it finds with what was there before.</div>")
        r_status = gr.HTML("")
        with gr.Row():
            r_go = gr.Button("Close and reopen FitPlan", variant="primary", elem_classes="big", scale=0, min_width=260)
            r_review = gr.Button("Review my waiting plan", elem_classes="big", scale=0, min_width=240, visible=False)
        r_table = gr.HTML("")
        r_tech = gr.Textbox(label="Behind the scenes", lines=6, interactive=False)

    VIEW_COLS = {"signin": signin_view, "signup": signup_view, "forgot": forgot_view, "reset": reset_view, "profile": profile_view, "saved": saved_view, "home": home_view,
                 "community": comm_view, "privacy": priv_view, "data": data_view, "restart": restart_view}
    NAV = {"home": nav_dash, "community": nav_comm, "privacy": nav_priv, "data": nav_data, "restart": nav_data}

    def show(view):
        out = {col: gr.update(visible=(v == view)) for v, col in VIEW_COLS.items()}
        out[app_nav] = gr.update(visible=view in NAV)
        for v, b_ in NAV.items():
            if v != "restart":
                out[b_] = gr.update(variant="primary" if (v == view or (v == "data" and view == "restart")) else "secondary")
        return out

    def wizard(step):
        return {step1: gr.update(visible=step == 1), step2: gr.update(visible=step == 2),
                step3: gr.update(visible=step == 3), p_steps: steps_html(step)}

    def profile_values(uid):
        p = get_profile(uid) or {}
        u = get_user(uid) or {}
        days = [d.strip() for d in (p.get("training_days") or "").split(",") if d.strip()]
        return {p_name: p.get("display_name") or u.get("name") or "", p_dob: p.get("dob") or None,
                p_sex: p.get("sex") or "Prefer not to say", p_units: "cm and kg",
                p_h: p.get("height_cm"), p_w: p.get("weight_kg"), p_goal: p.get("goal"),
                p_level: p.get("experience_level"), p_days: days, p_min: p.get("session_minutes") or 60,
                p_eq: p.get("equipment") or "Full gym", p_split: p.get("split") or "Let the coach decide",
                p_pref: p.get("exercise_preferences") or "", p_watch: "Yes" if p.get("has_watch") else "No",
                p_consent: bool(p.get("consent_llm")), p_share: bool(p.get("share_suggestions", 1))}

    def home_page(uid):
        return greet_html(uid)

    def fresh_dash():
        return {"feed": [], "trace": "", "report": {}, "plan": "", "status": "", "share": None, "reply": "",
                "feedback": "", "checker": ("ready", "Waiting for your request"),
                "coach": ("ready", "Waiting for your request"), "handed": False}

    def render(d, uid, show_tech):
        paused_plan = d.get("status") == "pending" and d.get("plan")
        reply = f"<div class='okbox' style='font-size:15px'>{esc(d['reply'])}</div>" if d.get("reply") else ""
        return {team: team_html(d["checker"][0], d["coach"][0], d["checker"][1], d["coach"][1], d["handed"]),
                feed_box: feed_html(d["feed"], show_tech), report_box: report_html(d["report"]),
                plan_box: plan_html(d["plan"], d["status"], d.get("feedback", "")), reply_box: reply,
                decide_row: gr.update(visible=bool(paused_plan)), changes_row: gr.update(visible=False),
                share_box: share_html(d["share"]), share_row: gr.update(visible=bool(d["share"])),
                home_html: greet_html(uid), dash: d}

    def add(d, text, who, kind="done", tech_=""):
        d["feed"] = [f for f in d["feed"] if f["kind"] != "now"]     # a finished step replaces its "working" line
        d["feed"].append({"text": text, "who": who, "kind": kind, "tech": tech_})

    def stream_graph(uid, tid, payload, d, show_tech):
        cfg = {"configurable": {"thread_id": tid}}
        it = iter(app.stream(payload, cfg, stream_mode=["updates", "debug"]))
        while True:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    chunk = next(it)
                except StopIteration:
                    chunk = None
                except Exception as e:
                    print(f"ERROR: {e}")
                    chunk = "error"
            log = buf.getvalue()
            d["trace"] += log
            if chunk == "error":
                add(d, "Something went wrong. Please try again.", "FitPlan", "stop", log.strip()[-200:])
                yield render(d, uid, show_tech)
                return
            if chunk is None:
                break
            mode, chunk = chunk
            if mode == "debug":
                if chunk["type"] != "task":
                    continue
                name = chunk["payload"].get("name")
                if name == "progress_analyst":
                    d["checker"] = ("working", "Reading your workouts, rest score and every lift")
                    d["coach"] = ("waiting", "Waiting for the progress report")
                    add(d, "Progress Checker is reviewing your last 4 weeks", "Progress Checker", "now", "task started: progress_analyst")
                elif name in ("planner", "tools", "revise"):
                    if d["coach"][0] != "working":
                        d["coach"] = ("working", "Choosing this week's weights from the report")
                        add(d, "Plan Coach is building your week", "Plan Coach", "now", f"task started: {name}")
                else:
                    continue
                yield render(d, uid, show_tech)
                continue
            for node, upd in chunk.items():
                last_log = " | ".join(l.strip() for l in log.splitlines() if l.strip())[:300]
                if node == "__interrupt__":
                    p = upd[0].value
                    if p["kind"] == "plan":
                        d["plan"], d["status"] = p["plan"], "pending"
                        d["report"] = p.get("report") or d["report"]
                        d["coach"] = ("done", "Plan ready for you to review")
                        add(d, "Plan ready. Waiting for your OK before saving", "You", "wait", "interrupt() at human_approval")
                    else:
                        d["share"] = p["record"]
                        add(d, "Waiting for your OK before sharing a workout", "You", "wait", "interrupt() at approve_share")
                    continue
                upd = upd or {}
                text, who = STEP_WORDS.get(node, (node, ""))
                if node == "planner":
                    msgs = upd.get("messages") or []
                    calls = [c["name"] for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
                    if calls:
                        nice = {"get_progression_targets": "your weight targets", "search_exercises": "exercises that fit",
                                "search_library": "the community library"}
                        text = "Looking up " + ", ".join(nice.get(c, c) for c in dict.fromkeys(calls))
                    elif upd.get("plan_issues"):
                        text = "Safety check found a problem, fixing it"
                    elif upd.get("plan"):
                        text = "Plan written and safety-checked: fits your days, time and weight targets"
                if node == "tools":
                    continue
                if node == "progress_analyst":
                    d["report"] = upd.get("progress_report") or {}
                    d["handed"] = True
                    d["checker"] = ("done", "Report finished and handed over")
                    add(d, text, who, "done", last_log)
                    add(d, "Progress report handed to the Plan Coach", "Hand-off", "done",
                        f"[handoff] ProgressReport v1 ({len(d['report'].get('findings', []))} findings)")
                    if upd.get("share_rid"):
                        add(d, "Suggested one of your steady workouts for the community (you decide later)",
                            "Progress Checker", "done", f"propose_share -> {upd['share_rid']} pending")
                    continue
                if node in ("safety_response", "acknowledge"):
                    d["reply"] = upd.get("reply", "")
                    d["checker"], d["coach"] = ("ready", "Not needed this time"), ("ready", "Not needed this time")
                if node == "starter_setup":
                    d["report"] = upd.get("progress_report") or {}
                if node == "save_plan":
                    d["status"] = "saved"
                if node == "not_saved":
                    d["status"] = "not_saved"
                if node == "revise":
                    d["coach"] = ("working", "Reworking the plan with your request")
                if node == "approve_share":
                    d["share"] = None
                    res = upd.get("share_status")
                    text = ("Shared your workout with the community" if res == "approved"
                            else "Kept your workout private. Nothing was shared")
                add(d, text, who, "done", last_log or f"[{node}]")
            yield render(d, uid, show_tech)
        yield render(d, uid, show_tech)

    def on_ask(msg_, uid, show_tech):
        msg_ = (msg_ or "").strip() or "Plan my next week"
        tid = f"ui-{uuid.uuid4().hex[:6]}"
        d = fresh_dash()
        add(d, f"You asked: \u201c{msg_}\u201d", "You", "done", f"thread_id={tid}")
        yield {**render(d, uid, show_tech), thread: tid, ask: ""}
        for out in stream_graph(uid, tid, {"session_user": uid, "athlete_id": uid, "request": msg_, "revisions": 0}, d, show_tech):
            yield out

    def on_decide(approved, feedback, uid, tid, d, show_tech):
        if not tid or not d:
            return
        d["feedback"] = feedback if (not approved and feedback) else d.get("feedback", "")
        add(d, "You approved the plan" if approved else (f"You asked for changes: {feedback}" if feedback else "You chose not to save"),
            "You", "done", f"Command(resume=approved={approved})")
        if not approved and feedback:
            d["plan"], d["status"] = "", ""
            d["coach"] = ("working", "Reworking the plan with your request")
        yield render(d, uid, show_tech)
        for out in stream_graph(uid, tid, Command(resume={"approved": approved, "feedback": feedback or ""}), d, show_tech):
            yield out

    def on_share(yes, uid, tid, d, show_tech):
        if not tid or not d:
            return
        for out in stream_graph(uid, tid, Command(resume={"approved": yes}), d, show_tech):
            yield out

    def go_dashboard(uid):
        d = fresh_dash()
        return {**show("home"), **render(d, uid, False), other_page: gr.update(visible=False), top: topbar(uid)}

    def go_dashboard_resume(uid, tid):
        """Back to the dashboard. If a plan is still waiting for approval, it is reloaded from the checkpoint file."""
        d = fresh_dash()
        if tid:
            st = app.get_state({"configurable": {"thread_id": tid}})
            if st.next and st.values.get("athlete_id") == uid:
                v = st.values
                d.update(report=v.get("progress_report") or {}, plan=v.get("plan", ""), handed=True,
                         status="pending" if "human_approval" in st.next else v.get("status", ""),
                         checker=("done", "Report finished and handed over"), coach=("done", "Plan ready for you to review"))
                if "approve_share" in st.next and v.get("share_rid"):
                    d["share"] = gateway._get(v["share_rid"])
                add(d, "Reloaded your conversation from the saved file", "FitPlan", "done", f"app.get_state(thread_id={tid})")
        return {**show("home"), **render(d, uid, False), top: topbar(uid)}

    # --- sign in ---
    def on_signin(u, p):
        ok, msg, uid = login(u, p)
        if not ok:
            return {si_msg: box("errbox", esc(msg))}
        out = {user: uid, si_msg: "", si_pw: "", top: topbar(uid)}
        if get_profile(uid):
            with contextlib.redirect_stdout(io.StringIO()):
                sync_profile(uid)
        if get_profile(uid):
            return {**out, **go_dashboard(uid)}
        return {**out, **show("profile"), **wizard(1), **profile_values(uid), p_msg: ""}

    # --- create account ---
    def on_signup(name, email, pw, pw2, agree):
        ok, msg, uid = create_account(name, email, pw, pw2, agree)
        if not ok:
            return {su_msg: box("errbox", esc(msg))}
        email = email.strip().lower()
        welcome_email(name.strip(), email)
        sent = f" We sent a confirmation email to <b>{esc(email)}</b>." if EMAIL_READY else ""
        return {**show("signin"), su_msg: "", su_pw: "", su_pw2: "", si_pw: "", si_msg: "", si_user: email,
                si_note: box("okbox", f"Account created for <b>{esc(email)}</b>.{sent} Sign in to continue.")}

    # --- forgot / reset password ---
    def on_forgot(email, request: gr.Request):
        ok, msg = request_password_reset(email, request)
        return {fp_msg: box("okbox" if ok else "errbox", esc(msg))}

    def open_reset(token):
        uid, err = check_reset_token(token)
        if not uid:
            return {**show("signin"), si_note: box("errbox", esc(err)), reset_token: ""}
        tok = token.split("reset=", 1)[1].split("&")[0] if "reset=" in token else token.strip()
        return {**show("reset"), reset_token: tok, rs_msg: "", rs_pw: "", rs_pw2: "",
                rs_head: f"<div class='h2'>Choose a new password</div><div class='sub'>For {esc(member_name(uid))}'s account. "
                         "Use at least 8 characters with a letter and a number.</div>"}

    def on_reset(token, pw, pw2):
        ok, msg = reset_password(token, pw, pw2)
        if not ok:
            return {rs_msg: box("errbox", esc(msg))}
        return {**show("signin"), reset_token: "", rs_pw: "", rs_pw2: "", si_pw: "", si_msg: "",
                si_note: box("okbox", esc(msg) + " We also emailed you to confirm the change.")}

    def on_load(request: gr.Request):
        token = dict(request.query_params).get("reset") if request else None
        return open_reset(token) if token else {si_note: gr.update()}

    # --- profile wizard ---
    def check_step1(name, dob, units, h, w):
        age = calc_age(dob)
        if not (name or "").strip():
            return "Please enter a profile name."
        if age is None:
            return "Please enter your date of birth."
        if age < MIN_AGE:
            return f"FitPlan is for members aged {MIN_AGE} and over."
        try:
            hh, ww = to_metric(units, h, w)
        except (TypeError, ValueError):
            return "Please enter your height and weight as numbers."
        if not (120 <= hh <= 230) or not (30 <= ww <= 300):
            return "Height should be 120-230 cm (4-7.5 ft) and weight 30-300 kg (66-660 lb)."
        return ""

    def on_next1(name, dob, units, h, w):
        err = check_step1(name, dob, units, h, w)
        return {p_msg: box("errbox", err)} if err else {p_msg: "", **wizard(2)}

    def on_next2(goal, level, days, *summary_args):
        err = ("Please pick your main goal." if not goal else "Please tell us how experienced you are." if not level
               else "Please pick at least one training day." if not days else "")
        if err:
            return {p_msg: box("errbox", err)}
        return {p_msg: "", **wizard(3), p_summary: summary_html(*summary_args)}

    def on_save(uid, name, dob, sex, units, h, w, goal, level, days, minutes, eq, split, pref, watch, consent, share):
        ok, msg = save_profile(uid, name, dob, sex, units, h, w, goal, level, days, minutes, eq, split,
                               pref, watch, consent, share)
        if not ok:
            return {p_msg: box("errbox", esc(msg))}
        with contextlib.redirect_stdout(io.StringIO()):
            sync_profile(uid)
        done = (f"<div class='done'><div class='tick'>&#10003;</div><div class='h2'>Your profile is saved, {esc(name)}</div>"
                f"<div class='sub'>Your AI team now knows your goal, your {len(days)} training days and your equipment. "
                f"It was saved to <b>fitplan_users.db</b> in your FitPlan folder.</div></div>")
        return {**show("saved"), saved_html: done, p_msg: "", top: topbar(uid)}

    def on_age(dob):
        a = calc_age(dob)
        return f"<div class='sub' style='font-size:13px'>{'Age: <b>' + str(a) + '</b>' if a is not None else 'Your age is worked out for you.'}</div>"

    def on_units(u):
        imperial = u == "ft and lb"
        return {p_h: gr.update(label="Height (ft)" if imperial else "Height (cm)"),
                p_w: gr.update(label="Weight (lb)" if imperial else "Weight (kg)")}

    # ---------------- wiring ----------------
    ALL = list(VIEW_COLS.values()) + [user, top, si_note, si_user, si_pw, si_msg, su_msg, su_pw, su_pw2,
                                      fp_email, fp_msg, reset_token, rs_head, rs_pw, rs_pw2, rs_msg,
                                      step1, step2, step3, p_steps, p_msg, p_summary, saved_html, home_html,
                                      p_name, p_dob, p_sex, p_units, p_h, p_w, p_goal, p_level, p_days, p_min,
                                      p_eq, p_split, p_pref, p_watch, p_consent, p_share,
                                      dash, thread, ask, team, reply_box, feed_box, report_box, plan_box,
                                      decide_row, changes_row, fb, share_box, share_row, other_page,
                                      app_nav, nav_dash, nav_comm, nav_priv, nav_data,
                                      c_pending, c_pending_row, c_note, c_filter, c_list, c_detail, c_edit_box, c_edit_ex,
                                      c_edit_note, c_edit_btn, pv_spaces, pv_action, pv_result, pv_log, pv_audit,
                                      d_files, d_which, d_table, d_tech_txt, r_status, r_review, r_table, r_tech]

    for ev in [si_btn.click, si_user.submit, si_pw.submit]:
        ev(on_signin, [si_user, si_pw], ALL)
    su_in = [su_name, su_email, su_pw, su_pw2, su_agree]
    for ev in [su_btn.click, su_name.submit, su_email.submit, su_pw.submit, su_pw2.submit]:
        ev(on_signup, su_in, ALL)
    su_pw.change(rules_html, [su_pw, su_pw2], su_rules)
    su_pw2.change(rules_html, [su_pw, su_pw2], su_rules)
    si_eye.click(None, None, None, js=eye_js(["si_pw"], "#si_eye"))
    su_eye.click(None, None, None, js=eye_js(["su_pw", "su_pw2"], ".su-eye"))
    su_eye2.click(None, None, None, js=eye_js(["su_pw", "su_pw2"], ".su-eye"))
    go_signup.click(lambda: {**show("signup"), si_msg: ""}, None, ALL)
    go_signin.click(lambda: {**show("signin"), su_msg: ""}, None, ALL)
    go_forgot.click(lambda u: {**show("forgot"), fp_msg: "", fp_email: u if "@" in (u or "") else ""}, si_user, ALL)
    for ev in [fp_btn.click, fp_email.submit]:
        ev(on_forgot, fp_email, ALL)
    for ev in [fp_paste_btn.click, fp_paste.submit]:
        ev(open_reset, fp_paste, ALL)
    for b_ in [fp_back, rs_back]:
        b_.click(lambda: {**show("signin"), fp_msg: "", rs_msg: ""}, None, ALL)
    rs_pw.change(rules_html, [rs_pw, rs_pw2], rs_rules)
    rs_pw2.change(rules_html, [rs_pw, rs_pw2], rs_rules)
    rs_eye.click(None, None, None, js=eye_js(["rs_pw", "rs_pw2"], ".rs-eye"))
    rs_eye2.click(None, None, None, js=eye_js(["rs_pw", "rs_pw2"], ".rs-eye"))
    for ev in [rs_btn.click, rs_pw.submit, rs_pw2.submit]:
        ev(on_reset, [reset_token, rs_pw, rs_pw2], ALL)
    demo.load(on_load, None, ALL)

    s1_in = [p_name, p_dob, p_units, p_h, p_w]
    for ev in [p_next1.click, p_name.submit, p_h.submit, p_w.submit]:
        ev(on_next1, s1_in, ALL)
    s2_in = [p_goal, p_level, p_days, p_name, p_dob, p_goal, p_level, p_days, p_min, p_eq, p_split]
    for ev in [p_next2.click, p_pref.submit]:
        ev(on_next2, s2_in, ALL)
    p_back2.click(lambda: {**wizard(1), p_msg: ""}, None, ALL)
    p_back3.click(lambda: {**wizard(2), p_msg: ""}, None, ALL)
    p_dob.change(on_age, p_dob, p_age)
    p_units.change(on_units, p_units, [p_h, p_w])
    save_in = [user, p_name, p_dob, p_sex, p_units, p_h, p_w, p_goal, p_level, p_days, p_min, p_eq, p_split,
               p_pref, p_watch, p_consent, p_share]
    p_save.click(on_save, save_in, ALL)
    edit_profile.click(lambda uid: {**show("profile"), **wizard(1), **profile_values(uid)}, user, ALL)
    go_home.click(go_dashboard, user, ALL)
    logout.click(lambda: {**show("signin"), user: None, top: topbar(), si_pw: "", si_msg: "", si_note: "",
                          dash: {}, thread: ""}, None, ALL)
    for ev in [ask.submit, ask_btn.click]:
        ev(on_ask, [ask, user, tech], ALL)
    ex1.click(lambda uid, t: (yield from on_ask("Plan my next week", uid, t)), [user, tech], ALL)
    ex2.click(lambda uid, t: (yield from on_ask("Make my sessions 45 minutes", uid, t)), [user, tech], ALL)
    ex3.click(lambda uid, t: (yield from on_ask("How am I progressing?", uid, t)), [user, tech], ALL)
    new_chat.click(lambda uid: render(fresh_dash(), uid, False) | {thread: ""}, user, ALL)
    tech.change(lambda d, t: feed_html((d or {}).get("feed", []), t), [dash, tech], feed_box)
    approve_btn.click(lambda uid, tid, d, t: (yield from on_decide(True, "", uid, tid, d, t)), [user, thread, dash, tech], ALL)
    nosave_btn.click(lambda uid, tid, d, t: (yield from on_decide(False, "", uid, tid, d, t)), [user, thread, dash, tech], ALL)
    changes_btn.click(lambda: {changes_row: gr.update(visible=True)}, None, ALL)
    for ev in [fb.submit, fb_btn.click]:
        ev(lambda f, uid, tid, d, t: (yield from on_decide(False, (f or "").strip() or "Please adjust the plan", uid, tid, d, t)),
           [fb, user, thread, dash, tech], ALL)
    share_yes.click(lambda uid, tid, d, t: (yield from on_share(True, uid, tid, d, t)), [user, thread, dash, tech], ALL)
    share_no.click(lambda uid, tid, d, t: (yield from on_share(False, uid, tid, d, t)), [user, thread, dash, tech], ALL)
    nav_dash.click(go_dashboard_resume, [user, thread], ALL)

    # ----- Community -----
    def community_page(uid, filt, key, tech_, note=""):
        with contextlib.redirect_stdout(io.StringIO()):
            gateway.review_library()              # the member always sees up-to-date statuses
        choices = community_choices(uid, filt)
        keys = [k for _, k in choices]
        key = key if key in keys else (keys[0] if keys else None)
        rec = next((r for r in gateway.library_all(("member", uid)) if record_key(r) == key), None)
        mine = bool(rec and rec["author"] == uid and rec["status"] in ("approved", "outdated"))
        pend = pending_for(uid)
        return {**show("community"), c_pending: share_html(pend) if pend else "", c_pending_row: gr.update(visible=bool(pend)),
                c_note: note, c_list: gr.update(choices=choices, value=key), c_detail: record_html(uid, key, tech_),
                c_edit_btn: gr.update(visible=mine), c_edit_box: gr.update(visible=False)}

    def community_decide(uid, yes, filt, key, tech_):
        pend = pending_for(uid)
        if not pend:
            return community_page(uid, filt, key, tech_)
        with contextlib.redirect_stdout(io.StringIO()):
            res = gateway.library_decide(("member", uid), pend["record_id"], yes)
        note = box("okbox", f"Shared. &ldquo;{esc(pend['title'])}&rdquo; is now ready for anyone's plan. Only you can change it."
                   if res == "approved" else "Kept private. Nothing was shared.")
        return community_page(uid, filt, record_key(pend) if res == "approved" else key, tech_, note)

    def community_open_edit(uid, key):
        rec = next((r for r in gateway.library_all(("member", uid)) if record_key(r) == key), None)
        return {c_edit_box: gr.update(visible=True), c_edit_ex: "\n".join(rec["exercises"]) if rec else "", c_edit_note: ""}

    def community_save_edit(uid, key, ex_text, note, filt, tech_):
        rec = next((r for r in gateway.library_all(("member", uid)) if record_key(r) == key), None)
        exs = [l.strip() for l in (ex_text or "").splitlines() if l.strip()]
        if not rec or not exs:
            return {c_note: box("errbox", "Please list at least one exercise.")}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                v = gateway.library_update(("member", uid), rec["record_id"], {"exercises": exs}, (note or "updated by the author").strip())
        except AccessDenied:
            return {c_note: box("errbox", "Only the person who shared this workout can change it.")}
        return community_page(uid, filt, f"{rec['record_id']}-v{v}", tech_,
                              box("okbox", f"Saved as version {v}. The old version is kept in the history and no longer used by AI plans."))

    nav_comm.click(lambda uid, t: community_page(uid, "All", None, t), [user, c_tech], ALL)
    c_filter.change(lambda uid, f, t: community_page(uid, f, None, t), [user, c_filter, c_tech], ALL)
    c_list.change(lambda uid, f, k, t: community_page(uid, f, k, t), [user, c_filter, c_list, c_tech], ALL)
    c_tech.change(lambda uid, k, t: record_html(uid, k, t), [user, c_list, c_tech], c_detail)
    c_yes.click(lambda uid, f, k, t: community_decide(uid, True, f, k, t), [user, c_filter, c_list, c_tech], ALL)
    c_no.click(lambda uid, f, k, t: community_decide(uid, False, f, k, t), [user, c_filter, c_list, c_tech], ALL)
    c_edit_btn.click(community_open_edit, [user, c_list], ALL)
    c_edit_save.click(community_save_edit, [user, c_list, c_edit_ex, c_edit_note, c_filter, c_tech], ALL)

    # ----- Privacy -----
    def privacy_page(uid, tech_):
        return {**show("privacy"), pv_spaces: spaces_html(uid, tech_), pv_log: gr.update(visible=tech_),
                pv_audit: gr.update(visible=tech_, value=audit_frame(uid)), pv_action: None,
                pv_result: "<div class='panel' style='color:#5B6B82'>Pick something to see what FitPlan allows.</div>"}

    def privacy_try(uid, action, tech_):
        if not action:
            return {pv_result: gr.update()}
        h, log_ = try_action(uid, action)
        return {pv_result: h, pv_log: gr.update(value=log_, visible=tech_), pv_audit: gr.update(value=audit_frame(uid), visible=tech_)}

    nav_priv.click(privacy_page, [user, pv_tech], ALL)
    pv_action.change(privacy_try, [user, pv_action, pv_tech], ALL)
    pv_tech.change(lambda uid, t: {pv_spaces: spaces_html(uid, t), pv_log: gr.update(visible=t),
                                   pv_audit: gr.update(visible=t, value=audit_frame(uid))}, [user, pv_tech], ALL)

    # ----- My data -----
    def data_page(uid, which, tech_):
        df, q = data_table(uid, which)
        return {**show("data"), d_files: files_html(tech_), d_table: df, d_tech_txt: gr.update(value=q, visible=tech_)}

    nav_data.click(lambda uid, t: data_page(uid, "My account", t) | {d_which: "My account"}, [user, d_tech], ALL)
    go_data.click(lambda uid, t: data_page(uid, "My profile", t) | {d_which: "My profile"}, [user, d_tech], ALL)
    d_which.change(data_page, [user, d_which, d_tech], ALL)
    d_tech.change(data_page, [user, d_which, d_tech], ALL)

    # ----- Restart check -----
    def restart_page(uid, tid):
        before = memory_counts(tid, uid)
        return {**show("restart"), r_status: box("okbox", "FitPlan is running. Press the button to check what survives a restart."),
                r_table: restart_table(before, None), r_review: gr.update(visible=False),
                r_tech: f"current process id {before['pid']}\nmemory file: {MEMORY_DB}\ncheckpoint file: {CHECKPOINT_DB}"}

    def restart_run(uid, tid):
        before = memory_counts(tid, uid)
        yield {r_status: box("okbox", "Closing FitPlan&hellip; then opening the saved files in a brand-new program&hellip;")}
        try:
            after = fresh_process_counts(tid, uid)
        except Exception as e:
            yield {r_status: box("errbox", f"The restart check could not run: {esc(e)}")}
            return
        same = all(before[k] == after[k] for k in RESTART_KEYS)
        yield {r_status: box("okbox" if same else "errbox", "Everything came back. All checks match what was there before."
                             if same else "Something did not match. See the table."),
               r_table: restart_table(before, after), r_review: gr.update(visible=bool(after.get("paused"))),
               r_tech: (f"before: process {before['pid']}  |  after: NEW process {after['pid']}\n"
                        f"your memory: {before['me']} -> {after['me']} items\n"
                        f"all members' private memory: {before['private_items']} -> {after['private_items']} items "
                        f"({before['members']} -> {after['members']} members)\n"
                        f"library records: {before['library']} -> {after['library']}\n"
                        f"versions: {', '.join(after['versions'])}\npaused conversation {tid or '-'}: {after['paused']}")}

    d_restart.click(restart_page, [user, thread], ALL)
    r_go.click(restart_run, [user, thread], ALL)
    r_review.click(lambda uid, tid: go_dashboard_resume(uid, tid), [user, thread], ALL)


# --------------------------------------------------------------------------
# Launch. Three ways to open FitPlan:
#  1. ngrok (a link that stays the SAME on every run): add the Colab secrets NGROK_AUTHTOKEN and
#     NGROK_DOMAIN (your free static domain, e.g. something.ngrok-free.app). Needs `pip install ngrok`.
#  2. Colab without ngrok: a temporary gradio.live link (changes on every run, lasts up to 1 week).
#  3. Hugging Face Space: its own fixed address (https://<user>-<space>.hf.space).
# --------------------------------------------------------------------------
NGROK_TOKEN, NGROK_DOMAIN = secret("NGROK_AUTHTOKEN"), secret("NGROK_DOMAIN")

if __name__ == "__main__" or os.environ.get("FITPLAN_LAUNCH") == "1":
    IN_COLAB = "google.colab" in sys.modules
    PORT = int(os.environ.get("PORT", "7860")) if ON_SPACE else int(os.environ.get("FITPLAN_PORT", "7860"))
    use_ngrok = bool(NGROK_TOKEN) and not ON_SPACE
    demo.launch(share=(not ON_SPACE) and (not use_ngrok) and os.environ.get("FITPLAN_SHARE", "1" if IN_COLAB else "0") == "1",
                server_name="0.0.0.0" if ON_SPACE else os.environ.get("FITPLAN_HOST", "127.0.0.1"),
                server_port=PORT, prevent_thread_lock=use_ngrok,
                theme=THEME, css=CSS + DASH_CSS + PAGE_CSS)
    if use_ngrok:
        try:
            import ngrok
            listener = ngrok.forward(PORT, authtoken=NGROK_TOKEN, domain=NGROK_DOMAIN or None)
            APP_URL = listener.url().rstrip("/")
            print("\n" + "=" * 70 + f"\nFitPlan is live at: {APP_URL}\n(the same link every time you run this)\n" + "=" * 70)
        except Exception as e:
            print(f"\nngrok could not start ({str(e)[:120]}). Check NGROK_AUTHTOKEN and NGROK_DOMAIN.\n"
                  "Opening a temporary gradio.live link instead.")
            demo.close()
            demo.launch(share=True, server_name="127.0.0.1", server_port=PORT, prevent_thread_lock=True,
                        theme=THEME, css=CSS + DASH_CSS + PAGE_CSS)
        if not IN_COLAB:
            demo.block_thread()
