---
title: FitPlan
emoji: 🏋️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Two AI agents plan your gym week, with shared memory
---

# FitPlan: an AI training team with shared memory

ITS 58100 Agentic AI, Fall 2026, Project 2. Prateesh Kuruba and Charan Kumar Vemuri.

Two LangGraph agents work together: the **Progress Checker** reviews your training and hands a ProgressReport
to the **Plan Coach**, which writes next week's plan. You approve the plan before anything is saved.
Members keep private memory, and workouts can be shared to a community library only with the author's approval.

All training data used for the tests is synthetic. Saving a plan to the calendar is a simulation.

## Settings this Space needs (Settings > Variables and secrets)

| Kind | Name | Value |
|---|---|---|
| Secret | `DEEPSEEK_API_KEY` | DeepSeek API key |
| Secret | `HF_TOKEN` | Hugging Face token with **write** access (saves the data backup) |
| Variable | `FITPLAN_BACKUP_REPO` | `your-username/fitplan-data` (a private dataset, created automatically) |
