# Multi-Agent Stock Assistant

Website Link: [https://github.com/Atul-Kumar002/Multi-Stock-Research-Assistant](https://stock-research-assistant-smoky.vercel.app/)

## Installation

After cloning, install dependencies:

```bash
cd frontend && npm install
cd ../faithful-fireball && npm install
```

## Secrets / API keys
This repository may include code that expects these environment variables:
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

**Do not commit your `.env` file** or any credential values. Add your keys locally using environment variables (or a local `.env` that is ignored by Git).

## Local database
The backend uses a local SQLite DB file (e.g. `backend/alphamind.db`). This file is **not** meant for public sharing and is ignored by Git.

## GitHub safety checklist (before pushing)
Run:
- `git status`
- ensure `backend/alphamind.db` and any `.env*` files are not staged/committed.

