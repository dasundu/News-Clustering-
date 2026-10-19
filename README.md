# News Clustering — Sri Lankan news event clustering

Groups articles from Sri Lankan English-language news outlets into **events** (one real-world
occurrence, reported by several outlets) whose IDs stay the same every time the pipeline runs
again, and groups related events into **topics**. Results are served as JSON over a REST API.

This is the event clustering component (C1) of the PerspectiveLK platform. The main project
repository is `DileepWick/perspective-lk`; this repository is the deployment copy of
`clustering/backend`, with its full commit history.

## How it runs

| Piece | What it does | Where it runs |
|---|---|---|
| Pipeline (`pipeline.py`) | scrape → extract → clean → embed → cluster | a home connection (`main.py`, every 2 hours); GitHub Actions by hand only ([pipeline.yml](.github/workflows/pipeline.yml)) |
| API (`api.py`) | serves events, topics, articles and claim search | Render free tier ([render.yaml](render.yaml)) |
| Database | articles, embeddings, clusters, topics | Neon (PostgreSQL + pgvector) |

The pipeline and the API never talk to each other directly; the database is the only shared
state. The API runs read-only (`PIPELINE_ENABLED=false`).

## Run locally

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows; venv/bin/python elsewhere
cp .env.example .env    # set DATABASE_URL and GEMINI_API_KEY

venv/Scripts/python.exe pipeline.py          # one pipeline run
venv/Scripts/python.exe main.py              # pipeline on a schedule
venv/Scripts/python.exe main.py --api        # API on http://localhost:8000 (docs at /docs)
```

## Deploy

1. **Secrets for the pipeline:** Settings → Secrets and variables → Actions → add
   `DATABASE_URL` and `GEMINI_API_KEY`. Then Actions → Clustering pipeline → Run workflow.
2. **API on Render:** New → Blueprint → choose this repository. Enter `DATABASE_URL` and
   `GEMINI_API_KEY` when asked. Check `https://<your-service>.onrender.com/stats`.
3. **Website:** set `CLUSTERING_API_URL` to the Render address in the website's hosting settings.

The free Render service sleeps after about 15 minutes idle and takes 30–60 seconds to wake.
Only one pipeline should write to the database at a time: stop the local scheduler before
running the GitHub Actions workflow by hand.

## Evaluation tools

| Script | What it measures |
|---|---|
| `compute_metrics.py` | internal cluster quality: coherence, separation, silhouette, coverage |
| `identity_retention.py` | how many event IDs survive each consecutive run, from `scraper.log` |
| `gold_set.py` | agreement with hand labels (ARI, NMI, pairwise and B-cubed F1) against baselines |
