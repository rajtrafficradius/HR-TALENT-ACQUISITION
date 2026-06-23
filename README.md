# RADIUS · Talent Acquisition

A focused HR tool that **sources job-change-ready candidates** at other companies —
concentrated on **sales, marketing, SEO and digital-marketing** roles — scores them
on six dimensions, stores them in MySQL, and lets HR browse, filter, and reveal
contacts through a clean **People View** and **Company View**.

Built to be **free-first**: candidate discovery runs entirely on Apollo's free People
Search (no credits). Email/phone are revealed only when you click **Enrich** on a
candidate, one at a time.

```
app.py      Flask: shared-password auth, all API routes, JobState, daily scheduler
core.py     Apollo client, 6-dimension scoring engine, intent providers, discovery,
            best-effort company crawler, OpenAI refinement, enrichment, scheduler loop
db.py       MySQL pool (PyMySQL + DBUtils, ping=4) + schema + repositories
index.html  React 18 + Babel SPA (no build step) — industrial black/white theme
```

---

## How it works

1. **Discovery (free).** A run pages Apollo `POST /mixed_people/api_search` across a
   matrix of `{role family} × {seniority group}` for your chosen regions. It stores
   each person and their company. **No contact credits are spent.** The daily
   scheduler does this automatically once per day.
2. **Scoring.** Every candidate gets six deterministic scores (0–100):
   `technical`, `role_fit`, `job_change_intent`, `company_quality`, `freshness`,
   and a weighted `overall`. Sort/filter by any of them.
3. **Job-change intent.** Inferred, and labelled by source:
   - `Heuristic` — from title/seniority/company signals (free discovery default).
   - `History · verified` — recomputed from real `employment_history` (tenure,
     job-hop frequency) after you Enrich a candidate.
   - `LinkedIn ✓` — **reserved** for a future LinkedIn integration (see below).
   A candidate is flagged **Open to shift** only when intent ≥ `HR_INTENT_OPEN_THRESHOLD`
   (default 60) — a high-confidence bar that heuristic alone rarely clears, so the
   badge is earned by evidence, not guessed.
4. **Enrich (costs Apollo credits).** Clicking **Enrich** calls `POST /people/match`
   to reveal email (and optionally phone), pull real name + LinkedIn + employment
   history, upgrade intent to `History · verified`, and **auto-fill the company's
   firmographics** (domain, employees, industry…) for free. Every reveal is logged
   and counted against per-day caps.

> **Free-data note.** Apollo's free search returns a thin payload (first name,
> obfuscated last name, title, company *name*, and `has_email`/`has_phone` flags).
> Real names, domains, seniority and firmographics arrive on Enrich. Companies are
> therefore keyed by name until enriched, then upgraded with the real domain.

---

## Local development

```bash
pip install -r requirements.txt
cp .env.example .env          # fill APOLLO_API_KEY, OPENAI_API_KEY, MYSQL_PUBLIC_URL,
                              # APP_PASSWORD, SECRET_KEY
python app.py                 # http://localhost:8000
python -m pytest test_scoring.py -q     # scoring engine unit tests
```

Locally, point at the database with `MYSQL_PUBLIC_URL` (the Railway public proxy).
The schema auto-creates on first boot.

---

## Deploy on Railway (via GitHub)

1. Create a **new MySQL** service on Railway (or reuse the dedicated one) and note its
   connection variables.
2. Push this repo to GitHub and create a Railway service from it.
3. Set the service **Variables**:
   - `APOLLO_API_KEY`, `OPENAI_API_KEY` (optional)
   - MySQL: set the discrete `MYSQLHOST/MYSQLPORT/MYSQLUSER/MYSQLPASSWORD/MYSQLDATABASE`
     (Railway injects these when you reference the MySQL service)
   - `SECRET_KEY` (`python -c "import secrets;print(secrets.token_hex(32))"`),
     `APP_PASSWORD`, `SESSION_COOKIE_SECURE=1`
   - tuning: `HR_PERSON_LOCATIONS`, `HR_TARGET_DEPARTMENTS`, caps (see `.env.example`)
4. Railway reads `railway.toml` → `gunicorn -w 1 app:app`. **`-w 1` is required** so the
   in-memory job registry, rate-limiter, and daily scheduler share one process.

Schema auto-creates on first boot; the daily discovery sweep starts automatically.

---

## Future LinkedIn integration (reserved, no migration needed)

Job-change intent is computed through a provider seam (`core.compute_intent`). Today
`ApolloIntentProvider` is active and `LinkedInIntentProvider` is **stubbed and inert**
while `LINKEDIN_ENABLED=0`. When you obtain a LinkedIn API/crawler:

1. Implement `LinkedInIntentProvider.fetch()` in `core.py` (read Open-to-Work / recent
   activity from the stored `linkedin_url`).
2. Set `LINKEDIN_API_KEY` and `LINKEDIN_ENABLED=1`.

The DB columns (`intent_source`, `linkedin_open_to_work`, `linkedin_signals_json`,
`linkedin_checked_at`) and the UI source badge already exist — verified candidates
will light up with **no schema change and no front-end work**.

---

## API surface (all JSON, session-cookie auth)

| Method | Path | Purpose |
|---|---|---|
| POST | `/login` `/logout`, GET `/whoami` | shared-password auth |
| GET | `/health` | liveness + DB status |
| GET | `/api/stats` `/api/filters` | KPIs + dropdown sources |
| GET | `/api/people` `/api/people/<id>` | filtered candidates + detail |
| GET | `/api/companies` `/api/companies/<id>` | companies + profile w/ people |
| POST | `/api/enrich/<id>` | Apollo reveal (costs credits, capped) |
| GET | `/api/credits` | Apollo credit balance |
| POST | `/api/discover`, GET `/api/status/<job>`, POST `/api/cancel/<job>` | run a sweep |
| GET | `/api/runs` | run history |
