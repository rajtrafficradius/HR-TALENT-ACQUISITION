"""app.py — Flask application for the Smart HR Talent Acquisition system.

Wires together:
  * single shared-password auth (session cookie)
  * the React SPA (index.html) + all JSON API routes
  * an in-memory JobState registry + daemon-thread discovery runs
  * the daily auto-refresh scheduler (started once, under gunicorn -w 1)

Run locally:   python app.py            (Flask dev server on :8000)
Production:    gunicorn -w 1 -b 0.0.0.0:$PORT app:app   (see railway.toml)
"""
from __future__ import annotations

import hmac
import logging
import os
import secrets
import threading
import time
import uuid
from typing import Dict, Optional

from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import core
import db

# ── logging (ASCII-safe; gunicorn captures stdout on Railway) ────────────────
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hr.app")

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Flask app + session config ───────────────────────────────────────────────
app = Flask(__name__, static_folder=None)
_secret = os.environ.get("SECRET_KEY")
if not _secret:
    _secret = secrets.token_hex(32)
    log.warning("SECRET_KEY not set - using an ephemeral key (sessions reset on restart). "
                "Set SECRET_KEY in production.")
app.secret_key = _secret
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "1") == "1",
    PERMANENT_SESSION_LIFETIME=60 * 60 * 12,
)
CORS(app, supports_credentials=True)

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
if not APP_PASSWORD:
    APP_PASSWORD = "change-me"
    log.warning("APP_PASSWORD not set - defaulting to 'change-me'. Set APP_PASSWORD in production.")

# ── DB + schema bootstrap ────────────────────────────────────────────────────
_db_ready = False
_startup_error: Optional[str] = None
try:
    db.init_pool()
    db.init_schema()
    n = db.RunRepo.reconcile_stuck()
    if n:
        log.info("Reconciled %d stuck 'running' run(s) on boot", n)
    _db_ready = True
    log.info("Database ready.")
except Exception as e:
    _startup_error = str(e)
    log.error("DB startup failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
#  Job registry (in-memory; safe under gunicorn -w 1)
# ═══════════════════════════════════════════════════════════════════════════


class JobState:
    def __init__(self):
        self.state = "running"          # running|done|cancelled|error
        self.progress = 0
        self.status_text = "Starting..."
        self.logs: list = []
        self.log_cursor = 0
        self.start_time = time.time()
        self.run_id: Optional[int] = None
        self.cancel_flag = False
        self.stats: dict = {}
        self.error = ""

    def log(self, msg: str) -> None:
        self.logs.append(msg)


_jobs: Dict[str, JobState] = {}
_jobs_lock = threading.Lock()
_DEFAULT_PARAMS: dict = {}  # scheduled runs read all config from env


def _new_job_id() -> str:
    return uuid.uuid4().hex[:8]


def _run_active() -> bool:
    with _jobs_lock:
        if any(j.state == "running" for j in _jobs.values()):
            return True
    try:
        return db.RunRepo.is_run_active()
    except Exception:
        return False


def _run_discovery_job(job_id: str, params: dict, trigger: str) -> None:
    job = _jobs[job_id]
    try:
        job.run_id = db.RunRepo.create(job_id, trigger, params)
        stats = core.run_discovery(job.run_id, params, job)
        if job.cancel_flag:
            job.state = "cancelled"
            db.RunRepo.finish(job.run_id, "cancelled", stats)
        else:
            job.state = "done"
            job.progress = 100
            db.RunRepo.finish(job.run_id, "done", stats)
        job.stats = stats
    except Exception as e:
        log.exception("discovery job failed")
        job.state = "error"
        job.error = str(e)
        job.log(f"ERROR: {e}")
        if job.run_id:
            try:
                db.RunRepo.finish(job.run_id, "error", job.stats, str(e))
            except Exception:
                pass


def _start_run(params: dict, trigger: str) -> Optional[str]:
    if _run_active():
        return None
    job_id = _new_job_id()
    with _jobs_lock:
        _jobs[job_id] = JobState()
        # keep registry bounded
        if len(_jobs) > 50:
            for k in list(_jobs.keys())[:-25]:
                if _jobs[k].state != "running":
                    _jobs.pop(k, None)
    threading.Thread(target=_run_discovery_job, args=(job_id, params, trigger),
                     daemon=True).start()
    return job_id


def _trigger_scheduled_run(params: Optional[dict] = None) -> bool:
    return _start_run(params or _DEFAULT_PARAMS, "scheduled") is not None


# ── scheduler bootstrap (exactly once) ───────────────────────────────────────
_sched_started = False
_sched_lock = threading.Lock()


def _start_scheduler_once() -> None:
    global _sched_started
    with _sched_lock:
        if _sched_started or not _db_ready:
            return
        if os.environ.get("HR_DAILY_REFRESH_ENABLED", "1") != "1":
            log.info("Daily refresh disabled (HR_DAILY_REFRESH_ENABLED!=1)")
            _sched_started = True
            return
        _sched_started = True
        stop = threading.Event()
        app.config["_sched_stop"] = stop
        threading.Thread(target=core.scheduler_loop, args=(stop, _trigger_scheduled_run),
                         daemon=True).start()
        log.info("Daily scheduler started.")


_start_scheduler_once()


# ═══════════════════════════════════════════════════════════════════════════
#  Auth
# ═══════════════════════════════════════════════════════════════════════════


def require_auth(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authed"):
            return jsonify({"error": "unauthorized"}), 401
        return view(*args, **kwargs)
    return wrapped


def _require_db():
    if not _db_ready:
        return jsonify({"error": "database_unavailable", "detail": _startup_error}), 503
    return None


@app.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    pw = data.get("password", "") or request.form.get("password", "")
    if hmac.compare_digest(str(pw), str(APP_PASSWORD)):
        session.permanent = True
        session["authed"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "invalid_password"}), 401


@app.post("/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/whoami")
def whoami():
    return jsonify({"authenticated": bool(session.get("authed"))})


# ═══════════════════════════════════════════════════════════════════════════
#  SPA + health
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/")
def index():
    return send_from_directory(_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "db_ready": _db_ready,
                    "db_healthy": db.healthcheck() if _db_ready else False,
                    "pool_error": db.pool_error(), "startup_error": _startup_error})


# ═══════════════════════════════════════════════════════════════════════════
#  Data API
# ═══════════════════════════════════════════════════════════════════════════


def _page_args() -> tuple:
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    try:
        page_size = int(request.args.get("page_size", 50))
    except ValueError:
        page_size = 50
    return page, page_size


@app.get("/api/stats")
@require_auth
def api_stats():
    if (r := _require_db()):
        return r
    return jsonify(db.StatsRepo.counts())


@app.get("/api/filters")
@require_auth
def api_filters():
    if (r := _require_db()):
        return r
    return jsonify({
        "departments": db.CandidateRepo.distinct_departments(),
        "seniorities": db.CandidateRepo.distinct_seniorities(),
        "categories": core.CATEGORY_LABELS,                       # full fixed 28-item list
        "active_categories": db.CandidateRepo.distinct_categories(),  # categories with data
        "groups": [{"name": g, "categories": core.GROUPS[g]} for g in core.GROUP_ORDER],
        "industries": db.CompanyRepo.distinct_industries(),
        "companies": db.CandidateRepo.companies_for_filter(),
        "countries": ["India", "Australia"],
        "company_country_counts": db.CompanyRepo.country_counts(),
        "size_bands": core.SIZE_LABELS,                       # 10-level size taxonomy
        "size_band_ranges": {b[2]: [b[0], b[1]] for b in core.SIZE_BANDS},
        "company_size_counts": db.CompanyRepo.size_band_counts(),
        "enrichment_statuses": ["not_enriched", "enriching", "enriched", "failed", "no_credits"],
        "all_departments": ["sales", "marketing", "seo", "digital_marketing", "other"],
    })


@app.get("/api/people")
@require_auth
def api_people():
    if (r := _require_db()):
        return r
    page, page_size = _page_args()
    filters = {k: request.args.get(k) for k in (
        "department", "category", "country", "seniority", "company_id", "enrichment_status",
        "min_overall", "min_intent", "min_company_score", "open_to_shift", "freshness", "q", "sort")
        if request.args.get(k) not in (None, "")}
    rows, total = db.CandidateRepo.list_page(filters, page, page_size)
    return jsonify({"rows": rows, "total": total, "page": page, "page_size": page_size,
                    "total_pages": max(1, -(-total // page_size))})


@app.get("/api/people/<int:cid>")
@require_auth
def api_person(cid):
    if (r := _require_db()):
        return r
    cand = db.CandidateRepo.get(cid)
    if not cand:
        return jsonify({"error": "not_found"}), 404
    company = db.CompanyRepo.get(cand["company_id"]) if cand.get("company_id") else None
    if company and not company.get("industry"):
        company["industry_derived"] = core.derive_industry(company["id"])

    # Interconnected web: resolve this person's PAST companies to DB rows (clickable).
    # employment_history only exists for Apollo-enriched candidates.
    past_companies = []
    eh = cand.get("employment_history_json") or []
    if isinstance(eh, list) and eh:
        cur_key = (company or {}).get("company_key") or core.company_key_for(cand.get("company_name") or "")
        pairs = []
        for e in eh:
            if not isinstance(e, dict):
                continue
            nm = e.get("organization_name")
            if not nm:
                continue
            key = core.company_key_for(nm)
            if e.get("current") or not key or key == cur_key:
                continue  # skip the current employer (already shown above)
            pairs.append((e, key))
        idmap = db.CompanyRepo.ids_by_keys([k for _, k in pairs]) if pairs else {}
        for e, key in pairs:
            past_companies.append({
                "name": e.get("organization_name"), "title": e.get("title"),
                "start_date": e.get("start_date"), "end_date": e.get("end_date"),
                "current": bool(e.get("current")), "company_id": idmap.get(key)})

    return jsonify({"candidate": cand, "company": company, "past_companies": past_companies,
                    "enrichment_log": db.EnrichmentLogRepo.for_candidate(cid)})


@app.get("/api/companies")
@require_auth
def api_companies():
    if (r := _require_db()):
        return r
    page, page_size = _page_args()
    filters = {k: request.args.get(k) for k in (
        "country", "category", "min_quality", "min_employees", "max_employees",
        "size_band", "q", "sort") if request.args.get(k) not in (None, "")}
    rows, total = db.CompanyRepo.list_page(filters, page, page_size)
    return jsonify({"rows": rows, "total": total, "page": page, "page_size": page_size,
                    "total_pages": max(1, -(-total // page_size))})


@app.get("/api/companies/<int:coid>")
@require_auth
def api_company(coid):
    if (r := _require_db()):
        return r
    company = db.CompanyRepo.get(coid)
    if not company:
        return jsonify({"error": "not_found"}), 404
    if not company.get("industry"):
        company["industry_derived"] = core.derive_industry(coid)
    return jsonify({"company": company, "people": db.CandidateRepo.for_company(coid)})


@app.post("/api/enrich/<int:cid>")
@require_auth
def api_enrich(cid):
    if (r := _require_db()):
        return r
    data = request.get_json(silent=True) or {}
    reveal_email = bool(data.get("reveal_email", True))
    reveal_phone = bool(data.get("reveal_phone", False))

    acquired = db.CandidateRepo.set_enriching(cid)
    if not acquired:
        cur = db.CandidateRepo.get(cid)
        if not cur:
            return jsonify({"error": "not_found"}), 404
        if cur["enrichment_status"] == "enriched":
            return jsonify({"ok": True, "status": "enriched", "candidate": cur, "cached": True})
        return jsonify({"ok": False, "status": cur["enrichment_status"],
                        "error": "already_in_progress"}), 409

    res = core.enrich_candidate(cid, reveal_email=reveal_email, reveal_phone=reveal_phone)
    if not res.get("ok"):
        code = 409 if res.get("error") == "no_credits" else 400
        return jsonify(res), code
    return jsonify(res)


@app.get("/api/credits")
@require_auth
def api_credits():
    rem = core.get_apollo().credits_remaining()
    return jsonify({"credits_remaining": rem, "available": rem >= 0})


@app.post("/api/discover")
@require_auth
def api_discover():
    if (r := _require_db()):
        return r
    data = request.get_json(silent=True) or {}
    params = {}
    if data.get("categories"):
        params["categories"] = [c for c in data["categories"] if c in core.CATEGORY_DEPT]
    elif data.get("groups"):
        params["groups"] = [g for g in data["groups"] if g in core.GROUPS]
    elif data.get("departments"):
        params["departments"] = [d for d in data["departments"]
                                 if d in ("sales", "marketing", "seo", "digital_marketing")]
    if data.get("person_locations"):
        params["person_locations"] = data["person_locations"]
        params["organization_locations"] = data["person_locations"]
    for k in ("max_pages", "max_candidates"):
        if data.get(k):
            try:
                params[k] = int(data[k])
            except (ValueError, TypeError):
                pass
    if data.get("use_seed_domains"):
        params["use_seed_domains"] = True
    if data.get("seed_domains"):
        params["seed_domains"] = [s.strip() for s in data["seed_domains"] if s.strip()]

    job_id = _start_run(params, "manual")
    if job_id is None:
        return jsonify({"error": "busy", "detail": "a discovery run is already active"}), 409
    return jsonify({"job_id": job_id}), 202


@app.get("/api/status/<job_id>")
@require_auth
def api_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown_job"}), 404
    new_logs = job.logs[job.log_cursor:]
    job.log_cursor = len(job.logs)
    out = {"state": job.state, "progress": job.progress, "status_text": job.status_text,
           "new_logs": new_logs, "elapsed_seconds": int(time.time() - job.start_time),
           "stats": job.stats}
    if job.state == "error":
        out["error"] = job.error
    return jsonify(out)


@app.post("/api/cancel/<job_id>")
@require_auth
def api_cancel(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "unknown_job"}), 404
    job.cancel_flag = True
    job.status_text = "Cancelling..."
    return jsonify({"status": "cancelling"})


@app.get("/api/hunt")
@require_auth
def api_hunt_status():
    return jsonify(core.hunt_status())


@app.post("/api/hunt")
@require_auth
def api_hunt_toggle():
    """Master auto-hunt toggle — when ON, the background scheduler continuously
    rotates the category taxonomy and runs FREE discovery slices (zero credits)."""
    data = request.get_json(silent=True) or {}
    core.set_auto_hunt(bool(data.get("enabled")))
    return jsonify({"ok": True, **core.hunt_status()})


@app.post("/api/linkedin-enrich/<int:cid>")
@require_auth
def api_linkedin_enrich(cid):
    """Confirm job-change intent + capture a concise profile from the candidate's
    public LinkedIn (best-effort) + AI. FREE — no Apollo credits."""
    if (r := _require_db()):
        return r
    res = core.enrich_linkedin(cid)
    if not res.get("ok"):
        return jsonify(res), (404 if res.get("error") == "not_found" else 400)
    return jsonify(res)


@app.get("/api/roster")
@require_auth
def api_roster_status():
    return jsonify(core.roster_status())


@app.post("/api/roster")
@require_auth
def api_roster_toggle():
    """Master toggle for the roster verifier — when ON, the scheduler walks every
    company and adds any Apollo employees missing from the DB (free)."""
    data = request.get_json(silent=True) or {}
    core.set_roster_verify(bool(data.get("enabled")))
    return jsonify({"ok": True, **core.roster_status()})


@app.get("/api/runs")
@require_auth
def api_runs():
    if (r := _require_db()):
        return r
    page, page_size = _page_args()
    rows, total = db.RunRepo.list_page(page, page_size)
    return jsonify({"rows": rows, "total": total, "page": page, "page_size": page_size,
                    "total_pages": max(1, -(-total // page_size))})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
