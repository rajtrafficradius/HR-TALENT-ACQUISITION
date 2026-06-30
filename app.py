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
from werkzeug.middleware.proxy_fix import ProxyFix

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
# Railway terminates TLS at its edge and forwards to the app over plain HTTP; trust the
# X-Forwarded-* headers so request.scheme/host reflect the real public HTTPS origin (needed so
# the Apollo phone webhook is built as a valid https:// URL — Apollo 400s on http).
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
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


def _public_base_url() -> str:
    """Public base URL for outbound callbacks. Apollo REQUIRES https for the phone webhook and
    rejects http with 400 ('Webhook URL is not a valid HTTPS URL'). Railway terminates TLS at its
    edge, so the raw request can look like http://. Resolution order: PUBLIC_BASE_URL env override →
    forwarded host/proto → force https for any non-local host."""
    env = (os.environ.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if env:
        return env if "://" in env else ("https://" + env)
    host = (request.headers.get("X-Forwarded-Host") or request.host or "").split(",")[0].strip()
    proto = (request.headers.get("X-Forwarded-Proto") or "").split(",")[0].strip() or request.scheme or "http"
    if host and not (host.startswith("localhost") or host.startswith("127.0.0.1")):
        proto = "https"  # public host sits behind Railway's TLS proxy
    return f"{proto}://{host}"


def _phone_webhook_url() -> str:
    """The exact (https) webhook URL Apollo is told to call for async phone reveals — or '' if no
    token is configured. Always HTTPS for public hosts so Apollo accepts the reveal and returns a
    request_id (which the server-side poller then uses to pull the number)."""
    tok = core.apollo_webhook_token()
    if not tok:
        return ""
    return _public_base_url().rstrip("/") + "/api/apollo-phone-webhook?token=" + tok


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
    # One-shot data migrations (run once on the Railway host where the DB is local & fast):
    #  (1) consolidate every stored category to the 12-taxonomy; (2) drop clearly-irrelevant
    #  companies (schools, textiles/garments, banks, mega-corps). Each guarded by a settings flag.
    try:
        if db.SettingsRepo.get("cat12_migrated") != "1":
            r = core.migrate_categories_to_12()
            db.SettingsRepo.set("cat12_migrated", "1")
            log.info("cat12 migration: remapped %d candidates / %d companies", r["candidates"], r["companies"])
        if db.SettingsRepo.get("irrelevant_cleaned_v2") != "1":
            removed = core.cleanup_hard_blocked_companies()
            db.SettingsRepo.set("irrelevant_cleaned_v2", "1")
            log.info("irrelevant cleanup: removed %d clearly-irrelevant companies", removed)
    except Exception as e:
        log.warning("startup data migration skipped (will retry next boot): %s", e)
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
        "categories": core.CATEGORY_LABELS,                       # full fixed taxonomy
        "active_categories": db.CandidateRepo.distinct_categories(),  # categories with data
        "category_company_counts": db.CompanyRepo.category_company_counts(),  # '(N)' per category
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
    for row in rows:  # LinkedIn-first effective contacts for the list (Avail icons + source)
        row.update(core.effective_contacts(row))
    return jsonify({"rows": rows, "total": total, "page": page, "page_size": page_size,
                    "total_pages": max(1, -(-total // page_size))})


@app.get("/api/people/export.csv")
@require_auth
def api_people_export():
    """Export the CURRENT filtered candidate list as CSV with LinkedIn-first effective contacts —
    the recruiter's working shortlist. Capped to keep it snappy."""
    import csv, io
    if (r := _require_db()):
        return r
    filters = {k: request.args.get(k) for k in (
        "department", "category", "country", "seniority", "company_id", "enrichment_status",
        "min_overall", "min_intent", "min_company_score", "open_to_shift", "freshness", "q", "sort")
        if request.args.get(k) not in (None, "")}
    limit = min(int(request.args.get("limit", 5000) or 5000), 10000)
    rows, _ = db.CandidateRepo.list_page(filters, 1, limit, max_size=limit)
    cols = [("full_name", "Name"), ("title", "Title"), ("company_name", "Company"),
            ("company_domain", "Company Website"), ("category", "Category"), ("seniority", "Seniority"),
            ("location_country", "Country"), ("job_change_intent_score", "Intent"),
            ("overall_candidate_score", "Overall"), ("open_to_shift", "Open to Shift"),
            ("email_effective", "Email"), ("email_source", "Email Source"),
            ("phone_effective", "Phone"), ("phone_source", "Phone Source"),
            ("linkedin_url", "LinkedIn")]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([h for _, h in cols])
    for row in rows:
        row.update(core.effective_contacts(row))
        w.writerow([row.get(k, "") if row.get(k) is not None else "" for k, _ in cols])
    from flask import Response
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=candidates.csv"})


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
        # Group by company so each past employer appears ONCE — multiple roles become sub-points.
        grouped = {}
        order = []
        for e in eh:
            if not isinstance(e, dict):
                continue
            nm = e.get("organization_name")
            if not nm:
                continue
            key = core.company_key_for(nm)
            if e.get("current") or not key or key == cur_key:
                continue  # skip the current employer (already shown above)
            role = {"title": e.get("title"), "start_date": e.get("start_date"),
                    "end_date": e.get("end_date")}
            if key not in grouped:
                grouped[key] = {"name": nm, "roles": [role],
                                "start_date": e.get("start_date"), "end_date": e.get("end_date")}
                order.append(key)
            else:
                g = grouped[key]
                g["roles"].append(role)
                # widen the span: earliest start, latest end
                if e.get("start_date") and (not g["start_date"] or str(e["start_date"]) < str(g["start_date"])):
                    g["start_date"] = e["start_date"]
                if not e.get("end_date") or (g["end_date"] and str(e.get("end_date") or "") > str(g["end_date"])):
                    g["end_date"] = e.get("end_date")
        idmap = db.CompanyRepo.ids_by_keys(order) if order else {}
        for key in order:
            g = grouped[key]
            past_companies.append({
                "name": g["name"], "title": g["roles"][0].get("title"),
                "roles": g["roles"], "start_date": g["start_date"], "end_date": g["end_date"],
                "current": False, "company_id": idmap.get(key)})

    # AI recruiter brief (paragraph) — generated once, then cached on the row.
    ai_paragraph = cand.get("ai_paragraph")
    ai_paragraph_source = cand.get("ai_paragraph_source")
    if not ai_paragraph:
        try:
            res = core.generate_candidate_paragraph(cand, company)
            ai_paragraph = res.get("paragraph")
            ai_paragraph_source = res.get("source")
            if ai_paragraph:
                db.CandidateRepo.set_ai_paragraph(cid, ai_paragraph, ai_paragraph_source or "")
        except Exception:
            ai_paragraph = None

    # LinkedIn-sourced contacts take precedence over Apollo (regardless of Apollo's state).
    cand.update(core.effective_contacts(cand))

    return jsonify({"candidate": cand, "company": company, "past_companies": past_companies,
                    "ai_paragraph": ai_paragraph, "ai_paragraph_source": ai_paragraph_source,
                    "enrichment_log": db.EnrichmentLogRepo.for_candidate(cid)})


@app.get("/api/people/<int:cid>/phone")
@require_auth
def api_person_phone(cid):
    """Light poll endpoint — the candidate panel calls this after 'Get mobile' to surface the
    async-delivered mobile without reloading the whole record."""
    if (r := _require_db()):
        return r
    cand = db.CandidateRepo.get(cid)
    if not cand:
        return jsonify({"error": "not_found"}), 404
    # A LinkedIn-sourced mobile wins immediately (no Apollo dependency at all).
    li_phone = (cand.get("linkedin_phone") or "").strip()
    if li_phone:
        return jsonify({"phone": li_phone, "has_phone": True, "pending": False,
                        "resolved": True, "source": "linkedin"})
    # If an Apollo reveal is still pending, ACTIVELY pull the result now (drives delivery from this
    # poll — no dependency on the scheduler or Apollo's inbound webhook reaching us).
    if cand.get("phone_pending") and not cand.get("phone"):
        try:
            r = core.poll_one_phone(cid)
            return jsonify({"phone": r.get("phone"),
                            "has_phone": bool(cand.get("has_phone") or r.get("phone")),
                            "pending": bool(r.get("pending")), "resolved": bool(r.get("resolved")),
                            "reason": r.get("reason"), "source": "apollo"})
        except Exception:
            pass
    return jsonify({"phone": cand.get("phone"), "has_phone": bool(cand.get("has_phone")),
                    "pending": bool(cand.get("phone_pending")), "resolved": bool(cand.get("phone")),
                    "source": "apollo" if cand.get("phone") else None})


@app.post("/api/people/<int:cid>/ai-refresh")
@require_auth
def api_person_ai_refresh(cid):
    """Regenerate the AI recruiter brief on demand (re-runs OpenAI / deterministic)."""
    if (r := _require_db()):
        return r
    cand = db.CandidateRepo.get(cid)
    if not cand:
        return jsonify({"error": "not_found"}), 404
    company = db.CompanyRepo.get(cand["company_id"]) if cand.get("company_id") else None
    res = core.generate_candidate_paragraph(cand, company)
    if res.get("paragraph"):
        db.CandidateRepo.set_ai_paragraph(cid, res["paragraph"], res.get("source") or "")
    return jsonify({"ok": True, "ai_paragraph": res.get("paragraph"), "ai_paragraph_source": res.get("source")})


@app.get("/api/score-explanations")
@require_auth
def api_score_explanations():
    """Plain-language derivation of every score — powers the 'i' explainer buttons."""
    return jsonify(core.SCORE_EXPLANATIONS)


@app.get("/api/diag/phone")
@require_auth
def api_diag_phone():
    """Diagnostics for the async phone flow: shows the exact webhook URL Apollo would be told to
    call (must be a PUBLIC https URL), whether the token is set, and the last phone attempts."""
    tok = core.apollo_webhook_token()
    webhook_url = _phone_webhook_url()
    base = _public_base_url()
    return jsonify({
        "webhook_url": webhook_url,
        "webhook_token_set": bool(tok),
        "public_base_url": base,
        "raw_host_url": request.host_url,
        "x_forwarded_proto": request.headers.get("X-Forwarded-Proto"),
        "is_public_https": webhook_url.startswith("https://") and "localhost" not in webhook_url
                           and "127.0.0.1" not in webhook_url,
        "apollo_configured": bool(getattr(core.get_apollo(), "api_key", "")),
        "phones_in_db": db.CandidateRepo.phone_populated_count(),
        "phone_pending": db.CandidateRepo.phone_pending_stats(),
        "recent_phone_attempts": db.EnrichmentLogRepo.recent_phone_attempts(10),
    })


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
    # Website fallback: if the company row has no domain but its people carry one, surface it so
    # the website box stops showing "pending" (and persist it for next time, free).
    if not company.get("root_domain"):
        d = db.CompanyRepo.domain_from_candidates(coid)
        if d:
            company["root_domain"] = d
            company["website_url"] = company.get("website_url") or ("https://" + d)
            try:
                db.CompanyRepo.set_domain(coid, d, company["website_url"])
            except Exception:
                pass
    # OpenAI company summary (what it does / how long / solutions) — generate once, then cache.
    if not company.get("ai_summary"):
        try:
            res = core.generate_company_summary(company)
            if res.get("summary"):
                company["ai_summary"] = res["summary"]
                company["ai_summary_source"] = res.get("source")
                db.CompanyRepo.set_ai_summary(coid, res["summary"], res.get("source") or "")
        except Exception:
            pass
    return jsonify({"company": company, "people": db.CandidateRepo.for_company(coid)})


@app.post("/api/enrich/<int:cid>")
@require_auth
def api_enrich(cid):
    if (r := _require_db()):
        return r
    data = request.get_json(silent=True) or {}
    reveal_email = bool(data.get("reveal_email", True))
    reveal_phone = bool(data.get("reveal_phone", True))  # phone arrives async via webhook

    acquired = db.CandidateRepo.set_enriching(cid)
    if not acquired:
        cur = db.CandidateRepo.get(cid)
        if not cur:
            return jsonify({"error": "not_found"}), 404
        if cur["enrichment_status"] == "enriched":
            return jsonify({"ok": True, "status": "enriched", "candidate": cur, "cached": True})
        return jsonify({"ok": False, "status": cur["enrichment_status"],
                        "error": "already_in_progress"}), 409

    # Phone is async: Apollo needs a valid HTTPS webhook to ACCEPT the reveal (and return a
    # request_id the poller uses). Build it as https regardless of Railway's internal http hop.
    webhook_url = _phone_webhook_url() if reveal_phone else ""
    res = core.enrich_candidate(cid, reveal_email=reveal_email, reveal_phone=reveal_phone,
                                webhook_url=webhook_url)
    if not res.get("ok"):
        code = 409 if res.get("error") == "no_credits" else 400
        return jsonify(res), code
    res["phone_pending"] = bool(reveal_phone and webhook_url)
    return jsonify(res)


@app.post("/api/reveal-phone/<int:cid>")
@require_auth
def api_reveal_phone(cid):
    """Backfill the mobile/direct phone for a candidate (works even when already enriched).
    Apollo delivers it async to the phone webhook a few minutes later."""
    if (r := _require_db()):
        return r
    webhook_url = _phone_webhook_url()
    res = core.reveal_phone_only(cid, webhook_url=webhook_url)
    if not res.get("ok"):
        err = res.get("error")
        code = 404 if err == "not_found" else (402 if err == "no_credits" else 400)
        return jsonify(res), code
    return jsonify(res)


@app.post("/api/apollo-phone-webhook")
def api_apollo_phone_webhook():
    """Receives Apollo's async phone-reveal callback and writes the number onto the
    candidate. Public (Apollo posts here) but guarded by a secret token in the query."""
    tok = request.args.get("token") or request.headers.get("X-Webhook-Token") or ""
    want = core.apollo_webhook_token()
    if not want or tok != want:
        return jsonify({"error": "forbidden"}), 403
    if (r := _require_db()):
        return r
    data = request.get_json(silent=True) or {}
    updated = core.handle_apollo_phone_webhook(data)
    return jsonify({"ok": True, "updated": updated})


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


@app.get("/api/coresignal-status")
@require_auth
def api_coresignal_status():
    """Whether CoreSignal is configured + the last-seen remaining credits."""
    return jsonify(core.coresignal_status())


@app.post("/api/coresignal-enrich/<int:cid>")
@require_auth
def api_coresignal_enrich(cid):
    """Manual, per-candidate LinkedIn enrichment via CoreSignal (employee_multi_source).
    PAID — consumes CoreSignal credits. Optional body {"employee_id": N} collects a
    specific record after a manual disambiguation pick."""
    if (r := _require_db()):
        return r
    body = request.get_json(silent=True) or {}
    employee_id = body.get("employee_id")
    res = core.enrich_coresignal(cid, employee_id=employee_id)
    if not res.get("ok"):
        err = res.get("error")
        if err == "not_found":
            return jsonify(res), 404
        if err == "not_configured":
            return jsonify(res), 503
        if err == "insufficient_credits":
            return jsonify(res), 402
        # needs_manual_pick / not_found-match / ambiguous / collect_failed → 200 so the
        # UI can render the options/message without treating it as a hard error.
        return jsonify(res), 200
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


@app.get("/api/roster-reprocess")
@require_auth
def api_roster_reprocess_status():
    return jsonify(core.roster_reprocess_status())


@app.post("/api/roster-reprocess")
@require_auth
def api_roster_reprocess_toggle():
    """Toggle the quality re-process: when ON, the scheduler grinds the whole DB, re-scoring
    and re-classifying every existing lead to the latest standards (FREE). Pass {run_once:N}
    to run a small batch synchronously right now for an immediate sample."""
    if (r := _require_db()):
        return r
    data = request.get_json(silent=True) or {}
    if data.get("run_once"):
        ran = core.roster_reprocess_batch(int(data.get("run_once") or 1))
        return jsonify({"ok": True, "ran": ran, **core.roster_reprocess_status()})
    core.set_roster_reprocess(bool(data.get("enabled")))
    return jsonify({"ok": True, **core.roster_reprocess_status()})


@app.get("/api/admin/cleanup-preview")
@require_auth
def api_cleanup_preview():
    """Dry-run: how many companies the cleanup would remove, with a sample (no deletion)."""
    if (r := _require_db()):
        return r
    rows = db.CompanyRepo.all_id_name(100000)
    hard = [r["name"] for r in rows if core.is_hard_blocked(r.get("name") or "")]
    soft = [r["name"] for r in rows if not core.is_relevant_company(r.get("name") or "")
            and not core.is_hard_blocked(r.get("name") or "")]
    return jsonify({"total_companies": len(rows),
                    "hard_block_remove": len(hard), "hard_sample": hard[:40],
                    "relevance_filter_remove": len(soft), "relevance_sample": soft[:40]})


@app.post("/api/admin/cleanup-companies")
@require_auth
def api_cleanup_companies():
    """Delete irrelevant companies + their candidates. mode=hard (default, conservative name block)
    or mode=relevance (also drops soft-blocked names without an agency/marketing allow-signal)."""
    if (r := _require_db()):
        return r
    mode = (request.get_json(silent=True) or {}).get("mode", "hard")
    removed = core.cleanup_irrelevant_companies() if mode == "relevance" \
        else core.cleanup_hard_blocked_companies()
    return jsonify({"ok": True, "mode": mode, "removed": removed})


@app.post("/api/admin/recategorize")
@require_auth
def api_recategorize():
    """Re-run the 28→12 category consolidation across all candidate + company rows."""
    if (r := _require_db()):
        return r
    res = core.migrate_categories_to_12()
    return jsonify({"ok": True, **res})


# ── Recruit: shortlist top candidates per category + LinkedIn outreach sequences ──
def _recruit_params(src: dict) -> tuple:
    try:
        per = int(src.get("per_category") or src.get("count") or 5)
    except (TypeError, ValueError):
        per = 5
    per = max(1, min(50, per))
    emphasis = core.normalize_emphasis(src.get("emphasis") or "balanced")
    region = (src.get("region") or src.get("country") or "").strip() or None
    cats = src.get("categories")
    if isinstance(cats, str):
        cats = [c.strip() for c in cats.split(",") if c.strip()]
    elif not isinstance(cats, list):
        cats = None
    return per, emphasis, region, (cats or None)


@app.post("/api/recruit")
@require_auth
def api_recruit():
    """Stateless shortlist: top N candidates per category for N openings, re-ranked by the
    recruit-fit composite. Returns the shortlist grouped by category."""
    if (r := _require_db()):
        return r
    per, emphasis, region, cats = _recruit_params(request.get_json(silent=True) or {})
    return jsonify(core.build_recruit_shortlist(per, emphasis, region, cats))


@app.get("/api/recruit/export.csv")
@require_auth
def api_recruit_export():
    """Export the current shortlist (all categories) as CSV with LinkedIn-first contacts."""
    import csv, io
    if (r := _require_db()):
        return r
    per, emphasis, region, cats = _recruit_params(request.args)
    result = core.build_recruit_shortlist(per, emphasis, region, cats)
    cols = [("category", "Category"), ("full_name", "Name"), ("title", "Title"),
            ("company_name", "Company"), ("seniority", "Seniority"), ("location_country", "Country"),
            ("recruit_fit", "Recruit Fit"), ("overall_candidate_score", "Overall"),
            ("job_change_intent_score", "Intent"), ("role_fit_score", "Role Fit"),
            ("technical_score", "Technical"), ("email_effective", "Email"),
            ("phone_effective", "Phone"), ("linkedin_url", "LinkedIn")]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([h for _, h in cols])
    for g in result["groups"]:
        for c in g["candidates"]:
            c.update(core.effective_contacts(c))
            c["category"] = g["category"]
            w.writerow([c.get(k, "") if c.get(k) is not None else "" for k, _ in cols])
    from flask import Response
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=recruit_shortlist.csv"})


def _seq_payload(candidate_id: int) -> dict:
    """Format the stored sequence for the client (adds a live char count per message)."""
    stored = db.RecruitRepo.get(candidate_id) or {}
    msgs = stored.get("messages") or []
    out = [{"phase": m["phase"], "body": m.get("body") or "", "source": m.get("source"),
            "char_count": len(m.get("body") or "")} for m in msgs]
    return {"ok": True, "source": stored.get("source"), "messages": out,
            "outreach": core.outreach_status()}


@app.post("/api/recruit/sequence/<int:cid>")
@require_auth
def api_recruit_sequence(cid):
    """Get-or-generate a candidate's 3-message LinkedIn sequence. {regenerate:true} re-writes all
    three; {regenerate:true, phase:N} re-writes only message N; otherwise returns the stored
    sequence (generating + persisting it the first time)."""
    if (r := _require_db()):
        return r
    cand = db.CandidateRepo.get(cid)
    if not cand:
        return jsonify({"error": "not_found"}), 404
    company = db.CompanyRepo.get(cand["company_id"]) if cand.get("company_id") else None
    data = request.get_json(silent=True) or {}
    existing = db.RecruitRepo.get(cid)

    if data.get("regenerate") and data.get("phase"):
        msg = core.regenerate_recruit_message(cand, company, int(data["phase"]))
        db.RecruitRepo.update_message(cid, msg["phase"], msg["body"], msg.get("source"))
        return jsonify(_seq_payload(cid))

    if data.get("regenerate") or not existing:
        gen = core.generate_recruit_sequence(cand, company)
        db.RecruitRepo.save(cid, gen.get("source"), gen["messages"])
        return jsonify(_seq_payload(cid))

    return jsonify(_seq_payload(cid))


@app.put("/api/recruit/sequence/<int:cid>")
@require_auth
def api_recruit_sequence_save(cid):
    """Persist recruiter edits to the three messages."""
    if (r := _require_db()):
        return r
    if not db.CandidateRepo.get(cid):
        return jsonify({"error": "not_found"}), 404
    data = request.get_json(silent=True) or {}
    msgs = [{"phase": int(m.get("phase")), "body": m.get("body") or "", "source": "edited"}
            for m in (data.get("messages") or []) if m.get("phase") in (1, 2, 3, "1", "2", "3")]
    if msgs:
        db.RecruitRepo.save(cid, "edited", msgs)
    return jsonify(_seq_payload(cid))


@app.post("/api/recruit/sequence/<int:cid>/send")
@require_auth
def api_recruit_send(cid):
    """RESERVED — automated LinkedIn DM send. Inert until the provider is configured; returns a
    disabled result so the UI keeps to manual copy-paste delivery."""
    if (r := _require_db()):
        return r
    cand = db.CandidateRepo.get(cid)
    if not cand:
        return jsonify({"error": "not_found"}), 404
    stored = db.RecruitRepo.get(cid) or {}
    res = core.outreach_send(cand, stored.get("messages") or [])
    return jsonify(res), (200 if res.get("ok") else 501)


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
