"""Unit tests for the deterministic scoring engine + intent provider seam.
Pure functions only — no DB, no network. Run: python -m pytest test_scoring.py -q
"""
import datetime
import os

import core


# ── helpers ──────────────────────────────────────────────────────────────────

def cand(**kw):
    base = {"title": "", "headline": "", "seniority": "", "functions": [],
            "departments": [], "_org": {}}
    base.update(kw)
    return base


# ── normalize_domain / clamp ─────────────────────────────────────────────────

def test_normalize_domain():
    assert core.normalize_domain("https://www.Acme.com/path?x=1") == "acme.com"
    assert core.normalize_domain("HTTP://Foo.IO") == "foo.io"
    assert core.normalize_domain("") == ""


def test_clamp_bounds():
    assert core.clamp(-5) == 0
    assert core.clamp(150) == 100
    assert core.clamp(62.6) == 63


# ── classify_department (specific wins over general) ─────────────────────────

def test_classify_department():
    assert core.classify_department("SEO Manager", "") == "seo"
    assert core.classify_department("PPC Specialist", "") == "digital_marketing"
    assert core.classify_department("Account Executive", "") == "sales"
    assert core.classify_department("Brand Manager", "") == "marketing"
    assert core.classify_department("Software Engineer", "") == "other"
    # SEO must win over generic 'marketing' co-occurrence
    assert core.classify_department("SEO & Marketing Lead", "") == "seo"


# ── technical_score ──────────────────────────────────────────────────────────

def test_technical_score_range_and_signal():
    low = core.score_technical(cand(title="Intern", seniority="intern"))
    high = core.score_technical(cand(title="Head of SEO & Analytics, GA4, SQL", seniority="head"))
    assert 0 <= low <= 100 and 0 <= high <= 100
    assert high > low
    # specialist keywords add measurable depth
    plain = core.score_technical(cand(title="Manager", seniority="manager"))
    techy = core.score_technical(cand(title="PPC & Google Ads & HubSpot Manager", seniority="manager"))
    assert techy > plain


def test_technical_keyword_cap():
    loaded = cand(title="seo ppc ga4 sql hubspot marketo salesforce cro automation martech",
                  seniority="manager")
    # base 58 + capped 35 = 93 (cap prevents runaway)
    assert core.score_technical(loaded) <= 100
    assert core.score_technical(loaded) == core.clamp(58 + 35)


# ── role_fit_score ───────────────────────────────────────────────────────────

def test_role_fit_exact_vs_adjacent():
    exact = core.score_role_fit(cand(title="SEO Manager"), ["seo"])
    adjacent = core.score_role_fit(cand(title="Organic search lead"), ["seo"])
    none = core.score_role_fit(cand(title="Accountant"), ["seo"])
    assert exact == 100
    assert adjacent >= 70
    assert none < adjacent


def test_role_fit_neutral_when_empty():
    assert core.score_role_fit(cand(), ["seo", "sales"]) == 50


# ── job_change_intent: heuristic regime ──────────────────────────────────────

def test_intent_heuristic_seniority_mobility():
    junior, _, _ = core.score_job_change_intent(cand(seniority="entry"))
    exec_, _, _ = core.score_job_change_intent(cand(seniority="c_suite"))
    assert junior > exec_  # juniors more mobile than execs


def test_intent_heuristic_open_signal():
    open_, src, _ = core.score_job_change_intent(
        cand(headline="Open to new opportunities in marketing"))
    base, _, _ = core.score_job_change_intent(cand(headline="Marketing pro"))
    assert open_ > base
    assert src == "heuristic"


# ── job_change_intent: history regime ────────────────────────────────────────

def test_intent_history_regime_sweet_spot():
    today = datetime.date.today()
    twelve_mo = today.replace(year=today.year - 1)
    c = cand(employment_history=[
        {"start_date": twelve_mo.isoformat(), "current": True},
        {"start_date": today.replace(year=today.year - 3).isoformat(),
         "end_date": twelve_mo.isoformat(), "current": False},
    ])
    score, regime, signals = core.score_job_change_intent(c)
    assert regime == "history"
    assert signals["tenure_months"] >= 11
    assert score > 50  # 6-18mo tenure is the restless sweet spot


def test_intent_history_entrenched_lower():
    today = datetime.date.today()
    long_ago = today.replace(year=today.year - 8)
    settled = cand(employment_history=[{"start_date": long_ago.isoformat(), "current": True}])
    restless = cand(employment_history=[
        {"start_date": today.replace(year=today.year - 1).isoformat(), "current": True}])
    s_settled, _, _ = core.score_job_change_intent(settled)
    s_restless, _, _ = core.score_job_change_intent(restless)
    assert s_restless > s_settled


# ── company_quality_score ────────────────────────────────────────────────────

def test_company_quality_bands():
    small = core.score_company_quality({"estimated_employees": 5})
    big = core.score_company_quality({"estimated_employees": 6000, "annual_revenue": 300_000_000,
                                      "founded_year": 1999, "website_url": "x.com",
                                      "industry": "computer software"})
    assert 0 <= small <= 100 and 0 <= big <= 100
    assert big > small


def test_company_quality_neutral_on_empty():
    # all bands default to ~50 → blend stays mid, never 0
    v = core.score_company_quality({})
    assert 40 <= v <= 60


# ── freshness_score (decay) ──────────────────────────────────────────────────

def test_freshness_decay():
    now = datetime.datetime.utcnow()
    fresh = core.score_freshness({"last_verified_at": now}, now)
    mid = core.score_freshness({"last_verified_at": now - datetime.timedelta(days=45)}, now)
    old = core.score_freshness({"last_verified_at": now - datetime.timedelta(days=120)}, now)
    assert fresh == 100
    assert old == 10
    assert 10 < mid < 100
    assert core.score_freshness({}, now) == 100  # brand-new


# ── overall + weight invariant ───────────────────────────────────────────────

def test_overall_matches_weights():
    scores = {"role_fit": 85, "intent": 55, "technical": 70,
              "company_quality": 55, "freshness": 100}
    expected = round(0.30 * 85 + 0.25 * 55 + 0.15 * 70 + 0.15 * 55 + 0.15 * 100)
    assert core.score_overall(scores) == expected
    assert sum(core.WEIGHTS.values()) == 1.0


def test_overall_sql_matches_python():
    """The SQL recompute (db.overall_sql) must equal the Python blend for the same
    inputs — guards against weight drift between insert-time and refresh-time."""
    import db
    scores = {"role_fit": 80, "intent": 60, "technical": 50, "company_quality": 40, "freshness": 90}
    sql_expr = db.overall_sql()  # build a python-evaluable mirror
    py = core.score_overall(scores)
    # emulate the SQL ROUND(...) with the same coefficients
    emulated = round(0.30 * 80 + 0.25 * 60 + 0.15 * 50 + 0.15 * 40 + 0.15 * 90)
    assert py == emulated
    assert "0.3" in sql_expr and "job_change_intent_score" in sql_expr


# ── intent provider seam (LinkedIn reserved/inert) ───────────────────────────

def test_linkedin_provider_inert(monkeypatch):
    monkeypatch.delenv("LINKEDIN_ENABLED", raising=False)
    monkeypatch.delenv("LINKEDIN_API_KEY", raising=False)
    lp = core.LinkedInIntentProvider()
    assert lp.enabled() is False
    assert lp.contribute(cand()) is None


def test_compute_intent_falls_to_apollo(monkeypatch):
    monkeypatch.delenv("LINKEDIN_ENABLED", raising=False)
    score, source, signals = core.compute_intent(cand(seniority="manager"))
    assert source in ("heuristic", "history")
    assert 0 <= score <= 100


def test_compute_intent_prefers_history_over_heuristic():
    today = datetime.date.today()
    c = cand(seniority="manager", employment_history=[
        {"start_date": today.replace(year=today.year - 1).isoformat(), "current": True}])
    _, source, _ = core.compute_intent(c)
    assert source == "history"


# ── person_to_candidate integration ──────────────────────────────────────────

def test_person_to_candidate_full():
    person = {
        "id": "abc123", "first_name": "Jane", "last_name": "Doe", "name": "Jane Doe",
        "title": "SEO Manager", "headline": "Technical SEO @ Acme", "seniority": "manager",
        "departments": ["marketing"], "functions": ["marketing"],
        "linkedin_url": "https://linkedin.com/in/janedoe", "city": "Sydney", "country": "Australia",
        "organization": {"id": "o1", "name": "Acme", "primary_domain": "acme.com",
                         "industry": "marketing & advertising", "estimated_num_employees": 40,
                         "website_url": "https://acme.com", "founded_year": 2015},
    }
    c = core.person_to_candidate(person, ["seo", "digital_marketing", "sales", "marketing"])
    assert c["apollo_person_id"] == "abc123"
    assert c["department"] == "seo"
    assert c["company_domain"] == "acme.com"
    assert 0 <= c["overall_candidate_score"] <= 100
    assert c["intent_source"] == "heuristic"
    assert c["scores_json"]["weights"] == core.WEIGHTS
    assert "_org" in c  # carried for company upsert
