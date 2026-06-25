"""core.py — business logic for the Smart HR Talent Acquisition system.

Contains, in dependency order:
  * config + small helpers + RateLimiter
  * ApolloClient            — FREE People Search + paid people/match reveal
  * department/role taxonomy + the 6 deterministic scoring functions
  * intent provider seam    — ApolloIntentProvider (active) + LinkedInIntentProvider (reserved)
  * OpenAI refinement       — optional, graceful heuristic fallback
  * company discovery       — curated seed list (primary) + best-effort G2/Clutch crawl
  * run_discovery()         — the candidate discovery pipeline (FREE, no reveals)
  * enrich_candidate()      — the per-candidate Enrich action (COSTS credits)
  * scheduler_loop()        — daily auto-refresh worker

FREE-first: all discovery uses Apollo `mixed_people/api_search` (no credits).
Credits are spent ONLY by enrich_candidate(), behind the UI Enrich button.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

import db
from db import WEIGHTS  # canonical scoring weights (single source of truth)

log = logging.getLogger("hr.core")

# ── config helpers ───────────────────────────────────────────────────────────


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _env_list(name: str, default: str = "") -> List[str]:
    raw = os.environ.get(name, default) or ""
    return [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]


def clamp(x: float) -> int:
    """Clamp to a 0-100 integer."""
    try:
        return max(0, min(100, int(round(x))))
    except (ValueError, TypeError):
        return 0


def now_utc() -> datetime.datetime:
    return datetime.datetime.utcnow()


def normalize_domain(value: str) -> str:
    """Strip scheme/www/path → bare registrable-ish domain, lowercased."""
    if not value:
        return ""
    v = value.strip().lower()
    v = re.sub(r"^https?://", "", v)
    v = re.sub(r"^www\.", "", v)
    v = v.split("/")[0].split("?")[0].split("#")[0].strip()
    return v


# ── RateLimiter ──────────────────────────────────────────────────────────────


class RateLimiter:
    """Minimum-interval throttle, thread-safe (shared across all callers)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


# ═══════════════════════════════════════════════════════════════════════════
#  Apollo client
# ═══════════════════════════════════════════════════════════════════════════


class ApolloClient:
    BASE_URL = "https://api.apollo.io/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key or ""
        self.limiter = RateLimiter(float(_env("APOLLO_MIN_INTERVAL", "0.5") or 0.5))
        self.counter: Dict[str, int] = {"search": 0, "match": 0, "usage": 0}
        self._logged_match_error = False

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "Cache-Control": "no-cache",
                "X-Api-Key": self.api_key}

    def _post(self, url: str, payload: dict, retries: int = 2) -> Optional[requests.Response]:
        for attempt in range(retries + 1):
            self.limiter.wait()
            try:
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=40)
            except requests.RequestException as e:
                log.warning("Apollo POST error (%s): %s", url, e)
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return None
            if resp.status_code == 429 and attempt < retries:
                wait = int(resp.headers.get("Retry-After", 0)) or (2 * (attempt + 1))
                log.warning("Apollo 429 - backing off %ss", wait)
                time.sleep(min(wait, 30))
                continue
            if resp.status_code >= 500 and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return resp
        return None

    def search_people(self, query: dict, page: int, per_page: int = 100) -> Tuple[List[dict], int]:
        """FREE People Search. Returns (people, total_entries). No credits consumed,
        no contact info revealed (search never reveals email/phone)."""
        url = f"{self.BASE_URL}/mixed_people/api_search"
        payload: Dict[str, Any] = {"page": page, "per_page": per_page}
        if query.get("person_titles"):
            payload["person_titles"] = query["person_titles"]
        if query.get("person_seniorities"):
            payload["person_seniorities"] = query["person_seniorities"]
        if query.get("q_keywords"):
            payload["q_keywords"] = query["q_keywords"]
        if query.get("person_locations"):
            payload["person_locations"] = query["person_locations"]
        if query.get("organization_locations"):
            payload["organization_locations"] = query["organization_locations"]
        if query.get("organization_num_employees_ranges"):
            payload["organization_num_employees_ranges"] = query["organization_num_employees_ranges"]
        if query.get("seed_domains"):
            payload["q_organization_domains_list"] = query["seed_domains"][:1000]
        resp = self._post(url, payload)
        if resp is None or resp.status_code != 200:
            if resp is not None and resp.status_code not in (200,):
                log.warning("Apollo search HTTP %s: %s", resp.status_code, resp.text[:200])
            return [], 0
        self.counter["search"] += 1
        data = resp.json()
        people = data.get("people") or []
        # api_search returns total_entries at the TOP level (no `pagination` object on
        # the free tier). Fall back to page length so callers always get a number.
        total = data.get("total_entries")
        if not isinstance(total, int):
            total = ((data.get("pagination") or {}).get("total_entries")) or len(people)
        return people, int(total or 0)

    def enrich_person(self, *, apollo_id: str = "", first_name: str = "", last_name: str = "",
                      domain: str = "", linkedin_url: str = "", reveal_email: bool = True,
                      reveal_phone: bool = False, webhook_url: str = "") -> dict:
        """Paid people/match reveal. The EMAIL is returned synchronously in this response.
        The PHONE is async: Apollo requires a webhook_url and delivers the number to it
        minutes later (never in this response) — so we only request phone when a webhook_url
        is provided, and on a non-credits 400/422 we drop ONLY the phone reveal and retry,
        so the synchronous email is never lost. Matching is most reliable by apollo_id."""
        url = f"{self.BASE_URL}/people/match"
        payload: Dict[str, Any] = {}
        if reveal_email:
            payload["reveal_personal_emails"] = True
        if reveal_phone and webhook_url:  # phone is webhook-only; never send it without one
            payload["reveal_phone_number"] = True
            payload["webhook_url"] = webhook_url
        if apollo_id:
            payload["id"] = apollo_id
        if first_name:
            payload["first_name"] = first_name
        if last_name and not (len(last_name.rstrip(".")) == 1 and last_name.rstrip(".").isalpha()):
            payload["last_name"] = last_name
        if domain:
            payload["domain"] = domain
        if linkedin_url:
            payload["linkedin_url"] = linkedin_url

        resp = self._post(url, payload)
        retried = False
        # On a non-credits 400/422, first drop the phone reveal (the usual cause — its
        # webhook), KEEPING the email reveal; if it still fails, drop email too for the
        # base record. Never lose the synchronous email just because phone failed.
        for _ in range(2):
            if resp is None or resp.status_code not in (400, 422):
                break
            body = resp.text[:300]
            if "insufficient credits" in body.lower():
                if not self._logged_match_error:
                    self._logged_match_error = True
                    log.error("Apollo EXPORT CREDITS EXHAUSTED: %s", body)
                return {"_ok": False, "_no_credits": True, "_http_status": resp.status_code}
            if payload.pop("reveal_phone_number", None) is not None:
                payload.pop("webhook_url", None)
            elif payload.pop("reveal_personal_emails", None) is None:
                break  # nothing left to strip
            resp = self._post(url, payload)
            retried = True

        if resp is None:
            return {"_ok": False, "_http_status": None, "_error": "network"}
        if resp.status_code != 200:
            return {"_ok": False, "_http_status": resp.status_code, "_error": resp.text[:200]}

        self.counter["match"] += 1
        person = (resp.json() or {}).get("person") or {}
        return {
            "_ok": True, "_http_status": 200, "_retried": retried,
            "email": _best_email(person),
            "phone": _best_phone(person),
            "employment_history": person.get("employment_history") or [],
            "person": person,
        }

    def credits_remaining(self) -> int:
        """Best-effort credit balance. Returns -1 on any failure (never blocks)."""
        try:
            self.limiter.wait()
            resp = requests.post(f"{self.BASE_URL}/usage_stats/api_usage",
                                 headers=self._headers(), timeout=20)
            if resp.status_code != 200:
                return -1
            self.counter["usage"] += 1
            data = resp.json() or {}
            # Apollo shapes vary; probe common keys for an email-credit balance.
            for path in (("credits",), ("usage",)):
                node = data
                for k in path:
                    node = node.get(k, {}) if isinstance(node, dict) else {}
                if isinstance(node, dict):
                    for k in ("remaining", "available", "left"):
                        if isinstance(node.get(k), (int, float)):
                            return int(node[k])
            return -1
        except Exception:
            return -1


def _best_email(person: dict) -> Optional[str]:
    e = person.get("email")
    if e and "@" in str(e) and "email_not_unlocked" not in str(e):
        return str(e)
    for pe in (person.get("personal_emails") or []):
        if pe and "@" in str(pe):
            return str(pe)
    return None


def _best_phone(person: dict) -> Optional[str]:
    for ph in (person.get("phone_numbers") or []):
        if isinstance(ph, dict):
            num = ph.get("sanitized_number") or ph.get("raw_number")
            if num:
                return str(num)
        elif ph:
            return str(ph)
    return None


_apollo_singleton: Optional[ApolloClient] = None
_apollo_lock = threading.Lock()


def get_apollo() -> ApolloClient:
    global _apollo_singleton
    with _apollo_lock:
        if _apollo_singleton is None:
            _apollo_singleton = ApolloClient(_env("APOLLO_API_KEY"))
        return _apollo_singleton


# ═══════════════════════════════════════════════════════════════════════════
#  CoreSignal — manual LinkedIn enrichment (employee_multi_source, cdapi/v2)
#  Auth: custom header `apikey`. Paid, per-candidate, explicitly user-triggered.
# ═══════════════════════════════════════════════════════════════════════════
class CoreSignalClient:
    """CoreSignal cdapi/v2 client. Header auth via `apikey` (NOT Bearer). Best-effort:
    never raises; returns {ok,status,data,error,credits}. Tracks x-credits-remaining."""
    BASE = "https://api.coresignal.com/cdapi/v2"

    def __init__(self, api_key: str, dataset: str = "employee_multi_source"):
        self.api_key = (api_key or "").strip()
        self.dataset = (dataset or "employee_multi_source").strip()
        self.credits_remaining: Optional[int] = None
        self._limiter = RateLimiter(0.2)

    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self, post: bool = False) -> dict:
        h = {"apikey": self.api_key, "accept": "application/json"}
        if post:
            h["Content-Type"] = "application/json"
        return h

    def _note_credits(self, resp) -> None:
        v = resp.headers.get("x-credits-remaining")
        if v is not None:
            try:
                self.credits_remaining = int(v)
            except (ValueError, TypeError):
                pass

    def _request(self, method: str, path: str, json_body=None) -> dict:
        if not self.configured():
            return {"ok": False, "status": 0, "data": None, "error": "not_configured", "credits": None}
        url = f"{self.BASE}/{path}"
        for attempt in range(2):
            try:
                self._limiter.wait()
                resp = requests.request(method, url, headers=self._headers(post=json_body is not None),
                                        json=json_body, timeout=30)
            except Exception as e:
                return {"ok": False, "status": 0, "data": None,
                        "error": f"network:{str(e)[:120]}", "credits": self.credits_remaining}
            self._note_credits(resp)
            sc = resp.status_code
            if sc == 429:
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                return {"ok": False, "status": 429, "data": None, "error": "rate_limited", "credits": self.credits_remaining}
            if sc == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = None
                return {"ok": True, "status": 200, "data": data, "error": None, "credits": self.credits_remaining}
            if sc == 402:
                return {"ok": False, "status": 402, "data": None, "error": "insufficient_credits", "credits": self.credits_remaining}
            if sc in (401, 403):
                return {"ok": False, "status": sc, "data": None, "error": "auth", "credits": self.credits_remaining}
            if sc in (404, 422, 454):
                return {"ok": False, "status": sc, "data": None, "error": "no_data", "credits": self.credits_remaining}
            return {"ok": False, "status": sc, "data": None, "error": f"http_{sc}", "credits": self.credits_remaining}
        return {"ok": False, "status": 0, "data": None, "error": "unknown", "credits": self.credits_remaining}  # unreachable safety net

    def collect(self, id_or_slug) -> dict:
        from urllib.parse import quote
        return self._request("GET", f"{self.dataset}/collect/{quote(str(id_or_slug), safe='')}")

    def search(self, body: dict, preview: bool = False) -> dict:
        sub = "search/es_dsl/preview" if preview else "search/es_dsl"
        return self._request("POST", f"{self.dataset}/{sub}", json_body=body)


_coresignal_singleton: Optional[CoreSignalClient] = None
_coresignal_lock = threading.Lock()


def get_coresignal() -> CoreSignalClient:
    """Singleton; rebuilds if the env key was set/rotated since (Railway may set it after
    boot, or a wrong key may be corrected) so a corrected key is picked up without a hard
    restart — preserving the cached credits_remaining when the key is unchanged."""
    global _coresignal_singleton
    with _coresignal_lock:
        key = _env("CORESIGNAL_API_KEY")
        if _coresignal_singleton is None or _coresignal_singleton.api_key != key:
            _coresignal_singleton = CoreSignalClient(
                key, _env("CORESIGNAL_DATASET", "employee_multi_source"))
        return _coresignal_singleton


def _linkedin_slug(url: str) -> Optional[str]:
    """Bare vanity slug from a LinkedIn profile URL (linkedin.com/in/<slug>)."""
    if not url:
        return None
    m = re.search(r"linkedin\.com/(?:in|pub)/([^/?#]+)", url, re.I)
    if not m:
        return None
    return (m.group(1).strip().strip("/") or None)


def _registrable_domain(host_or_url: str) -> str:
    """Normalize a website/host to a comparable domain (lowercase, no scheme/www/path)."""
    if not host_or_url:
        return ""
    s = str(host_or_url).strip().lower()
    s = re.sub(r"^[a-z]+://", "", s)
    s = s.split("/")[0].split("?")[0]
    s = re.sub(r"^www\.", "", s)
    return s.strip()


def _domain_match(a: str, b: str) -> bool:
    a, b = _registrable_domain(a), _registrable_domain(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return ".".join(a.split(".")[-2:]) == ".".join(b.split(".")[-2:])


def _title_tokens(t: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (t or "").lower())) - {"the", "of", "and", "a", "at", "to"}


def _title_overlap(a: str, b: str) -> int:
    return len(_title_tokens(a) & _title_tokens(b))


def _cs_build_search_body(full_name: str, title: str, company_name: str,
                          company_domain: str, relax: int = 0) -> dict:
    """ES DSL body. relax 0: name+title+(company/domain); 1: name+company; 2: name+domain; 3: name only."""
    must = [{"match": {"full_name": {"query": full_name, "operator": "and"}}}]
    should: list = []
    dom = _registrable_domain(company_domain)
    if relax == 0 and title:
        must.append({"match": {"active_experience_title": {"query": title, "operator": "and"}}})
    if relax in (0, 1) and company_name:
        should.append({"match": {"active_experience_company_name": {"query": company_name, "operator": "and"}}})
    if relax in (0, 1, 2) and dom:
        should.append({"query_string": {"query": f"*{dom}*", "default_field": "active_experience_company_website"}})
    bool_q: dict = {"must": must}
    if should:
        bool_q["should"] = should
        bool_q["minimum_should_match"] = 1
    return {"query": {"bool": bool_q}, "sort": ["_score"]}


def _cs_preview_company_site(p: dict) -> str:
    return p.get("active_experience_company_website") or p.get("company_website") or ""


def _cs_preview_title(p: dict) -> str:
    return p.get("active_experience_title") or p.get("headline") or ""


def _cs_pick_best(previews: list, company_domain: str, title: str):
    """Return (best_preview, confidence, method, ambiguous)."""
    if not previews:
        return None, "none", "no_results", False
    if len(previews) == 1:
        return previews[0], "medium", "single_result", False
    dommatched = [p for p in previews if _domain_match(_cs_preview_company_site(p), company_domain)]
    if len(dommatched) == 1:
        return dommatched[0], "high", "domain_match", False
    if len(dommatched) > 1:
        best = max(dommatched, key=lambda p: _title_overlap(_cs_preview_title(p), title))
        return best, "high", "domain_match+title", False
    ranked = sorted(previews, key=lambda p: _title_overlap(_cs_preview_title(p), title), reverse=True)
    if _title_overlap(_cs_preview_title(ranked[0]), title) > 0 and (
            len(ranked) == 1 or _title_overlap(_cs_preview_title(ranked[0]), title)
            > _title_overlap(_cs_preview_title(ranked[1]), title)):
        return ranked[0], "medium", "title_overlap", False
    # Multiple candidates, NO domain match and NO clear title-overlap winner: a top-_score
    # pick on (especially) a name-only query is essentially arbitrary. Never auto-spend a
    # paid collect on a stranger — force the user to pick the right person.
    ps = sorted(previews, key=lambda p: p.get("_score") or 0, reverse=True)
    return ps[0], "low", "ambiguous", True


def _cs_preview_brief(p: dict) -> dict:
    return {"id": p.get("id"), "full_name": p.get("full_name"),
            "headline": p.get("headline"), "title": _cs_preview_title(p),
            "company_name": p.get("company_name"),
            "company_website": _cs_preview_company_site(p),
            "location": p.get("location_full") or p.get("location_country"),
            "score": p.get("_score"),
            "profile_url": p.get("professional_network_url")}


def _cs_profile_url(rec: dict) -> Optional[str]:
    u = rec.get("professional_network_url") or ""
    if "linkedin.com" in u.lower():
        return u
    sh = rec.get("professional_network_canonical_shorthand_name") or rec.get("shorthand_name")
    if sh:
        return f"https://www.linkedin.com/in/{sh}"
    return None


def _looks_like_linkedin(u: Optional[str]) -> bool:
    return bool(u and "linkedin.com" in u.lower())


# ═══════════════════════════════════════════════════════════════════════════
#  Org extraction
# ═══════════════════════════════════════════════════════════════════════════


def _apollo_revenue(org: dict) -> Optional[int]:
    for k in ("annual_revenue", "organization_revenue", "estimated_annual_revenue"):
        v = org.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    rng = org.get("revenue_range") or {}
    if isinstance(rng, dict):
        lo = rng.get("min")
        if isinstance(lo, (int, float)) and lo > 0:
            return int(lo)
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:120]


def company_key_for(name: str, domain: str = "") -> str:
    """Stable name-based dedup key. We key by NAME (not domain) because the free
    Apollo search only gives the org name and the domain is resolved later (Clearbit);
    keying by name keeps dedup stable so resolving a domain never creates a duplicate.
    Falls back to a normalized domain only when the name is empty."""
    sl = _slug(name)
    if sl:
        return f"name:{sl}"
    d = normalize_domain(domain)
    return d or ""


def extract_org(person: dict) -> dict:
    """Pull the embedded organization object → company dict. Robust to the THIN
    free-search org (which carries only `name`); firmographics fill in on enrich."""
    org = person.get("organization") or person.get("account") or {}
    if not isinstance(org, dict):
        org = {}
    name = org.get("name") or ""
    domain = normalize_domain(org.get("primary_domain") or org.get("website_url") or "")
    return {
        "apollo_org_id": org.get("id"),
        "name": name,
        "root_domain": domain or None,
        "company_key": company_key_for(name, domain),
        "website_url": org.get("website_url"),
        "linkedin_url": org.get("linkedin_url"),
        "industry": org.get("industry"),
        "estimated_employees": org.get("estimated_num_employees"),
        "annual_revenue": _apollo_revenue(org),
        "founded_year": org.get("founded_year"),
        "hq_city": org.get("city"),
        "hq_country": org.get("country"),
        "_raw": org,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Department / role taxonomy
# ═══════════════════════════════════════════════════════════════════════════

ROLE_FAMILIES: Dict[str, dict] = {
    "sales": {
        "keyword": "sales",
        "titles": ["Sales Manager", "Account Executive", "Business Development Manager",
                   "Sales Director", "Head of Sales", "VP Sales", "Sales Representative",
                   "Sales Development Representative", "Business Development Representative"],
        "strong": ["sales", "account executive", "business development", "sdr", "bdr",
                   "sales development", "revenue", "quota"],
    },
    "marketing": {
        "keyword": "marketing",
        "titles": ["Marketing Manager", "Marketing Director", "Head of Marketing",
                   "Demand Generation Manager", "Growth Marketing Manager", "CMO",
                   "Brand Manager", "Content Marketing Manager", "Product Marketing Manager"],
        "strong": ["marketing", "demand gen", "demand generation", "growth", "brand",
                   "content marketing", "cmo", "communications"],
    },
    "seo": {
        "keyword": "SEO",
        "titles": ["SEO Manager", "SEO Specialist", "SEO Analyst", "Head of SEO",
                   "Search Engine Optimization", "Organic Growth Manager", "SEO Lead",
                   "SEO Executive", "SEO Consultant"],
        "strong": ["seo", "search engine optimization", "organic search", "organic growth",
                   "technical seo", "link building", "serp"],
    },
    "digital_marketing": {
        "keyword": "digital marketing",
        "titles": ["Digital Marketing Manager", "PPC Manager", "Paid Media Manager",
                   "Performance Marketing Manager", "Paid Search Manager",
                   "Social Media Marketing Manager", "Digital Marketing Specialist",
                   "Digital Marketing Executive", "SEM Manager"],
        "strong": ["digital marketing", "ppc", "paid media", "performance marketing",
                   "paid search", "sem", "google ads", "social media marketing",
                   "paid social", "media buyer"],
    },
}
TARGET_FAMILY_ORDER = ["seo", "digital_marketing", "sales", "marketing"]  # specific → general

SENIORITY_POINTS = {
    "owner": 90, "founder": 90, "c_suite": 90, "partner": 85, "vp": 80, "head": 78,
    "director": 72, "senior": 62, "manager": 58, "entry": 40, "intern": 30,
}

TECH_SPECIALIST = ["technical seo", "seo", "ppc", "paid media", "google ads", "google analytics",
                   "ga4", "analytics", "sql", "hubspot", "marketo", "salesforce", "cro",
                   "conversion rate", "marketing automation", "automation", "martech",
                   "programmatic", "tag manager", "schema", "looker", "semrush", "ahrefs",
                   "data studio", "sem"]
TECH_GENERIC = ["marketing", "growth", "demand", "content", "social", "brand", "campaign",
                "email marketing", "advertising"]


def _text(c: dict) -> str:
    return f"{c.get('title') or ''} {c.get('headline') or ''}".lower()


def classify_department(title: str = "", headline: str = "", departments: Optional[list] = None,
                        functions: Optional[list] = None) -> str:
    """Map a person to one of sales/marketing/seo/digital_marketing/other.
    Specific families (seo, digital_marketing) win over generic marketing."""
    text = f"{title or ''} {headline or ''}".lower()
    for fam in TARGET_FAMILY_ORDER:
        cfg = ROLE_FAMILIES[fam]
        if any(t.lower() in text for t in cfg["titles"]) or any(k in text for k in cfg["strong"]):
            return fam
    deps = " ".join(str(d).lower() for d in (departments or []))
    if "sales" in deps:
        return "sales"
    if "marketing" in deps:
        return "marketing"
    return "other"


# ── Fine-grained category taxonomy (28 categories in 6 groups) ───────────────
# Each: label, dept (coarse enum), group (UI), kw (q_keywords), titles[] (Apollo
# person_titles), match[] (classification tokens). ORDER = specific → general so
# classify_category() returns the most specific match first.
CATEGORIES = [
    {"label": "Technical SEO", "dept": "seo", "group": "SEO", "kw": "technical SEO",
     "titles": ["Technical SEO Specialist", "Technical SEO Manager", "SEO Developer"],
     "match": ["technical seo"]},
    {"label": "Local SEO", "dept": "seo", "group": "SEO", "kw": "local SEO",
     "titles": ["Local SEO Specialist", "Local SEO Manager"], "match": ["local seo"]},
    {"label": "Link Building and Digital PR", "dept": "seo", "group": "SEO", "kw": "link building",
     "titles": ["Link Building Specialist", "Digital PR Manager", "Outreach Specialist"],
     "match": ["link building", "digital pr", "outreach specialist"]},
    {"label": "SEO", "dept": "seo", "group": "SEO", "kw": "SEO",
     "titles": ["SEO Manager", "SEO Specialist", "SEO Executive", "SEO Analyst", "Head of SEO"],
     "match": ["seo", "search engine optim", "organic search"]},
    {"label": "Google Ads", "dept": "digital_marketing", "group": "Paid & Performance", "kw": "Google Ads",
     "titles": ["Google Ads Specialist", "Google Ads Manager", "AdWords Specialist"],
     "match": ["google ads", "adwords"]},
    {"label": "Paid Media / PPC", "dept": "digital_marketing", "group": "Paid & Performance", "kw": "PPC",
     "titles": ["PPC Manager", "Paid Media Manager", "PPC Specialist", "Paid Search Manager"],
     "match": ["ppc", "paid media", "paid search", "biddable"]},
    {"label": "Social Media Marketing", "dept": "digital_marketing", "group": "Paid & Performance",
     "kw": "social media", "titles": ["Social Media Manager", "Social Media Marketing Specialist",
                                       "Paid Social Manager"], "match": ["social media", "paid social"]},
    {"label": "Performance Marketing", "dept": "digital_marketing", "group": "Paid & Performance",
     "kw": "performance marketing", "titles": ["Performance Marketing Manager", "Growth Marketing Manager"],
     "match": ["performance marketing", "growth marketing"]},
    {"label": "Email Marketing", "dept": "digital_marketing", "group": "Paid & Performance",
     "kw": "email marketing", "titles": ["Email Marketing Manager", "Email Marketing Specialist",
                                         "CRM Marketing Manager"],
     "match": ["email marketing", "crm marketing", "lifecycle marketing"]},
    {"label": "Marketing Automation", "dept": "digital_marketing", "group": "Paid & Performance",
     "kw": "marketing automation", "titles": ["Marketing Automation Specialist",
                                              "Marketing Automation Manager", "HubSpot Specialist"],
     "match": ["marketing automation", "marketo", "hubspot", "pardot"]},
    {"label": "Digital Marketing", "dept": "digital_marketing", "group": "Marketing & Content",
     "kw": "digital marketing", "titles": ["Digital Marketing Manager", "Digital Marketing Specialist",
                                           "Digital Marketing Executive"], "match": ["digital marketing"]},
    {"label": "Conversion Rate Optimization", "dept": "digital_marketing", "group": "Marketing & Content",
     "kw": "conversion rate optimization", "titles": ["CRO Specialist",
                                                      "Conversion Rate Optimization Manager", "CRO Manager"],
     "match": ["conversion rate", "cro specialist", "cro manager"]},
    {"label": "Data Analytics and Reporting", "dept": "digital_marketing", "group": "Marketing & Content",
     "kw": "marketing analytics", "titles": ["Marketing Analyst", "Data Analyst", "Analytics Manager",
                                             "Reporting Analyst"],
     "match": ["analytics", "data analyst", "reporting analyst", "data studio", "looker"]},
    {"label": "Content Writer", "dept": "marketing", "group": "Marketing & Content", "kw": "content writer",
     "titles": ["Content Writer", "Content Specialist", "Content Marketing Manager"],
     "match": ["content writer", "content marketing", "content specialist"]},
    {"label": "Content Designer", "dept": "marketing", "group": "Marketing & Content",
     "kw": "content designer", "titles": ["Content Designer", "Content Strategist"],
     "match": ["content designer", "content strategist"]},
    {"label": "Copywriting", "dept": "marketing", "group": "Marketing & Content", "kw": "copywriter",
     "titles": ["Copywriter", "Senior Copywriter", "Copywriting Lead"], "match": ["copywriter", "copywriting"]},
    {"label": "Marketing", "dept": "marketing", "group": "Marketing & Content", "kw": "marketing",
     "titles": ["Marketing Manager", "Marketing Director", "Head of Marketing", "CMO", "Brand Manager"],
     "match": ["marketing", "brand manager", "cmo", "communications"]},
    {"label": "WordPress Development", "dept": "other", "group": "Creative & Web", "kw": "WordPress",
     "titles": ["WordPress Developer", "WordPress Designer"], "match": ["wordpress"]},
    {"label": "Web Development", "dept": "other", "group": "Creative & Web", "kw": "web developer",
     "titles": ["Web Developer", "Frontend Developer", "Full Stack Developer"],
     "match": ["web developer", "frontend", "front-end", "full stack", "web development"]},
    {"label": "UI/UX Design", "dept": "other", "group": "Creative & Web", "kw": "UX designer",
     "titles": ["UI Designer", "UX Designer", "UI/UX Designer", "Product Designer"],
     "match": ["ui/ux", "ux design", "ui design", "product designer", "user experience"]},
    {"label": "Graphic Design", "dept": "other", "group": "Creative & Web", "kw": "graphic designer",
     "titles": ["Graphic Designer", "Visual Designer", "Senior Graphic Designer"],
     "match": ["graphic design", "visual designer"]},
    {"label": "Video Editing and Production", "dept": "other", "group": "Creative & Web", "kw": "video editor",
     "titles": ["Video Editor", "Video Producer", "Motion Designer"],
     "match": ["video editor", "video produc", "motion designer", "videographer"]},
    {"label": "E-commerce", "dept": "other", "group": "Sales & Ops", "kw": "ecommerce",
     "titles": ["Ecommerce Manager", "E-commerce Specialist", "Shopify Developer"],
     "match": ["ecommerce", "e-commerce", "shopify"]},
    {"label": "Account Management", "dept": "sales", "group": "Sales & Ops", "kw": "account manager",
     "titles": ["Account Manager", "Client Services Manager", "Account Director"],
     "match": ["account manager", "account director", "client services", "client success"]},
    {"label": "Sales and Business Development", "dept": "sales", "group": "Sales & Ops",
     "kw": "business development", "titles": ["Business Development Manager", "Sales Manager",
                                             "Account Executive", "Sales Director"],
     "match": ["business development", "sales manager", "account executive", "sales director",
               "bdr", "sdr", "sales executive"]},
    {"label": "Project Management", "dept": "other", "group": "Sales & Ops", "kw": "project manager",
     "titles": ["Project Manager", "Digital Project Manager", "Program Manager"],
     "match": ["project manager", "program manager", "project management", "scrum master"]},
    {"label": "Talent Acquisition", "dept": "other", "group": "People", "kw": "talent acquisition",
     "titles": ["Talent Acquisition Specialist", "Recruiter", "Recruitment Manager",
                "Talent Acquisition Manager"], "match": ["talent acquisition", "recruiter", "recruitment"]},
    {"label": "HR", "dept": "other", "group": "People", "kw": "human resources",
     "titles": ["HR Manager", "Human Resources Manager", "HR Business Partner"],
     "match": ["human resources", "hr manager", "hr business partner", "people operations"]},
]
CATEGORY_DEPT = {c["label"]: c["dept"] for c in CATEGORIES}
CATEGORY_LABELS = [c["label"] for c in CATEGORIES]
# UI groups (preserve definition order)
GROUPS: Dict[str, List[str]] = {}
for _c in CATEGORIES:
    GROUPS.setdefault(_c["group"], []).append(_c["label"])
GROUP_ORDER = list(GROUPS.keys())
CATEGORY_GROUP = {label: g for g, labels in GROUPS.items() for label in labels}
# human "industry" label per group — used as a FREE fallback when Apollo has no industry
GROUP_INDUSTRY = {
    "SEO": "SEO / Search Marketing",
    "Paid & Performance": "Performance / Paid Media",
    "Marketing & Content": "Digital Marketing & Content",
    "Creative & Web": "Creative & Web Development",
    "Sales & Ops": "Sales & Business Development",
    "People": "HR & Recruiting",
}


def derive_industry(company_id: int) -> Optional[str]:
    """Infer a company's industry FREE from the categories of its own candidates
    (used only when Apollo has no real industry). Returns a label or None."""
    try:
        cat = db.CandidateRepo.dominant_category(company_id)
    except Exception:
        return None
    if not cat:
        return None
    return GROUP_INDUSTRY.get(CATEGORY_GROUP.get(cat, ""), None)


def classify_category(title: str = "", headline: str = "") -> Optional[str]:
    """Return the most specific matching category label, or None."""
    text = f"{title or ''} {headline or ''}".lower()
    if not text.strip():
        return None
    for c in CATEGORIES:
        if any(tok in text for tok in c["match"]):
            return c["label"]
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Company size bands, relevance filter, free domain resolution
# ═══════════════════════════════════════════════════════════════════════════

# (min, max, label) — matches the HR size taxonomy. `max` is inclusive; the giant
# band (10001+) is excluded from discovery by default to drop Amazon/Tech-Mahindra
# scale firms (turn on with HR_INCLUDE_GIANTS=1).
SIZE_BANDS = [
    (1, 1, "Solo / Self-employed"),
    (2, 10, "Micro business"),
    (11, 50, "Small business"),
    (51, 200, "SME / Growing business"),
    (201, 500, "Lower mid-market"),
    (501, 1000, "Mid-market"),
    (1001, 5000, "Enterprise"),
    (5001, 10000, "Large enterprise"),
    (10001, 5000000, "Global enterprise"),
]
SIZE_LABELS = [b[2] for b in SIZE_BANDS] + ["Unknown / Unverified"]
_GIANT_LABEL = "Global enterprise"


def size_band_label(employees: Optional[int]) -> str:
    if not isinstance(employees, int) or employees <= 0:
        return "Unknown / Unverified"
    for lo, hi, label in SIZE_BANDS:
        if lo <= employees <= hi:
            return label
    return "Global enterprise"


def discovery_size_bands() -> List[Tuple[int, int, str]]:
    """The employee bands used as a discovery query axis (giants excluded by default)."""
    bands = SIZE_BANDS if _env("HR_INCLUDE_GIANTS", "0") == "1" else SIZE_BANDS[:-1]
    return bands


# ── Relevance filter: we're an SEO/digital-marketing firm, so drop banks, govt,
#    healthcare, education, mega-corporations, etc. Two tiers + an allow-list:
#    HARD blocks always win (banks, giants); SOFT blocks are overridden by a clear
#    agency/marketing/tech allow-signal. Robust, no fragile industry-tag IDs. ───────
_HARD_BLOCK = [
    # finance (a "marketing" team at a bank is not who we source)
    "bank", "banking", "insurance", "assurance", "mutual fund", "securities",
    "capital markets", "stock exchange", "asset management", "wealth management",
    "credit union", "non banking", "nbfc", "fintech",
    # public sector / institutions
    "government", "ministry", "municipal", "council", "public sector", "defence",
    "defense", "police", "university", "college", "institute of technology",
    "school district",
    # healthcare
    "hospital", "clinic", "healthcare", "pharmaceutic", "pharma ", "medical center",
    "diagnostics",
    # well-known mega-corporations (200k+ / not agency talent pools)
    "amazon", "tech mahindra", "infosys", "wipro", "tata consultancy", "tcs ",
    "hcl tech", "hcltech", "cognizant", "capgemini", "accenture", "genpact",
    "deloitte", "ernst & young", "kpmg", "pricewaterhouse", "pwc ", "ibm ",
    "reliance", "adani", "flipkart", "walmart", "jpmorgan", "wells fargo",
    "concentrix", "teleperformance", "foxconn",
    "nocree",
]
_SOFT_BLOCK = [
    "airlines", "airways", "aviation", "railways", "petroleum", "oil & gas",
    "oil and gas", "power plant", "electricity board", "steel", "cement", "mining",
    "automobile", "manufacturing plant", "freight", "logistics & supply",
    "real estate", "construction", "hospitality", "restaurant", "hotel",
]
_ALLOW_TOKENS = ["seo", "digital marketing", "digital agency", "marketing", "advertis",
                 "agency", "media", "creative", "design", "software", "web develop",
                 "growth", "performance marketing", "analytics", "studio", "interactive",
                 "e-commerce", "ecommerce", "tech labs", "martech"]


def is_relevant_company(name: str) -> bool:
    """True if the company looks relevant to an SEO/digital-marketing talent search."""
    n = (name or "").lower().strip()
    if not n:
        return True
    if any(b in n for b in _HARD_BLOCK):
        return False
    if any(a in n for a in _ALLOW_TOKENS):
        return True
    return not any(b in n for b in _SOFT_BLOCK)


# ── Free company-domain resolution via Clearbit autocomplete (no key, no Apollo
#    credits). Maps a company NAME → its primary domain so we can show the website. ─
_domain_cache: Dict[str, Optional[str]] = {}
_domain_limiter = RateLimiter(0.25)


def resolve_company_domain(name: str) -> Optional[str]:
    """Best-effort free name→domain lookup (Clearbit autocomplete). Cached; returns
    None on any failure. Never raises, never costs credits."""
    n = (name or "").strip()
    if not n:
        return None
    key = n.lower()
    if key in _domain_cache:
        return _domain_cache[key]
    domain = None
    try:
        _domain_limiter.wait()
        r = requests.get("https://autocomplete.clearbit.com/v1/companies/suggest",
                         params={"query": n}, timeout=10)
        if r.status_code == 200:
            for item in (r.json() or []):
                d = normalize_domain(item.get("domain") or "")
                if d:
                    domain = d
                    break
    except Exception:
        domain = None
    _domain_cache[key] = domain
    return domain


def linkedin_search_url(full_name: str, company: str = "") -> str:
    """A LinkedIn people-search URL (free) for when we don't have the exact profile."""
    from urllib.parse import quote
    q = quote(f"{full_name or ''} {company or ''}".strip())
    return f"https://www.linkedin.com/search/results/people/?keywords={q}"


# ═══════════════════════════════════════════════════════════════════════════
#  Scoring engine — 6 deterministic pure functions (0-100, neutral defaults)
# ═══════════════════════════════════════════════════════════════════════════


def score_technical(c: dict) -> int:
    base = SENIORITY_POINTS.get((c.get("seniority") or "").lower(), 40)
    text = _text(c)
    kw = sum(6 for k in TECH_SPECIALIST if k in text)
    kw += sum(3 for k in TECH_GENERIC if k in text)
    kw = min(kw, 35)
    funcs = {str(f).lower() for f in (c.get("functions") or [])}
    func_bonus = 5 if funcs & {"marketing", "sales", "engineering"} else 0
    return clamp(base + kw + func_bonus)


def score_role_fit(c: dict, target_families: Optional[List[str]] = None) -> int:
    families = target_families or list(ROLE_FAMILIES.keys())
    text = _text(c)
    if not text.strip():
        return 50
    best = 0
    for fam in families:
        cfg = ROLE_FAMILIES.get(fam)
        if not cfg:
            continue
        if any(t.lower() in text for t in cfg["titles"]):
            best = max(best, 100)
        elif cfg["keyword"].lower() in text:
            best = max(best, 85)
        elif any(k in text for k in cfg["strong"]):
            best = max(best, 70)
    # Recognised in the 28-category taxonomy (e.g. web dev, UI/UX, HR) → solid base
    # even if it isn't one of the 4 core marketing families.
    cat = classify_category(c.get("title", ""), c.get("headline", ""))
    if cat and best < 65:
        best = 65
    if best == 0:
        best = 50
    if best <= 50 and not cat:
        dept = classify_department(c.get("title", ""), c.get("headline", ""),
                                   c.get("departments"), c.get("functions"))
        if dept == "other":
            best = round(best * 0.7)
    return clamp(best)


def _parse_date(v: Any) -> Optional[datetime.date]:
    if not v:
        return None
    s = str(v).strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # last resort: leading year
    m = re.match(r"(\d{4})", s)
    if m:
        try:
            return datetime.date(int(m.group(1)), 1, 1)
        except ValueError:
            return None
    return None


def _months_between(a: datetime.date, b: datetime.date) -> int:
    return max(0, (b.year - a.year) * 12 + (b.month - a.month))


def score_job_change_intent(c: dict, now: Optional[datetime.datetime] = None) -> Tuple[int, str, dict]:
    """INFERRED intent. Returns (score, regime, signals). Two regimes:
    history (employment_history present) or heuristic (discovery default)."""
    now = now or now_utc()
    today = now.date()
    history = c.get("employment_history") or []
    parsed = []
    for h in history:
        if not isinstance(h, dict):
            continue
        sd = _parse_date(h.get("start_date"))
        ed = _parse_date(h.get("end_date"))
        parsed.append({"start": sd, "end": ed, "current": bool(h.get("current"))})

    if parsed and any(p["start"] for p in parsed):
        # ── history regime ──
        current = next((p for p in parsed if p["current"] and p["start"]), None)
        if current is None:
            current = max((p for p in parsed if p["start"]), key=lambda p: p["start"])
        tenure_m = _months_between(current["start"], today) if current and current["start"] else 24
        five_yr_ago = today.replace(year=today.year - 5)
        num_jobs_5y = sum(1 for p in parsed if p["start"] and p["start"] >= five_yr_ago)
        tenures = [_months_between(p["start"], p["end"]) for p in parsed
                   if p["start"] and p["end"] and p["end"] >= p["start"]]
        avg_tenure = (sum(tenures) / len(tenures)) if tenures else None

        s = 0
        if tenure_m < 3:
            s += 5
        elif tenure_m < 6:
            s += 15
        elif tenure_m < 18:
            s += 25
        elif tenure_m < 36:
            s += 18
        else:
            s += 10
        if num_jobs_5y >= 4:
            s += 20
        elif num_jobs_5y == 3:
            s += 12
        elif num_jobs_5y == 2:
            s += 5
        if avg_tenure is not None and avg_tenure < 18:
            s += 10
        signals = {"regime": "history", "tenure_months": tenure_m,
                   "jobs_last_5y": num_jobs_5y,
                   "avg_tenure_months": round(avg_tenure) if avg_tenure else None}
        return clamp(50 + s - 25), "history", signals

    # ── heuristic regime (discovery default) ──
    text = _text(c)
    sen = (c.get("seniority") or "").lower()
    s = 0
    if sen in ("entry", "intern"):
        s += 8
    elif sen in ("manager", "senior"):
        s += 5
    elif sen == "director":
        s += 2
    elif sen in ("vp", "head", "c_suite", "owner", "founder", "partner"):
        s -= 8
    if any(t in text for t in ("contractor", "freelance", "freelancer", "consultant")):
        s += 10
    org = c.get("_org") or {}
    industry = str(org.get("industry") or "").lower()
    emp = org.get("estimated_employees") or 0
    if industry in ("marketing & advertising", "marketing and advertising", "internet") \
            and isinstance(emp, int) and 0 < emp < 50:
        s += 6
    if any(t in text for t in ("open to", "seeking", "looking for new", "#opentowork")):
        s += 15
    signals = {"regime": "heuristic", "seniority": sen}
    return clamp(50 + s), "heuristic", signals


def score_company_quality(org: dict) -> int:
    emp = org.get("estimated_employees")
    if not isinstance(emp, int):
        size = 50
    elif emp <= 10:
        size = 40
    elif emp <= 50:
        size = 55
    elif emp <= 200:
        size = 70
    elif emp <= 1000:
        size = 82
    elif emp <= 5000:
        size = 90
    else:
        size = 95
    rev = org.get("annual_revenue")
    if not isinstance(rev, (int, float)) or rev <= 0:
        rev_pts = 50
    elif rev < 1_000_000:
        rev_pts = 45
    elif rev < 10_000_000:
        rev_pts = 62
    elif rev < 50_000_000:
        rev_pts = 75
    elif rev < 250_000_000:
        rev_pts = 85
    else:
        rev_pts = 95
    fy = org.get("founded_year")
    if not isinstance(fy, int) or fy < 1900:
        age_pts = 55
    else:
        years = max(0, now_utc().year - fy)
        age_pts = 45 if years < 2 else 60 if years <= 5 else 78 if years <= 15 else 85
    website_pts = 5 if (org.get("website_url") or org.get("root_domain")) else 0
    industry_pts = 5 if str(org.get("industry") or "").lower() in (
        "marketing & advertising", "marketing and advertising", "internet",
        "computer software", "information technology & services") else 0
    return clamp(0.40 * size + 0.35 * rev_pts + 0.25 * age_pts + website_pts + industry_pts)


def score_freshness(c: dict, now: Optional[datetime.datetime] = None) -> int:
    now = now or now_utc()
    ref = c.get("last_verified_at") or c.get("discovered_at")
    if ref is None:
        return 100  # freshly discovered this instant
    if isinstance(ref, str):
        ref = _parse_date(ref) or now.date()
    if isinstance(ref, datetime.datetime):
        ref = ref.date()
    days = max(0, (now.date() - ref).days)
    if days <= 7:
        return 100
    if days >= 90:
        return 10
    return clamp(100 - (days - 7) * (90 / 83))


def score_overall(scores: dict) -> int:
    return clamp(
        WEIGHTS["role_fit"] * scores.get("role_fit", 50)
        + WEIGHTS["intent"] * scores.get("intent", 50)
        + WEIGHTS["technical"] * scores.get("technical", 50)
        + WEIGHTS["company_quality"] * scores.get("company_quality", 50)
        + WEIGHTS["freshness"] * scores.get("freshness", 100)
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Intent provider seam (LinkedIn reserved)
# ═══════════════════════════════════════════════════════════════════════════

_SOURCE_PRIORITY = {"heuristic": 1, "history": 2, "linkedin": 3}


class IntentProvider:
    name = "base"

    def enabled(self) -> bool:
        return True

    def contribute(self, c: dict) -> Optional[Tuple[int, str, dict]]:
        raise NotImplementedError


class ApolloIntentProvider(IntentProvider):
    """Active provider — heuristic at discovery, history after enrichment."""
    name = "apollo"

    def contribute(self, c: dict) -> Optional[Tuple[int, str, dict]]:
        score, regime, signals = score_job_change_intent(c)
        return score, regime, signals


class LinkedInIntentProvider(IntentProvider):
    """RESERVED — inert until a LinkedIn API/crawler is integrated.

    When LINKEDIN_ENABLED=1 and LINKEDIN_API_KEY is set, implement fetch() to read
    Open-to-Work / recent activity / new certifications / recent title change from the
    stored linkedin_url and return a strong contribution with source='linkedin'.
    The DB columns (intent_source, linkedin_open_to_work, linkedin_signals_json,
    linkedin_checked_at) already exist — no migration required.
    """
    name = "linkedin"

    def enabled(self) -> bool:
        return _env("LINKEDIN_ENABLED", "0") == "1" and bool(_env("LINKEDIN_API_KEY"))

    def fetch(self, linkedin_url: str) -> Optional[dict]:  # pragma: no cover - future
        # TODO(LinkedIn integration): call the LinkedIn API/crawler here and return
        # {"open_to_work": bool, "signals": {...}}. Respect LINKEDIN_MAX_LOOKUPS_PER_RUN.
        return None

    def contribute(self, c: dict) -> Optional[Tuple[int, str, dict]]:
        if not self.enabled():
            return None
        data = self.fetch(c.get("linkedin_url") or "")
        if not data:
            return None
        score = 90 if data.get("open_to_work") else 65
        return score, "linkedin", {"regime": "linkedin", **(data.get("signals") or {})}


_PROVIDERS: List[IntentProvider] = [ApolloIntentProvider(), LinkedInIntentProvider()]


def compute_intent(c: dict) -> Tuple[int, str, dict]:
    """Blend all enabled intent providers; the highest-priority source wins
    (linkedin > history > heuristic)."""
    best: Optional[Tuple[int, str, dict]] = None
    for p in _PROVIDERS:
        try:
            if not p.enabled():
                continue
            contrib = p.contribute(c)
        except Exception as e:  # pragma: no cover
            log.warning("intent provider %s failed: %s", p.name, e)
            contrib = None
        if not contrib:
            continue
        if best is None or _SOURCE_PRIORITY.get(contrib[1], 0) > _SOURCE_PRIORITY.get(best[1], 0):
            best = contrib
    return best or (50, "heuristic", {"regime": "heuristic"})


# ═══════════════════════════════════════════════════════════════════════════
#  OpenAI refinement (optional, graceful fallback)
# ═══════════════════════════════════════════════════════════════════════════


def openai_available() -> bool:
    return bool(_env("OPENAI_API_KEY"))


_SEN_LABEL = {"c_suite": "C-suite", "vp": "VP", "head": "Head", "director": "Director",
              "manager": "Manager", "senior": "Senior", "entry": "Junior", "intern": "Intern",
              "owner": "Owner", "founder": "Founder", "partner": "Partner"}
_DEPT_LABEL = {"seo": "SEO", "digital_marketing": "Digital Marketing", "sales": "Sales",
               "marketing": "Marketing", "other": "Specialist"}


def heuristic_classify(c: dict) -> dict:
    """Free, deterministic ai_meta — used by default and as OpenAI fallback."""
    dept = classify_department(c.get("title", ""), c.get("headline", ""),
                               c.get("departments"), c.get("functions"))
    sen = (c.get("seniority") or "").lower() or "unknown"
    org_name = (c.get("_org") or {}).get("name") or c.get("company_domain") or "their current company"
    sen_l = _SEN_LABEL.get(sen, sen.replace("_", " ").title())
    blurb = f"{sen_l} {_DEPT_LABEL.get(dept, 'Specialist')} at {org_name}.".strip()
    return {"_source": "heuristic", "department": dept, "role_family": dept,
            "seniority_level": sen, "technical_level": None, "why": blurb}


def openai_classify(batch: List[dict]) -> List[dict]:
    """Refine up to ~20 candidates with gpt-4o-mini (JSON mode). Falls back to
    heuristic_classify per item on any failure / missing key."""
    if not batch:
        return []
    if not openai_available():
        return [heuristic_classify(c) for c in batch]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        items = [{"i": i, "title": c.get("title", ""), "headline": c.get("headline", ""),
                  "industry": (c.get("_org") or {}).get("industry", ""),
                  "seniority": c.get("seniority", "")} for i, c in enumerate(batch)]
        sys = ("You classify professionals into a fixed HR taxonomy. Use ONLY the provided "
               "fields; never invent. Output STRICT JSON: {\"results\":[{\"i\":int,"
               "\"department\":\"sales|marketing|seo|digital_marketing|other\","
               "\"role_family\":same enum,\"seniority_level\":\"intern|entry|manager|senior|"
               "director|vp|c_suite\",\"technical_level\":0-100,\"why\":\"<=20 words why this "
               "person is a strong job-change candidate\"}]}")
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": json.dumps(items)}],
            timeout=40,
        )
        parsed = json.loads(resp.choices[0].message.content)
        by_i = {int(r.get("i")): r for r in (parsed.get("results") or []) if "i" in r}
        out = []
        for i, c in enumerate(batch):
            r = by_i.get(i)
            if r:
                r["_source"] = "openai"
                out.append(r)
            else:
                out.append(heuristic_classify(c))
        return out
    except Exception as e:
        log.warning("openai_classify failed (%s) — using heuristics", e)
        return [heuristic_classify(c) for c in batch]


def _apply_ai_nudge(scores: dict, ai_meta: dict) -> dict:
    """OpenAI nudges technical & role_fit by ±15 max (never dominates)."""
    tl = ai_meta.get("technical_level")
    if isinstance(tl, (int, float)):
        delta = max(-15, min(15, (tl - scores["technical"]) * 0.5))
        scores["technical"] = clamp(scores["technical"] + delta)
    return scores


# ═══════════════════════════════════════════════════════════════════════════
#  Candidate mapping + scoring
# ═══════════════════════════════════════════════════════════════════════════


def person_to_candidate(person: dict, target_families: List[str],
                        now: Optional[datetime.datetime] = None,
                        ai_meta: Optional[dict] = None, ctx: Optional[dict] = None) -> dict:
    """Map a raw Apollo person (+ embedded org) → a fully-scored candidate dict.

    `ctx` (from the search cell) backfills fields the THIN free search omits:
    seniority (queried but not returned), department family, and country."""
    now = now or now_utc()
    ctx = ctx or {}
    org = extract_org(person)
    title = person.get("title") or ""
    headline = person.get("headline") or ""
    departments = person.get("departments") or []
    functions = person.get("functions") or []
    seniority = person.get("seniority") or ctx.get("seniority") or ""
    first = person.get("first_name") or ""
    last = person.get("last_name") or person.get("last_name_obfuscated") or ""
    full = person.get("name") or f"{first} {last}".strip() or (first or "Unknown")
    country = person.get("country") or ctx.get("country")

    base = {"title": title, "headline": headline, "seniority": seniority,
            "functions": functions, "departments": departments, "_org": org,
            "company_domain": org.get("root_domain"), "linkedin_url": person.get("linkedin_url")}

    category = classify_category(title, headline) or ctx.get("category")
    dept = CATEGORY_DEPT.get(category) if category else None
    if not dept:
        dept = classify_department(title, headline, departments, functions)
        if dept == "other" and ctx.get("family"):
            dept = ctx["family"]
    # Free firmographics are thin, but the employee-band query tells us company size —
    # inject the band midpoint so company_quality reflects real size.
    if not org.get("estimated_employees") and isinstance(ctx.get("size_min"), int) \
            and isinstance(ctx.get("size_max"), int):
        org["estimated_employees"] = (ctx["size_min"] + min(ctx["size_max"], 100000)) // 2
    cq = score_company_quality(org)
    technical = score_technical(base)
    role_fit = score_role_fit(base, target_families)
    intent, intent_source, intent_signals = compute_intent(base)
    freshness = 100  # freshly discovered

    if ai_meta is None:
        ai_meta = heuristic_classify(base)
    scores_pack = {"role_fit": role_fit, "intent": intent, "technical": technical,
                   "company_quality": cq, "freshness": freshness}
    scores_pack = _apply_ai_nudge(scores_pack, ai_meta)
    overall = score_overall(scores_pack)
    threshold = _env_int("HR_INTENT_OPEN_THRESHOLD", 60)

    scores_json = {**scores_pack, "overall": overall, "intent_regime": intent_source,
                   "intent_signals": intent_signals, "weights": WEIGHTS}

    return {
        "apollo_person_id": str(person.get("id") or ""),
        "company_name": org.get("name") or None,
        "company_domain": org.get("root_domain"),
        "company_key": org.get("company_key"),
        "full_name": full or "Unknown",
        "first_name": first, "last_name": last,
        "title": title, "headline": headline[:500] if headline else None,
        "department": dept, "category": category, "departments_json": departments, "seniority": seniority,
        "linkedin_url": person.get("linkedin_url"), "photo_url": person.get("photo_url"),
        "location_city": person.get("city"), "location_country": country,
        "has_email": bool(person.get("has_email")),
        "has_phone": bool(person.get("has_direct_phone")),
        "technical_score": scores_pack["technical"], "role_fit_score": role_fit,
        "job_change_intent_score": intent, "company_quality_score": cq,
        "freshness_score": freshness, "overall_candidate_score": overall,
        "scores_json": scores_json, "ai_meta_json": ai_meta,
        "open_to_shift": 1 if intent >= threshold else 0,
        "intent_source": intent_source, "confidence": 70 if person.get("id") else 40,
        "payload_json": {"title": title, "headline": headline, "seniority": seniority,
                         "organization": org.get("_raw")},
        "_org": org,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Company discovery (seed list primary, best-effort crawl)
# ═══════════════════════════════════════════════════════════════════════════

SEED_DOMAINS = [
    # Global martech / SaaS
    "hubspot.com", "semrush.com", "ahrefs.com", "moz.com", "salesforce.com", "marketo.com",
    "mailchimp.com", "hootsuite.com", "buffer.com", "sproutsocial.com", "klaviyo.com",
    "activecampaign.com", "zoho.com", "freshworks.com", "pipedrive.com",
    # Global agencies
    "wpromote.com", "neilpatel.com", "ogilvy.com", "publicisgroupe.com", "dentsu.com",
    "wearesocial.com", "iprospect.com", "merkle.com", "razorfish.com", "huge.com",
    "rga.com", "vmlyr.com", "360i.com", "performics.com", "tinuiti.com",
    # Australia agencies / digital
    "webprofits.com.au", "kingkong.co", "rocketagency.com.au", "trafficradius.com.au",
    "studiohawk.com.au", "prospermedia.com.au", "sponsoredlinx.com.au", "reload.com.au",
    "impressive.com.au", "fame.agency", "onlinemarketinggurus.com.au", "digitalsurfer.com.au",
    "localsearch.com.au", "uplift.com.au", "megaphonemarketing.com.au", "redsearch.com.au",
    # India agencies / digital
    "webchutney.com", "pinstorm.com", "socialbeat.in", "techmagnate.com", "iquanti.com",
    "interactiveavenues.com", "watconsult.com", "ralecon.com", "infidigit.com",
    "pagetraffic.com", "digitalsuccess.us", "brandlogist.com", "adlift.com", "regalix.com",
]


def discover_company_domains(extra_seed: Optional[List[str]] = None,
                             crawl_enabled: Optional[bool] = None,
                             max_requests: Optional[int] = None,
                             logf=None) -> List[str]:
    """Return target company domains. Seed list is the reliable primary source; a
    best-effort G2/Clutch crawl augments it and degrades silently on any failure."""
    def _log(m):
        if logf:
            logf(m)
    domains = {normalize_domain(d) for d in SEED_DOMAINS}
    for d in (extra_seed or _env_list("HR_SEED_DOMAINS")):
        nd = normalize_domain(d)
        if nd:
            domains.add(nd)
    if crawl_enabled is None:
        crawl_enabled = _env("HR_COMPANY_CRAWL_ENABLED", "0") == "1"
    if crawl_enabled:
        try:
            crawled = _crawl_directory_domains(max_requests or _env_int("HR_COMPANY_CRAWL_MAX_REQUESTS", 60), _log)
            domains.update(crawled)
            _log(f"Crawl added {len(crawled)} domains")
        except Exception as e:
            _log(f"Crawl failed ({e}); using seed list only")
    return sorted(d for d in domains if d)


_CRAWL_LIMITER = RateLimiter(1.5)
_DOMAIN_RE = re.compile(r"https?://(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})", re.I)
_BLOCK_DOMAINS = {"clutch.co", "g2.com", "google.com", "facebook.com", "twitter.com",
                  "linkedin.com", "youtube.com", "instagram.com", "apple.com"}


def _crawl_directory_domains(max_requests: int, logf) -> List[str]:
    """Best-effort scrape of public G2/Clutch category pages for outbound company
    domains ONLY (never people). Any block/error → caller falls back to seed list."""
    from bs4 import BeautifulSoup
    urls = [
        "https://clutch.co/agencies/digital-marketing",
        "https://clutch.co/seo-firms",
        "https://clutch.co/agencies/social-media-marketing",
        "https://clutch.co/agencies/ppc",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    found: set = set()
    reqs = 0
    for u in urls:
        if reqs >= max_requests:
            break
        _CRAWL_LIMITER.wait()
        reqs += 1
        try:
            r = requests.get(u, headers=headers, timeout=20)
            if r.status_code in (403, 429):
                logf(f"Crawl blocked {r.status_code} on {u}")
                break
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                m = _DOMAIN_RE.match(a["href"])
                if m:
                    d = normalize_domain(m.group(1))
                    if d and d not in _BLOCK_DOMAINS and "clutch.co" not in d:
                        found.add(d)
        except requests.RequestException as e:
            logf(f"Crawl error on {u}: {e}")
            continue
    return list(found)


# ═══════════════════════════════════════════════════════════════════════════
#  Discovery pipeline (FREE — no reveals)
# ═══════════════════════════════════════════════════════════════════════════

# Seniority is a QUERY axis (the free search filters by it but doesn't return it),
# so each group carries a representative seniority value we stamp onto results.
SENIORITY_GROUPS = [
    {"name": "exec", "seniorities": ["owner", "founder", "c_suite", "partner", "vp", "head"], "repr": "vp"},
    {"name": "director", "seniorities": ["director"], "repr": "director"},
    {"name": "manager", "seniorities": ["manager", "senior"], "repr": "manager"},
    {"name": "junior", "seniorities": ["entry", "intern"], "repr": "entry"},
]


_CAT_BY_LABEL = {c["label"]: c for c in CATEGORIES}


# Broad professional-seniority filter applied to every cell (keeps results to real
# professionals without exploding the query count). Exact seniority is filled on enrich.
BROAD_SENIORITIES = ["owner", "founder", "c_suite", "partner", "vp", "head",
                     "director", "manager", "senior"]


def build_search_queries(cfg: dict) -> List[dict]:
    """Cartesian product of {category} × {country} × {employee-band}. Each cell is a
    separate paged Apollo search. The per-country split stamps a definite region; the
    per-employee-band split (organization_num_employees_ranges) stamps a definite
    company SIZE for FREE (the thin search omits the count) and excludes giant firms.
    `ctx` carries category/department/country/size to backfill the omitted fields."""
    cat_labels = cfg.get("categories") or CATEGORY_LABELS
    locations = cfg.get("person_locations") or [None]   # None = global (no geo filter)
    org_locations = cfg.get("organization_locations") or []
    seed_domains = cfg.get("seed_domains") or []
    bands = cfg.get("size_bands") or discovery_size_bands()
    queries = []
    for country in locations:
        for label in cat_labels:
            cat = _CAT_BY_LABEL.get(label)
            if not cat:
                continue
            for lo, hi, slabel in bands:
                tag = f"{label}/{slabel}" + (f"/{country}" if country else "")
                queries.append({
                    "department": cat["dept"], "category": label, "label": tag,
                    "person_titles": cat["titles"],
                    "person_seniorities": BROAD_SENIORITIES,
                    "q_keywords": cat["kw"],
                    "person_locations": [country] if country else [],
                    "organization_locations": org_locations,
                    "organization_num_employees_ranges": [f"{lo},{hi}"],
                    "seed_domains": seed_domains,
                    "ctx": {"family": cat["dept"], "category": label, "country": country,
                            "size_label": slabel, "size_min": lo, "size_max": hi},
                })
    return queries


def resolve_categories(params: dict) -> List[str]:
    """Resolve a run's target categories from params: explicit `categories` >
    `groups` (expanded) > legacy `departments` (mapped) > env HR_TARGET_GROUPS > all."""
    cats = params.get("categories")
    if cats:
        keep = [c for c in cats if c in CATEGORY_DEPT]
        if keep:
            return keep
    if params.get("groups"):
        out = [lab for g in params["groups"] for lab in GROUPS.get(g, [])]
        if out:
            return out
    if params.get("departments"):
        deps = set(params["departments"])
        out = [c["label"] for c in CATEGORIES if c["dept"] in deps]
        if out:
            return out
    env_groups = _env_list("HR_TARGET_GROUPS")
    if env_groups:
        out = [lab for g in env_groups for lab in GROUPS.get(g, [])]
        if out:
            return out
    return CATEGORY_LABELS


def run_discovery(run_id: int, params: dict, job) -> dict:
    """Execute a discovery run. Streams logs into `job`. Returns stats dict.
    NEVER reveals contact info (zero credit spend)."""
    now = now_utc()
    target_families = list(ROLE_FAMILIES.keys())  # for role_fit scoring (core 4 areas)
    categories = resolve_categories(params)
    cfg = {
        "categories": categories,
        "person_locations": params.get("person_locations") or _env_list("HR_PERSON_LOCATIONS"),
        "organization_locations": params.get("organization_locations") or _env_list("HR_ORG_LOCATIONS"),
        "seed_domains": params.get("seed_domains") or [],
    }
    max_pages = int(params.get("max_pages") or _env_int("HR_MAX_PAGES_PER_DEPT", 10))
    max_candidates = int(params.get("max_candidates") or _env_int("HR_MAX_CANDIDATES_PER_RUN", 3000))
    use_seed = bool(params.get("use_seed_domains"))

    if use_seed:
        cfg["seed_domains"] = discover_company_domains(params.get("seed_domains"), logf=job.log)
        job.log(f"Focus mode: {len(cfg['seed_domains'])} target company domains")

    queries = build_search_queries(cfg)
    apollo = get_apollo()
    seen_ids: set = set()
    stats = {"companies_new": 0, "candidates_new": 0, "candidates_refreshed": 0,
             "apollo_search_calls": 0}
    processed = 0
    per_page = 100
    cap_page = 500  # 50k / 100
    # Per-cell cap: bound how many candidates each (category × region × size-band) cell
    # contributes, so a capped run samples ALL bands/categories evenly instead of
    # exhausting on the first cell (otherwise every company comes back "Solo").
    per_cell_cap = _env_int("HR_MAX_PER_CELL", 40)

    job.log(f"Discovery start - {len(queries)} query cells, regions={cfg['person_locations'] or 'global'}")

    for qi, q in enumerate(queries):
        if job.cancel_flag or processed >= max_candidates:
            break
        job.status_text = f"Searching {q['label']}"
        cell_count = 0
        for page in range(1, max_pages + 1):
            if job.cancel_flag or processed >= max_candidates or cell_count >= per_cell_cap:
                break
            people, total = apollo.search_people(q, page, per_page)
            stats["apollo_search_calls"] += 1
            job.log(f"  {q['label']} p{page}: {len(people)} people")
            if not people:
                break
            ctx = q.get("ctx") or {}
            for person in people:
                pid = str(person.get("id") or "")
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                org_name = ((person.get("organization") or {}).get("name")) or ""
                if not is_relevant_company(org_name):
                    continue  # skip banks/govt/healthcare/etc. (and their people)
                cand = person_to_candidate(person, target_families, now, ctx=ctx)
                org = cand.pop("_org", {})
                company_id = None
                # Size stamped from the employee-band query cell (free, exact band).
                smin, smax = ctx.get("size_min"), ctx.get("size_max")
                est_emp = org.get("estimated_employees")
                if not est_emp and isinstance(smin, int) and isinstance(smax, int):
                    est_emp = (smin + min(smax, 100000)) // 2
                if org.get("company_key"):
                    comp = {"company_key": org["company_key"], "name": org.get("name") or "Unknown",
                            "apollo_org_id": org.get("apollo_org_id"),
                            "root_domain": org.get("root_domain"),
                            "industry": org.get("industry"),
                            "estimated_employees": est_emp,
                            "size_band": ctx.get("size_label") or size_band_label(est_emp),
                            "size_min": smin, "size_max": smax,
                            "annual_revenue": org.get("annual_revenue"),
                            "founded_year": org.get("founded_year"),
                            "hq_city": org.get("hq_city"), "hq_country": org.get("hq_country"),
                            "country": cand.get("location_country"),
                            "company_quality_score": cand["company_quality_score"],
                            "source": "apollo", "confidence": 60, "payload_json": org.get("_raw")}
                    try:
                        company_id, comp_new = db.CompanyRepo.upsert(comp, run_id)
                        if comp_new:
                            stats["companies_new"] += 1
                    except Exception as e:
                        log.warning("company upsert failed: %s", e)
                cand["company_id"] = company_id
                try:
                    _cid, is_new = db.CandidateRepo.upsert(cand, run_id)
                    stats["candidates_new" if is_new else "candidates_refreshed"] += 1
                    processed += 1
                    cell_count += 1
                except Exception as e:
                    log.warning("candidate upsert failed: %s", e)
                if processed >= max_candidates or cell_count >= per_cell_cap:
                    break
            # Robust stop: partial page = last page, per-cell cap, or 50k page cap.
            if len(people) < per_page or page >= cap_page or cell_count >= per_cell_cap:
                if page >= cap_page:
                    job.log(f"  {q['label']}: hit Apollo 50k cap - filter is broad")
                break
        job.progress = int(((qi + 1) / max(1, len(queries))) * 100)

    job.stats = stats
    job.log(f"Discovery done - {stats['candidates_new']} new, "
            f"{stats['candidates_refreshed']} refreshed, {stats['companies_new']} new companies, "
            f"{stats['apollo_search_calls']} free searches")
    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  Enrichment (COSTS credits — gated behind the UI Enrich button)
# ═══════════════════════════════════════════════════════════════════════════


def backfill_company_domains(limit: int = 50, logf=None) -> int:
    """Resolve domains (free Clearbit) for companies missing one. Marks each attempted
    so failures aren't retried forever. Returns count newly resolved."""
    rows = db.CompanyRepo.missing_domain(limit)
    found = 0
    for r in rows:
        d = resolve_company_domain(r.get("name") or "")
        website = f"https://{d}" if d else None
        try:
            db.CompanyRepo.set_domain(r["id"], d, website)
            if d:
                found += 1
        except Exception as e:
            log.warning("set_domain failed for %s: %s", r.get("id"), e)
    if logf and rows:
        logf(f"Domain backfill: {found}/{len(rows)} resolved")
    return found


def sync_company_roster(company: dict, max_people: Optional[int] = None, logf=None) -> int:
    """Pull EVERY person Apollo has for a company (domain-scoped, no title filter) and
    attach them all to that company — so each company has its full employee roster, not
    just the handful found by category discovery. FREE (search only). Returns count."""
    cid = company.get("id")
    name = company.get("name") or ""
    domain = company.get("root_domain")
    if not cid:
        return 0
    max_people = max_people or _env_int("HR_MAX_ROSTER_PER_COMPANY", 400)
    now = now_utc()
    ctx = {"country": company.get("country"), "size_min": company.get("size_min"),
           "size_max": company.get("size_max"), "size_label": company.get("size_band")}
    if domain:
        query = {"seed_domains": [domain]}      # q_organization_domains_list — precise
        strict_slug = None
    else:
        query = {"q_keywords": name}            # fallback: keyword, filtered to exact name
        strict_slug = _slug(name)
    apollo = get_apollo()
    seen: set = set()
    count = 0
    for page in range(1, 200):
        people, _ = apollo.search_people(query, page, 100)
        if not people:
            break
        for person in people:
            pid = str(person.get("id") or "")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            if strict_slug:
                org_name = ((person.get("organization") or {}).get("name")) or ""
                if _slug(org_name) != strict_slug:
                    continue
            cand = person_to_candidate(person, list(ROLE_FAMILIES.keys()), now, ctx=ctx)
            cand.pop("_org", None)
            cand["company_id"] = cid
            cand["company_name"] = name
            cand["company_domain"] = domain
            try:
                db.CandidateRepo.upsert(cand, None)
                count += 1
            except Exception as e:
                log.warning("roster upsert failed: %s", e)
            if count >= max_people:
                break
        if count >= max_people or len(people) < 100:
            break
    db.CompanyRepo.mark_roster_synced(cid, count)
    if logf:
        logf(f"  roster {name[:30]}: {count} people")
    return count


def roster_sync_batch(limit: int = 4, logf=None) -> Tuple[int, int]:
    """Roster-sync the next batch of companies needing it. Returns (people_added, companies)."""
    companies = db.CompanyRepo.roster_pending(limit)
    total = 0
    for co in companies:
        try:
            total += sync_company_roster(co, logf=logf)
        except Exception as e:
            log.warning("roster sync failed for %s: %s", co.get("name"), e)
            try:
                db.CompanyRepo.mark_roster_synced(co["id"], 0)  # don't get stuck on it
            except Exception:
                pass
    return total, len(companies)


_web_limiter = RateLimiter(1.0)


def fetch_company_web(domain: str) -> dict:
    """Best-effort FREE fetch of a company homepage → {description, og_image}. Never
    raises. Extracts og:description / meta description / <title>."""
    import html as _html
    if not domain:
        return {}
    try:
        _web_limiter.wait()
        r = requests.get("https://" + domain,
                         headers={"User-Agent": _LI_UA, "Accept-Language": "en-US,en"},
                         timeout=10, allow_redirects=True)
    except Exception:
        return {}
    if r.status_code != 200:
        return {}
    h = r.text[:200000]

    def _prop(p):
        m = re.search(r'<meta\s+property="' + re.escape(p) + r'"\s+content="([^"]*)"', h, re.I)
        return _html.unescape(m.group(1).strip()) if m else None

    def _name(p):
        m = re.search(r'<meta\s+name="' + re.escape(p) + r'"\s+content="([^"]*)"', h, re.I)
        return _html.unescape(m.group(1).strip()) if m else None

    title = None
    mt = re.search(r"<title[^>]*>(.*?)</title>", h, re.S | re.I)
    if mt:
        title = _html.unescape(re.sub(r"\s+", " ", mt.group(1)).strip())
    desc = _prop("og:description") or _name("description") or title
    if desc:
        desc = desc[:280]
    img = _prop("og:image")
    return {"description": desc, "og_image": (img[:512] if img else None)}


def enrich_company_web(limit: int = 20, logf=None) -> int:
    """Paced FREE homepage 'About' enricher (mirrors backfill_company_domains)."""
    rows = db.CompanyRepo.web_pending(limit)
    n = 0
    for r in rows:
        info = fetch_company_web(r.get("root_domain") or "")
        try:
            db.CompanyRepo.set_web(r["id"], info.get("description"), info.get("og_image"))
            if info.get("description"):
                n += 1
        except Exception as e:
            log.warning("set_web failed for %s: %s", r.get("id"), e)
    if logf and rows:
        logf(f"Web enrich: {n}/{len(rows)} described")
    return n


def cleanup_irrelevant_companies(limit: int = 5000) -> int:
    """Delete existing companies (and their candidates) that fail the relevance filter
    (banks, govt, healthcare, etc.). One-time/periodic housekeeping."""
    rows = db.CompanyRepo.all_id_name(limit)
    drop = [r["id"] for r in rows if not is_relevant_company(r.get("name") or "")]
    if drop:
        db.CompanyRepo.delete_with_candidates(drop)
    return len(drop)


def enrich_candidate(candidate_id: int, reveal_email: bool = True,
                     reveal_phone: bool = False, webhook_url: str = "") -> dict:
    """Reveal contact info for ONE candidate via Apollo people/match, then recompute
    intent with the now-available employment_history. Caller must have already
    transitioned status → 'enriching' (db.CandidateRepo.set_enriching)."""
    row = db.CandidateRepo.get_basic(candidate_id)
    if not row:
        return {"ok": False, "error": "not_found"}

    # ── per-day reveal caps ──
    counts = db.RevealCounterRepo.today()
    cap_email = _env_int("ENRICH_MAX_REVEALS_PER_DAY_EMAIL", 150)
    cap_phone = _env_int("ENRICH_MAX_REVEALS_PER_DAY_PHONE", 60)
    if reveal_email and counts["email_reveals"] >= cap_email:
        db.CandidateRepo.set_status(candidate_id, "not_enriched")
        return {"ok": False, "error": "daily_email_cap_reached"}
    if reveal_phone and counts["phone_reveals"] >= cap_phone:
        db.CandidateRepo.set_status(candidate_id, "not_enriched")
        return {"ok": False, "error": "daily_phone_cap_reached"}

    apollo = get_apollo()
    credits_before = apollo.credits_remaining()
    res = apollo.enrich_person(
        apollo_id=row.get("apollo_person_id") or "",
        first_name=row.get("first_name") or "", last_name=row.get("last_name") or "",
        domain=row.get("company_domain") or "", linkedin_url=row.get("linkedin_url") or "",
        reveal_email=reveal_email, reveal_phone=reveal_phone, webhook_url=webhook_url)

    log_entry = {"candidate_id": candidate_id, "apollo_person_id": row.get("apollo_person_id"),
                 "reveal_email": reveal_email, "reveal_phone": reveal_phone,
                 "http_status": res.get("_http_status"), "credits_before": credits_before,
                 "response_json": res.get("person")}

    if res.get("_no_credits"):
        db.CandidateRepo.set_status(candidate_id, "no_credits")
        db.EnrichmentLogRepo.log({**log_entry, "result": "no_credits"})
        return {"ok": False, "error": "no_credits", "status": "no_credits"}

    if not res.get("_ok"):
        db.CandidateRepo.set_status(candidate_id, "failed")
        db.EnrichmentLogRepo.log({**log_entry, "result": "failed",
                                  "error_text": str(res.get("_error"))[:500]})
        return {"ok": False, "error": "enrich_failed", "status": "failed"}

    email = res.get("email")
    phone = res.get("phone")
    emp = res.get("employment_history") or []
    person = res.get("person") or {}
    org = extract_org(person)

    # Rich fields now available from people/match (absent in the thin free search).
    real_name = person.get("name") or None
    seniority = person.get("seniority") or None
    linkedin = person.get("linkedin_url") or None
    country = person.get("country") or None
    title = person.get("title") or ""
    headline = person.get("headline") or ""
    functions = person.get("functions") or []

    # Recompute scores with the richer data (real seniority/headline/firmographics).
    score_base = {"title": title, "headline": headline, "seniority": seniority or "",
                  "functions": functions, "departments": person.get("departments") or [],
                  "_org": org, "employment_history": emp,
                  "linkedin_url": linkedin or row.get("linkedin_url")}
    technical = score_technical(score_base) if title else row.get("technical_score", 50)
    company_quality = score_company_quality(org) if org.get("estimated_employees") \
        else row.get("company_quality_score", 50)
    intent, intent_source, intent_signals = compute_intent(score_base)
    overall = score_overall({
        "role_fit": row.get("role_fit_score", 50), "intent": intent, "technical": technical,
        "company_quality": company_quality, "freshness": row.get("freshness_score", 100)})
    scores_json = {"role_fit": row.get("role_fit_score"), "intent": intent,
                   "technical": technical, "company_quality": company_quality,
                   "freshness": row.get("freshness_score"), "overall": overall,
                   "intent_regime": intent_source, "intent_signals": intent_signals,
                   "weights": WEIGHTS}

    db.CandidateRepo.apply_enrichment(
        candidate_id, email=email, phone=phone, status="enriched", full_name=real_name,
        seniority=seniority, linkedin_url=linkedin, location_country=country,
        employment_history=emp, intent_score=intent, intent_source=intent_source,
        scores_json=scores_json, overall=overall, company_quality=company_quality)

    # Opportunistically enrich the company's firmographics (domain, employees, etc.)
    # now that people/match exposed them — Company View improves for FREE on every reveal.
    if row.get("company_id") and org.get("estimated_employees"):
        try:
            db.CompanyRepo.update_firmographics(row["company_id"], org, company_quality)
        except Exception as e:
            log.warning("company firmographics update failed: %s", e)

    credits_after = apollo.credits_remaining()
    spent = (credits_before - credits_after) if (credits_before >= 0 and credits_after >= 0) else None
    db.RevealCounterRepo.incr(email=1 if (reveal_email and email) else 0,
                              phone=1 if (reveal_phone and phone) else 0)
    db.EnrichmentLogRepo.log({
        **log_entry, "result": "retried_no_reveal" if res.get("_retried") else "success",
        "credits_after": credits_after, "credits_spent": spent,
        "email_revealed": bool(email), "phone_revealed": bool(phone)})

    return {"ok": True, "status": "enriched", "email_revealed": bool(email),
            "phone_revealed": bool(phone), "candidate": db.CandidateRepo.get(candidate_id)}


# ═══════════════════════════════════════════════════════════════════════════
#  LinkedIn enrichment — confirm job-change intent from the public profile
#  (best-effort public fetch + OpenAI structuring grounded in REAL data).
#  No LinkedIn API key yet → this is the active implementation of the reserved
#  LinkedIn seam; swap fetch_linkedin_public() for a LinkedIn API later.
# ═══════════════════════════════════════════════════════════════════════════

_LI_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_li_limiter = RateLimiter(2.0)  # be gentle with LinkedIn


def fetch_linkedin_public(url: str) -> dict:
    """Best-effort fetch of a PUBLIC LinkedIn profile. Returns og: tags + JSON-LD when
    available; flags a login wall. Never raises. (Datacenter IPs like Railway may be
    walled — caller degrades to AI-over-Apollo-data.)"""
    import html as _html
    if not url or "/in/" not in url:
        return {"_ok": False, "_reason": "no_profile_url"}
    try:
        _li_limiter.wait()
        r = requests.get(url, headers={"User-Agent": _LI_UA, "Accept-Language": "en-US,en"},
                         timeout=15, allow_redirects=True)
    except Exception as e:
        return {"_ok": False, "_reason": f"fetch_error:{e}"}
    if "authwall" in (r.url or "").lower() or r.status_code in (999, 403, 451):
        return {"_ok": False, "_reason": "login_wall", "_status": r.status_code}
    if r.status_code != 200:
        return {"_ok": False, "_reason": f"http_{r.status_code}"}
    html_text = r.text

    def _meta(prop):
        m = re.search(r'<meta\s+property="' + re.escape(prop) + r'"\s+content="([^"]*)"', html_text)
        return _html.unescape(m.group(1)) if m else None

    ld = None
    m = re.search(r'<script type="application/ld\+json">(.*?)</script>', html_text, re.S)
    if m:
        try:
            ld = json.loads(m.group(1))
        except Exception:
            ld = None
    return {"_ok": True, "og_title": _meta("og:title"), "og_description": _meta("og:description"),
            "ld": ld}


def _ld_person(ld) -> dict:
    """Pull a Person object out of LinkedIn JSON-LD (which is usually a @graph list)."""
    if not ld:
        return {}
    nodes = ld.get("@graph") if isinstance(ld, dict) else (ld if isinstance(ld, list) else [])
    for n in (nodes or []):
        if isinstance(n, dict) and "Person" in str(n.get("@type", "")):
            return n
    return ld if isinstance(ld, dict) else {}


def openai_linkedin_assess(candidate: dict, fetched: dict) -> dict:
    """Assess job-change intent + extract a concise profile from the REAL fetched
    LinkedIn text + Apollo data. Uses ONLY provided data (no fabrication). Degrades to
    a heuristic when OpenAI/key is unavailable."""
    eh = candidate.get("employment_history_json") or []
    hist = [{"title": h.get("title"), "company": h.get("organization_name"),
             "start": h.get("start_date"), "end": ("present" if h.get("current") else h.get("end_date"))}
            for h in eh if isinstance(h, dict)][:8]
    person_ld = _ld_person(fetched.get("ld"))
    li_text = " | ".join(filter(None, [fetched.get("og_title"), fetched.get("og_description")]))
    walled = not fetched.get("_ok")

    if not openai_available():
        # heuristic fallback over real data
        score, regime, sig = score_job_change_intent({
            "employment_history": eh, "title": candidate.get("title", ""),
            "headline": candidate.get("headline", "")})
        return {"_source": "heuristic", "open_to_work": score >= 65,
                "intent_likelihood": score, "summary": fetched.get("og_description") or candidate.get("headline"),
                "signals": [f"intent regime: {regime}"], "experience": hist,
                "linkedin_fetched": not walled,
                "note": "OpenAI key absent — intent inferred from Apollo data."}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        payload = {
            "name": candidate.get("full_name"), "current_title": candidate.get("title"),
            "headline": candidate.get("headline"), "company": candidate.get("company_name"),
            "apollo_employment_history": hist,
            "linkedin_public_text": li_text or None,
            "linkedin_jsonld": {k: person_ld.get(k) for k in
                                ("jobTitle", "worksFor", "alumniOf", "description", "address")
                                if person_ld.get(k)} or None,
            "linkedin_login_walled": walled,
        }
        sysmsg = ("You are an HR analyst confirming a candidate's job-change intent from "
                  "their LinkedIn + employment data. Use ONLY the provided data — NEVER invent "
                  "experience, skills, education, or facts. If LinkedIn text is missing/walled, "
                  "base intent on the Apollo employment history and say so. Output STRICT JSON: "
                  '{"open_to_work":bool,"intent_likelihood":0-100,"confidence":"low|medium|high",'
                  '"summary":"<=40 words, only from provided data","signals":["short factual signals"],'
                  '"experience":[{"title":..,"company":..,"start":..,"end":..}],'
                  '"skills":["only if explicitly present, else empty"],'
                  '"recommendation":"<=20 words for the recruiter"}')
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.2, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": json.dumps(payload, default=str)}], timeout=45)
        out = json.loads(resp.choices[0].message.content)
        out["_source"] = "openai"
        out["linkedin_fetched"] = not walled
        if not out.get("experience"):
            out["experience"] = hist
        return out
    except Exception as e:
        score, regime, _ = score_job_change_intent({"employment_history": eh,
                                                     "title": candidate.get("title", ""),
                                                     "headline": candidate.get("headline", "")})
        note = ("OpenAI key invalid/unreachable — intent from Apollo employment history; "
                "set a valid OPENAI_API_KEY for AI profile structuring.") \
            if "401" in str(e) or "api_key" in str(e).lower() else str(e)[:160]
        return {"_source": "heuristic", "open_to_work": score >= 65, "intent_likelihood": score,
                "summary": fetched.get("og_description") or candidate.get("headline"),
                "signals": [f"intent regime: {regime}"], "experience": hist,
                "linkedin_fetched": not walled, "note": note}


def enrich_linkedin(candidate_id: int) -> dict:
    """The 'Enrich via LinkedIn' action: confirm intent + capture a concise profile,
    store it, and recompute intent with source='linkedin'. FREE (no Apollo credits)."""
    row = db.CandidateRepo.get(candidate_id)
    if not row:
        return {"ok": False, "error": "not_found"}
    url = row.get("linkedin_url")
    fetched = fetch_linkedin_public(url) if url else {"_ok": False, "_reason": "no_url"}
    assess = openai_linkedin_assess(row, fetched)

    intent = assess.get("intent_likelihood")
    intent = clamp(intent) if isinstance(intent, (int, float)) else int(row.get("job_change_intent_score") or 50)
    overall = score_overall({"role_fit": row.get("role_fit_score", 50), "intent": intent,
                             "technical": row.get("technical_score", 50),
                             "company_quality": row.get("company_quality_score", 50),
                             "freshness": row.get("freshness_score", 100)})
    signals = {"checked_at": now_utc().isoformat(), "linkedin_url": url,
               "fetched_ok": bool(fetched.get("_ok")), "fetch_reason": fetched.get("_reason"),
               "og_title": fetched.get("og_title"), "og_description": fetched.get("og_description"),
               "assessment": assess}
    scores_json = {**(row.get("scores_json") or {}), "intent": intent, "overall": overall,
                   "intent_regime": "linkedin", "linkedin": assess}
    db.CandidateRepo.apply_linkedin(candidate_id, open_to_work=assess.get("open_to_work"),
                                    signals=signals, intent_score=intent,
                                    scores_json=scores_json, overall=overall)
    return {"ok": True, "candidate": db.CandidateRepo.get(candidate_id), "assessment": assess,
            "linkedin_fetched": bool(fetched.get("_ok"))}


# ── CoreSignal enrichment flow ───────────────────────────────────────────────
def coresignal_resolve(candidate: dict) -> dict:
    """Resolve a known candidate → ONE CoreSignal employee record. Branch A (linkedin_url
    direct collect) → Branch B (preview-first search → disambiguate → collect). Returns
    {ok, record, match, needs_manual_pick, candidates, error, status, credits}."""
    cs = get_coresignal()
    if not cs.configured():
        return {"ok": False, "error": "not_configured", "record": None, "credits": None}
    name = (candidate.get("full_name") or "").strip()
    title = (candidate.get("title") or "").strip()
    company = (candidate.get("company_name") or "").strip()
    domain = (candidate.get("company_domain") or "").strip()
    if not name:
        return {"ok": False, "error": "no_name", "record": None, "credits": cs.credits_remaining}

    def _fatal(r):  # auth / credit errors that should stop the flow
        if r["status"] == 402:
            return {"ok": False, "error": "insufficient_credits", "record": None, "status": 402, "credits": r["credits"]}
        if r["status"] in (401, 403):
            return {"ok": False, "error": "auth", "record": None, "status": r["status"], "credits": r["credits"]}
        return None

    # BRANCH A — direct collect by LinkedIn slug (cheapest, most precise)
    slug = _linkedin_slug(candidate.get("linkedin_url") or "")
    if slug:
        r = cs.collect(slug)
        if r["ok"] and isinstance(r["data"], dict) and r["data"].get("id"):
            return {"ok": True, "record": r["data"], "needs_manual_pick": False, "candidates": [],
                    "match": {"confidence": "high", "method": "linkedin_url", "id": r["data"].get("id")},
                    "error": None, "status": 200, "credits": r["credits"]}
        f = _fatal(r)
        if f:
            return f
        # else 404/no_data → fall through to Branch B

    # BRANCH B — preview-first search, disambiguate, then collect ONE id
    previews: list = []
    for relax in (0, 1, 2, 3):
        body = _cs_build_search_body(name, title, company, domain, relax=relax)
        pv = cs.search(body, preview=True)
        if not pv["ok"]:
            f = _fatal(pv)
            if f:
                return f
            continue
        if isinstance(pv["data"], list) and pv["data"]:
            previews = pv["data"][:15]
            break
    if not previews:
        return {"ok": False, "error": "no_match", "record": None, "candidates": [], "credits": cs.credits_remaining}

    best, conf, method, ambiguous = _cs_pick_best(previews, domain, title)
    if ambiguous:
        return {"ok": False, "needs_manual_pick": True, "record": None,
                "candidates": [_cs_preview_brief(p) for p in previews], "error": "ambiguous",
                "credits": cs.credits_remaining}
    c = cs.collect(best.get("id"))
    if c["ok"] and isinstance(c["data"], dict):
        return {"ok": True, "record": c["data"], "needs_manual_pick": False, "candidates": [],
                "match": {"confidence": conf, "method": method, "id": best.get("id")},
                "error": None, "status": 200, "credits": c["credits"]}
    f = _fatal(c)
    if f:
        return f
    return {"ok": False, "error": "collect_failed", "record": None, "credits": c["credits"]}


def _cs_exp_brief(exp: list, n: int) -> list:
    out = []
    for e in (exp or []):
        if not isinstance(e, dict):
            continue
        out.append({"title": e.get("position_title") or e.get("title"),
                    "company": e.get("company_name"), "start": e.get("date_from"),
                    "end": (e.get("date_to") or "present"), "months": e.get("duration_months"),
                    "current": bool(e.get("active_experience"))})
    return out[:n]


def _cs_store_fields(rec: dict) -> dict:
    """Compact subset of the CoreSignal record kept for UI display."""
    return {
        "full_name": rec.get("full_name"), "headline": rec.get("headline"),
        "summary": rec.get("summary"), "location": rec.get("location_full"),
        "country": rec.get("location_country"), "current_title": rec.get("active_experience_title"),
        "department": rec.get("active_experience_department"),
        "management_level": rec.get("active_experience_management_level"),
        "is_decision_maker": rec.get("is_decision_maker"),
        "total_experience_months": rec.get("total_experience_duration_months"),
        "connections": rec.get("connections_count"), "followers": rec.get("followers_count"),
        "skills": (rec.get("inferred_skills") or rec.get("skills") or [])[:25],
        "experience": _cs_exp_brief(rec.get("experience"), 12),
        "education": [{"institution": e.get("institution_name") or e.get("title"),
                       "degree": e.get("degree") or e.get("subtitle"),
                       "end": e.get("date_to") or e.get("date_to_year")}
                      for e in (rec.get("education") or []) if isinstance(e, dict)][:8],
        "certifications": [{"title": c.get("title"), "issuer": c.get("issuer") or c.get("subtitle")}
                           for c in (rec.get("certifications") or []) if isinstance(c, dict)][:10],
        "languages": rec.get("languages"),
        "recent_started": rec.get("experience_recently_started"),
        "recent_closed": rec.get("experience_recently_closed"),
        "updated_at": rec.get("updated_at") or rec.get("checked_at"),
        "profile_url": _cs_profile_url(rec),
    }


def _cs_openai_payload(rec: dict, candidate: dict) -> dict:
    exp = _cs_exp_brief(rec.get("experience"), 8)
    return {
        "name": rec.get("full_name") or candidate.get("full_name"),
        "headline": rec.get("headline"), "about": rec.get("summary"),
        "current_title": rec.get("active_experience_title") or candidate.get("title"),
        "current_company": (exp[0]["company"] if exp else candidate.get("company_name")),
        "department": rec.get("active_experience_department"),
        "management_level": rec.get("active_experience_management_level"),
        "is_decision_maker": rec.get("is_decision_maker"),
        "location": rec.get("location_full"),
        "total_experience_months": rec.get("total_experience_duration_months"),
        "skills": rec.get("inferred_skills") or rec.get("skills"),
        "experience": exp,
        "education": [{"institution": e.get("institution_name") or e.get("title"),
                       "degree": e.get("degree") or e.get("subtitle"),
                       "end_year": e.get("date_to_year") or e.get("date_to")}
                      for e in (rec.get("education") or []) if isinstance(e, dict)][:6],
        "certifications": [{"title": c.get("title"), "issuer": c.get("issuer") or c.get("subtitle"),
                            "year": c.get("date_from_year") or c.get("date_from")}
                           for c in (rec.get("certifications") or []) if isinstance(c, dict)][:8],
        "languages": rec.get("languages"),
        "recent_role_started": rec.get("experience_recently_started"),
        "recent_role_closed": rec.get("experience_recently_closed"),
        "experience_change_last_identified_at": rec.get("experience_change_last_identified_at"),
        "data_source": "coresignal_multi_source",
        "record_updated_at": rec.get("updated_at") or rec.get("checked_at"),
    }


def _cs_heuristic_assess(rec: dict, candidate: dict, note: str = "") -> dict:
    exp = rec.get("experience") or []
    hist = [{"title": e.get("position_title") or e.get("title"),
             "organization_name": e.get("company_name"), "start_date": e.get("date_from"),
             "end_date": e.get("date_to"), "current": bool(e.get("active_experience"))}
            for e in exp if isinstance(e, dict)]
    score, regime, _ = score_job_change_intent({
        "employment_history": hist,
        "title": rec.get("active_experience_title") or candidate.get("title", ""),
        "headline": rec.get("headline") or ""})
    started = rec.get("experience_recently_started") or []
    keyexp = _cs_exp_brief(exp, 5)
    months = rec.get("total_experience_duration_months") or 0
    return {"_source": "coresignal",
            "summary": rec.get("summary") or rec.get("headline"),
            "current_role": {"title": rec.get("active_experience_title"),
                             "company": (keyexp[0]["company"] if keyexp else candidate.get("company_name"))},
            "seniority_level": rec.get("active_experience_management_level") or candidate.get("seniority"),
            "department": rec.get("active_experience_department") or candidate.get("department"),
            "years_experience": (round(months / 12, 1) if months else None),
            "top_skills": (rec.get("inferred_skills") or rec.get("skills") or [])[:8],
            "key_experience": [{"title": e["title"], "company": e["company"],
                                "start": e["start"], "end": e["end"]} for e in keyexp],
            "education": [{"institution": e.get("institution_name") or e.get("title"),
                           "degree": e.get("degree") or e.get("subtitle")}
                          for e in (rec.get("education") or []) if isinstance(e, dict)][:4],
            "open_to_work": (score >= 65 or bool(started)),
            "intent_likelihood": score, "confidence": "medium",
            "signals": ([f"recently started role at {started[0].get('company_name')}"] if started else [])
            + [f"intent regime: {regime}"],
            "recommendation": "Review the CoreSignal LinkedIn profile.",
            "data_completeness": "high" if exp else "low", "note": note}


def openai_coresignal_assess(candidate: dict, record: dict) -> dict:
    """Concise structured LinkedIn assessment from a CoreSignal record via gpt-4o-mini
    (JSON mode). Uses ONLY provided data; degrades to a heuristic when OpenAI is absent."""
    if not openai_available():
        return _cs_heuristic_assess(record, candidate,
                                    note="OpenAI key absent — summary derived directly from CoreSignal data.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        payload = _cs_openai_payload(record, candidate)
        sysmsg = ("You are an HR analyst producing a concise LinkedIn-based assessment of a known "
                  "candidate from CoreSignal structured data. Use ONLY the provided fields — NEVER "
                  "invent experience, skills, education, employers, or dates. If a field is absent, "
                  "omit it or use null. Output STRICT JSON: "
                  '{"summary":"<=45 words, only from provided data",'
                  '"current_role":{"title":str,"company":str},'
                  '"seniority_level":"intern|entry|manager|senior|director|vp|c_suite",'
                  '"department":"sales|marketing|seo|digital_marketing|engineering|other",'
                  '"years_experience":number,'
                  '"top_skills":[str],"key_experience":[{"title":str,"company":str,"start":str,"end":str}],'
                  '"education":[{"institution":str,"degree":str}],'
                  '"open_to_work":bool,"intent_likelihood":0-100,"confidence":"low|medium|high",'
                  '"signals":["short factual job-change signals from the data"],'
                  '"recommendation":"<=20 words for the recruiter",'
                  '"data_completeness":"low|medium|high"}')
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.2, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": json.dumps(payload, default=str)}], timeout=45)
        out = json.loads(resp.choices[0].message.content)
        out["_source"] = "coresignal+openai"
        return out
    except Exception as e:
        note = ("OpenAI key invalid/unreachable — summary from CoreSignal data only.") \
            if ("401" in str(e) or "api_key" in str(e).lower()) else str(e)[:160]
        return _cs_heuristic_assess(record, candidate, note=note)


def enrich_coresignal(candidate_id: int, employee_id=None) -> dict:
    """The 'Enrich via CoreSignal' action: resolve the candidate to a CoreSignal LinkedIn
    record, summarize with OpenAI, store it, and recompute intent. PAID (CoreSignal credits).
    Pass employee_id to collect a specific record (used after a manual disambiguation pick)."""
    row = db.CandidateRepo.get(candidate_id)
    if not row:
        return {"ok": False, "error": "not_found"}
    cs = get_coresignal()
    if not cs.configured():
        return {"ok": False, "error": "not_configured",
                "message": "Set CORESIGNAL_API_KEY on Railway to enable CoreSignal LinkedIn enrichment."}

    if employee_id:
        c = cs.collect(employee_id)
        if not (c["ok"] and isinstance(c["data"], dict)):
            err = "insufficient_credits" if c["status"] == 402 else (c["error"] or "collect_failed")
            return {"ok": False, "error": err, "credits_remaining": c["credits"]}
        res = {"ok": True, "record": c["data"], "credits": c["credits"],
               "match": {"confidence": "high", "method": "manual_pick", "id": employee_id}}
    else:
        res = coresignal_resolve(row)
        # If name+company couldn't find a profile AND we have no LinkedIn URL, ask Apollo
        # for the person's LinkedIn URL (no contact reveal), then retry the precise lookup.
        via_apollo = False
        if (not res.get("ok") and res.get("error") == "no_match" and not row.get("linkedin_url")
                and _env("CORESIGNAL_APOLLO_LOOKUP", "1") == "1"):
            url = apollo_linkedin_lookup(row)
            if url:
                row["linkedin_url"] = url
                try:
                    db.CandidateRepo.set_linkedin_url(candidate_id, url)
                except Exception as e:
                    log.warning("set_linkedin_url failed: %s", e)
                res = coresignal_resolve(row)
                via_apollo = res.get("ok", False)
        res["_via_apollo"] = via_apollo

    if not res.get("ok"):
        out = {"ok": False, "error": res.get("error"), "credits_remaining": res.get("credits")}
        if res.get("needs_manual_pick"):
            out["needs_manual_pick"] = True
            out["candidates"] = res.get("candidates") or []
        return out

    record = res["record"]
    assess = openai_coresignal_assess(row, record)
    intent = assess.get("intent_likelihood")
    intent = clamp(intent) if isinstance(intent, (int, float)) else int(row.get("job_change_intent_score") or 50)
    overall = score_overall({"role_fit": row.get("role_fit_score", 50), "intent": intent,
                             "technical": row.get("technical_score", 50),
                             "company_quality": row.get("company_quality_score", 50),
                             "freshness": row.get("freshness_score", 100)})
    resolved_url = _cs_profile_url(record) or row.get("linkedin_url")
    cs_store = {"checked_at": now_utc().isoformat(), "coresignal_id": record.get("id"),
                "match": res.get("match"), "profile_url": resolved_url,
                "raw": _cs_store_fields(record), "assessment": assess,
                "credits_remaining": res.get("credits"), "dataset": cs.dataset}
    scores_json = {**(row.get("scores_json") or {}), "intent": intent, "overall": overall,
                   "intent_regime": "coresignal", "coresignal": assess}
    db.CandidateRepo.apply_coresignal(
        candidate_id, coresignal_json=cs_store,
        coresignal_id=(str(record.get("id")) if record.get("id") is not None else None),
        open_to_work=assess.get("open_to_work"), intent_score=intent,
        scores_json=scores_json, overall=overall,
        linkedin_url=(resolved_url if _looks_like_linkedin(resolved_url) else None))
    return {"ok": True, "candidate": db.CandidateRepo.get(candidate_id), "assessment": assess,
            "match": res.get("match"), "coresignal": cs_store, "credits_remaining": res.get("credits"),
            "linkedin_via_apollo": bool(res.get("_via_apollo"))}


def apollo_linkedin_lookup(candidate: dict) -> Optional[str]:
    """ONE Apollo people/match WITHOUT revealing email/phone, purely to obtain the
    person's LinkedIn URL so CoreSignal can do a precise (Branch A) profile lookup.
    Many free-search candidates have no linkedin_url; a targeted match usually does.
    Returns the URL or None. Never raises. (Uses an Apollo match, not a contact reveal.)"""
    name = (candidate.get("full_name") or "").strip()
    if not name:
        return None
    ap = get_apollo()
    if not getattr(ap, "api_key", ""):
        return None
    parts = name.split()
    first = parts[0]
    last = " ".join(parts[1:]) if len(parts) > 1 else ""
    try:
        # Match by apollo_person_id when available — name+domain alone does NOT resolve the
        # exact person (Apollo returns an empty match with no linkedin_url).
        res = ap.enrich_person(apollo_id=candidate.get("apollo_person_id") or "",
                               first_name=first, last_name=last,
                               domain=candidate.get("company_domain") or "",
                               reveal_email=False, reveal_phone=False)
    except Exception as e:
        log.warning("apollo linkedin lookup failed: %s", e)
        return None
    if not res.get("_ok"):
        return None
    url = (res.get("person") or {}).get("linkedin_url")
    return url or None


def apollo_webhook_token() -> str:
    """Stable secret for the Apollo async phone webhook. Uses APOLLO_WEBHOOK_SECRET if set,
    else derives one from SECRET_KEY. Empty when neither is configured (then phone reveal is
    skipped and only the synchronous email is returned)."""
    t = _env("APOLLO_WEBHOOK_SECRET")
    if t:
        return t
    sk = _env("SECRET_KEY")
    if sk:
        import hashlib
        return hashlib.sha256(("apollo-phone:" + sk).encode()).hexdigest()[:32]
    return ""


def handle_apollo_phone_webhook(data: dict) -> int:
    """Apollo posts the async phone-reveal result here (minutes after the match). Defensively
    locate the people + their phone numbers and write them onto candidates by apollo_person_id.
    Returns how many candidates were updated."""
    if not isinstance(data, dict):
        return 0
    people = data.get("people") or data.get("contacts") or data.get("matches") or []
    if not people and isinstance(data.get("person"), dict):
        people = [data["person"]]
    if isinstance(people, dict):
        people = [people]
    updated = 0
    for p in people:
        if not isinstance(p, dict):
            continue
        pid = p.get("id") or p.get("person_id") or p.get("apollo_id")
        phone = _best_phone(p)
        if pid and phone:
            try:
                if db.CandidateRepo.set_phone_by_apollo_id(str(pid), phone):
                    updated += 1
            except Exception as e:
                log.warning("phone webhook update failed: %s", e)
    if not updated:
        log.info("apollo phone webhook: no usable phone in payload (keys=%s)", list(data.keys())[:8])
    return updated


def coresignal_status() -> dict:
    cs = get_coresignal()
    return {"configured": cs.configured(), "dataset": cs.dataset,
            "credits_remaining": cs.credits_remaining}


# ═══════════════════════════════════════════════════════════════════════════
#  Daily auto-refresh scheduler
# ═══════════════════════════════════════════════════════════════════════════

SCHED_POLL_SECONDS = 120  # short poll → responsive to the auto-hunt toggle


def auto_hunt_enabled() -> bool:
    return db.SettingsRepo.get_bool("auto_hunt", _env("HR_AUTO_HUNT_DEFAULT", "0") == "1")


def set_auto_hunt(on: bool) -> None:
    db.SettingsRepo.set("auto_hunt", "1" if on else "0")
    if on:
        db.SettingsRepo.set("hunt_started_at", now_utc().isoformat())


# ── Roster verifier (separate toggle): for every company, ensure every Apollo
#    employee is in the DB. Gated so the user controls this heavy free crawl. ──────
def roster_verify_enabled() -> bool:
    return db.SettingsRepo.get_bool("roster_verify", _env("HR_ROSTER_VERIFY_DEFAULT", "0") == "1")


def set_roster_verify(on: bool) -> None:
    db.SettingsRepo.set("roster_verify", "1" if on else "0")
    if on:
        db.SettingsRepo.set("roster_started_at", now_utc().isoformat())


def roster_status() -> dict:
    s = db.SettingsRepo.get_many(["roster_verify", "roster_started_at", "roster_last_at"])
    counts = db.CompanyRepo.roster_counts()
    rv = s.get("roster_verify")
    enabled = (rv in ("1", "true", "True", "on")) if rv is not None \
        else (_env("HR_ROSTER_VERIFY_DEFAULT", "0") == "1")
    return {"enabled": enabled, "last_at": s.get("roster_last_at"),
            "started_at": s.get("roster_started_at"), **counts}


def hunt_status() -> dict:
    """Single-query status snapshot (fast even over a high-latency DB proxy)."""
    s = db.SettingsRepo.get_many(["auto_hunt", "hunt_last_at", "hunt_started_at",
                                  "hunt_cursor", "hunt_cycles"])
    total = len(CATEGORY_LABELS)
    try:
        cursor = int(s.get("hunt_cursor") or 0)
    except (ValueError, TypeError):
        cursor = 0
    av = s.get("auto_hunt")
    enabled = (av in ("1", "true", "True", "on")) if av is not None \
        else (_env("HR_AUTO_HUNT_DEFAULT", "0") == "1")
    n = max(1, _env_int("HUNT_CATEGORIES_PER_CYCLE", 3))
    nextcats = [CATEGORY_LABELS[(cursor + i) % total] for i in range(min(n, total))] if total else []
    try:
        cycles = int(s.get("hunt_cycles") or 0)
    except (ValueError, TypeError):
        cycles = 0
    return {
        "enabled": enabled, "last_hunt_at": s.get("hunt_last_at"),
        "started_at": s.get("hunt_started_at"), "cursor": cursor % total if total else 0,
        "next_categories": nextcats, "total_categories": total, "cycles": cycles,
        "interval_seconds": _env_int("HUNT_INTERVAL_SECONDS", 600),
    }


def _peek_hunt_categories() -> List[str]:
    order = CATEGORY_LABELS
    if not order:
        return []
    cursor = db.SettingsRepo.get_int("hunt_cursor", 0) % len(order)
    n = max(1, _env_int("HUNT_CATEGORIES_PER_CYCLE", 3))
    return [order[(cursor + i) % len(order)] for i in range(min(n, len(order)))]


def _next_hunt_params():
    order = CATEGORY_LABELS
    cursor = db.SettingsRepo.get_int("hunt_cursor", 0) % len(order)
    n = max(1, _env_int("HUNT_CATEGORIES_PER_CYCLE", 3))
    cats = [order[(cursor + i) % len(order)] for i in range(min(n, len(order)))]
    params = {"categories": cats, "person_locations": _env_list("HR_PERSON_LOCATIONS"),
              "max_candidates": _env_int("HUNT_MAX_PER_CYCLE", 600),
              "max_pages": _env_int("HUNT_PAGES_PER_CELL", 1), "trigger": "hunt"}
    return params, (cursor + n) % len(order)


def scheduler_loop(stop_event: threading.Event, trigger_scheduled_run) -> None:
    """Background worker. When the auto-hunt toggle is ON, it continuously rotates
    through the category taxonomy, launching a FREE discovery slice every
    HUNT_INTERVAL_SECONDS (zero credits — search only). Always keeps freshness
    scores decaying hourly. `trigger_scheduled_run(params)` returns True if started,
    False if a run is already active."""
    log.info("Scheduler thread started (poll=%ss)", SCHED_POLL_SECONDS)
    while not stop_event.wait(SCHED_POLL_SECONDS):
        try:
            # hourly freshness decay (cheap single UPDATE), independent of hunting
            fa = db.SettingsRepo.get("freshness_at")
            due = fa is None
            if fa:
                try:
                    due = (now_utc() - datetime.datetime.fromisoformat(fa)).total_seconds() >= 3600
                except Exception:
                    due = True
            if due:
                db.CandidateRepo.recompute_freshness_all(_env_int("HR_INTENT_OPEN_THRESHOLD", 60))
                db.SettingsRepo.set("freshness_at", now_utc().isoformat())

            # Progressively resolve company websites (free Clearbit), paced every tick.
            try:
                backfill_company_domains(_env_int("HUNT_DOMAIN_BACKFILL_PER_TICK", 40))
            except Exception as e:
                log.warning("domain backfill error: %s", e)

            # Progressively pull each company's homepage 'About' (free), paced.
            if _env("HR_WEB_ENRICH_ENABLED", "1") == "1":
                try:
                    enrich_company_web(_env_int("HUNT_WEB_ENRICH_PER_TICK", 20))
                except Exception as e:
                    log.warning("web enrich error: %s", e)

            # Roster verifier (separate toggle): pull each company's FULL employee
            # roster from Apollo so none are missing. Only when the user enables it.
            if roster_verify_enabled():
                try:
                    added, n = roster_sync_batch(_env_int("HUNT_ROSTER_PER_TICK", 4))
                    db.SettingsRepo.set("roster_last_at", now_utc().isoformat())
                    if added:
                        log.info("roster verify: +%d people across %d companies", added, n)
                except Exception as e:
                    log.warning("roster sync error: %s", e)

            if not auto_hunt_enabled():
                continue
            # pace hunt cycles by HUNT_INTERVAL_SECONDS
            last = db.SettingsRepo.get("hunt_last_at")
            interval = _env_int("HUNT_INTERVAL_SECONDS", 600)
            if last:
                try:
                    if (now_utc() - datetime.datetime.fromisoformat(last)).total_seconds() < interval:
                        continue
                except Exception:
                    pass
            params, new_cursor = _next_hunt_params()
            if trigger_scheduled_run(params):
                db.SettingsRepo.set("hunt_cursor", str(new_cursor))
                db.SettingsRepo.set("hunt_last_at", now_utc().isoformat())
                db.SettingsRepo.set("hunt_cycles", str(db.SettingsRepo.get_int("hunt_cycles", 0) + 1))
                log.info("Auto-hunt cycle: %s", params["categories"])
        except Exception as e:  # pragma: no cover
            log.warning("scheduler tick error: %s", e)
