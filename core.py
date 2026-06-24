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
                      reveal_phone: bool = False) -> dict:
        """Paid people/match reveal. Returns a structured dict with _ok / _no_credits /
        _http_status / email / phone / employment_history. Retries once WITHOUT reveal
        flags on 400/422 (still returns employment_history cheaply)."""
        url = f"{self.BASE_URL}/people/match"
        payload: Dict[str, Any] = {}
        if reveal_email:
            payload["reveal_personal_emails"] = True
        if reveal_phone:
            payload["reveal_phone_number"] = True
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
        if resp is not None and resp.status_code in (400, 422):
            body = resp.text[:300]
            if "insufficient credits" in body.lower():
                if not self._logged_match_error:
                    self._logged_match_error = True
                    log.error("Apollo EXPORT CREDITS EXHAUSTED: %s", body)
                return {"_ok": False, "_no_credits": True, "_http_status": resp.status_code}
            payload.pop("reveal_phone_number", None)
            payload.pop("reveal_personal_emails", None)
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


def cleanup_irrelevant_companies(limit: int = 5000) -> int:
    """Delete existing companies (and their candidates) that fail the relevance filter
    (banks, govt, healthcare, etc.). One-time/periodic housekeeping."""
    rows = db.CompanyRepo.all_id_name(limit)
    drop = [r["id"] for r in rows if not is_relevant_company(r.get("name") or "")]
    if drop:
        db.CompanyRepo.delete_with_candidates(drop)
    return len(drop)


def enrich_candidate(candidate_id: int, reveal_email: bool = True,
                     reveal_phone: bool = False) -> dict:
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
        reveal_email=reveal_email, reveal_phone=reveal_phone)

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
#  Daily auto-refresh scheduler
# ═══════════════════════════════════════════════════════════════════════════

SCHED_POLL_SECONDS = 120  # short poll → responsive to the auto-hunt toggle


def auto_hunt_enabled() -> bool:
    return db.SettingsRepo.get_bool("auto_hunt", _env("HR_AUTO_HUNT_DEFAULT", "0") == "1")


def set_auto_hunt(on: bool) -> None:
    db.SettingsRepo.set("auto_hunt", "1" if on else "0")
    if on:
        db.SettingsRepo.set("hunt_started_at", now_utc().isoformat())


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
