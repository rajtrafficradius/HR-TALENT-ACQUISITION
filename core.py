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
        phone_reveal_error = None  # captured so the real Apollo phone-reject reason isn't swallowed
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
                phone_reveal_error = f"http_{resp.status_code}: {body}"
                log.warning("Apollo phone reveal rejected: %s", phone_reveal_error)
            elif payload.pop("reveal_personal_emails", None) is None:
                break  # nothing left to strip
            resp = self._post(url, payload)
            retried = True

        if resp is None:
            return {"_ok": False, "_http_status": None, "_error": "network",
                    "_phone_reveal_error": phone_reveal_error}
        if resp.status_code != 200:
            return {"_ok": False, "_http_status": resp.status_code, "_error": resp.text[:200],
                    "_phone_reveal_error": phone_reveal_error}

        self.counter["match"] += 1
        data = resp.json() or {}
        person = data.get("person") or {}
        org = person.get("organization") or {}
        co_phone = safe_str(org.get("phone") or org.get("sanitized_phone"))
        pp = org.get("primary_phone")
        if not co_phone and isinstance(pp, dict):
            co_phone = safe_str(pp.get("number") or pp.get("sanitized_number"))
        phone_requested = ("reveal_phone_number" in payload)
        # Capture the async request_id robustly (Apollo's shape varies) so we can POLL
        # GET /webhook_result/{request_id} later — reliable even if the inbound webhook never reaches us.
        req_id = ""
        if phone_requested:
            for cand_id in (data.get("request_id"), data.get("id"),
                            (data.get("request") or {}).get("id") if isinstance(data.get("request"), dict) else None,
                            (data.get("enrichment") or {}).get("request_id") if isinstance(data.get("enrichment"), dict) else None):
                if cand_id:
                    req_id = safe_str(cand_id); break
        return {
            "_ok": True, "_http_status": 200, "_retried": retried,
            "_phone_reveal_error": phone_reveal_error,
            "_phone_requested": phone_requested or bool(webhook_url and not phone_reveal_error),
            "_phone_request_id": req_id,
            "email": _best_email(person, first_name, last_name),
            "phone": _best_phone(person, co_phone),
            "employment_history": person.get("employment_history") or [],
            "person": person,
        }

    def poll_webhook_result(self, request_id: str) -> Optional[dict]:
        """Poll Apollo for an async reveal result: GET /api/v1/webhook_result/{request_id}.
        This is the RELIABLE phone path — we pull the result ourselves instead of waiting for
        Apollo's webhook to reach our server. Returns the JSON payload (same shape as the
        webhook: {status, people:[{phone_numbers:[...]}]}) or None if not ready / on error.
        Results are retained ~30 days by Apollo. Costs no extra credits (already spent)."""
        if not request_id:
            return None
        try:
            self.limiter.wait()
            resp = requests.get(f"{self.BASE_URL}/webhook_result/{request_id}",
                                headers=self._headers(), timeout=25)
        except Exception as e:
            log.warning("poll_webhook_result network error (%s): %s", request_id, e)
            return None
        if resp is None or resp.status_code != 200:
            return None
        try:
            return resp.json() or None
        except Exception:
            return None

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


def safe_str(x) -> str:
    return "" if x is None else str(x).strip()


# Consumer/personal email domains — an address here is "personal", anything else at a
# company domain is treated as an official/work email. (Ported from LEAD FORGE V5.py.)
PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "yahoo.com.au", "yahoo.ca",
    "yahoo.co.in", "ymail.com", "rocketmail.com", "hotmail.com", "hotmail.co.uk",
    "hotmail.com.au", "hotmail.ca", "outlook.com", "outlook.com.au", "live.com",
    "live.com.au", "msn.com", "passport.com", "icloud.com", "me.com", "mac.com", "aol.com",
    "protonmail.com", "proton.me", "pm.me", "fastmail.com", "fastmail.fm", "zoho.com",
    "tutanota.com", "tutamail.com", "hey.com", "mail.com", "email.com", "bigpond.com",
    "bigpond.net.au", "telstra.com", "optusnet.com.au", "tpg.com.au", "tpg.com",
    "internode.on.net", "aapt.net.au", "iprimus.com.au", "westnet.com.au", "dodo.com.au",
    "btinternet.com", "btopenworld.com", "sky.com", "talktalk.net", "virgin.net",
    "ntlworld.com", "blueyonder.co.uk", "rediffmail.com", "indiatimes.com", "gmx.com",
    "gmx.net", "gmx.de", "web.de", "t-online.de", "seznam.cz", "yandex.com", "yandex.ru",
}


def is_personal_email(email: str) -> bool:
    """True ONLY for known consumer domains (gmail, yahoo, …). first@company.com is NOT
    personal — it's an official/work email."""
    if not email or "@" not in email:
        return False
    return email.lower().split("@")[-1].strip() in PERSONAL_EMAIL_DOMAINS


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:(?:\+|00)\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?){2,5}\d{2,4}")


def _valid_email(s: str) -> Optional[str]:
    s = (s or "").strip().strip(".,;:<>()[]\"' ")
    if not s or "@" not in s:
        return None
    m = _EMAIL_RE.search(s)
    return m.group(0).lower() if m else None


def _valid_phone(s: str) -> Optional[str]:
    """Accept a phone only if it has 8–15 digits (drops years, ZIPs, ids)."""
    if not s:
        return None
    m = _PHONE_RE.search(str(s))
    if not m:
        return None
    cand = m.group(0).strip()
    digits = re.sub(r"\D", "", cand)
    if not (8 <= len(digits) <= 15):
        return None
    return cand


def _harvest_contacts_from_obj(obj, emails: set, phones: set, depth: int = 0, phone_ctx: bool = False) -> None:
    """Walk a CoreSignal record collecting contacts. Emails: scanned from EVERY string (the @
    pattern is unambiguous). Phones: only under a phone-ish key (to avoid grabbing random numbers
    like years/ids) — the key context propagates into nested lists/dicts."""
    if depth > 5 or obj is None:
        return
    if isinstance(obj, str):
        e = _valid_email(obj)
        if e:
            emails.add(e)
        if phone_ctx:
            p = _valid_phone(obj)
            if p:
                phones.add(p)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            child_phone = phone_ctx or ("phone" in kl or "mobile" in kl or "contact_number" in kl)
            _harvest_contacts_from_obj(v, emails, phones, depth + 1, child_phone)
    elif isinstance(obj, list):
        for it in obj[:80]:
            _harvest_contacts_from_obj(it, emails, phones, depth + 1, phone_ctx)


def extract_linkedin_contacts(record: dict, candidate: Optional[dict] = None) -> dict:
    """Pull any phone/email present on a candidate's LinkedIn profile from the CoreSignal record.
    Two free passes: (1) structured contact-ish fields anywhere in the record; (2) OpenAI over the
    free-text summary/headline (people often put a contact in their 'About'). Validated, deduped.
    Returns {email, phone, source}. Never raises."""
    emails: set = set()
    phones: set = set()
    try:
        _harvest_contacts_from_obj(record, emails, phones)
    except Exception:
        pass
    # OpenAI pass over the free text only (cheap, grounded, no invention).
    if openai_available():
        text = " \n".join(filter(None, [
            (record or {}).get("summary"), (record or {}).get("headline"),
            (record or {}).get("about"), (record or {}).get("contact_info"),
            (record or {}).get("location_full")]))[:4000]
        if text.strip():
            try:
                from openai import OpenAI
                client = OpenAI(api_key=_env("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content":
                               "Extract a contact EMAIL and PHONE only if they literally appear in the "
                               "text. Never guess or fabricate. Return STRICT JSON "
                               '{"email":"<addr or null>","phone":"<number or null>"}.'},
                              {"role": "user", "content": text}], timeout=20)
                out = json.loads(resp.choices[0].message.content)
                e = _valid_email(out.get("email") or "")
                p = _valid_phone(out.get("phone") or "")
                if e:
                    emails.add(e)
                if p:
                    phones.add(p)
            except Exception as ex:
                log.warning("linkedin contact OpenAI extract failed: %s", ex)
    # Prefer a non-consumer/work-looking email if several; else any.
    email = None
    if emails:
        official = [e for e in emails if not is_personal_email(e)]
        email = sorted(official or list(emails))[0]
    phone = sorted(phones, key=len, reverse=True)[0] if phones else None
    return {"email": email, "phone": phone, "source": "linkedin"}


def apply_linkedin_contacts(candidate_id: int, record: dict, candidate: Optional[dict] = None) -> dict:
    """Extract LinkedIn contacts and persist them (precedence over Apollo; never overwritten)."""
    got = extract_linkedin_contacts(record, candidate)
    try:
        db.CandidateRepo.set_linkedin_contacts(candidate_id, got.get("email"), got.get("phone"))
    except Exception as e:
        log.warning("set_linkedin_contacts failed (%s): %s", candidate_id, e)
    return got


def effective_contacts(c: dict) -> dict:
    """LinkedIn-sourced contact takes precedence over Apollo for BOTH email and phone, regardless
    of Apollo's enrichment state. Returns the values to display + their source."""
    li_e = (c.get("linkedin_email") or "").strip()
    li_p = (c.get("linkedin_phone") or "").strip()
    ap_e = (c.get("email") or "").strip()
    ap_p = (c.get("phone") or "").strip()
    return {
        "email_effective": li_e or ap_e or None,
        "phone_effective": li_p or ap_p or None,
        "email_source": "linkedin" if li_e else ("apollo" if ap_e else None),
        "phone_source": "linkedin" if li_p else ("apollo" if ap_p else None),
    }


def _email_contains_person_name(email_addr: str, first_name: str = "", last_name: str = "") -> bool:
    if not email_addr or "@" not in email_addr:
        return False
    local = email_addr.lower().split("@")[0]
    clean = local.replace(".", "").replace("-", "").replace("_", "")
    fl = (first_name or "").lower().strip()
    ll = (last_name or "").lower().strip()
    if fl and len(fl) >= 2 and (fl in local or fl in clean):
        return True
    if ll and len(ll) >= 2 and (ll in local or ll in clean):
        return True
    return False


def _pick_best_email_from_apollo(person: dict, first_name: str = "", last_name: str = ""):
    """Pick the proper OFFICIAL/personal email (never the masked 'email_not_unlocked'
    placeholder). Priority: business-primary → business → personal-primary → personal →
    company-domain org/contact → name-based personal_emails → consumer → any. Ported from
    LEAD FORGE V5.py. Returns (email, is_from_personal_list)."""
    def _ok(e):
        return bool(e) and "@" in e and "email_not_unlocked" not in e
    structured = person.get("emails") or []
    flat_personal = [safe_str(e) for e in (person.get("personal_emails") or []) if _ok(safe_str(e))]
    org_email = safe_str(person.get("email"))
    contact_email = safe_str(person.get("contact_email"))
    org_email = org_email if _ok(org_email) else ""
    contact_email = contact_email if _ok(contact_email) else ""

    if structured:
        bp, bo, pp, po = [], [], [], []
        for em in structured:
            if not isinstance(em, dict):
                continue
            addr = safe_str(em.get("email"))
            if not _ok(addr):
                continue
            etype = (em.get("email_type") or em.get("type") or "").lower()
            etag = (em.get("email_tag") or em.get("tag") or em.get("label") or "").lower()
            estatus = (em.get("email_status") or em.get("status") or "").lower()
            is_primary = ("primary" in etag or "primary" in estatus or "primary" in etype
                          or em.get("position") == 0)
            is_business = ("business" in etype or "professional" in etype or "work" in etype
                           or "primary" in etype)
            if is_business and not is_personal_email(addr):
                (bp if is_primary else bo).append(addr)
            elif "personal" in etype or is_personal_email(addr):
                (pp if is_primary else po).append(addr)
        for bucket in (bp, bo, pp, po):
            if bucket:
                return bucket[0], True

    for em in (org_email, contact_email):
        if em and not is_personal_email(em):
            return em, False

    if flat_personal:
        name_based, consumer = [], []
        for em in flat_personal:
            (consumer if is_personal_email(em) else name_based).append(em)
        if name_based:
            return name_based[0], True
        if consumer:
            return consumer[0], True

    for em in (org_email, contact_email):
        if em:
            return em, False
    return "", False


# Phone type → quality score: personal/direct numbers win over the company switchboard.
_PHONE_TYPE_SCORES = {
    "mobile": 50, "direct": 40, "work_direct": 40, "direct_dial": 40, "personal": 35,
    "home": 30, "other": 15, "work": 15, "work_hq": 5, "company_hq": 5, "corporate": 5,
    "headquarters": 5, "main": 5,
}


def _pick_best_phone_from_apollo(person: dict, company_phone: str = ""):
    """Pick the most personal/direct phone (mobile > direct > personal > home > … > HQ),
    excluding the generic company switchboard. Handles both the standard (type) and async
    webhook (type_cd) shapes, plus singular fallback fields. Ported from LEAD FORGE V5.py.
    Returns (phone, quality_score)."""
    co_digits = re.sub(r"\D", "", company_phone) if company_phone else ""
    phones = person.get("phone_numbers") or []
    if not phones:
        for field in ("phone_number", "direct_phone_number", "sanitized_phone", "phone"):
            singular = safe_str(person.get(field))
            if singular:
                if co_digits and re.sub(r"\D", "", singular) == co_digits:
                    continue
                return singular, 25
        return "", 0
    scored = []
    for pn in phones:
        if not isinstance(pn, dict):
            continue
        number = safe_str(pn.get("sanitized_number") or pn.get("number") or pn.get("raw_number"))
        if not number:
            continue
        ptype = (pn.get("type") or pn.get("type_cd") or "").lower().strip()
        pstatus = (pn.get("status") or pn.get("status_cd") or "").lower()
        plabel = (pn.get("label") or pn.get("tag") or "").lower()
        is_primary = pn.get("is_primary") or pn.get("position") == 0
        is_default = ("default" in pstatus or "default" in plabel or "default" in ptype)
        score = _PHONE_TYPE_SCORES.get(ptype, 20)
        if is_primary:
            score += 3
        if is_default:
            score += 20
        if co_digits and re.sub(r"\D", "", number) == co_digits:
            score = min(score, 5)
        scored.append((score, number))
    if not scored:
        return "", 0
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored[0][0]


def _best_email(person: dict, first_name: str = "", last_name: str = "") -> Optional[str]:
    return _pick_best_email_from_apollo(person, first_name, last_name)[0] or None


def _best_phone(person: dict, company_phone: str = "") -> Optional[str]:
    return _pick_best_phone_from_apollo(person, company_phone)[0] or None


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


# ── Category taxonomy: 12 consolidated categories in 4 UI groups ─────────────
# Each: label, dept (coarse enum), group (UI), kw (q_keywords), titles[] (Apollo
# person_titles for discovery), match[] (classification tokens). ORDER = specific →
# general so classify_category() returns the most specific match first (e.g. a
# "Digital Marketing Manager" matches Digital Marketing before the catch-all Marketing
# & Branding). Each category's titles[] is the UNION of its merged sub-roles, so
# discovery still searches every specific title — only the labelling is consolidated.
CATEGORIES = [
    {"label": "SEO", "dept": "seo", "group": "Search & Performance", "kw": "SEO",
     "titles": ["SEO Manager", "SEO Specialist", "SEO Executive", "SEO Analyst", "Head of SEO",
                "Technical SEO Specialist", "Technical SEO Manager", "SEO Developer",
                "Local SEO Specialist", "Link Building Specialist", "Digital PR Manager",
                "Outreach Specialist"],
     "match": ["technical seo", "local seo", "link building", "digital pr", "outreach specialist",
               "seo", "search engine optim", "organic search", "search marketing"]},
    {"label": "Paid Media & PPC", "dept": "digital_marketing", "group": "Search & Performance",
     "kw": "PPC", "titles": ["PPC Manager", "Paid Media Manager", "Paid Search Manager",
                             "Google Ads Specialist", "Google Ads Manager", "AdWords Specialist",
                             "Performance Marketing Manager", "Growth Marketing Manager"],
     "match": ["google ads", "adwords", "ppc", "paid media", "paid search", "biddable",
               "performance marketing", "growth marketing", "sem manager", "media buyer"]},
    {"label": "Social Media Marketing", "dept": "digital_marketing", "group": "Search & Performance",
     "kw": "social media", "titles": ["Social Media Manager", "Social Media Marketing Specialist",
                                       "Paid Social Manager", "Community Manager"],
     "match": ["social media", "paid social", "community manager", "influencer marketing"]},
    {"label": "Digital Marketing", "dept": "digital_marketing", "group": "Marketing & Content",
     "kw": "digital marketing", "titles": ["Digital Marketing Manager", "Digital Marketing Specialist",
                                           "Digital Marketing Executive", "Email Marketing Manager",
                                           "Marketing Automation Specialist", "CRO Specialist",
                                           "Ecommerce Manager", "Shopify Developer"],
     "match": ["digital marketing", "email marketing", "crm marketing", "lifecycle marketing",
               "marketing automation", "marketo", "hubspot", "pardot", "conversion rate",
               "cro specialist", "cro manager", "ecommerce", "e-commerce", "shopify"]},
    {"label": "Data & Analytics", "dept": "digital_marketing", "group": "Marketing & Content",
     "kw": "marketing analytics", "titles": ["Marketing Analyst", "Data Analyst", "Analytics Manager",
                                             "Reporting Analyst", "Web Analytics Manager"],
     "match": ["marketing analytics", "web analytics", "data analyst", "reporting analyst",
               "data studio", "looker", "ga4", "google analytics"]},
    {"label": "Content & Copywriting", "dept": "marketing", "group": "Marketing & Content",
     "kw": "content writer", "titles": ["Content Writer", "Content Marketing Manager",
                                        "Content Specialist", "Content Designer", "Content Strategist",
                                        "Copywriter", "Senior Copywriter"],
     "match": ["content writer", "content marketing", "content specialist", "content designer",
               "content strategist", "copywriter", "copywriting"]},
    {"label": "Marketing & Branding", "dept": "marketing", "group": "Marketing & Content",
     "kw": "marketing", "titles": ["Marketing Manager", "Marketing Director", "Head of Marketing",
                                   "CMO", "Brand Manager"],
     "match": ["marketing", "brand manager", "branding", "cmo", "communications", "marcom"]},
    {"label": "Web Development", "dept": "other", "group": "Creative & Web", "kw": "web developer",
     "titles": ["Web Developer", "Frontend Developer", "Full Stack Developer", "WordPress Developer",
                "WordPress Designer"],
     "match": ["wordpress", "web developer", "frontend", "front-end", "full stack", "web development"]},
    {"label": "Design (UI/UX & Graphic)", "dept": "other", "group": "Creative & Web", "kw": "UX designer",
     "titles": ["UI Designer", "UX Designer", "UI/UX Designer", "Product Designer", "Graphic Designer",
                "Visual Designer"],
     "match": ["ui/ux", "ux design", "ui design", "product designer", "user experience",
               "graphic design", "visual designer"]},
    {"label": "Video & Creative Production", "dept": "other", "group": "Creative & Web",
     "kw": "video editor", "titles": ["Video Editor", "Video Producer", "Motion Designer",
                                      "Videographer"],
     "match": ["video editor", "video produc", "motion designer", "videographer", "animator"]},
    {"label": "Sales & Account Management", "dept": "sales", "group": "Client & People",
     "kw": "business development", "titles": ["Business Development Manager", "Sales Manager",
                                             "Account Executive", "Sales Director", "Account Manager",
                                             "Client Services Manager", "Account Director",
                                             "Project Manager", "Program Manager"],
     "match": ["business development", "sales manager", "account executive", "sales director",
               "bdr", "sdr", "sales executive", "account manager", "account director",
               "client services", "client success", "project manager", "program manager",
               "project management", "scrum master"]},
    {"label": "HR & Recruiting", "dept": "other", "group": "Client & People", "kw": "talent acquisition",
     "titles": ["Talent Acquisition Specialist", "Recruiter", "Recruitment Manager",
                "Talent Acquisition Manager", "HR Manager", "Human Resources Manager", "HR Business Partner"],
     "match": ["talent acquisition", "recruiter", "recruitment", "human resources", "hr manager",
               "hr business partner", "people operations"]},
]
CATEGORY_DEPT = {c["label"]: c["dept"] for c in CATEGORIES}
CATEGORY_LABELS = [c["label"] for c in CATEGORIES]
CATEGORY_SET = set(CATEGORY_LABELS)

# Map every legacy (28-taxonomy) label → its consolidated 12-taxonomy label. Used to migrate
# existing candidate/company rows in one bulk UPDATE and to normalise any stored old label.
CATEGORY_MERGE_MAP = {
    "Technical SEO": "SEO", "Local SEO": "SEO", "Link Building and Digital PR": "SEO", "SEO": "SEO",
    "Google Ads": "Paid Media & PPC", "Paid Media / PPC": "Paid Media & PPC",
    "Performance Marketing": "Paid Media & PPC",
    "Social Media Marketing": "Social Media Marketing",
    "Email Marketing": "Digital Marketing", "Marketing Automation": "Digital Marketing",
    "Digital Marketing": "Digital Marketing", "Conversion Rate Optimization": "Digital Marketing",
    "E-commerce": "Digital Marketing",
    "Data Analytics and Reporting": "Data & Analytics",
    "Content Writer": "Content & Copywriting", "Content Designer": "Content & Copywriting",
    "Copywriting": "Content & Copywriting",
    "Marketing": "Marketing & Branding",
    "WordPress Development": "Web Development", "Web Development": "Web Development",
    "UI/UX Design": "Design (UI/UX & Graphic)", "Graphic Design": "Design (UI/UX & Graphic)",
    "Video Editing and Production": "Video & Creative Production",
    "Account Management": "Sales & Account Management",
    "Sales and Business Development": "Sales & Account Management",
    "Project Management": "Sales & Account Management",
    "Talent Acquisition": "HR & Recruiting", "HR": "HR & Recruiting",
}


def canonical_category(cat: Optional[str]) -> Optional[str]:
    """Normalise any category string to one of the 12 (maps legacy labels; passes valid ones)."""
    if not cat:
        return None
    if cat in CATEGORY_SET:
        return cat
    return CATEGORY_MERGE_MAP.get(cat)
# UI groups (preserve definition order)
GROUPS: Dict[str, List[str]] = {}
for _c in CATEGORIES:
    GROUPS.setdefault(_c["group"], []).append(_c["label"])
GROUP_ORDER = list(GROUPS.keys())
CATEGORY_GROUP = {label: g for g, labels in GROUPS.items() for label in labels}
# human "industry" label per group — used as a FREE fallback when Apollo has no industry
GROUP_INDUSTRY = {
    "Search & Performance": "Search & Performance Marketing",
    "Marketing & Content": "Digital Marketing & Content",
    "Creative & Web": "Creative & Web Development",
    "Client & People": "Agency Client Services & HR",
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
    # education / schools (the user explicitly does NOT want schools, e.g. Delhi Public School)
    "school district", "public school", "high school", "senior secondary", "sr. sec",
    "vidyalaya", "kendriya", "navodaya", "montessori", "kindergarten", "playschool",
    "play school", "pre school", "preschool", "grammar school", "convent", "gurukul",
    "edutech", "ed-tech", "coaching centre", "coaching center", "tuition",
    # healthcare
    "hospital", "clinic", "healthcare", "pharmaceutic", "pharma ", "medical center",
    "diagnostics",
    # textiles / garments / apparel manufacturing (explicitly unwanted)
    "textile", "garment", "apparel", "spinning mill", "knitwear", "readymade",
    "hosiery", "yarn", "weaving",
    # well-known mega-corporations / consumer brands (not agency talent pools)
    "amazon", "tech mahindra", "infosys", "wipro", "tata consultancy", "tcs ",
    "hcl tech", "hcltech", "cognizant", "capgemini", "accenture", "genpact",
    "deloitte", "ernst & young", "kpmg", "pricewaterhouse", "pwc ", "ibm ",
    "reliance", "adani", "flipkart", "myntra", "walmart", "jpmorgan", "wells fargo",
    "concentrix", "teleperformance", "foxconn", "swiggy", "zomato", "paytm", "byju",
    "unacademy", "vedantu", "jio", "airtel", "vodafone", "godrej", "hindustan unilever",
    "nestle", "britannia", "dabur", "patanjali", "asian paints", "maruti", "hyundai",
    "samsung", "lg electronics", "panasonic", "itc limited", "bajaj auto",
    "nocree",
]
_SOFT_BLOCK = [
    "airlines", "airways", "aviation", "railways", "petroleum", "oil & gas",
    "oil and gas", "power plant", "electricity board", "steel", "cement", "mining",
    "automobile", "automotive", "dealership", "manufacturing", "manufacturing plant",
    "industries", "industrial", "factory", "freight", "logistics & supply",
    "real estate", "construction", "hospitality", "restaurant", "hotel",
    # consumer goods / retail (explicitly unwanted unless clearly an agency)
    "fmcg", "consumer goods", "supermarket", "hypermarket", "grocery", "retail chain",
    "jeweller", "jewellery", "jewelry", "fertilizer", "agro", "dairy", "sugar mill",
    "paper mill", "chemicals", "plastics", "packaging",
]
_ALLOW_TOKENS = ["seo", "digital marketing", "digital agency", "marketing", "advertis",
                 "agency", "media", "creative", "design", "software", "web develop",
                 "growth", "performance marketing", "analytics", "studio", "interactive",
                 "e-commerce", "ecommerce", "tech labs", "martech", "branding"]


def is_hard_blocked(name: str) -> bool:
    """True only for names that are NEVER relevant (no allow-override) — used for the immediate,
    conservative bulk cleanup so the obvious offenders (schools, textiles, mega-corps) are dropped
    while subtler cases are judged with context by the OpenAI-aware crawler."""
    n = (name or "").lower().strip()
    return bool(n) and any(b in n for b in _HARD_BLOCK)


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


def openai_classify_company(name: str, description: str = "", industry: str = "",
                            dominant_cats: Optional[List[str]] = None) -> dict:
    """Use OpenAI to (a) judge whether a company is RELEVANT to an SEO/digital-marketing talent
    search (we source from agencies, marketing/creative/web/software firms — NOT schools, textiles,
    garments, banks, hospitals, manufacturers, retail/FMCG, mega-corps) and (b) place it in ONE of
    the 12 categories. Grounded in the provided facts; never invents. Falls back to the keyword
    relevance filter + dominant candidate category when OpenAI is absent.
    Returns {relevant: bool, category: <one of CATEGORY_LABELS or None>, source}."""
    dominant_cats = [canonical_category(c) for c in (dominant_cats or [])]
    dominant_cats = [c for c in dominant_cats if c]
    kw_relevant = is_relevant_company(name)
    kw_cat = (canonical_category(classify_category(name, " ".join(filter(None, [description, industry]))))
              or (dominant_cats[0] if dominant_cats else None))
    if not openai_available():
        return {"relevant": kw_relevant, "category": kw_cat, "source": "keyword"}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        sysmsg = (
            "You categorise companies for a recruitment agency that sources SEO / digital-marketing / "
            "creative / web / software talent. Decide if the company is a RELEVANT employer to source "
            "such talent from. RELEVANT = digital/marketing/advertising/SEO/creative/design/web/"
            "software/media/PR/ecommerce-services agencies and tech companies. NOT RELEVANT = schools/"
            "colleges/coaching, textiles/garments/apparel, manufacturing/industrial, banks/finance/"
            "insurance, hospitals/pharma, retail/FMCG/consumer brands, real estate, hospitality, "
            "logistics, government, and huge non-agency corporations. Then assign EXACTLY ONE category "
            "from this fixed list (use the closest fit): " + "; ".join(CATEGORY_LABELS) + ". "
            "Use ONLY the provided facts; do not invent. Return STRICT JSON: "
            '{"relevant":true|false,"category":"<one label from the list, or null if not relevant>",'
            '"reason":"<=12 words"}.')
        facts = {"name": name, "description": (description or "")[:600], "industry": industry,
                 "employee_disciplines": dominant_cats}
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": json.dumps(facts, default=str)}], timeout=30)
        out = json.loads(resp.choices[0].message.content)
        rel = bool(out.get("relevant"))
        cat = canonical_category(out.get("category")) if rel else None
        if rel and not cat:
            cat = kw_cat  # model said relevant but gave an off-list category → keyword fallback
        # Hard safety net: never let OpenAI keep a clearly-blocked employer.
        if is_hard_blocked(name):
            rel = False; cat = None
        return {"relevant": rel, "category": cat, "source": "openai"}
    except Exception as e:
        log.warning("openai_classify_company failed (%s) — keyword fallback", e)
        return {"relevant": kw_relevant, "category": kw_cat, "source": "keyword"}


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


# Plain-language derivations for the "i" explainer next to every score. Each entry:
# {summary, factors[]} — kept truthful to the actual scoring functions below.
SCORE_EXPLANATIONS = {
    "overall": {
        "summary": "A weighted blend of the five sub-scores — role fit leads, job-change "
                   "readiness is next, then technical depth, company quality and data freshness.",
        "factors": ["Role fit 30%", "Job-change intent 25%", "Technical 15%",
                    "Company quality 15%", "Freshness 15%"]},
    "role_fit": {
        "summary": "How closely the person's role matches the target functions (sales, "
                   "marketing, SEO, digital marketing). Exact-title and recognised-category "
                   "roles score highest; unrelated 'other' roles are discounted.",
        "factors": ["Exact target title = 100", "Strong keyword/category match = 70–85",
                    "Recognised taxonomy category ≥ 65", "Unrelated 'other' role ×0.7"]},
    "intent": {
        "summary": "Likelihood the person is open to a move. With real employment history it "
                   "is evidence-based (tenure + job frequency); otherwise it is inferred from "
                   "title, seniority and industry signals.",
        "factors": ["History regime: months in role, jobs in last 5y, average tenure",
                    "Heuristic regime: seniority, contractor/agency signals, open-to-work cues"]},
    "technical": {
        "summary": "Hands-on/technical depth from job-title keywords and engineering/marketing "
                   "functions, anchored by seniority.",
        "factors": ["Seniority baseline", "Specialist tool/skill keywords (+6 each, capped)",
                    "Generic keywords (+3 each)", "Engineering/marketing function bonus"]},
    "company_quality": {
        "summary": "Quality of the person's current employer — a blend of company size, "
                   "revenue and longevity, plus bonuses for a live website and a "
                   "marketing-adjacent/tech industry.",
        "factors": ["Size 40%", "Revenue 35%", "Company age 25%",
                    "Live website (+8)", "Relevant industry (+5)"]},
    "freshness": {
        "summary": "How recent the underlying data is. Verified within 7 days scores full; "
                   "older data decays toward a floor.",
        "factors": ["≤7 days = 100", "≥90 days = 10", "Linear decay in between"]},
}


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
    # A live, resolvable website is a real quality signal for a lead (reachable, legitimate
    # business) — weighted meaningfully so company grade reflects it.
    website_pts = 8 if (org.get("website_url") or org.get("root_domain")) else 0
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


class LinkedInOutreachProvider:
    """RESERVED — automated LinkedIn direct messaging. INERT until both LINKEDIN_DM_ENABLED=1 and
    LINKEDIN_DM_API_KEY are set. Mirrors the LinkedInIntentProvider reservation pattern: the
    sequence generation and persistence already exist, so switching this on later needs no schema
    change and no rework — only an implementation of send(). Delivery is manual copy-paste today."""
    name = "linkedin_dm"

    def enabled(self) -> bool:
        return _env("LINKEDIN_DM_ENABLED", "0") == "1" and bool(_env("LINKEDIN_DM_API_KEY"))

    def send(self, candidate: dict, messages: list) -> dict:  # pragma: no cover - future
        # TODO(LinkedIn DM): call the LinkedIn messaging API/automation here, sending ONLY the
        # recruiter-reviewed stored messages, respecting per-day send limits and connection state.
        raise NotImplementedError("Automated LinkedIn DM send not implemented")


_OUTREACH_PROVIDER = LinkedInOutreachProvider()


def outreach_status() -> dict:
    """Whether automated LinkedIn DM sending is available (reserved seam; manual for now)."""
    return {"enabled": _OUTREACH_PROVIDER.enabled(), "provider": _OUTREACH_PROVIDER.name,
            "mode": "automated" if _OUTREACH_PROVIDER.enabled() else "manual_copy_paste"}


def outreach_send(candidate: dict, messages: list) -> dict:
    """Inert dispatcher for automated LinkedIn DMs. Returns a disabled result unless the reserved
    provider is switched on — delivery is manual copy-paste from the recruiter's own LinkedIn."""
    if not _OUTREACH_PROVIDER.enabled():
        return {"ok": False, "disabled": True, "reason": "manual_only",
                "message": "Automated LinkedIn DMs are reserved — copy each message and send it "
                           "from your own LinkedIn."}
    try:
        return _OUTREACH_PROVIDER.send(candidate, messages)  # pragma: no cover - future
    except NotImplementedError:
        return {"ok": False, "disabled": True, "reason": "stub",
                "message": "LinkedIn DM provider is enabled but send() is not implemented yet."}


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


def _candidate_facts(cand: dict, company: Optional[dict]) -> dict:
    """Compact, strictly job-relevant facts for the AI brief — NEVER protected attributes."""
    ai = cand.get("ai_meta_json") or {}
    sj = cand.get("scores_json") or {}
    csj = cand.get("coresignal_json") or {}
    cs = csj.get("assessment") or {}
    csraw = csj.get("raw") or {}
    sig = sj.get("intent_signals") or {}
    return {
        "name": cand.get("full_name"), "title": cand.get("title"),
        "headline": cand.get("headline"), "current_company": cand.get("company_name"),
        "role_family": ai.get("role_family"),
        "seniority": cand.get("seniority") or ai.get("seniority_level"),
        "technical_level": ai.get("technical_level"),
        "department": cand.get("department") or ai.get("department"),
        "category": cand.get("category"), "intent_regime": sj.get("intent_regime"),
        "intent_score": cand.get("job_change_intent_score"),
        "open_to_shift": bool(cand.get("open_to_shift")),
        "months_in_role": sig.get("tenure_months"), "jobs_last_5y": sig.get("jobs_last_5y"),
        "avg_tenure_months": sig.get("avg_tenure_months"),
        "company_industry": (company or {}).get("industry") or (company or {}).get("industry_derived"),
        "company_size": (company or {}).get("size_band"),
        "company_quality": cand.get("company_quality_score"),
        "overall_score": cand.get("overall_candidate_score"),
        "linkedin_summary": cs.get("professional_summary") or csraw.get("summary"),
        "expertise": (cs.get("expertise") or csraw.get("skills") or [])[:10],
        "ai_why": ai.get("why"),
    }


def _deterministic_paragraph(f: dict) -> str:
    """Always-available candidate narrative composed from facts (no OpenAI needed)."""
    name = f.get("name") or "This candidate"
    sen = (f.get("seniority") or "").replace("_", " ")
    sen = "" if sen in ("", "unknown") else sen + " "
    role = f.get("title") or f.get("role_family") or "professional"
    lead = f"{name} is a {sen}{role}"
    if f.get("current_company"):
        lead += f" at {f['current_company']}"
    if f.get("company_industry"):
        lead += f" ({f['company_industry']})"
    bits = [lead.strip() + "."]
    if f.get("linkedin_summary"):
        bits.append(str(f["linkedin_summary"]).strip().rstrip(".") + ".")
    if f.get("intent_regime") == "history" and f.get("months_in_role") is not None:
        s = f"Evidence-based job-change read: ~{f['months_in_role']} months in the current role"
        if f.get("jobs_last_5y"):
            s += f", {f['jobs_last_5y']} roles in the last 5 years"
        bits.append(s + f" (intent {f.get('intent_score')}/100).")
    else:
        bits.append(f"Job-change intent {f.get('intent_score')}/100"
                    + (" — flagged open to shift." if f.get("open_to_shift")
                       else " (inferred from role and seniority)."))
    if f.get("ai_why"):
        bits.append(str(f["ai_why"]).strip().rstrip(".") + ".")
    return " ".join(b for b in bits if b)[:800]


def generate_candidate_paragraph(cand: dict, company: Optional[dict] = None) -> dict:
    """Concise 2-4 sentence recruiter brief for a candidate. Uses gpt-4o-mini when available —
    strictly grounded in the provided facts and FORBIDDEN from inferring or using any protected
    characteristic — and always falls back to a deterministic paragraph. Returns {paragraph, source}."""
    f = _candidate_facts(cand, company)
    if not openai_available():
        return {"paragraph": _deterministic_paragraph(f), "source": "derived"}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        sysmsg = ("You are an expert technical recruiter writing a concise candidate brief for "
                  "another recruiter. Use ONLY the provided facts — NEVER invent experience, skills, "
                  "employers or numbers. Write 2-4 plain sentences: who they are (role & seniority), "
                  "what they likely do well, the job-change read (cite the strongest signal), and one "
                  "reason to reach out now. Reason ONLY on job-relevant evidence. NEVER infer or "
                  "mention age, gender, race, ethnicity, nationality, religion, disability, "
                  "marital/family status, or any protected characteristic or proxy. Return STRICT "
                  'JSON: {"paragraph":"<the brief>"}.')
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.3, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": json.dumps(f, default=str)}], timeout=30)
        para = (json.loads(resp.choices[0].message.content).get("paragraph") or "").strip()
        if para:
            return {"paragraph": para[:900], "source": "ai"}
    except Exception as e:
        log.warning("ai paragraph failed (%s) — using deterministic", e)
    return {"paragraph": _deterministic_paragraph(f), "source": "derived"}


def _company_facts(company: dict) -> dict:
    """Grounding facts for the company summary — from the company row + the real services its
    tracked employees perform (a free proxy for 'solutions'). No invented data."""
    cid = company.get("id")
    cats = []
    if cid:
        try:
            cats = db.CandidateRepo.top_categories_for_company(cid, 4)
        except Exception:
            cats = []
    founded = company.get("founded_year")
    age_years = None
    if isinstance(founded, int) and 1900 < founded <= now_utc().year:
        age_years = now_utc().year - founded
    return {
        "name": company.get("name"),
        "website": company.get("root_domain") or company.get("website_url"),
        "industry": company.get("industry"),
        "homepage_about": company.get("description"),   # real meta/og text from their site
        "founded_year": founded if isinstance(founded, int) else None,
        "years_in_business": age_years,
        "employee_band": company.get("size_band"),
        "estimated_employees": company.get("estimated_employees"),
        "country": company.get("country") or company.get("hq_country"),
        "team_disciplines": cats,   # what their people actually do → their solutions/services
    }


def _deterministic_company_summary(f: dict) -> str:
    """Always-available company blurb from facts (no OpenAI needed)."""
    name = f.get("name") or "This company"
    bits = []
    lead = name
    if f.get("industry"):
        lead += f" operates in {f['industry']}"
    elif f.get("team_disciplines"):
        lead += f" works across {', '.join(f['team_disciplines'][:3])}"
    if f.get("country"):
        lead += f", based in {f['country']}"
    bits.append(lead.strip() + ".")
    if f.get("years_in_business"):
        bits.append(f"In business ~{f['years_in_business']} years (founded {f.get('founded_year')}).")
    elif f.get("founded_year"):
        bits.append(f"Founded {f['founded_year']}.")
    if f.get("homepage_about"):
        bits.append(str(f["homepage_about"]).strip().rstrip(".") + ".")
    if f.get("team_disciplines"):
        bits.append(f"Tracked team strengths: {', '.join(f['team_disciplines'][:4])}.")
    if f.get("employee_band"):
        bits.append(f"Team size: {f['employee_band']}.")
    return " ".join(b for b in bits if b)[:700] or f"{name}."


def generate_company_summary(company: dict) -> dict:
    """Short 'what this company does, how long it's been around, and its solutions' summary for
    the company view. Uses gpt-4o-mini when available (strictly grounded in the provided facts,
    no invention), else a deterministic blurb. Returns {summary, source}."""
    if not company or not company.get("name"):
        return {"summary": "", "source": "derived"}
    f = _company_facts(company)
    if not openai_available():
        return {"summary": _deterministic_company_summary(f), "source": "derived"}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        sysmsg = ("You are a B2B analyst writing a SHORT company snapshot for a recruiter. Use ONLY "
                  "the provided facts (company row + homepage text + the disciplines its employees "
                  "work in) — NEVER invent products, clients, revenue, history or numbers. Write 2-3 "
                  "plain sentences covering: what the company does, how long it has been operating "
                  "(only if a founded year/age is given), and the solutions/services it offers "
                  "(infer the service mix from team_disciplines + homepage_about, but do not fabricate "
                  'specific named products). Return STRICT JSON: {"summary":"<the snapshot>"}.')
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.3, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": json.dumps(f, default=str)}], timeout=30)
        summ = (json.loads(resp.choices[0].message.content).get("summary") or "").strip()
        if summ:
            return {"summary": summ[:900], "source": "ai"}
    except Exception as e:
        log.warning("ai company summary failed (%s) — using deterministic", e)
    return {"summary": _deterministic_company_summary(f), "source": "derived"}


# ═══════════════════════════════════════════════════════════════════════════
#  Recruit — shortlist top candidates per category + LinkedIn outreach sequences
#  Reuses the SAME stored six-dimension scores; re-ranks with a recruit-fit
#  composite that drops freshness and leans on role-fit / intent / technical.
# ═══════════════════════════════════════════════════════════════════════════

# Emphasis presets (freshness deliberately excluded — staleness shouldn't sink a great hire).
RECRUIT_EMPHASIS = {
    "balanced": {"role_fit": 0.40, "intent": 0.30, "technical": 0.20, "company_quality": 0.10},
    "intent":   {"intent": 0.50, "role_fit": 0.25, "technical": 0.15, "company_quality": 0.10},
    "skills":   {"technical": 0.45, "role_fit": 0.35, "intent": 0.10, "company_quality": 0.10},
}
_EMPHASIS_ALIASES = {
    "balanced": "balanced", "balance": "balanced",
    "most likely to move": "intent", "most_likely_to_move": "intent", "intent": "intent",
    "move": "intent", "likely": "intent",
    "best skills match": "skills", "best_skills_match": "skills", "skills": "skills",
    "skills match": "skills", "best skills": "skills",
}


def normalize_emphasis(s: str) -> str:
    return _EMPHASIS_ALIASES.get((s or "").strip().lower(), "balanced")


def recruit_fit_score(c: dict, emphasis: str = "balanced") -> int:
    """Composite recruit-fit from the stored scores (freshness dropped)."""
    w = RECRUIT_EMPHASIS.get(emphasis, RECRUIT_EMPHASIS["balanced"])
    v = (w.get("role_fit", 0) * (c.get("role_fit_score") or 0)
         + w.get("intent", 0) * (c.get("job_change_intent_score") or 0)
         + w.get("technical", 0) * (c.get("technical_score") or 0)
         + w.get("company_quality", 0) * (c.get("company_quality_score") or 0))
    return clamp(v)


def build_recruit_shortlist(per_category: int, emphasis: str = "balanced",
                            country: Optional[str] = None) -> dict:
    """For each of the 12 categories, pull a generous overall-ordered pool (index-friendly),
    re-rank by the recruit-fit composite in Python, and take the top N. Stateless & FREE."""
    per_category = max(1, min(50, int(per_category or 1)))
    emphasis = normalize_emphasis(emphasis)
    country = (country or "").strip() or None
    pool_size = min(300, max(40, per_category * 5))
    groups, total = [], 0
    for cat in CATEGORY_LABELS:
        try:
            pool = db.CandidateRepo.recruit_pool(cat, country, pool_size)
        except Exception as e:
            log.warning("recruit_pool failed for %s: %s", cat, e)
            pool = []
        for c in pool:
            c["recruit_fit"] = recruit_fit_score(c, emphasis)
        pool.sort(key=lambda x: (x.get("recruit_fit") or 0, x.get("overall_candidate_score") or 0),
                  reverse=True)
        top = pool[:per_category]
        groups.append({"category": cat, "count": len(top), "candidates": top})
        total += len(top)
    return {"emphasis": emphasis, "per_category": per_category, "region": country,
            "total": total, "groups": groups}


# ── LinkedIn outreach sequence (3 phased messages, grounded, deterministic fallback) ──
_PHASE_SPEC = {
    1: ("connection-request opener, MAX 300 characters, with a SPECIFIC hook tied to their real "
        "role or company"),
    2: ("the value message: why you're reaching out, the kind of role you have in mind, and what's "
        "in it for them"),
    3: ("a soft, graceful nudge with a clear, low-pressure call to action"),
}


def _first_name(name: str) -> str:
    return (name or "").strip().split(" ")[0] if name else "there"


def _recruit_facts(cand: dict, company: Optional[dict]) -> dict:
    """Grounding facts for outreach — reuses the candidate-brief facts (which already prefer the
    richer CoreSignal summary/expertise) and adds the cached AI brief."""
    f = _candidate_facts(cand, company)
    f["first_name"] = _first_name(cand.get("full_name"))
    f["ai_brief"] = cand.get("ai_paragraph")
    return f


def _truncate_chars(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    cut = s[:n]
    sp = cut.rsplit(" ", 1)[0]
    return (sp if len(sp) >= n * 0.6 else cut).rstrip(" ,;:-")


def _deterministic_sequence(f: dict) -> list:
    """Always-available 3-message sequence composed strictly from the provided facts."""
    fn = f.get("first_name") or "there"
    title = f.get("title") or f.get("role_family") or "your work"
    company = f.get("current_company")
    exp = f.get("expertise") or []
    focus = exp[0] if exp else (f.get("category") or "your field")
    role_area = f.get("category") or f.get("role_family") or "growth"
    sen = (f.get("seniority") or "").replace("_", " ").strip()

    hook = f"your work as {title}" if title else "your background"
    if company:
        hook += f" at {company}"
    m1 = _truncate_chars(
        f"Hi {fn}, {hook} stood out to me" + (f" — especially around {focus}" if focus else "")
        + ". I'm connecting with strong people in this space and would love to add you.", 300)

    m2 = (f"Thanks for connecting, {fn}. I work with teams hiring in {role_area}, and given your "
          + (f"{sen} " if sen and sen != "unknown" else "")
          + (f"experience with {focus}" if focus else "experience")
          + ", I thought there could be a genuine fit. Happy to share the specifics — scope, the "
          "team, and how the comp and flexibility compare to where you are now.")

    m3 = (f"No pressure at all, {fn} — even if the timing isn't right, I'd value staying in touch. "
          "Would you be open to a quick 15-minute call this week to compare notes?")

    return [{"phase": 1, "body": m1, "source": "derived"},
            {"phase": 2, "body": m2, "source": "derived"},
            {"phase": 3, "body": m3, "source": "derived"}]


_RECRUIT_SYS = (
    "You are an expert technical recruiter writing a 3-step LinkedIn outreach sequence to a passive "
    "candidate. Use ONLY the provided facts — NEVER invent experience, employers, skills, numbers or "
    "titles, and NEVER mention or infer age, gender, race, ethnicity, nationality, religion, "
    "disability, marital/family status or any protected characteristic. The three messages must be "
    "DISTINCT and phased so the person feels genuinely and specifically approached, not spammed, and "
    "must name real details from their profile. "
    "m1 = " + _PHASE_SPEC[1] + ". m2 = " + _PHASE_SPEC[2] + ". m3 = " + _PHASE_SPEC[3] + ". "
    "Warm, human, concise, and specific. Keep m1 at or under 300 characters. "
    'Return STRICT JSON {"m1":"...","m2":"...","m3":"..."}.')


def generate_recruit_sequence(cand: dict, company: Optional[dict] = None) -> dict:
    """Personalised 3-message LinkedIn sequence, grounded strictly in the candidate's stored facts
    (preferring CoreSignal LinkedIn data when present). gpt-4o-mini with a deterministic fallback.
    Returns {messages:[{phase,body,source}], source}."""
    f = _recruit_facts(cand, company)
    if not openai_available():
        msgs = _deterministic_sequence(f)
        return {"messages": msgs, "source": "derived"}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.5, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _RECRUIT_SYS},
                      {"role": "user", "content": json.dumps(f, default=str)}], timeout=35)
        out = json.loads(resp.choices[0].message.content)
        bodies = [out.get("m1"), out.get("m2"), out.get("m3")]
        if all((b or "").strip() for b in bodies):
            msgs = [{"phase": 1, "body": _truncate_chars(bodies[0], 300), "source": "ai"},
                    {"phase": 2, "body": (bodies[1] or "").strip()[:1500], "source": "ai"},
                    {"phase": 3, "body": (bodies[2] or "").strip()[:1500], "source": "ai"}]
            return {"messages": msgs, "source": "ai"}
    except Exception as e:
        log.warning("recruit sequence failed (%s) — using deterministic", e)
    return {"messages": _deterministic_sequence(f), "source": "derived"}


def regenerate_recruit_message(cand: dict, company: Optional[dict], phase: int) -> dict:
    """Regenerate ONE message (phase 1/2/3), grounded, with a deterministic fallback.
    Returns {phase, body, source}."""
    phase = int(phase)
    if phase not in (1, 2, 3):
        phase = 1
    f = _recruit_facts(cand, company)
    det = {m["phase"]: m for m in _deterministic_sequence(f)}[phase]
    if not openai_available():
        return {"phase": phase, "body": det["body"], "source": "derived"}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        sysmsg = (
            "You are an expert technical recruiter. Write ONE LinkedIn outreach message — " + _PHASE_SPEC[phase]
            + ". Use ONLY the provided facts; NEVER invent experience, employers, skills, numbers or "
            "titles; NEVER mention or infer any protected characteristic. Name real details from the "
            "profile. Warm, human, specific."
            + (" Keep it at or under 300 characters." if phase == 1 else "")
            + ' Return STRICT JSON {"message":"..."}.')
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.7, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": json.dumps(f, default=str)}], timeout=30)
        body = (json.loads(resp.choices[0].message.content).get("message") or "").strip()
        if body:
            if phase == 1:
                body = _truncate_chars(body, 300)
            return {"phase": phase, "body": body[:1500], "source": "ai"}
    except Exception as e:
        log.warning("recruit message regen failed (%s) — using deterministic", e)
    return {"phase": phase, "body": det["body"], "source": "derived"}


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
        # FREE first: many company rows have NULL root_domain while their own candidates carry
        # a company_domain (from discovery) — recover it before spending a Clearbit lookup. This
        # is the main reason the website box stayed "pending" for companies that clearly have one.
        d = db.CompanyRepo.domain_from_candidates(r["id"]) or resolve_company_domain(r.get("name") or "")
        d = normalize_domain(d) if d else None
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
            # Converge both crawlers on ONE upgrade path: re-queue each rostered company so the
            # quality re-process pipeline (categorise, website, summary, rescore + LinkedIn
            # contacts) runs over its people too — no record left on an older process.
            try:
                db.CompanyRepo.queue_reprocess(co["id"])
            except Exception:
                pass
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


def cleanup_irrelevant_companies(limit: int = 100000) -> int:
    """Delete existing companies (and their candidates) that fail the relevance filter
    (banks, govt, healthcare, etc.). One-time/periodic housekeeping."""
    rows = db.CompanyRepo.all_id_name(limit)
    drop = [r["id"] for r in rows if not is_relevant_company(r.get("name") or "")]
    if drop:
        db.CompanyRepo.delete_with_candidates(drop)
    return len(drop)


def cleanup_hard_blocked_companies(limit: int = 100000) -> int:
    """Delete companies whose NAME is clearly never-relevant (schools, textiles/garments, banks,
    mega-corps, …) — the conservative immediate sweep (no allow-override, no OpenAI). The nuanced
    cases are judged with context by the re-process crawler. Returns count removed."""
    rows = db.CompanyRepo.all_id_name(limit)
    drop = [r["id"] for r in rows if is_hard_blocked(r.get("name") or "")]
    if drop:
        db.CompanyRepo.delete_with_candidates(drop)
    return len(drop)


def migrate_categories_to_12() -> dict:
    """One-shot: consolidate every stored candidate + company category to the 12-taxonomy."""
    nc = db.CandidateRepo.remap_categories(CATEGORY_MERGE_MAP)
    nk = db.CompanyRepo.remap_categories(CATEGORY_MERGE_MAP)
    return {"candidates": nc, "companies": nk}


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

    # Phone is async — if we requested it and it didn't come back synchronously, persist the
    # request_id so the reconciler can poll Apollo's webhook_result endpoint and fill it in.
    if reveal_phone and not phone and res.get("_phone_request_id"):
        try:
            db.CandidateRepo.set_phone_request(candidate_id, res["_phone_request_id"])
        except Exception as e:
            log.warning("set_phone_request failed (%s): %s", candidate_id, e)

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
def _cs_linkedin_search_body(slug: str, url: str = "") -> dict:
    """Precise ES DSL to find ONE employee by their LinkedIn shorthand/URL (unique key) —
    used when collect-by-slug 404s because CoreSignal's canonical shorthand != the LinkedIn
    vanity slug."""
    should = [
        {"match": {"professional_network_canonical_shorthand_name": slug}},
        {"query_string": {"query": "*" + slug + "*", "default_field": "professional_network_url"}},
    ]
    return {"query": {"bool": {"should": should, "minimum_should_match": 1}}, "sort": ["_score"]}


def coresignal_resolve(candidate: dict) -> dict:
    """Resolve a known candidate → ONE CoreSignal employee record. Branch A (resolve by the
    LinkedIn URL: collect by slug → by URL → precise URL/shorthand search) → Branch B
    (name+company preview search → disambiguate → collect). When a LinkedIn URL is present we
    never fall to a fuzzy name-based manual pick. Returns
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

    # BRANCH A — resolve by the LinkedIn URL precisely; never fall to a fuzzy name pick.
    li_url = (candidate.get("linkedin_url") or "").strip()
    slug = _linkedin_slug(li_url)
    had_url = bool(slug)
    if slug:
        # 1) deterministic collect by the bare slug, then by the full URL (both accepted)
        for key in (slug, li_url):
            if not key:
                continue
            r = cs.collect(key)
            if r["ok"] and isinstance(r["data"], dict) and r["data"].get("id"):
                return {"ok": True, "record": r["data"], "needs_manual_pick": False, "candidates": [],
                        "match": {"confidence": "high", "method": "linkedin_url", "id": r["data"].get("id")},
                        "error": None, "status": 200, "credits": r["credits"]}
            f = _fatal(r)
            if f:
                return f
        # 2) collect-by-slug 404s when CoreSignal's canonical shorthand != the LinkedIn vanity
        #    slug — so SEARCH by the LinkedIn URL/shorthand (unique) and collect the match.
        pv = cs.search(_cs_linkedin_search_body(slug, li_url), preview=True)
        if pv["ok"] and isinstance(pv["data"], list) and pv["data"]:
            best_li = pv["data"][0]
            c = cs.collect(best_li.get("id"))
            if c["ok"] and isinstance(c["data"], dict):
                return {"ok": True, "record": c["data"], "needs_manual_pick": False, "candidates": [],
                        "match": {"confidence": "high", "method": "linkedin_search", "id": best_li.get("id")},
                        "error": None, "status": 200, "credits": c["credits"]}
            f = _fatal(c)
            if f:
                return f
        elif not pv["ok"]:
            f = _fatal(pv)
            if f:
                return f
        # All LinkedIn-URL paths failed → fall to name+company below, but AUTO-pick (no prompt).

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
    if ambiguous and not had_url:
        # only bother the user with a manual pick when we had NO explicit LinkedIn URL to go on
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
        # Retain any contact fields the profile exposes so the LinkedIn-first contact route can
        # (re)derive them later from the stored record, not just at enrich time.
        "professional_emails": rec.get("professional_emails"),
        "emails": rec.get("emails"),
        "primary_professional_email": rec.get("primary_professional_email"),
        "recommended_personal_email": rec.get("recommended_personal_email"),
        "phone": rec.get("phone") or rec.get("phone_number"),
        "phones": rec.get("phones") or rec.get("phone_numbers"),
        "contact_info": rec.get("contact_info"),
    }


def _cs_openai_payload(rec: dict, candidate: dict) -> dict:
    exp = _cs_exp_brief(rec.get("experience"), 12)
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
    full = _cs_exp_brief(exp, 12)
    prev = list(dict.fromkeys([e["company"] for e in full if e.get("company") and not e.get("current")]))
    months = rec.get("total_experience_duration_months") or 0
    return {"_source": "coresignal",
            "professional_summary": rec.get("summary") or rec.get("headline"),
            "expertise": (rec.get("inferred_skills") or rec.get("skills") or [])[:12],
            "current_role": {"title": rec.get("active_experience_title"),
                             "company": (full[0]["company"] if full else candidate.get("company_name"))},
            "career_history": [{"title": e["title"], "company": e["company"],
                                "start": e["start"], "end": e["end"]} for e in full],
            "previous_companies": prev[:12],
            "seniority_level": rec.get("active_experience_management_level") or candidate.get("seniority"),
            "department": rec.get("active_experience_department") or candidate.get("department"),
            "years_experience": (round(months / 12, 1) if months else None),
            "top_skills": (rec.get("inferred_skills") or rec.get("skills") or [])[:12],
            "education": [{"institution": e.get("institution_name") or e.get("title"),
                           "degree": e.get("degree") or e.get("subtitle")}
                          for e in (rec.get("education") or []) if isinstance(e, dict)][:6],
            "certifications": [{"title": c.get("title"), "issuer": c.get("issuer") or c.get("subtitle")}
                               for c in (rec.get("certifications") or []) if isinstance(c, dict)][:8],
            "languages": rec.get("languages"),
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
        sysmsg = ("You are an HR analyst producing a COMPREHENSIVE yet concise LinkedIn profile of a "
                  "known candidate from CoreSignal structured data. Use ONLY the provided fields — NEVER "
                  "invent experience, skills, education, employers, certifications, or dates. If a field "
                  "is absent, omit it or use null/[]. Be thorough: include the FULL career history and all "
                  "previous companies. Output STRICT JSON: "
                  '{"professional_summary":"3-4 sentence professional bio from the data: who they are, '
                  'where they work, their focus and standing",'
                  '"expertise":["areas of expertise / specialties, broader than raw skills"],'
                  '"current_role":{"title":str,"company":str},'
                  '"career_history":[{"title":str,"company":str,"start":str,"end":str,'
                  '"focus":"<=12 word note on the role"}],'  # EVERY role provided, newest first
                  '"previous_companies":["past employers, excluding the current one"],'
                  '"seniority_level":"intern|entry|manager|senior|director|vp|c_suite",'
                  '"department":"sales|marketing|seo|digital_marketing|engineering|other",'
                  '"years_experience":number,'
                  '"top_skills":[str],"education":[{"institution":str,"degree":str}],'
                  '"certifications":[{"title":str,"issuer":str}],"languages":[str],'
                  '"open_to_work":bool,"intent_likelihood":0-100,"confidence":"low|medium|high",'
                  '"signals":["short factual job-change signals from the data"],'
                  '"recommendation":"<=25 words for the recruiter",'
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
    # PRIMARY contact route: pull any phone/email on the LinkedIn profile (free) and persist it —
    # it takes precedence over Apollo and is never overwritten on later runs.
    try:
        apply_linkedin_contacts(candidate_id, record, row)
    except Exception as e:
        log.warning("apply_linkedin_contacts failed (%s): %s", candidate_id, e)
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
    """Stable secret for the Apollo async phone webhook. Uses APOLLO_WEBHOOK_SECRET, else
    derives a token from SECRET_KEY or APP_PASSWORD (always set) so phone reveal works without
    extra config. Empty only if literally nothing is configured."""
    t = _env("APOLLO_WEBHOOK_SECRET")
    if t:
        return t
    for src in ("SECRET_KEY", "APP_PASSWORD"):
        sk = _env(src)
        if sk:
            import hashlib
            return hashlib.sha256(("apollo-phone:" + sk).encode()).hexdigest()[:32]
    return ""


def handle_apollo_phone_webhook(data: dict) -> int:
    """Apollo posts the async phone-reveal result here (minutes after the match). Defensively
    locate the people + their phone numbers and write them onto candidates by apollo_person_id.
    Returns how many candidates were updated. Logs verbosely so delivery can be diagnosed."""
    if not isinstance(data, dict):
        log.warning("apollo phone webhook: non-dict payload")
        return 0
    people = data.get("people") or data.get("contacts") or data.get("matches") or []
    if not people and isinstance(data.get("person"), dict):
        people = [data["person"]]
    if isinstance(people, dict):
        people = [people]
    log.info("apollo phone webhook: status=%s people=%d keys=%s",
             data.get("status"), (len(people) if isinstance(people, list) else 0), list(data.keys())[:8])
    updated = 0
    for p in people:
        if not isinstance(p, dict):
            continue
        pid = p.get("id") or p.get("person_id") or p.get("apollo_id") or p.get("contact_id")
        phone = _best_phone(p)
        if pid and phone:
            try:
                if db.CandidateRepo.set_phone_by_apollo_id(str(pid), phone):
                    updated += 1
                    log.info("apollo phone webhook: set phone for apollo_id=%s", pid)
            except Exception as e:
                log.warning("phone webhook update failed: %s", e)
    if not updated:
        log.warning("apollo phone webhook: NO candidate updated (people=%s, sample=%s)",
                    len(people) if isinstance(people, list) else 0,
                    json.dumps(people[:1], default=str)[:400] if people else "[]")
    return updated


def reveal_phone_only(candidate_id: int, webhook_url: str = "") -> dict:
    """Request JUST the mobile/direct phone for a candidate (Apollo; async via webhook). Works
    regardless of enrichment_status — used to backfill phones for candidates enriched email-only.
    Returns {ok, phone, phone_pending, message}. Never reveals email (cheaper)."""
    row = db.CandidateRepo.get(candidate_id)
    if not row:
        return {"ok": False, "error": "not_found"}
    apollo = get_apollo()
    if not getattr(apollo, "api_key", ""):
        return {"ok": False, "error": "apollo_not_configured"}
    res = apollo.enrich_person(
        apollo_id=row.get("apollo_person_id") or "",
        first_name=row.get("first_name") or "", last_name=row.get("last_name") or "",
        domain=row.get("company_domain") or "", linkedin_url=row.get("linkedin_url") or "",
        reveal_email=False, reveal_phone=True, webhook_url=webhook_url)
    if res.get("_no_credits"):
        return {"ok": False, "error": "no_credits"}
    if not res.get("_ok"):
        return {"ok": False, "error": "apollo_failed", "detail": res.get("_phone_reveal_error")}
    phone = res.get("phone")  # occasionally present synchronously
    if phone:
        try:
            db.CandidateRepo.set_phone(candidate_id, phone)
        except Exception as e:
            log.warning("reveal_phone_only set failed: %s", e)
        return {"ok": True, "phone": phone, "phone_pending": False,
                "candidate": db.CandidateRepo.get(candidate_id)}
    # Apollo rejected the phone reveal (e.g. not entitled) — report it.
    if res.get("_phone_reveal_error"):
        return {"ok": False, "error": "phone_reveal_rejected",
                "detail": res.get("_phone_reveal_error")}
    # Async path: record the request_id so the reconciler can POLL Apollo for the number
    # (works even if the inbound webhook never reaches us). This is the reliable fix.
    rid = res.get("_phone_request_id")
    if rid:
        try:
            db.CandidateRepo.set_phone_request(candidate_id, rid)
        except Exception as e:
            log.warning("reveal_phone_only set_phone_request failed: %s", e)
    return {"ok": True, "phone": None, "phone_pending": bool(rid or webhook_url),
            "message": ("Mobile requested — Apollo reveals it async; it’s auto-filled within "
                        "a few minutes (no refresh needed)." if (rid or webhook_url)
                        else "Phone reveal unavailable — check Apollo entitlement.")}


def poll_one_phone(candidate_id: int) -> dict:
    """Actively poll Apollo's webhook_result for ONE candidate's pending phone and write it if it's
    ready. Called from the frontend's poll loop so delivery is driven by the user's own polling —
    it does NOT depend on the background scheduler or on Apollo's inbound webhook reaching us.
    FREE (no new credit). Returns {phone, pending, resolved, reason}."""
    row = db.CandidateRepo.get(candidate_id) or {}
    if row.get("phone"):
        return {"phone": row["phone"], "pending": False, "resolved": True, "reason": "have_phone"}
    rid = row.get("phone_request_id")
    if not row.get("phone_pending"):
        return {"phone": None, "pending": False, "resolved": True, "reason": "not_pending"}
    if not rid:
        # No async request_id captured (Apollo didn't return one) — only the inbound webhook can
        # fill this. Keep showing pending; the webhook handler writes by apollo_person_id.
        return {"phone": None, "pending": True, "resolved": False, "reason": "no_request_id"}
    apollo = get_apollo()
    if not getattr(apollo, "api_key", ""):
        return {"phone": None, "pending": True, "resolved": False, "reason": "apollo_unconfigured"}
    data = apollo.poll_webhook_result(rid)
    if not isinstance(data, dict):
        return {"phone": None, "pending": True, "resolved": False, "reason": "not_ready"}
    people = data.get("people") or data.get("contacts") or []
    if isinstance(people, dict):
        people = [people]
    got = None
    for p in people:
        if isinstance(p, dict):
            got = _best_phone(p)
            if got:
                break
    status = str(data.get("status") or "").lower()
    if got:
        try:
            db.CandidateRepo.set_phone(candidate_id, got)
        except Exception as e:
            log.warning("poll_one_phone set failed (%s): %s", candidate_id, e)
        return {"phone": got, "pending": False, "resolved": True, "reason": "filled"}
    if status in ("success", "complete", "completed", "finished") or data.get("people") is not None:
        # Apollo finished but has no mobile for this person — stop polling, report clearly.
        try:
            db.CandidateRepo.clear_phone_pending(candidate_id)
        except Exception:
            pass
        return {"phone": None, "pending": False, "resolved": True, "reason": "no_phone_available"}
    return {"phone": None, "pending": True, "resolved": False, "reason": "not_ready"}


def reconcile_pending_phones(limit: int = 25) -> int:
    """Pull async phone reveals that Apollo has finished, by POLLING
    GET /api/v1/webhook_result/{request_id} for every candidate still awaiting a number.
    This is the RELIABLE delivery path — it does not depend on Apollo's webhook reaching us,
    so phones land even when the inbound webhook is blocked/misconfigured. FREE (no new credit;
    the reveal was already paid). Returns how many phones were filled. Safe to call every tick."""
    apollo = get_apollo()
    if not getattr(apollo, "api_key", ""):
        return 0
    try:
        rows = db.CandidateRepo.pending_phone_requests(limit)
    except Exception as e:
        log.warning("pending_phone_requests failed: %s", e)
        return 0
    filled = 0
    for r in rows:
        rid = r.get("phone_request_id")
        cid = r.get("id")
        data = apollo.poll_webhook_result(rid)
        if not isinstance(data, dict):
            continue  # not ready yet — try again next tick
        people = data.get("people") or data.get("contacts") or []
        if isinstance(people, dict):
            people = [people]
        phone = None
        for p in people:
            if isinstance(p, dict):
                phone = _best_phone(p)
                if phone:
                    break
        status = str(data.get("status") or "").lower()
        if phone:
            try:
                if db.CandidateRepo.set_phone(cid, phone):
                    filled += 1
                    log.info("phone reconciler: filled phone for candidate %s (req %s)", cid, rid)
                else:
                    db.CandidateRepo.clear_phone_pending(cid)
            except Exception as e:
                log.warning("phone reconciler set failed (%s): %s", cid, e)
        elif status in ("success", "complete", "completed", "finished") or data.get("people") is not None:
            # Apollo finished but found no phone for this person — stop polling it.
            try:
                db.CandidateRepo.clear_phone_pending(cid)
            except Exception:
                pass
    # Abandon requests too old to ever resolve.
    try:
        db.CandidateRepo.expire_stale_phone_pending(_env_int("PHONE_RECONCILE_MAX_AGE_HOURS", 48))
    except Exception:
        pass
    return filled


def _bg_webhook_url() -> str:
    """Build the Apollo phone webhook URL for BACKGROUND jobs (no request context). Uses
    PUBLIC_BASE_URL (must be the public https host) + the webhook token. Empty if not configured."""
    base = (_env("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    tok = apollo_webhook_token()
    if not base or not tok:
        return ""
    if "://" not in base:
        base = "https://" + base
    return base + "/api/apollo-phone-webhook?token=" + tok


def proactive_phone_reveal_batch(limit: int = 8) -> int:
    """Proactively request mobiles in the BACKGROUND for candidates Apollo says have one, so phones
    fill in for everyone over time rather than only on a manual click. The reconciler then polls the
    result. Cost-controlled: OFF unless HR_PROACTIVE_PHONE_ENABLED=1, strictly bounded by the daily
    phone cap. Returns reveals requested. Needs PUBLIC_BASE_URL so Apollo gets a valid https webhook."""
    if _env("HR_PROACTIVE_PHONE_ENABLED", "0") != "1":
        return 0
    apollo = get_apollo()
    if not getattr(apollo, "api_key", ""):
        return 0
    webhook = _bg_webhook_url()
    if not webhook:
        log.warning("proactive phone: set PUBLIC_BASE_URL (https) to enable background reveals")
        return 0
    try:
        counts = db.RevealCounterRepo.today()
        cap = _env_int("ENRICH_MAX_REVEALS_PER_DAY_PHONE", 60)
        remaining = cap - int(counts.get("phone_reveals", 0))
    except Exception:
        remaining = _env_int("ENRICH_MAX_REVEALS_PER_DAY_PHONE", 60)
    if remaining <= 0:
        return 0
    rows = db.CandidateRepo.phone_reveal_candidates(min(limit, remaining))
    requested = 0
    for r in rows:
        try:
            res = apollo.enrich_person(
                apollo_id=r.get("apollo_person_id") or "", first_name=r.get("first_name") or "",
                last_name=r.get("last_name") or "", domain=r.get("company_domain") or "",
                linkedin_url=r.get("linkedin_url") or "", reveal_email=False, reveal_phone=True,
                webhook_url=webhook)
        except Exception as e:
            log.warning("proactive phone reveal failed (%s): %s", r.get("id"), e)
            continue
        if res.get("phone"):
            db.CandidateRepo.set_phone(r["id"], res["phone"])
        elif res.get("_phone_request_id"):
            db.CandidateRepo.set_phone_request(r["id"], res["_phone_request_id"])
        else:
            # mark pending so we don't immediately re-pick it; the inbound webhook may still deliver
            db.CandidateRepo.set_phone_request(r["id"], "")
        try:
            db.RevealCounterRepo.incr(email=0, phone=1)  # count against the daily cap (cost guard)
        except Exception:
            pass
        requested += 1
    if requested:
        log.info("proactive phone: requested %d mobile reveal(s)", requested)
    return requested


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


def rescore_candidate(candidate_id: int, target_families: Optional[List[str]] = None) -> bool:
    """Re-score & re-classify an EXISTING candidate from its stored fields using the CURRENT
    scoring logic + taxonomy. FREE (no Apollo / no reveal). Preserves a LinkedIn/CoreSignal-
    verified intent (higher quality than a recompute). Returns True if updated."""
    row = db.CandidateRepo.get(candidate_id)
    if not row:
        return False
    company = db.CompanyRepo.get(row["company_id"]) if row.get("company_id") else None
    title = row.get("title") or ""
    headline = row.get("headline") or ""
    # Prefer the company's own resolved domain (set by the website backfill) over the
    # candidate's, so a now-known website feeds the company-quality score on re-process.
    root_domain = (company or {}).get("root_domain") or row.get("company_domain")
    org = {"name": row.get("company_name"), "root_domain": root_domain,
           "industry": (company or {}).get("industry"),
           "estimated_employees": (company or {}).get("estimated_employees"),
           "annual_revenue": (company or {}).get("annual_revenue"),
           "founded_year": (company or {}).get("founded_year"),
           "website_url": (company or {}).get("website_url") or (f"https://{root_domain}" if root_domain else None)}
    base = {"title": title, "headline": headline, "seniority": row.get("seniority") or "",
            "functions": [], "departments": row.get("departments_json") or [], "_org": org,
            "employment_history": row.get("employment_history_json") or [],
            "company_domain": row.get("company_domain"), "linkedin_url": row.get("linkedin_url")}
    category = classify_category(title, headline) or canonical_category(row.get("category"))
    dept = CATEGORY_DEPT.get(category) if category else None
    if not dept:
        dept = classify_department(title, headline, row.get("departments_json") or [], [])
    # Recompute company quality whenever we have a size OR a resolved website (website is now a
    # graded signal); otherwise keep the stored value rather than regressing to a guess.
    cq = score_company_quality(org) if (org.get("estimated_employees") or org.get("root_domain")) \
        else (row.get("company_quality_score") or 50)
    technical = score_technical(base)
    role_fit = score_role_fit(base, target_families)
    intent, intent_source, intent_signals = compute_intent(base)
    if row.get("coresignal_enriched") or row.get("linkedin_enriched"):
        intent = int(row.get("job_change_intent_score") or intent)
        intent_source = row.get("intent_source") or intent_source
        intent_signals = (row.get("scores_json") or {}).get("intent_signals") or intent_signals
        # Lead-upgrade crawler applies the LinkedIn-first contact route to every enriched lead
        # (free — re-derives from already-stored CoreSignal data; never overwrites once found).
        if not row.get("linkedin_contact_checked"):
            rec = (row.get("coresignal_json") or {}).get("raw") or (row.get("coresignal_json") or {})
            try:
                apply_linkedin_contacts(candidate_id, rec, row)
            except Exception as e:
                log.warning("rescore linkedin contacts (%s): %s", candidate_id, e)
    freshness = score_freshness(row)
    ai_meta = row.get("ai_meta_json") or heuristic_classify(base)
    pack = {"role_fit": role_fit, "intent": intent, "technical": technical,
            "company_quality": cq, "freshness": freshness}
    pack = _apply_ai_nudge(pack, ai_meta)
    overall = score_overall(pack)
    scores_json = {**pack, "overall": overall, "intent_regime": intent_source,
                   "intent_signals": intent_signals, "weights": WEIGHTS}
    db.CandidateRepo.apply_rescore(
        candidate_id, category=category, department=dept, technical=pack["technical"],
        role_fit=role_fit, intent=intent, company_quality=cq, freshness=freshness,
        overall=overall, intent_source=intent_source, scores_json=scores_json)
    return True


def recompute_company_category(company_id: int) -> Optional[str]:
    """Set the authoritative per-company category (one of the 12). Uses OpenAI when available —
    grounded in the company name, homepage text, industry and the disciplines of its own employees
    — and falls back to the score-weighted dominant candidate category + homepage keywords. FREE-ish
    (one cheap gpt-4o-mini call per company on re-process)."""
    co = db.CompanyRepo.get(company_id)
    dominant = canonical_category(db.CandidateRepo.weighted_dominant_category(company_id))
    cat, source = dominant, "candidates"
    if co:
        try:
            top_cats = db.CandidateRepo.top_categories_for_company(company_id, 4)
        except Exception:
            top_cats = []
        res = openai_classify_company(co.get("name") or "", co.get("description") or "",
                                      co.get("industry") or "", top_cats)
        if res.get("category"):
            cat, source = res["category"], res.get("source") or "openai"
        elif not cat:
            # keyword homepage fallback
            web_text = " ".join(filter(None, [co.get("description"), co.get("industry"), co.get("name")]))
            cat = canonical_category(classify_category("", web_text)) if web_text else None
            source = "web"
    cat = canonical_category(cat)
    if cat:
        db.CompanyRepo.set_category(company_id, cat, source)
    return cat


def roster_reprocess_batch(limit: int = 2, logf=None) -> dict:
    """Bring EXISTING records up to current quality standards (FREE — no reveal credits): for
    each due company refresh free firmographics (domain + homepage 'About'), re-score &
    re-classify every attached candidate, then set the authoritative company category."""
    companies = db.CompanyRepo.reprocess_pending(limit)
    n_co = n_cand = n_web = n_sum = n_drop = 0
    for co in companies:
        # 0) Relevance prune. Hard-blocked names go immediately; otherwise let OpenAI judge with
        #    context (description + employee disciplines) and delete the irrelevant ones + their
        #    candidates. This removes schools/textiles/garments/retail/mega-corps from the DB.
        try:
            if is_hard_blocked(co.get("name") or ""):
                db.CompanyRepo.delete_with_candidates([co["id"]]); n_drop += 1; continue
            try:
                top0 = db.CandidateRepo.top_categories_for_company(co["id"], 4)
            except Exception:
                top0 = []
            verdict = openai_classify_company(co.get("name") or "", co.get("description") or "",
                                              co.get("industry") or "", top0)
            if not verdict.get("relevant"):
                db.CompanyRepo.delete_with_candidates([co["id"]]); n_drop += 1; continue
        except Exception as e:
            log.warning("reprocess relevance %s: %s", co.get("id"), e)
        try:
            # 1) Resolve the website — FREE from candidates first (fixes "pending"), then Clearbit.
            if not co.get("root_domain"):
                d = db.CompanyRepo.domain_from_candidates(co["id"]) or resolve_company_domain(co.get("name") or "")
                d = normalize_domain(d) if d else None
                if d:
                    db.CompanyRepo.set_domain(co["id"], d, f"https://{d}")
                    co["root_domain"] = d
            # 2) Pull the homepage 'About' text (grounds both category + AI summary).
            if co.get("root_domain") and not co.get("description"):
                info = fetch_company_web(co["root_domain"])
                if info.get("description"):
                    db.CompanyRepo.set_web(co["id"], info.get("description"), info.get("og_image"))
                    co["description"] = info.get("description")
                    n_web += 1
        except Exception as e:
            log.warning("reprocess firmographics %s: %s", co.get("id"), e)
        # 3) Re-score & re-classify every candidate (website now feeds company quality).
        for cand in db.CandidateRepo.for_company(co["id"], limit=10000):
            try:
                if rescore_candidate(cand["id"]):
                    n_cand += 1
            except Exception as e:
                log.warning("rescore %s: %s", cand.get("id"), e)
        # 4) Authoritative company category (needs candidate categories from step 3).
        try:
            recompute_company_category(co["id"])
        except Exception as e:
            log.warning("recompute_company_category %s: %s", co.get("id"), e)
        # 5) OpenAI company summary (what it does / how long / solutions) — cached on the row.
        try:
            if not co.get("ai_summary"):
                fresh = db.CompanyRepo.get(co["id"]) or co
                res = generate_company_summary(fresh)
                if res.get("summary"):
                    db.CompanyRepo.set_ai_summary(co["id"], res["summary"], res.get("source") or "")
                    n_sum += 1
        except Exception as e:
            log.warning("company summary %s: %s", co.get("id"), e)
        db.CompanyRepo.mark_reprocessed(co["id"])
        n_co += 1
    if logf and companies:
        logf(f"Re-process: {n_cand} candidates upgraded · {n_web} websites · {n_sum} summaries · "
             f"{n_drop} irrelevant removed across {n_co} companies")
    return {"companies": n_co, "candidates": n_cand, "websites": n_web, "summaries": n_sum,
            "removed": n_drop}


def roster_reprocess_enabled() -> bool:
    return db.SettingsRepo.get_bool("roster_reprocess", _env("ROSTER_REPROCESS_ENABLED", "0") == "1")


def set_roster_reprocess(on: bool) -> None:
    db.SettingsRepo.set("roster_reprocess", "1" if on else "0")
    if on:
        db.SettingsRepo.set("roster_reprocess_started_at", now_utc().isoformat())


def roster_reprocess_status() -> dict:
    s = db.SettingsRepo.get_many(["roster_reprocess", "roster_reprocess_started_at", "roster_reprocess_last_at"])
    counts = db.CompanyRepo.reprocess_counts()
    rv = s.get("roster_reprocess")
    enabled = (rv in ("1", "true", "True", "on")) if rv is not None else (_env("ROSTER_REPROCESS_ENABLED", "0") == "1")
    return {"enabled": enabled, "last_at": s.get("roster_reprocess_last_at"),
            "started_at": s.get("roster_reprocess_started_at"),
            "companies_total": counts["total"], "companies_done": counts["done"],
            "companies_remaining": max(0, counts["total"] - counts["done"])}


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

            # Reliably deliver async phone reveals by POLLING Apollo's webhook_result endpoint
            # for every candidate awaiting a number (free; works even if the inbound webhook is
            # blocked). This is the real fix for "Get mobile never arrives".
            if _env("PHONE_RECONCILE_ENABLED", "1") == "1":
                try:
                    got = reconcile_pending_phones(_env_int("PHONE_RECONCILE_PER_TICK", 25))
                    if got:
                        log.info("phone reconciler: filled %d mobile number(s)", got)
                except Exception as e:
                    log.warning("phone reconcile error: %s", e)

            # Proactively request mobiles in the background (OFF unless HR_PROACTIVE_PHONE_ENABLED=1;
            # strictly bounded by the daily phone cap) so phones fill for everyone, not just on click.
            try:
                proactive_phone_reveal_batch(_env_int("PHONE_PROACTIVE_PER_TICK", 8))
            except Exception as e:
                log.warning("proactive phone error: %s", e)

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

            # Re-process & upgrade existing leads to current quality standards (separate toggle,
            # FREE). Paced slow so it grinds the whole DB without hammering anything.
            if roster_reprocess_enabled():
                try:
                    r = roster_reprocess_batch(_env_int("ROSTER_REPROCESS_PER_TICK", 2))
                    db.SettingsRepo.set("roster_reprocess_last_at", now_utc().isoformat())
                    if r.get("candidates"):
                        log.info("roster reprocess: upgraded %d candidates / %d companies",
                                 r["candidates"], r["companies"])
                except Exception as e:
                    log.warning("roster reprocess error: %s", e)

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
