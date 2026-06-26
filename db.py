"""MySQL persistence layer for the Smart HR Talent Acquisition system.

Responsibilities
  * Pool connections via PyMySQL + DBUtils.PooledDB (ping=4 → reconnect before
    every query, the fix for Railway's ~60s idle-connection timeout).
  * Create/verify the schema (runs, companies, candidates, enrichment_log,
    reveal_counter).
  * Thin repository classes encapsulating every SQL statement.

Free-tier reality (drives the schema): Apollo's FREE People Search returns only a
THIN payload — first_name, obfuscated last name, title, org NAME, and has_email/
has_phone availability flags (no domain, seniority, location values, or
firmographics). So companies are keyed by `company_key` (domain when known, else a
name-slug); domain + firmographics fill in opportunistically when a candidate at
that company is enriched via people/match (the paid Enrich action).

Environment variables (priority order):
  1. Discrete:  MYSQLHOST, MYSQLPORT, MYSQLUSER, MYSQLPASSWORD, MYSQLDATABASE
  2. URL:       MYSQL_URL
  3. URL:       MYSQL_PUBLIC_URL  (Railway public proxy — used from laptop/CI)
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, unquote

try:
    import pymysql
    from pymysql.cursors import DictCursor
    from dbutils.pooled_db import PooledDB
    _DEPS_OK = True
    _DEPS_ERR: Optional[str] = None
except ImportError as _e:  # pragma: no cover
    _DEPS_OK = False
    _DEPS_ERR = str(_e)

log = logging.getLogger("hr.db")

# ── Canonical scoring weights (single source of truth for Python + SQL) ──────
WEIGHTS: Dict[str, float] = {
    "role_fit": 0.30, "intent": 0.25, "technical": 0.15,
    "company_quality": 0.15, "freshness": 0.15,
}


def overall_sql(role_fit="role_fit_score", intent="job_change_intent_score",
                technical="technical_score", company="company_quality_score",
                freshness="freshness_score") -> str:
    w = WEIGHTS
    return (f"ROUND({w['role_fit']}*{role_fit} + {w['intent']}*{intent} + "
            f"{w['technical']}*{technical} + {w['company_quality']}*{company} + "
            f"{w['freshness']}*{freshness})")


def freshness_sql(col="last_verified_at") -> str:
    d = f"DATEDIFF(NOW(), {col})"
    return (f"CASE WHEN {col} IS NULL THEN 50 WHEN {d} <= 7 THEN 100 "
            f"WHEN {d} >= 90 THEN 10 "
            f"ELSE GREATEST(10, LEAST(100, ROUND(100 - ({d} - 7) * (90/83)))) END")


# ── Exceptions ───────────────────────────────────────────────────────────────


class DBUnavailable(RuntimeError):
    """MySQL is configured but unreachable."""


class DBConfigError(RuntimeError):
    """MySQL env vars are missing or malformed."""


# ── Config resolution ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "DBConfig":
        host = os.environ.get("MYSQLHOST") or ""
        user = os.environ.get("MYSQLUSER") or ""
        password = os.environ.get("MYSQLPASSWORD") or ""
        database = os.environ.get("MYSQLDATABASE") or ""
        port_str = os.environ.get("MYSQLPORT") or ""
        if host and user and database:
            try:
                port = int(port_str) if port_str else 3306
            except ValueError:
                port = 3306
            return cls(host, port, user, password, database)
        for url_var in ("MYSQL_URL", "MYSQL_PUBLIC_URL", "DATABASE_URL"):
            url = os.environ.get(url_var)
            if url:
                return cls._parse_url(url)
        raise DBConfigError(
            "MySQL not configured. Set MYSQLHOST/MYSQLUSER/MYSQLPASSWORD/"
            "MYSQLDATABASE or MYSQL_URL (or MYSQL_PUBLIC_URL for local/proxy).")

    @staticmethod
    def _parse_url(url: str) -> "DBConfig":
        p = urlparse(url)
        if p.scheme not in ("mysql", "mysql+pymysql"):
            raise DBConfigError(f"Unsupported URL scheme: {p.scheme}")
        host = p.hostname or ""
        user = unquote(p.username or "")
        password = unquote(p.password or "")
        port = p.port or 3306
        database = (p.path or "").lstrip("/")
        if not (host and user and database):
            raise DBConfigError("Malformed MySQL URL: missing host/user/database")
        return DBConfig(host, port, user, password, database)


# ── Pool lifecycle ───────────────────────────────────────────────────────────

_pool: Optional[Any] = None
_pool_lock = threading.Lock()
_pool_error: Optional[str] = None


def init_pool(size: int = 12) -> None:
    """Create the shared PyMySQL connection pool. Idempotent."""
    global _pool, _pool_error
    if not _DEPS_OK:
        _pool_error = f"PyMySQL/DBUtils not installed: {_DEPS_ERR}"
        raise DBUnavailable(_pool_error)
    with _pool_lock:
        if _pool is not None:
            return
        cfg = DBConfig.from_env()
        try:
            _pool = PooledDB(
                creator=pymysql, maxconnections=size, mincached=2, maxcached=4,
                blocking=True, ping=4, host=cfg.host, port=cfg.port, user=cfg.user,
                password=cfg.password, database=cfg.database, charset="utf8mb4",
                # autocommit=True is REQUIRED: with autocommit=False + MySQL's default
                # REPEATABLE READ, a pooled connection that ran a SELECT keeps a stale
                # snapshot and won't see commits from other pooled connections — causing
                # stale reads right after a write. autocommit gives every statement its
                # own transaction (no multi-statement atomicity is needed here).
                autocommit=True, connect_timeout=12, read_timeout=40,
                write_timeout=40, cursorclass=DictCursor)
            with _pool.connection() as probe:
                with probe.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            _pool_error = None
            log.info("MySQL pool ready (%s:%s/%s, size=%d)",
                     cfg.host, cfg.port, cfg.database, size)
        except Exception as e:
            _pool = None
            _pool_error = (f"{type(e).__name__}: {e} "
                           f"[host={cfg.host}:{cfg.port}, db={cfg.database}, user={cfg.user}]")
            log.error("MySQL pool init failed: %s", _pool_error)
            raise DBUnavailable(_pool_error) from e


def is_available() -> bool:
    return _pool is not None


def pool_error() -> Optional[str]:
    return _pool_error


def healthcheck() -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


@contextmanager
def get_conn():
    """Borrow a pooled connection. Caller manages its own transactions."""
    if _pool is None:
        init_pool()
    if _pool is None:
        raise DBUnavailable(_pool_error or "pool not initialized")
    conn = _pool.connection()
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── JSON helpers ─────────────────────────────────────────────────────────────


def _dumps(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return json.dumps(v, ensure_ascii=False, default=str)
    except Exception:
        return None


def _loads(v: Any) -> Any:
    if v is None or v == "":
        return None
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return None


_CAND_JSON = ("departments_json", "scores_json", "ai_meta_json",
              "employment_history_json", "linkedin_signals_json", "coresignal_json", "payload_json")
_COMP_JSON = ("payload_json",)


def _parse_json_cols(row: Optional[dict], cols: Sequence[str]) -> Optional[dict]:
    if not row:
        return row
    for c in cols:
        if c in row:
            row[c] = _loads(row[c])
    return row


# ── Schema ───────────────────────────────────────────────────────────────────

_TBL = "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"

_SCHEMA_SQL: Sequence[str] = (
    # 1) runs
    f"""
    CREATE TABLE IF NOT EXISTS runs (
        id                   BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        job_uuid             CHAR(8) NOT NULL,
        trigger_type         ENUM('manual','scheduled') NOT NULL DEFAULT 'manual',
        state                ENUM('running','done','cancelled','error') NOT NULL DEFAULT 'running',
        params_json          JSON NULL,
        companies_new        INT NOT NULL DEFAULT 0,
        candidates_new       INT NOT NULL DEFAULT 0,
        candidates_refreshed INT NOT NULL DEFAULT 0,
        apollo_search_calls  INT NOT NULL DEFAULT 0,
        duration_seconds     INT NOT NULL DEFAULT 0,
        error_text           TEXT NULL,
        started_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        finished_at          DATETIME NULL,
        PRIMARY KEY (id),
        UNIQUE KEY uk_run_job (job_uuid),
        KEY idx_run_started (started_at),
        KEY idx_run_trigger_state (trigger_type, state)
    ) {_TBL}
    """,
    # 2) companies (keyed by company_key; domain/firmographics fill on enrich)
    f"""
    CREATE TABLE IF NOT EXISTS companies (
        id                    BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        company_key           VARCHAR(255) NOT NULL,
        apollo_org_id         VARCHAR(64) NULL,
        name                  VARCHAR(255) NOT NULL,
        root_domain           VARCHAR(255) NULL,
        website_url           VARCHAR(512) NULL,
        linkedin_url          VARCHAR(512) NULL,
        industry              VARCHAR(128) NULL,
        estimated_employees   INT NULL,
        size_band             VARCHAR(48) NULL,
        size_min              INT NULL,
        size_max              INT NULL,
        annual_revenue        BIGINT NULL,
        founded_year          SMALLINT NULL,
        hq_city               VARCHAR(128) NULL,
        hq_country            VARCHAR(64) NULL,
        country               VARCHAR(64) NULL,
        company_quality_score TINYINT UNSIGNED NOT NULL DEFAULT 0,
        enriched              TINYINT(1) NOT NULL DEFAULT 0,
        domain_checked        TINYINT(1) NOT NULL DEFAULT 0,
        web_checked           TINYINT(1) NOT NULL DEFAULT 0,
        description           TEXT NULL,
        og_image              VARCHAR(512) NULL,
        roster_synced_at      DATETIME NULL,
        roster_count          INT NOT NULL DEFAULT 0,
        source                ENUM('apollo','seed','g2','clutch','manual') NOT NULL DEFAULT 'apollo',
        run_id                BIGINT UNSIGNED NULL,
        confidence            TINYINT UNSIGNED NOT NULL DEFAULT 0,
        payload_json          JSON NULL,
        discovered_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_verified_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uk_company_key (company_key),
        KEY idx_company_apollo (apollo_org_id),
        KEY idx_company_domain (root_domain),
        KEY idx_company_quality (company_quality_score),
        KEY idx_company_country (country),
        KEY idx_company_size (estimated_employees),
        KEY idx_company_band (size_band),
        KEY idx_company_domchk (domain_checked),
        KEY idx_company_industry (industry)
    ) {_TBL}
    """,
    # 3) candidates (scores inlined; LinkedIn columns reserved)
    f"""
    CREATE TABLE IF NOT EXISTS candidates (
        id                      BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        apollo_person_id        VARCHAR(64) NOT NULL,
        company_id              BIGINT UNSIGNED NULL,
        company_name            VARCHAR(255) NULL,
        company_domain          VARCHAR(255) NULL,
        full_name               VARCHAR(255) NOT NULL,
        first_name              VARCHAR(128) NULL,
        last_name               VARCHAR(128) NULL,
        title                   VARCHAR(255) NULL,
        headline                VARCHAR(512) NULL,
        department              ENUM('sales','marketing','seo','digital_marketing','other') NOT NULL DEFAULT 'other',
        category                VARCHAR(64) NULL,
        departments_json        JSON NULL,
        seniority               VARCHAR(64) NULL,
        linkedin_url            VARCHAR(512) NULL,
        photo_url               VARCHAR(512) NULL,
        location_city           VARCHAR(128) NULL,
        location_country        VARCHAR(64) NULL,
        has_email               TINYINT(1) NOT NULL DEFAULT 0,
        has_phone               TINYINT(1) NOT NULL DEFAULT 0,
        technical_score         TINYINT UNSIGNED NOT NULL DEFAULT 0,
        role_fit_score          TINYINT UNSIGNED NOT NULL DEFAULT 0,
        job_change_intent_score TINYINT UNSIGNED NOT NULL DEFAULT 0,
        company_quality_score   TINYINT UNSIGNED NOT NULL DEFAULT 0,
        freshness_score         TINYINT UNSIGNED NOT NULL DEFAULT 0,
        overall_candidate_score TINYINT UNSIGNED NOT NULL DEFAULT 0,
        scores_json             JSON NULL,
        ai_meta_json            JSON NULL,
        enrichment_status       ENUM('not_enriched','enriching','enriched','failed','no_credits') NOT NULL DEFAULT 'not_enriched',
        email                   VARCHAR(255) NULL,
        phone                   VARCHAR(64) NULL,
        employment_history_json JSON NULL,
        enriched_at             DATETIME NULL,
        open_to_shift           TINYINT(1) NOT NULL DEFAULT 0,
        intent_source           ENUM('heuristic','history','linkedin') NOT NULL DEFAULT 'heuristic',
        linkedin_enriched       TINYINT(1) NOT NULL DEFAULT 0,
        linkedin_open_to_work   TINYINT(1) NULL,
        linkedin_signals_json   JSON NULL,
        linkedin_checked_at     DATETIME NULL,
        coresignal_enriched     TINYINT(1) NOT NULL DEFAULT 0,
        coresignal_id           VARCHAR(64) NULL,
        coresignal_json         JSON NULL,
        coresignal_checked_at   DATETIME NULL,
        source                  ENUM('apollo','manual') NOT NULL DEFAULT 'apollo',
        run_id                  BIGINT UNSIGNED NULL,
        confidence              TINYINT UNSIGNED NOT NULL DEFAULT 0,
        payload_json            JSON NULL,
        discovered_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_verified_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uk_candidate_apollo (apollo_person_id),
        KEY idx_cand_company (company_id),
        KEY idx_cand_department (department),
        KEY idx_cand_category (category),
        KEY idx_cand_country (location_country),
        KEY idx_cand_overall (overall_candidate_score),
        KEY idx_cand_intent (job_change_intent_score),
        KEY idx_cand_enrich (enrichment_status),
        KEY idx_cand_open (open_to_shift),
        KEY idx_cand_verified (last_verified_at),
        CONSTRAINT fk_cand_company FOREIGN KEY (company_id)
            REFERENCES companies(id) ON DELETE SET NULL
    ) {_TBL}
    """,
    # 4) enrichment_log — credit/audit ledger
    f"""
    CREATE TABLE IF NOT EXISTS enrichment_log (
        id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        candidate_id     BIGINT UNSIGNED NOT NULL,
        apollo_person_id VARCHAR(64) NULL,
        reveal_email     TINYINT(1) NOT NULL DEFAULT 0,
        reveal_phone     TINYINT(1) NOT NULL DEFAULT 0,
        http_status      INT NULL,
        result           ENUM('success','partial','no_credits','failed','retried_no_reveal') NOT NULL,
        credits_before   INT NULL,
        credits_after    INT NULL,
        credits_spent    INT NULL,
        email_revealed   TINYINT(1) NOT NULL DEFAULT 0,
        phone_revealed   TINYINT(1) NOT NULL DEFAULT 0,
        error_text       TEXT NULL,
        response_json    JSON NULL,
        created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        KEY idx_enrlog_candidate (candidate_id),
        KEY idx_enrlog_created (created_at),
        CONSTRAINT fk_enrlog_cand FOREIGN KEY (candidate_id)
            REFERENCES candidates(id) ON DELETE CASCADE
    ) {_TBL}
    """,
    # 5) reveal_counter — per-day reveal caps
    f"""
    CREATE TABLE IF NOT EXISTS reveal_counter (
        day           DATE NOT NULL,
        email_reveals INT NOT NULL DEFAULT 0,
        phone_reveals INT NOT NULL DEFAULT 0,
        PRIMARY KEY (day)
    ) {_TBL}
    """,
    # 6) app_settings — small key/value store (auto-hunt toggle, hunt cursor, ...)
    f"""
    CREATE TABLE IF NOT EXISTS app_settings (
        k          VARCHAR(64) NOT NULL,
        v          TEXT NULL,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (k)
    ) {_TBL}
    """,
)

# Columns added after v1 — applied to already-existing tables by migrate().
# (table, column, ALTER ddl). Idempotent: only run when the column is absent.
_MIGRATIONS = (
    ("companies", "country", "ALTER TABLE companies ADD COLUMN country VARCHAR(64) NULL"),
    ("candidates", "category", "ALTER TABLE candidates ADD COLUMN category VARCHAR(64) NULL"),
    ("companies", "size_band", "ALTER TABLE companies ADD COLUMN size_band VARCHAR(48) NULL"),
    ("companies", "size_min", "ALTER TABLE companies ADD COLUMN size_min INT NULL"),
    ("companies", "size_max", "ALTER TABLE companies ADD COLUMN size_max INT NULL"),
    ("companies", "domain_checked", "ALTER TABLE companies ADD COLUMN domain_checked TINYINT(1) NOT NULL DEFAULT 0"),
    ("companies", "roster_synced_at", "ALTER TABLE companies ADD COLUMN roster_synced_at DATETIME NULL"),
    ("companies", "roster_count", "ALTER TABLE companies ADD COLUMN roster_count INT NOT NULL DEFAULT 0"),
    ("candidates", "linkedin_enriched", "ALTER TABLE candidates ADD COLUMN linkedin_enriched TINYINT(1) NOT NULL DEFAULT 0"),
    ("companies", "web_checked", "ALTER TABLE companies ADD COLUMN web_checked TINYINT(1) NOT NULL DEFAULT 0"),
    ("companies", "description", "ALTER TABLE companies ADD COLUMN description TEXT NULL"),
    ("companies", "og_image", "ALTER TABLE companies ADD COLUMN og_image VARCHAR(512) NULL"),
    ("candidates", "coresignal_enriched", "ALTER TABLE candidates ADD COLUMN coresignal_enriched TINYINT(1) NOT NULL DEFAULT 0"),
    ("candidates", "coresignal_id", "ALTER TABLE candidates ADD COLUMN coresignal_id VARCHAR(64) NULL"),
    ("candidates", "coresignal_json", "ALTER TABLE candidates ADD COLUMN coresignal_json JSON NULL"),
    ("candidates", "coresignal_checked_at", "ALTER TABLE candidates ADD COLUMN coresignal_checked_at DATETIME NULL"),
    ("candidates", "ai_paragraph", "ALTER TABLE candidates ADD COLUMN ai_paragraph TEXT NULL"),
    ("candidates", "ai_paragraph_source", "ALTER TABLE candidates ADD COLUMN ai_paragraph_source VARCHAR(16) NULL"),
    ("companies", "category", "ALTER TABLE companies ADD COLUMN category VARCHAR(64) NULL"),
    ("companies", "category_source", "ALTER TABLE companies ADD COLUMN category_source VARCHAR(16) NULL"),
)
_MIGRATION_INDEXES = (
    ("companies", "idx_company_webchk", "ALTER TABLE companies ADD KEY idx_company_webchk (web_checked)"),
    ("candidates", "idx_cand_cs", "ALTER TABLE candidates ADD KEY idx_cand_cs (coresignal_enriched)"),
    ("companies", "idx_company_category", "ALTER TABLE companies ADD KEY idx_company_category (category)"),
    ("companies", "idx_company_roster", "ALTER TABLE companies ADD KEY idx_company_roster (roster_synced_at)"),
    ("candidates", "idx_cand_li", "ALTER TABLE candidates ADD KEY idx_cand_li (linkedin_enriched)"),
    ("companies", "idx_company_country", "ALTER TABLE companies ADD KEY idx_company_country (country)"),
    ("candidates", "idx_cand_category", "ALTER TABLE candidates ADD KEY idx_cand_category (category)"),
    ("candidates", "idx_cand_country", "ALTER TABLE candidates ADD KEY idx_cand_country (location_country)"),
    ("companies", "idx_company_size", "ALTER TABLE companies ADD KEY idx_company_size (estimated_employees)"),
    ("companies", "idx_company_band", "ALTER TABLE companies ADD KEY idx_company_band (size_band)"),
    ("companies", "idx_company_domchk", "ALTER TABLE companies ADD KEY idx_company_domchk (domain_checked)"),
)


def migrate() -> None:
    """Add post-v1 columns/indexes to already-existing tables. Idempotent."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DATABASE() AS d")
            schema = cur.fetchone()["d"]
            for table, col, ddl in _MIGRATIONS:
                cur.execute("SELECT COUNT(*) AS c FROM information_schema.columns "
                            "WHERE table_schema=%s AND table_name=%s AND column_name=%s",
                            (schema, table, col))
                if cur.fetchone()["c"] == 0:
                    cur.execute(ddl)
                    log.info("migrate: added %s.%s", table, col)
            for table, idx, ddl in _MIGRATION_INDEXES:
                cur.execute("SELECT COUNT(*) AS c FROM information_schema.statistics "
                            "WHERE table_schema=%s AND table_name=%s AND index_name=%s",
                            (schema, table, idx))
                if cur.fetchone()["c"] == 0:
                    try:
                        cur.execute(ddl)
                        log.info("migrate: added index %s", idx)
                    except Exception as e:
                        log.warning("migrate index %s skipped: %s", idx, e)
        conn.commit()


def init_schema() -> None:
    """Create all tables if absent (FK-safe order), then apply migrations. Idempotent."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for stmt in _SCHEMA_SQL:
                cur.execute(stmt)
        conn.commit()
    migrate()
    log.info("Schema verified (6 tables).")


def drop_all() -> None:
    """DESTRUCTIVE — drop every table (dev reset only)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for t in ("enrichment_log", "candidates", "companies", "runs", "reveal_counter",
                      "app_settings"):
                cur.execute(f"DROP TABLE IF EXISTS {t}")
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit()


# ── Pagination helper ────────────────────────────────────────────────────────


def _page_bounds(page: int, page_size: int, max_size: int = 200) -> Tuple[int, int, int]:
    page = max(1, int(page or 1))
    page_size = max(1, min(max_size, int(page_size or 50)))
    return page, page_size, (page - 1) * page_size


# ── Repositories ─────────────────────────────────────────────────────────────


class RunRepo:
    @staticmethod
    def create(job_uuid: str, trigger_type: str, params: Optional[dict]) -> int:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO runs (job_uuid, trigger_type, state, params_json) "
                            "VALUES (%s,%s,'running',%s)",
                            (job_uuid[:8], trigger_type, _dumps(params)))
                rid = cur.lastrowid
            conn.commit()
        return int(rid)

    @staticmethod
    def finish(run_id: int, state: str, stats: Optional[dict] = None,
               error_text: Optional[str] = None) -> None:
        stats = stats or {}
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE runs SET state=%s, companies_new=%s, candidates_new=%s, "
                    "candidates_refreshed=%s, apollo_search_calls=%s, "
                    "duration_seconds=TIMESTAMPDIFF(SECOND, started_at, NOW()), "
                    "error_text=%s, finished_at=NOW() WHERE id=%s",
                    (state, int(stats.get("companies_new", 0)),
                     int(stats.get("candidates_new", 0)),
                     int(stats.get("candidates_refreshed", 0)),
                     int(stats.get("apollo_search_calls", 0)),
                     (error_text or "")[:60000] or None, run_id))
            conn.commit()

    @staticmethod
    def last_run_at():
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(started_at) AS t FROM runs")
                row = cur.fetchone()
        return row["t"] if row else None

    @staticmethod
    def is_run_active() -> bool:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM runs WHERE state='running'")
                row = cur.fetchone()
        return bool(row and row["c"])

    @staticmethod
    def reconcile_stuck() -> int:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE runs SET state='error', "
                            "error_text='reconciled: process restarted', finished_at=NOW() "
                            "WHERE state='running'")
                n = cur.rowcount
            conn.commit()
        return int(n)

    @staticmethod
    def get(run_id: int) -> Optional[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM runs WHERE id=%s", (run_id,))
                row = cur.fetchone()
        if row:
            row["params_json"] = _loads(row.get("params_json"))
        return row

    @staticmethod
    def list_page(page: int = 1, page_size: int = 50) -> Tuple[List[dict], int]:
        page, page_size, off = _page_bounds(page, page_size)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM runs")
                total = int(cur.fetchone()["c"])
                cur.execute(
                    "SELECT id, job_uuid, trigger_type, state, companies_new, candidates_new, "
                    "candidates_refreshed, apollo_search_calls, duration_seconds, error_text, "
                    "started_at, finished_at FROM runs ORDER BY started_at DESC LIMIT %s OFFSET %s",
                    (page_size, off))
                rows = list(cur.fetchall() or [])
        return rows, total


class CompanyRepo:
    @staticmethod
    def upsert(c: dict, run_id: Optional[int]) -> Tuple[int, bool]:
        """INSERT…ON DUPLICATE KEY (company_key). Returns (company_id, is_new).
        COALESCE keeps already-known firmographics from being wiped by thin data."""
        vals = (
            (c.get("company_key") or "")[:255], c.get("apollo_org_id"),
            (c.get("name") or "")[:255], c.get("root_domain"), c.get("website_url"),
            c.get("linkedin_url"), c.get("industry"), c.get("estimated_employees"),
            c.get("size_band"), c.get("size_min"), c.get("size_max"),
            c.get("annual_revenue"), c.get("founded_year"), c.get("hq_city"),
            c.get("hq_country"), c.get("country"), int(c.get("company_quality_score") or 0),
            c.get("source", "apollo"), run_id, int(c.get("confidence") or 0),
            _dumps(c.get("payload_json")))
        sql = (
            "INSERT INTO companies (company_key,apollo_org_id,name,root_domain,website_url,"
            "linkedin_url,industry,estimated_employees,size_band,size_min,size_max,"
            "annual_revenue,founded_year,hq_city,"
            "hq_country,country,company_quality_score,source,run_id,confidence,payload_json) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE name=VALUES(name),"
            "apollo_org_id=COALESCE(VALUES(apollo_org_id),apollo_org_id),"
            "root_domain=COALESCE(VALUES(root_domain),root_domain),"
            "website_url=COALESCE(VALUES(website_url),website_url),"
            "linkedin_url=COALESCE(VALUES(linkedin_url),linkedin_url),"
            "industry=COALESCE(VALUES(industry),industry),"
            "estimated_employees=COALESCE(VALUES(estimated_employees),estimated_employees),"
            "size_band=COALESCE(VALUES(size_band),size_band),"
            "size_min=COALESCE(VALUES(size_min),size_min),"
            "size_max=COALESCE(VALUES(size_max),size_max),"
            "annual_revenue=COALESCE(VALUES(annual_revenue),annual_revenue),"
            "founded_year=COALESCE(VALUES(founded_year),founded_year),"
            "hq_city=COALESCE(VALUES(hq_city),hq_city),"
            "hq_country=COALESCE(VALUES(hq_country),hq_country),"
            "country=COALESCE(VALUES(country),country),"
            "company_quality_score=GREATEST(company_quality_score,VALUES(company_quality_score)),"
            "last_verified_at=NOW(),id=LAST_INSERT_ID(id)")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals)
                cid = cur.lastrowid
                is_new = (cur.rowcount == 1)
            conn.commit()
        return int(cid), is_new

    @staticmethod
    def update_firmographics(company_id: int, org: dict, quality: int) -> None:
        """Opportunistic enrichment: fill domain + firmographics from people/match."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE companies SET "
                    "apollo_org_id=COALESCE(%s,apollo_org_id),"
                    "root_domain=COALESCE(%s,root_domain),website_url=COALESCE(%s,website_url),"
                    "linkedin_url=COALESCE(%s,linkedin_url),industry=COALESCE(%s,industry),"
                    "estimated_employees=COALESCE(%s,estimated_employees),"
                    "annual_revenue=COALESCE(%s,annual_revenue),"
                    "founded_year=COALESCE(%s,founded_year),hq_city=COALESCE(%s,hq_city),"
                    "hq_country=COALESCE(%s,hq_country),country=COALESCE(country,%s),"
                    "company_quality_score=%s,enriched=1,last_verified_at=NOW() WHERE id=%s",
                    (org.get("apollo_org_id"), org.get("root_domain"), org.get("website_url"),
                     org.get("linkedin_url"), org.get("industry"),
                     org.get("estimated_employees"), org.get("annual_revenue"),
                     org.get("founded_year"), org.get("hq_city"), org.get("hq_country"),
                     org.get("hq_country"), int(quality), company_id))
            conn.commit()

    @staticmethod
    def missing_domain(limit: int = 50) -> List[dict]:
        # Resolve the most-populated companies first (best UX — top of the list).
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT co.id, co.name FROM companies co "
                    "WHERE co.root_domain IS NULL AND co.domain_checked=0 "
                    "ORDER BY (SELECT COUNT(*) FROM candidates c WHERE c.company_id=co.id) DESC, "
                    "co.id DESC LIMIT %s", (limit,))
                return list(cur.fetchall() or [])

    @staticmethod
    def set_domain(company_id: int, domain: Optional[str], website: Optional[str]) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE companies SET root_domain=%s, "
                            "website_url=COALESCE(%s,website_url), domain_checked=1, "
                            "last_verified_at=NOW() WHERE id=%s", (domain, website, company_id))
            conn.commit()

    @staticmethod
    def web_pending(limit: int = 40) -> List[dict]:
        """Companies with a domain but no homepage 'About' fetched yet (rostered first)."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, root_domain FROM companies "
                    "WHERE root_domain IS NOT NULL AND web_checked=0 "
                    "ORDER BY roster_count DESC, id DESC LIMIT %s", (limit,))
                return list(cur.fetchall() or [])

    @staticmethod
    def set_web(company_id: int, description: Optional[str], og_image: Optional[str]) -> None:
        # Defensive length caps: og_image is VARCHAR(512); an over-length value would
        # raise DataError under STRICT mode, aborting the UPDATE (web_checked never set).
        og_image = og_image[:512] if og_image else None
        description = description if description else None
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE companies SET description=COALESCE(%s,description), "
                            "og_image=COALESCE(%s,og_image), web_checked=1, last_verified_at=NOW() "
                            "WHERE id=%s", (description, og_image, company_id))
            conn.commit()

    @staticmethod
    def ids_by_keys(keys) -> Dict[str, int]:
        """Resolve many company_keys → ids in one round-trip (for the interconnected web)."""
        keys = list(dict.fromkeys(k for k in keys if k))  # dedupe, drop empties
        if not keys:
            return {}
        ph = ",".join(["%s"] * len(keys))
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT company_key, id FROM companies WHERE company_key IN ({ph})", keys)
                return {r["company_key"]: int(r["id"]) for r in (cur.fetchall() or [])}

    @staticmethod
    def roster_pending(limit: int = 5) -> List[dict]:
        """Companies whose full employee roster hasn't been synced yet. Domain-having
        companies first (precise search), then by how many candidates they already have."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, root_domain, country, size_band, size_min, size_max, "
                    "estimated_employees FROM companies WHERE roster_synced_at IS NULL "
                    "ORDER BY (root_domain IS NOT NULL) DESC, "
                    "(SELECT COUNT(*) FROM candidates c WHERE c.company_id=companies.id) DESC, "
                    "id DESC LIMIT %s", (limit,))
                return list(cur.fetchall() or [])

    @staticmethod
    def roster_counts() -> dict:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) c FROM companies"); total = int(cur.fetchone()["c"])
                cur.execute("SELECT COUNT(*) c FROM companies WHERE roster_synced_at IS NOT NULL")
                done = int(cur.fetchone()["c"])
                cur.execute("SELECT COALESCE(SUM(roster_count),0) s FROM companies")
                people = int(cur.fetchone()["s"])
        return {"companies_total": total, "companies_rostered": done,
                "companies_pending": total - done, "people_rostered": people}

    @staticmethod
    def mark_roster_synced(company_id: int, count: int) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE companies SET roster_synced_at=NOW(), roster_count=%s, "
                            "last_verified_at=NOW() WHERE id=%s", (int(count), company_id))
            conn.commit()

    @staticmethod
    def reset_roster(company_id: int) -> None:
        """Mark a company for re-roster (e.g., after a long interval)."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE companies SET roster_synced_at=NULL WHERE id=%s", (company_id,))
            conn.commit()

    @staticmethod
    def all_id_name(limit: int = 5000) -> List[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, name FROM companies ORDER BY id DESC LIMIT %s", (limit,))
                return list(cur.fetchall() or [])

    @staticmethod
    def delete_with_candidates(ids) -> int:
        ids = list(ids)
        if not ids:
            return 0
        ph = ",".join(["%s"] * len(ids))
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM candidates WHERE company_id IN ({ph})", ids)
                cur.execute(f"DELETE FROM companies WHERE id IN ({ph})", ids)
                n = cur.rowcount
            conn.commit()
        return int(n)

    @staticmethod
    def size_band_counts() -> dict:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(size_band,'Unknown / Unverified') AS b, COUNT(*) AS c "
                            "FROM companies GROUP BY b")
                return {r["b"]: int(r["c"]) for r in (cur.fetchall() or [])}

    @staticmethod
    def category_company_counts() -> dict:
        """Companies per authoritative category (fast, indexed). Empty until the roster
        re-process backfills companies.category; the frontend shows counts only when present,
        and the category filter still works via the candidate fallback in list_page."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT category, COUNT(*) AS c FROM companies "
                            "WHERE category IS NOT NULL AND category<>'' GROUP BY category")
                return {r["category"]: int(r["c"]) for r in (cur.fetchall() or [])}

    @staticmethod
    def get(company_id: int) -> Optional[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM companies WHERE id=%s", (company_id,))
                row = cur.fetchone()
        return _parse_json_cols(row, _COMP_JSON)

    @staticmethod
    def get_by_key(company_key: str) -> Optional[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM companies WHERE company_key=%s", (company_key,))
                row = cur.fetchone()
        return _parse_json_cols(row, _COMP_JSON)

    @staticmethod
    def list_page(filters: dict, page: int = 1, page_size: int = 50) -> Tuple[List[dict], int]:
        page, page_size, off = _page_bounds(page, page_size)
        where, params = ["1=1"], []
        if filters.get("min_quality"):
            where.append("company_quality_score>=%s"); params.append(int(filters["min_quality"]))
        if filters.get("q"):
            where.append("(name LIKE %s OR root_domain LIKE %s)")
            like = f"%{filters['q']}%"; params += [like, like]
        # Country tab: company's own country OR any of its candidates' country.
        if filters.get("country"):
            where.append("(country=%s OR EXISTS (SELECT 1 FROM candidates c "
                         "WHERE c.company_id=companies.id AND c.location_country=%s))")
            params += [filters["country"], filters["country"]]
        # Category: use the authoritative per-company category when it's been set, else fall
        # back to "has a candidate of that category" so the filter works before backfill.
        if filters.get("category"):
            where.append("(companies.category=%s OR ((companies.category IS NULL OR companies.category='') "
                         "AND EXISTS (SELECT 1 FROM candidates c "
                         "WHERE c.company_id=companies.id AND c.category=%s)))")
            params += [filters["category"], filters["category"]]
        # Employee-count slider + size-category filters.
        if filters.get("min_employees"):
            where.append("estimated_employees>=%s"); params.append(int(filters["min_employees"]))
        if filters.get("max_employees"):
            where.append("estimated_employees<=%s"); params.append(int(filters["max_employees"]))
        if filters.get("size_band"):
            if filters["size_band"] == "Unknown / Unverified":
                where.append("(size_band IS NULL OR size_band=%s)"); params.append(filters["size_band"])
            else:
                where.append("size_band=%s"); params.append(filters["size_band"])
        wsql = " AND ".join(where)
        sort_map = {"quality": "company_quality_score DESC",
                    "employees": "estimated_employees DESC", "name": "name ASC",
                    "recent": "last_verified_at DESC", "candidates": "candidate_count DESC",
                    "open": "open_count DESC, candidate_count DESC"}
        order = sort_map.get(filters.get("sort", "candidates"), "candidate_count DESC")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) AS c FROM companies WHERE {wsql}", params)
                total = int(cur.fetchone()["c"])
                cur.execute(
                    "SELECT id, name, company_key, root_domain, website_url, linkedin_url, "
                    "industry, description, estimated_employees, size_band, size_min, size_max, "
                    "annual_revenue, founded_year, hq_city, hq_country, roster_count, "
                    "country, company_quality_score, enriched, source, discovered_at, last_verified_at, "
                    "(SELECT COUNT(*) FROM candidates c WHERE c.company_id=companies.id) "
                    "AS candidate_count, "
                    "(SELECT COUNT(*) FROM candidates c WHERE c.company_id=companies.id "
                    "AND c.open_to_shift=1) AS open_count "
                    f"FROM companies WHERE {wsql} ORDER BY {order} LIMIT %s OFFSET %s",
                    params + [page_size, off])
                rows = list(cur.fetchall() or [])
        return rows, total

    @staticmethod
    def country_counts() -> dict:
        """Company counts per country (for the country tabs)."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM companies")
                total = int(cur.fetchone()["c"])
                out = {"All": total}
                for ctry in ("India", "Australia"):
                    cur.execute("SELECT COUNT(*) AS c FROM companies WHERE country=%s OR EXISTS "
                                "(SELECT 1 FROM candidates c WHERE c.company_id=companies.id "
                                "AND c.location_country=%s)", (ctry, ctry))
                    out[ctry] = int(cur.fetchone()["c"])
        return out

    @staticmethod
    def distinct_industries() -> List[str]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT industry FROM companies "
                            "WHERE industry IS NOT NULL AND industry<>'' ORDER BY industry")
                return [r["industry"] for r in (cur.fetchall() or [])]


class CandidateRepo:
    _LIST_COLS = (
        "id, apollo_person_id, full_name, title, department, category, seniority, company_id, "
        "company_name, company_domain, has_email, has_phone, technical_score, role_fit_score, "
        "job_change_intent_score, company_quality_score, freshness_score, "
        "overall_candidate_score, enrichment_status, linkedin_enriched, coresignal_enriched, "
        "open_to_shift, intent_source, "
        "email, phone, linkedin_url, location_city, location_country, "
        "discovered_at, last_verified_at, enriched_at")

    @staticmethod
    def upsert(c: dict, run_id: Optional[int]) -> Tuple[int, bool]:
        """INSERT…ON DUPLICATE KEY (apollo_person_id). Preserves enrichment +
        first-seen; never downgrades a history/linkedin intent to heuristic."""
        vals = (
            (c.get("apollo_person_id") or "")[:64], c.get("company_id"),
            (c.get("company_name") or None), (c.get("company_domain") or None),
            (c.get("full_name") or "Unknown")[:255], (c.get("first_name") or None),
            (c.get("last_name") or None), (c.get("title") or None),
            (c.get("headline") or None), c.get("department", "other"),
            (c.get("category") or None),
            _dumps(c.get("departments_json")), (c.get("seniority") or None),
            (c.get("linkedin_url") or None), (c.get("photo_url") or None),
            (c.get("location_city") or None), (c.get("location_country") or None),
            int(bool(c.get("has_email"))), int(bool(c.get("has_phone"))),
            int(c.get("technical_score") or 0), int(c.get("role_fit_score") or 0),
            int(c.get("job_change_intent_score") or 0),
            int(c.get("company_quality_score") or 0), int(c.get("freshness_score") or 0),
            int(c.get("overall_candidate_score") or 0), _dumps(c.get("scores_json")),
            _dumps(c.get("ai_meta_json")), int(c.get("open_to_shift") or 0),
            c.get("intent_source", "heuristic"), run_id, int(c.get("confidence") or 0),
            _dumps(c.get("payload_json")))
        threshold = int(os.environ.get("HR_INTENT_OPEN_THRESHOLD", "60"))
        sql = (
            "INSERT INTO candidates (apollo_person_id,company_id,company_name,company_domain,"
            "full_name,first_name,last_name,title,headline,department,category,departments_json,seniority,"
            "linkedin_url,photo_url,location_city,location_country,has_email,has_phone,"
            "technical_score,role_fit_score,job_change_intent_score,company_quality_score,"
            "freshness_score,overall_candidate_score,scores_json,ai_meta_json,open_to_shift,"
            "intent_source,run_id,confidence,payload_json) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
            "%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE "
            "company_id=COALESCE(VALUES(company_id),company_id),"
            "company_name=COALESCE(VALUES(company_name),company_name),"
            "company_domain=COALESCE(VALUES(company_domain),company_domain),"
            "full_name=VALUES(full_name),first_name=VALUES(first_name),last_name=VALUES(last_name),"
            "title=VALUES(title),headline=COALESCE(VALUES(headline),headline),"
            "department=VALUES(department),category=COALESCE(VALUES(category),category),"
            "departments_json=COALESCE(VALUES(departments_json),departments_json),"
            "seniority=COALESCE(VALUES(seniority),seniority),"
            "linkedin_url=COALESCE(VALUES(linkedin_url),linkedin_url),"
            "photo_url=COALESCE(VALUES(photo_url),photo_url),"
            "location_city=COALESCE(VALUES(location_city),location_city),"
            "location_country=COALESCE(VALUES(location_country),location_country),"
            "has_email=GREATEST(has_email,VALUES(has_email)),"
            "has_phone=GREATEST(has_phone,VALUES(has_phone)),"
            "technical_score=VALUES(technical_score),role_fit_score=VALUES(role_fit_score),"
            "company_quality_score=VALUES(company_quality_score),freshness_score=VALUES(freshness_score),"
            "job_change_intent_score=IF(intent_source IN ('history','linkedin'),"
            "job_change_intent_score,VALUES(job_change_intent_score)),"
            "scores_json=VALUES(scores_json),ai_meta_json=COALESCE(VALUES(ai_meta_json),ai_meta_json),"
            f"overall_candidate_score={overall_sql()},"
            "open_to_shift=IF(job_change_intent_score>=%s,1,0),"
            "confidence=VALUES(confidence),payload_json=VALUES(payload_json),"
            "last_verified_at=NOW(),id=LAST_INSERT_ID(id)")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals + (threshold,))
                cid = cur.lastrowid
                is_new = (cur.rowcount == 1)
            conn.commit()
        return int(cid), is_new

    @staticmethod
    def get(candidate_id: int) -> Optional[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM candidates WHERE id=%s", (candidate_id,))
                row = cur.fetchone()
        return _parse_json_cols(row, _CAND_JSON)

    @staticmethod
    def get_basic(candidate_id: int) -> Optional[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, apollo_person_id, first_name, last_name, full_name, company_id, "
                    "company_domain, linkedin_url, enrichment_status, email, phone, "
                    "technical_score, role_fit_score, company_quality_score, freshness_score, "
                    "intent_source FROM candidates WHERE id=%s", (candidate_id,))
                return cur.fetchone()

    @staticmethod
    def list_page(filters: dict, page: int = 1, page_size: int = 50) -> Tuple[List[dict], int]:
        page, page_size, off = _page_bounds(page, page_size)
        where, params = ["1=1"], []
        if filters.get("department"):
            where.append("department=%s"); params.append(filters["department"])
        if filters.get("category"):
            where.append("category=%s"); params.append(filters["category"])
        if filters.get("country"):
            where.append("location_country=%s"); params.append(filters["country"])
        if filters.get("seniority"):
            where.append("seniority=%s"); params.append(filters["seniority"])
        if filters.get("company_id"):
            where.append("company_id=%s"); params.append(int(filters["company_id"]))
        es = filters.get("enrichment_status")
        if es == "apollo":
            where.append("enrichment_status='enriched'")
        elif es == "linkedin":
            where.append("linkedin_enriched=1")
        elif es == "both":
            where.append("enrichment_status='enriched' AND linkedin_enriched=1")
        elif es == "any_enriched":
            where.append("(enrichment_status='enriched' OR linkedin_enriched=1)")
        elif es == "not_enriched":
            where.append("enrichment_status<>'enriched' AND linkedin_enriched=0")
        elif es:
            where.append("enrichment_status=%s"); params.append(es)
        if filters.get("min_overall"):
            where.append("overall_candidate_score>=%s"); params.append(int(filters["min_overall"]))
        if filters.get("min_intent"):
            where.append("job_change_intent_score>=%s"); params.append(int(filters["min_intent"]))
        if filters.get("min_company_score"):
            where.append("company_quality_score>=%s"); params.append(int(filters["min_company_score"]))
        if str(filters.get("open_to_shift", "")) in ("1", "true", "True"):
            where.append("open_to_shift=1")
        if filters.get("freshness"):
            where.append("last_verified_at >= (NOW() - INTERVAL %s DAY)")
            params.append(int(filters["freshness"]))
        if filters.get("q"):
            where.append("(full_name LIKE %s OR title LIKE %s OR company_name LIKE %s)")
            like = f"%{filters['q']}%"; params += [like, like, like]
        wsql = " AND ".join(where)
        sort_map = {"overall": "overall_candidate_score DESC",
                    "intent": "job_change_intent_score DESC",
                    "company_quality": "company_quality_score DESC",
                    "technical": "technical_score DESC", "role_fit": "role_fit_score DESC",
                    "freshness": "freshness_score DESC", "recent": "discovered_at DESC",
                    "name": "full_name ASC"}
        order = sort_map.get(filters.get("sort", "overall"), "overall_candidate_score DESC")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) AS c FROM candidates WHERE {wsql}", params)
                total = int(cur.fetchone()["c"])
                cur.execute(f"SELECT {CandidateRepo._LIST_COLS} FROM candidates WHERE {wsql} "
                            f"ORDER BY {order}, id DESC LIMIT %s OFFSET %s",
                            params + [page_size, off])
                rows = list(cur.fetchall() or [])
        return rows, total

    @staticmethod
    def for_company(company_id: int, limit: int = 500) -> List[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {CandidateRepo._LIST_COLS} FROM candidates "
                            "WHERE company_id=%s ORDER BY overall_candidate_score DESC, id DESC "
                            "LIMIT %s", (company_id, limit))
                return list(cur.fetchall() or [])

    @staticmethod
    def set_enriching(candidate_id: int) -> bool:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE candidates SET enrichment_status='enriching' WHERE id=%s "
                            "AND enrichment_status IN ('not_enriched','failed','no_credits')",
                            (candidate_id,))
                acquired = (cur.rowcount == 1)
            conn.commit()
        return acquired

    @staticmethod
    def apply_linkedin(candidate_id: int, *, open_to_work: Optional[bool],
                       signals: Optional[dict], intent_score: Optional[int],
                       scores_json: Optional[dict], overall: Optional[int]) -> None:
        sets = ["linkedin_enriched=1", "linkedin_checked_at=NOW()", "intent_source='linkedin'"]
        params: list = []
        if open_to_work is not None:
            sets.append("linkedin_open_to_work=%s"); params.append(1 if open_to_work else 0)
        if signals is not None:
            sets.append("linkedin_signals_json=%s"); params.append(_dumps(signals))
        if intent_score is not None:
            sets.append("job_change_intent_score=%s"); params.append(int(intent_score))
        if scores_json is not None:
            sets.append("scores_json=%s"); params.append(_dumps(scores_json))
        if overall is not None:
            sets.append("overall_candidate_score=%s"); params.append(int(overall))
        if intent_score is not None:
            sets.append("open_to_shift=IF(%s>=%s,1,0)")
            params += [int(intent_score), int(os.environ.get("HR_INTENT_OPEN_THRESHOLD", "60"))]
        params.append(candidate_id)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE candidates SET {', '.join(sets)} WHERE id=%s", params)
            conn.commit()

    @staticmethod
    def apply_coresignal(candidate_id: int, *, coresignal_json: Optional[dict],
                         coresignal_id: Optional[str], open_to_work: Optional[bool],
                         intent_score: Optional[int], scores_json: Optional[dict],
                         overall: Optional[int], linkedin_url: Optional[str] = None) -> None:
        """Persist a CoreSignal LinkedIn enrichment. Stores the CoreSignal payload in its
        own columns AND updates the shared intent fields (it IS richer LinkedIn data), so
        the candidate's intent/open-to-shift reflect it. Mirrors apply_linkedin."""
        sets = ["coresignal_enriched=1", "coresignal_checked_at=NOW()",
                "linkedin_enriched=1", "linkedin_checked_at=NOW()", "intent_source='linkedin'"]
        params: list = []
        if coresignal_json is not None:
            sets.append("coresignal_json=%s"); params.append(_dumps(coresignal_json))
        if coresignal_id:
            sets.append("coresignal_id=%s"); params.append(str(coresignal_id)[:64])
        if linkedin_url:
            sets.append("linkedin_url=%s"); params.append(linkedin_url[:512])
        if open_to_work is not None:
            sets.append("linkedin_open_to_work=%s"); params.append(1 if open_to_work else 0)
        if intent_score is not None:
            sets.append("job_change_intent_score=%s"); params.append(int(intent_score))
        if scores_json is not None:
            sets.append("scores_json=%s"); params.append(_dumps(scores_json))
        if overall is not None:
            sets.append("overall_candidate_score=%s"); params.append(int(overall))
        if intent_score is not None:
            sets.append("open_to_shift=IF(%s>=%s,1,0)")
            params += [int(intent_score), int(os.environ.get("HR_INTENT_OPEN_THRESHOLD", "60"))]
        params.append(candidate_id)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE candidates SET {', '.join(sets)} WHERE id=%s", params)
            conn.commit()

    @staticmethod
    def set_phone_by_apollo_id(apollo_person_id: str, phone: str) -> bool:
        """Write a phone delivered by Apollo's async webhook onto the matching candidate
        (by apollo_person_id), without overwriting an existing number. Returns True if a row
        was updated."""
        if not (apollo_person_id and phone):
            return False
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE candidates SET phone=%s WHERE apollo_person_id=%s "
                            "AND (phone IS NULL OR phone='')", (phone[:64], apollo_person_id))
                n = cur.rowcount
            conn.commit()
        return n > 0

    @staticmethod
    def phone_populated_count() -> int:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM candidates WHERE phone IS NOT NULL AND phone<>''")
                return int(cur.fetchone()["c"])

    @staticmethod
    def set_ai_paragraph(candidate_id: int, paragraph: str, source: str) -> None:
        """Cache the AI recruiter brief on the row so it isn't regenerated every panel open."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE candidates SET ai_paragraph=%s, ai_paragraph_source=%s WHERE id=%s",
                            ((paragraph or "")[:2000], (source or "")[:16], candidate_id))
            conn.commit()

    @staticmethod
    def set_linkedin_url(candidate_id: int, url: str) -> None:
        """Light update of just the LinkedIn URL (used when Apollo resolves it for a
        candidate that lacked one, so CoreSignal can do a precise profile lookup)."""
        if not url:
            return
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE candidates SET linkedin_url=%s WHERE id=%s",
                            (url[:512], candidate_id))
            conn.commit()

    @staticmethod
    def set_status(candidate_id: int, status: str) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE candidates SET enrichment_status=%s WHERE id=%s",
                            (status, candidate_id))
            conn.commit()

    @staticmethod
    def apply_enrichment(candidate_id: int, *, email: Optional[str], phone: Optional[str],
                         status: str, full_name: Optional[str] = None,
                         seniority: Optional[str] = None, linkedin_url: Optional[str] = None,
                         location_country: Optional[str] = None,
                         employment_history: Optional[list] = None,
                         intent_score: Optional[int] = None, intent_source: Optional[str] = None,
                         scores_json: Optional[dict] = None, overall: Optional[int] = None,
                         company_quality: Optional[int] = None) -> None:
        sets, params = ["enrichment_status=%s", "enriched_at=NOW()"], [status]
        if email is not None:
            sets.append("email=%s"); params.append(email[:255])
        if phone is not None:
            sets.append("phone=%s"); params.append(phone[:64])
        if full_name:
            sets.append("full_name=%s"); params.append(full_name[:255])
        if seniority:
            sets.append("seniority=%s"); params.append(seniority[:64])
        if linkedin_url:
            sets.append("linkedin_url=%s"); params.append(linkedin_url[:512])
        if location_country:
            sets.append("location_country=%s"); params.append(location_country[:64])
        if company_quality is not None:
            sets.append("company_quality_score=%s"); params.append(int(company_quality))
        if employment_history is not None:
            sets.append("employment_history_json=%s"); params.append(_dumps(employment_history))
        if intent_score is not None:
            sets.append("job_change_intent_score=%s"); params.append(int(intent_score))
        if intent_source is not None:
            sets.append("intent_source=%s"); params.append(intent_source)
        if scores_json is not None:
            sets.append("scores_json=%s"); params.append(_dumps(scores_json))
        if overall is not None:
            sets.append("overall_candidate_score=%s"); params.append(int(overall))
            sets.append("open_to_shift=IF(%s>=%s,1,0)")
            params += [int(intent_score or 0), int(os.environ.get("HR_INTENT_OPEN_THRESHOLD", "60"))]
        params.append(candidate_id)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE candidates SET {', '.join(sets)} WHERE id=%s", params)
            conn.commit()

    @staticmethod
    def recompute_freshness_all(threshold: int) -> int:
        sql = (f"UPDATE candidates SET freshness_score=({freshness_sql()}), "
               f"overall_candidate_score=({overall_sql(freshness='(' + freshness_sql() + ')')}), "
               "open_to_shift=IF(job_change_intent_score>=%s,1,0)")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int(threshold),))
                n = cur.rowcount
            conn.commit()
        return int(n)

    @staticmethod
    def distinct_departments() -> List[str]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT department FROM candidates ORDER BY department")
                return [r["department"] for r in (cur.fetchall() or [])]

    @staticmethod
    def distinct_seniorities() -> List[str]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT seniority FROM candidates "
                            "WHERE seniority IS NOT NULL AND seniority<>'' ORDER BY seniority")
                return [r["seniority"] for r in (cur.fetchall() or [])]

    @staticmethod
    def distinct_categories() -> List[str]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT category FROM candidates "
                            "WHERE category IS NOT NULL AND category<>'' ORDER BY category")
                return [r["category"] for r in (cur.fetchall() or [])]

    @staticmethod
    def dominant_category(company_id: int) -> Optional[str]:
        """Most common category among a company's candidates (for derived industry)."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT category, COUNT(*) AS n FROM candidates "
                            "WHERE company_id=%s AND category IS NOT NULL AND category<>'' "
                            "GROUP BY category ORDER BY n DESC LIMIT 1", (company_id,))
                row = cur.fetchone()
        return row["category"] if row else None

    @staticmethod
    def companies_for_filter(limit: int = 500) -> List[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT co.id, co.name, co.root_domain, COUNT(c.id) AS n "
                    "FROM companies co JOIN candidates c ON c.company_id=co.id "
                    "GROUP BY co.id, co.name, co.root_domain ORDER BY n DESC, co.name ASC LIMIT %s",
                    (limit,))
                return list(cur.fetchall() or [])


class EnrichmentLogRepo:
    @staticmethod
    def log(e: dict) -> int:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO enrichment_log (candidate_id,apollo_person_id,reveal_email,"
                    "reveal_phone,http_status,result,credits_before,credits_after,credits_spent,"
                    "email_revealed,phone_revealed,error_text,response_json) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (e["candidate_id"], e.get("apollo_person_id"),
                     int(bool(e.get("reveal_email"))), int(bool(e.get("reveal_phone"))),
                     e.get("http_status"), e["result"], e.get("credits_before"),
                     e.get("credits_after"), e.get("credits_spent"),
                     int(bool(e.get("email_revealed"))), int(bool(e.get("phone_revealed"))),
                     (e.get("error_text") or None), _dumps(e.get("response_json"))))
                lid = cur.lastrowid
            conn.commit()
        return int(lid)

    @staticmethod
    def for_candidate(candidate_id: int, limit: int = 20) -> List[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, reveal_email, reveal_phone, http_status, result, credits_spent, "
                    "email_revealed, phone_revealed, error_text, created_at FROM enrichment_log "
                    "WHERE candidate_id=%s ORDER BY created_at DESC LIMIT %s", (candidate_id, limit))
                return list(cur.fetchall() or [])

    @staticmethod
    def recent_phone_attempts(limit: int = 10) -> List[dict]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT candidate_id, http_status, result, phone_revealed, error_text, created_at "
                    "FROM enrichment_log WHERE reveal_phone=1 ORDER BY created_at DESC LIMIT %s", (limit,))
                return list(cur.fetchall() or [])


class RevealCounterRepo:
    @staticmethod
    def today() -> dict:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT email_reveals, phone_reveals FROM reveal_counter WHERE day=CURDATE()")
                row = cur.fetchone()
        return {"email_reveals": int(row["email_reveals"]) if row else 0,
                "phone_reveals": int(row["phone_reveals"]) if row else 0}

    @staticmethod
    def incr(email: int = 0, phone: int = 0) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO reveal_counter (day,email_reveals,phone_reveals) "
                    "VALUES (CURDATE(),%s,%s) ON DUPLICATE KEY UPDATE "
                    "email_reveals=email_reveals+VALUES(email_reveals),"
                    "phone_reveals=phone_reveals+VALUES(phone_reveals)", (int(email), int(phone)))
            conn.commit()


class StatsRepo:
    @staticmethod
    def counts() -> dict:
        out: Dict[str, Any] = {}
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM candidates")
                out["candidates_total"] = int(cur.fetchone()["c"])
                cur.execute("SELECT COUNT(*) AS c FROM companies")
                out["companies_total"] = int(cur.fetchone()["c"])
                cur.execute("SELECT COUNT(*) AS c FROM candidates WHERE enrichment_status='enriched'")
                out["enriched"] = int(cur.fetchone()["c"])
                cur.execute("SELECT COUNT(*) AS c FROM candidates WHERE open_to_shift=1")
                out["open_to_shift"] = int(cur.fetchone()["c"])
                # of the open-to-shift, how many had that intent VERIFIED via LinkedIn data
                # (CoreSignal or the public-profile check)
                cur.execute("SELECT COUNT(*) AS c FROM candidates WHERE open_to_shift=1 "
                            "AND (coresignal_enriched=1 OR linkedin_enriched=1)")
                out["open_to_shift_li"] = int(cur.fetchone()["c"])
                cur.execute("SELECT department, COUNT(*) AS c FROM candidates GROUP BY department")
                out["by_department"] = {r["department"]: int(r["c"]) for r in (cur.fetchall() or [])}
                cur.execute("SELECT state, started_at FROM runs ORDER BY started_at DESC LIMIT 1")
                last = cur.fetchone()
                out["last_run_at"] = last["started_at"] if last else None
                out["last_run_state"] = last["state"] if last else None
        return out


class SettingsRepo:
    """Tiny key/value store (auto-hunt toggle, hunt cursor, counters)."""

    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT v FROM app_settings WHERE k=%s", (key,))
                row = cur.fetchone()
        return row["v"] if row else default

    @staticmethod
    def get_many(keys) -> Dict[str, Optional[str]]:
        """Read several settings in ONE round-trip (avoids per-key proxy latency)."""
        keys = list(keys)
        if not keys:
            return {}
        ph = ",".join(["%s"] * len(keys))
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT k, v FROM app_settings WHERE k IN ({ph})", keys)
                rows = cur.fetchall() or []
        return {r["k"]: r["v"] for r in rows}

    @staticmethod
    def set(key: str, value: Optional[str]) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO app_settings (k,v) VALUES (%s,%s) "
                            "ON DUPLICATE KEY UPDATE v=VALUES(v)", (key, value))
            conn.commit()

    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        v = SettingsRepo.get(key)
        return default if v is None else v in ("1", "true", "True", "on")

    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        v = SettingsRepo.get(key)
        try:
            return int(v) if v is not None else default
        except (ValueError, TypeError):
            return default
