"""Consent-gated, one-consultant-at-a-time consultation request delivery."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import smtplib
import sqlite3
import ssl
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from .pdf_report import create_consultation_pdf

logger = logging.getLogger(__name__)


class ConsultationConfigurationError(RuntimeError):
    pass


class DuplicateRequestError(RuntimeError):
    pass


class ConsultationDeliveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConsultationConfig:
    recipients: tuple[str, ...]
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    sender_email: str
    fingerprint_secret: str
    database_path: Path


def _setting(name: str, default: Any = None) -> Any:
    if name in os.environ:
        return os.environ[name]
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def load_config(project_root: Path | None = None) -> ConsultationConfig:
    root = project_root or Path(__file__).resolve().parents[2]
    raw_recipients = _setting("CONSULTANT_EMAILS", [])
    if isinstance(raw_recipients, str):
        recipients = tuple(item.strip() for item in raw_recipients.split(",") if item.strip())
    elif isinstance(raw_recipients, (tuple, list)):
        recipients = tuple(str(item).strip() for item in raw_recipients if str(item).strip())
    else:
        recipients = ()
    smtp_host = str(_setting("SMTP_HOST", "")).strip()
    smtp_username = str(_setting("SMTP_USERNAME", "")).strip()
    smtp_password = str(_setting("SMTP_PASSWORD", "")).strip()
    sender = str(_setting("SMTP_FROM_EMAIL", smtp_username)).strip()
    try:
        port = int(_setting("SMTP_PORT", 587))
    except (TypeError, ValueError):
        port = 0
    fingerprint_secret = str(_setting("CONSULTATION_FINGERPRINT_SECRET", "")).strip()
    configured_path = str(_setting("CONSULTATION_DB_PATH", "data/private/consultation-requests.sqlite3"))
    database_path = Path(configured_path)
    if not database_path.is_absolute():
        database_path = root / database_path

    errors = []
    if not recipients:
        errors.append("CONSULTANT_EMAILS")
    if not smtp_host or not smtp_username or not smtp_password or not sender or not (1 <= port <= 65535):
        errors.append("SMTP settings")
    if len(fingerprint_secret) < 32:
        errors.append("CONSULTATION_FINGERPRINT_SECRET")
    if errors:
        raise ConsultationConfigurationError("Consultation email is not configured: " + ", ".join(errors))
    return ConsultationConfig(
        recipients=recipients,
        smtp_host=smtp_host,
        smtp_port=port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        sender_email=sender,
        fingerprint_secret=fingerprint_secret,
        database_path=database_path,
    )


def normalize_phone(phone: str) -> str:
    translated = phone.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
    digits = "".join(character for character in translated if character.isdigit())
    if not 7 <= len(digits) <= 15:
        raise ValueError("شماره تماس معتبر نیست. شماره را با کد کشور وارد کنید.")
    return digits


class LeadStore:
    """Keep only a keyed phone fingerprint and routing state for duplicate control."""

    def __init__(self, config: ConsultationConfig) -> None:
        self.config = config
        config.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS consultation_requests (
                    phone_fingerprint TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    assigned_recipient TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS consultation_routing (
                    key TEXT PRIMARY KEY,
                    next_recipient_index INTEGER NOT NULL
                )
            """)

    def _connect(self):
        db = sqlite3.connect(self.config.database_path, timeout=15, isolation_level=None)
        db.execute("PRAGMA busy_timeout = 15000")
        return db

    def _fingerprint(self, phone: str) -> str:
        return hmac.new(
            self.config.fingerprint_secret.encode("utf-8"),
            phone.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def reserve(self, phone: str) -> tuple[str, str]:
        fingerprint = self._fingerprint(phone)
        request_id = uuid.uuid4().hex[:12].upper()
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT status FROM consultation_requests WHERE phone_fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if existing:
                db.execute("ROLLBACK")
                raise DuplicateRequestError("درخواستی با این شماره قبلاً ثبت شده یا در حال بررسی است.")
            row = db.execute(
                "SELECT next_recipient_index FROM consultation_routing WHERE key = 'round_robin'"
            ).fetchone()
            index = int(row[0]) % len(self.config.recipients) if row else 0
            recipient = self.config.recipients[index]
            next_index = (index + 1) % len(self.config.recipients)
            db.execute(
                "INSERT INTO consultation_routing(key, next_recipient_index) VALUES('round_robin', ?) "
                "ON CONFLICT(key) DO UPDATE SET next_recipient_index = excluded.next_recipient_index",
                (next_index,),
            )
            db.execute(
                "INSERT INTO consultation_requests VALUES(?, ?, ?, 'sending', ?, ?)",
                (fingerprint, request_id, recipient, now, now),
            )
            db.execute("COMMIT")
        return request_id, recipient

    def mark(self, phone: str, status: str) -> None:
        if status not in {"sent", "failed"}:
            raise ValueError("Unsupported consultation status")
        with closing(self._connect()) as db:
            db.execute(
                "UPDATE consultation_requests SET status = ?, updated_at = ? WHERE phone_fingerprint = ?",
                (status, datetime.now(timezone.utc).isoformat(), self._fingerprint(phone)),
            )


def _send_email(config: ConsultationConfig, recipient: str, name: str, request_id: str, pdf_data: bytes) -> None:
    message = EmailMessage()
    message["From"] = config.sender_email
    message["To"] = recipient
    message["Subject"] = f"درخواست مشاوره جدید - {name}"
    message.set_content(
        "درخواست مشاوره جدید ثبت شده است. خلاصه اطلاعات متقاضی در فایل PDF پیوست قرار دارد.\n"
        f"شماره درخواست: {request_id}\n"
        "این پیام برای یک مشاور اختصاص‌یافته ارسال شده است؛ لطفاً درخواست را به مشاور دیگری ارسال نکنید."
    )
    message.add_attachment(
        pdf_data,
        maintype="application",
        subtype="pdf",
        filename=f"consultation-request-{request_id}.pdf",
    )
    context = ssl.create_default_context()
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=25) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(config.smtp_username, config.smtp_password)
        server.send_message(message, from_addr=config.sender_email, to_addrs=[recipient])


class ConsultationService:
    def __init__(self, config: ConsultationConfig, project_root: Path | None = None) -> None:
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.store = LeadStore(config)

    @classmethod
    def from_app_secrets(cls, project_root: Path | None = None) -> "ConsultationService":
        return cls(load_config(project_root), project_root)

    def submit(self, profile: dict[str, Any], *, name: str, phone: str, consent: bool) -> str:
        if not consent:
            raise ValueError("برای ارسال درخواست، رضایت صریح شما لازم است.")
        cleaned_name = re.sub(r"\s+", " ", name).strip()
        if not 2 <= len(cleaned_name) <= 120:
            raise ValueError("نام را کامل وارد کنید.")
        normalized_phone = normalize_phone(phone)
        request_id, recipient = self.store.reserve(normalized_phone)
        now = datetime.now(timezone.utc)
        report_profile = dict(profile)
        report_profile["applicant_name"] = cleaned_name
        report_profile["contact_phone"] = phone.strip()
        try:
            pdf_data = create_consultation_pdf(
                report_profile, request_id=request_id, created_at=now, project_root=self.project_root
            )
            _send_email(self.config, recipient, cleaned_name, request_id, pdf_data)
        except Exception as exc:
            self.store.mark(normalized_phone, "failed")
            logger.error("Consultation request delivery failed; request_id=%s error_type=%s", request_id, type(exc).__name__)
            raise ConsultationDeliveryError(
                "ارسال درخواست انجام نشد. اطلاعات ثبت‌شده برای جلوگیری از ارسال تکراری قفل شده است؛ لطفاً با پشتیبانی تماس بگیرید."
            ) from exc
        try:
            self.store.mark(normalized_phone, "sent")
        except Exception as exc:
            # The email was delivered. Keep the UI result successful; the row
            # remains in the non-repeatable sending state if persistence failed.
            logger.error("Consultation delivered but status update failed; request_id=%s error_type=%s", request_id, type(exc).__name__)
        return request_id
