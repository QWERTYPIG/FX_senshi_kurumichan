
import os
import threading
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

BASE_URL = "https://final.html2026.csie.org"


def login() -> requests.Session:
    email = os.environ["email"]
    password = os.environ["passwd"]

    session = requests.Session()

    session.headers.update({
        "Origin": BASE_URL,
        "Referer": f"{BASE_URL}/login",
        "Accept": "*/*",
    })

    try:
        response = session.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "email": email,
                "password": password,
            },
            timeout=30,
        )

        response.raise_for_status()

        if "application/json" in response.headers.get(
            "Content-Type", ""
        ):
            result = response.json()
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(
                    f"Login failed: {result['error']}"
                )

        print("Login request succeeded!")
        return session

    except Exception:
        session.close()
        raise


class RequestNotAccepted(RuntimeError):
    """The operation did not reach the endpoint or was explicitly refused."""


class CompetitionSession:
    """Reuse one authenticated session across manager workers."""

    def __init__(self):
        self._session = None
        self._lock = threading.Lock()
        self._retry_at = 0

    def _rate_limited(self, response):
        retry_after = response.headers.get("Retry-After", "")
        try:
            seconds = int(retry_after)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(retry_after).timestamp() - time.time()
            except (ValueError, OverflowError):
                seconds = 60
        self._retry_at = time.monotonic() + max(1, seconds)
        return RequestNotAccepted(f"官方 API 回傳 429，{max(1, int(seconds))} 秒後可重試")

    def request(self, method, url, **kwargs):
        with self._lock:
            if time.monotonic() < self._retry_at:
                raise RequestNotAccepted("官方 API 暫時限流，等待 Retry-After 後重試")
            if self._session is None:
                try:
                    self._session = login()
                except Exception as exc:
                    response = getattr(exc, "response", None)
                    if response is not None and response.status_code == 429:
                        raise self._rate_limited(response) from exc
                    raise RequestNotAccepted(f"官方登入失敗：{exc}") from exc
            response = self._session.request(method, url, **kwargs)
            if response.status_code == 429:
                raise self._rate_limited(response)
            if response.status_code == 401:
                self._session.close()
                self._session = None
                raise RequestNotAccepted("官方登入已過期，下次請求會重新登入")
            return response


if __name__ == "__main__":
    with login() as session:
        print("Session cookies:", list(session.cookies.keys()))
