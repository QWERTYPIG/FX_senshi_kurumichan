
import os
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


if __name__ == "__main__":
    with login() as session:
        print("Session cookies:", list(session.cookies.keys()))
