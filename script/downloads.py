import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import requests
import urllib3

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
BASE_URL = "https://140.112.31.182:8444"
OUTPUT = Path("data.zip")

def download_data(output=None):
    output = Path(output) if output is not None else OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.verify = False
        response = session.post(
            f"{BASE_URL}/login",
            data={"key": os.environ["data_access_key"]},
            timeout=30,
        )
        response.raise_for_status()

        response = session.get(
            f"{BASE_URL}/data.zip",
            timeout=120,
        )
        response.raise_for_status()

        temp = output.with_suffix(".zip.part")
        try:
            temp.write_bytes(response.content)

            if not zipfile.is_zipfile(temp):
                raise RuntimeError(
                    "not zip"
                )

            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)

        print(f"Downloaded: {output}")
        return output


if __name__ == "__main__":
    download_data()

