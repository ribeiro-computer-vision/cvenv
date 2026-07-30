"""One resumable file downloader (unifies two near-identical copies that had
grown independently in the source projects).

Tries curl (``-C -`` resumes a partial file, ``--retry`` handles transient
failures within a single invocation), then wget (``-c`` resumes), falling back
to ``urllib.request.urlretrieve`` (no resume) only if neither binary exists.
A partial file from curl/wget is left in place between our own retries on
purpose so the next attempt resumes; urlretrieve's failure path deletes it,
since a truncated file there is useless for a resume anyway.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


def download_resumable(url: str, dest, attempts: int = 5, base_delay: float = 5.0):
    """Download ``url`` to ``dest`` with resume + retry/backoff."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("curl"):
        cmd = ["curl", "-fL", "-C", "-", "--retry", str(attempts),
               "--retry-delay", str(int(base_delay)), "--retry-connrefused",
               "-o", str(dest), url]
    elif shutil.which("wget"):
        cmd = ["wget", "-c", f"--tries={attempts}",
               f"--waitretry={int(base_delay)}", "-O", str(dest), url]
    else:
        cmd = None

    if cmd is not None:
        try:
            subprocess.run(cmd, check=True)
            return dest
        except subprocess.CalledProcessError as e:
            print(f"⚠️ {cmd[0]} failed ({e}); falling back to urlretrieve (no resume).")

    last_err = None
    for i in range(1, attempts + 1):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:
            last_err = e
            if dest.exists():
                dest.unlink()
            if i < attempts:
                delay = base_delay * (2 ** (i - 1))
                print(f"⚠️ download attempt {i}/{attempts} failed ({e}); "
                      f"retrying in {delay:.0f}s…")
                time.sleep(delay)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_err
