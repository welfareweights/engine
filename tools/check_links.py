"""Link gate: every hyperlink in shipped docs must resolve before we ship it.

Scans the public-facing docs for URLs and checks each one. Exit codes make it
a CI gate: 0 all good, 1 any dead link. Bot-blocked responses (403/429/405)
are reported as BLOCKED and pass with a warning; a site that rejects robots
is not a dead link, and several legitimate sources (charity evaluators,
publishers) block automated fetches. Dead means 404/410, other 4xx/5xx, DNS
failure, or timeout.

Run:  .venv/bin/python tools/check_links.py            (all shipped docs)
      .venv/bin/python tools/check_links.py FILE...    (specific files)
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    ENGINE / "README.md",
    ENGINE / "ROADMAP.md",
    *sorted((ENGINE / "docs").glob("*.md")),
    *sorted((ENGINE / "docs").glob("*.tex")),
    *sorted((ENGINE / "docs").glob("*.html")),
]

URL_RE = re.compile(r"https?://[^\s\]}\"'<>\\]+")
BLOCKED_STATUSES = {403, 405, 429}


def urls_in(path: Path) -> set[str]:
    text = path.read_text()
    if path.suffix == ".tex":
        # LaTeX escapes: \% -> %, \& -> &, \_ -> _, \# -> # (before matching,
        # or the backslash truncates the match)
        text = re.sub(r"\\([%&_#])", r"\1", text)
    found = set()
    for m in URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?")
        # DOIs legitimately contain parens; markdown wraps URLs in them.
        # Strip trailing ')' only while unbalanced.
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        found.add(url)
    return found


def check(url: str) -> tuple[str, str]:
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "Mozilla/5.0 (link checker; welfareweights)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return "OK", str(resp.status)
    except urllib.error.HTTPError as e:
        if e.code in BLOCKED_STATUSES:
            # Retry as GET once; some servers reject HEAD but serve GET.
            try:
                get_req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 (link checker; welfareweights)"}
                )
                with urllib.request.urlopen(get_req, timeout=20) as resp:
                    return "OK", f"{resp.status} (GET)"
            except urllib.error.HTTPError as e2:
                if e2.code in BLOCKED_STATUSES:
                    return "BLOCKED", str(e2.code)
                return "DEAD", str(e2.code)
            except Exception as e2:  # noqa: BLE001
                return "DEAD", str(e2)
        return "DEAD", str(e.code)
    except Exception as e:  # noqa: BLE001
        return "DEAD", str(e)


def main() -> int:
    files = [Path(a) for a in sys.argv[1:]] or DEFAULT_FILES
    by_url: dict[str, list[str]] = {}
    for f in files:
        for u in urls_in(f):
            by_url.setdefault(u, []).append(f.name)
    dead = 0
    for url in sorted(by_url):
        status, detail = check(url)
        mark = {"OK": " ", "BLOCKED": "?", "DEAD": "!"}[status]
        print(f"[{mark}] {status:7s} {detail:12s} {url}   ({', '.join(sorted(set(by_url[url])))})")
        dead += status == "DEAD"
    print(f"\n{len(by_url)} unique links; {dead} dead")
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
