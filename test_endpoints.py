"""
test_endpoints.py — quick smoke test against a running sky-annotations-worker.

Usage:
  python test_endpoints.py                        # hits localhost:8000
  python test_endpoints.py https://your.railway.app
"""

import sys
import json
import urllib.request
import urllib.error

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"


def get(path: str) -> dict:
    url = BASE + path
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "reason": e.reason}
    except Exception as e:
        return {"error": str(e)}


def post(path: str, body: dict) -> dict:
    url  = BASE + path
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        return {"http_error": e.code, "body": body_text}
    except Exception as e:
        return {"error": str(e)}


def check(label: str, result: dict, expect_key: str = "ok") -> None:
    ok = expect_key in result and result[expect_key] is True
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status}  {label}")
    if not ok:
        print(f"         response: {json.dumps(result)[:200]}")


print(f"\n🔍  Testing sky-annotations-worker at {BASE}\n")

# Health
r = get("/health")
check("/health → ok=true, service matches", r)
if r.get("service") != "sky-annotations-worker":
    print(f"         ⚠️  service='{r.get('service')}' (expected 'sky-annotations-worker')")

# OCR run — expects 422 (missing body fields) not 500
r = post("/ocr/run", {})
check("/ocr/run with empty body → 422 validation", {"ok": r.get("http_error") == 422})

# Timestamps run — same
r = post("/timestamps/run", {})
check("/timestamps/run with empty body → 422 validation", {"ok": r.get("http_error") == 422})

# AI run — same
r = post("/ai/run", {})
check("/ai/run with empty body → 422 validation", {"ok": r.get("http_error") == 422})

# Clips render — same
r = post("/clips/render", {})
check("/clips/render with empty body → 422 validation", {"ok": r.get("http_error") == 422})

# Slide preview with missing params → 422
r = get("/slide-preview")
check("/slide-preview with no params → 422 validation", {"ok": r.get("http_error") == 422})

# Clip file with unknown name → 404 or 500 (not crash)
r = get("/clips/file/nonexistent_clip.mp4")
check("/clips/file/{unknown} → 404 (not crash)", {"ok": r.get("http_error") in (404, 500)})

print("\nAll checks complete.\n")
