"""Operator side of the session provider.

    python scripts/session_answer.py list                 pending request ids, sizes, schema names
    python scripts/session_answer.py show <id>            print the request (system, user, schema)
    python scripts/session_answer.py answer <id> <file>   validate <file> against the request's schema, then respond
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

D = Path(os.environ.get("VOICEPRINT_SESSION_DIR", Path(__file__).resolve().parents[1] / ".voiceprint" / "session"))


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "list":
        for f in sorted((D / "requests").glob("*.json"), key=lambda p: p.stat().st_mtime):
            r = json.loads(f.read_text(encoding="utf-8"))
            title = (r.get("schema") or {}).get("title", "text")
            print(f"{r['id']}  {title:12s} user={len(r['user'])} chars  max_tokens={r['max_tokens']}")
    elif cmd == "show":
        r = json.loads((D / "requests" / f"{sys.argv[2]}.json").read_text(encoding="utf-8"))
        print("=== SYSTEM ===\n" + r["system"] + "\n=== USER ===\n" + r["user"])
        if r.get("schema"):
            print("=== SCHEMA ===\n" + json.dumps(r["schema"]))
    elif cmd == "answer":
        rid, src = sys.argv[2], Path(sys.argv[3])
        r = json.loads((D / "requests" / f"{rid}.json").read_text(encoding="utf-8"))
        text = src.read_text(encoding="utf-8").strip()
        if r.get("schema"):
            json.loads(text)  # must be JSON; the pipeline then validates it against the pydantic schema
        (D / "responses" / f"{rid}.tmp").write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
        (D / "responses" / f"{rid}.tmp").replace(D / "responses" / f"{rid}.json")
        print(f"answered {rid}")


if __name__ == "__main__":
    main()
