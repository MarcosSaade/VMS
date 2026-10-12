"""`.env.example` must parse the same way in Docker Compose and in python-dotenv.

Compose's env-file parser keeps an inline comment after an empty value as the value itself
(`LLM_MODEL=   # empty means ...` gives LLM_MODEL="# empty means ..."), while python-dotenv
reads it as empty. A fresh clone copies this file to .env, so such a line made the
containerized query service pick an LLM named after the comment (found by a fresh-clone
`make demo`). Every comment goes on its own line, so a value can be emptied without
turning its comment into the value.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INLINE_COMMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*[ \t]#")


def test_no_inline_comments() -> None:
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    offending = [f"{i}: {line}" for i, line in enumerate(lines, 1) if INLINE_COMMENT.match(line)]
    assert offending == [], "\n".join(offending)
