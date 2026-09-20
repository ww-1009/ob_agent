"""SSE 帧序列化。统一走 data-only 通道，type 放在 JSON 内。"""
from __future__ import annotations

import json
from typing import Any, Mapping


def sse_frame(payload: Mapping[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
