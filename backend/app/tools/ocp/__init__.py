"""OCP 客户端工厂。"""
from __future__ import annotations

from app.config import Settings
from app.tools.ocp.mock import MockOcpClient


def get_ocp_client(settings: Settings):
    if settings.ocp.provider == "real":
        from app.tools.ocp.real import RealOcpClient  # Task 9 实现

        return RealOcpClient(settings.ocp)
    return MockOcpClient()
