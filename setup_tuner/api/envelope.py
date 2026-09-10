"""API 响应信封工具 —— 统一 ``{code, message, data}`` 信封格式。

对齐 design.md 2.8.1「统一前缀 /api/v1，响应统一 {code, message, data} 信封；
错误码稳定（4xx 业务失败、5xx 服务异常）」。

提供：
    - :func:`ok`    — 成功响应信封（code=0）
    - :func:`fail`  — 业务失败响应信封（code=非零，HTTP 4xx）
    - :func:`error` — 服务异常响应信封（code=负数，HTTP 5xx）
    - :class:`ApiError` — 业务异常（被全局异常处理器捕获转为 fail 信封）
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 信封模型
# ---------------------------------------------------------------------------
class Envelope(BaseModel):
    """统一响应信封 ``{code, message, data}``。

    - ``code = 0``  表示成功；
    - ``code > 0``  表示业务失败（4xx）；
    - ``code < 0``  表示服务异常（5xx）。
    """

    code: int = Field(default=0, description="0=成功, >0=业务失败, <0=服务异常")
    message: str = Field(default="ok", description="人类可读的提示消息")
    data: Any = Field(default=None, description="业务数据载荷")


# ---------------------------------------------------------------------------
# 信封构造辅助
# ---------------------------------------------------------------------------
def ok(data: Any = None, message: str = "ok") -> dict[str, Any]:
    """构造成功信封。"""
    return {"code": 0, "message": message, "data": data}


def fail(
    message: str,
    code: int = 1,
    data: Any = None,
    http_status: int = 400,
) -> HTTPException:
    """构造业务失败 HTTPException（携带信封 body）。

    Args:
        message: 失败提示消息。
        code: 业务错误码（>0）。
        data: 可选附加数据。
        http_status: HTTP 状态码（4xx）。

    Returns:
        HTTPException，其 detail 为信封 dict。
    """
    body = {"code": code, "message": message, "data": data}
    return HTTPException(status_code=http_status, detail=body)


def error(
    message: str,
    code: int = -1,
    data: Any = None,
    http_status: int = 500,
) -> HTTPException:
    """构造服务异常 HTTPException（携带信封 body）。

    Args:
        message: 异常提示消息。
        code: 服务错误码（<0）。
        data: 可选附加数据。
        http_status: HTTP 状态码（5xx）。

    Returns:
        HTTPException，其 detail 为信封 dict。
    """
    body = {"code": code, "message": message, "data": data}
    return HTTPException(status_code=http_status, detail=body)


# ---------------------------------------------------------------------------
# 业务异常
# ---------------------------------------------------------------------------
class ApiError(Exception):
    """业务异常 —— 被全局异常处理器捕获转为 fail 信封。

    Args:
        message: 失败提示消息。
        code: 业务错误码（>0，默认 1）。
        data: 可选附加数据。
        http_status: HTTP 状态码（4xx，默认 400）。
    """

    def __init__(
        self,
        message: str,
        code: int = 1,
        data: Any = None,
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data
        self.http_status = http_status