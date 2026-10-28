from typing import Optional


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, retryable: bool = False,
                 request_id: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.request_id = request_id


def error_body(code: str, message: str, request_id: str, retryable: bool = False) -> dict:
    return {
        "detail": message,
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "retryable": retryable,
        },
    }
