"""HTTP client with exponential backoff retry for transient errors."""
import time
import requests

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BASE_DELAY  = 1.0  # seconds


def post_with_retry(url: str, headers: dict, data: str, timeout: int = 60) -> requests.Response:
    """POST with exponential backoff on 429/5xx responses."""
    delay = _BASE_DELAY
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, data=data, timeout=timeout)
            if resp.status_code not in _RETRY_STATUSES:
                return resp
            # Honour Retry-After header if present
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)
            delay *= 2
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
            delay *= 2
    if last_exc:
        raise last_exc
    return resp  # last response even if it was an error status


def post_with_retry_stream(url: str, headers: dict, data: str, timeout: int = 120) -> requests.Response:
    """POST with stream=True and retry on connection errors (not on mid-stream errors)."""
    delay = _BASE_DELAY
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, data=data, timeout=timeout, stream=True)
            if resp.status_code not in _RETRY_STATUSES:
                return resp
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)
            delay *= 2
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
            delay *= 2
    if last_exc:
        raise last_exc
    return resp
