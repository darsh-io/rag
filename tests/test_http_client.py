"""HTTP retry client unit tests using mocked responses."""
import pytest
import requests as req_lib
from unittest.mock import patch, MagicMock
from src.http_client import post_with_retry, post_with_retry_stream


def _mock_response(status_code, json_body=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.headers = headers or {}
    return resp


def test_success_on_first_try():
    with patch("src.http_client.requests.post", return_value=_mock_response(200, {"ok": True})) as mock_post:
        r = post_with_retry("http://test", {}, "{}")
        assert r.status_code == 200
        assert mock_post.call_count == 1


def test_retries_on_429():
    responses = [_mock_response(429), _mock_response(429), _mock_response(200, {"ok": True})]
    with patch("src.http_client.requests.post", side_effect=responses):
        with patch("src.http_client.time.sleep"):
            r = post_with_retry("http://test", {}, "{}")
            assert r.status_code == 200


def test_retries_on_503():
    responses = [_mock_response(503), _mock_response(200, {"data": []})]
    with patch("src.http_client.requests.post", side_effect=responses):
        with patch("src.http_client.time.sleep"):
            r = post_with_retry("http://test", {}, "{}")
            assert r.status_code == 200


def test_max_retries_exhausted():
    responses = [_mock_response(503)] * 4
    with patch("src.http_client.requests.post", side_effect=responses):
        with patch("src.http_client.time.sleep"):
            r = post_with_retry("http://test", {}, "{}")
            assert r.status_code == 503


def test_retry_after_header_respected():
    responses = [
        _mock_response(429, headers={"Retry-After": "0.01"}),
        _mock_response(200, {"ok": True}),
    ]
    with patch("src.http_client.requests.post", side_effect=responses):
        with patch("src.http_client.time.sleep") as mock_sleep:
            r = post_with_retry("http://test", {}, "{}")
            assert r.status_code == 200
            mock_sleep.assert_called_once_with(0.01)


def test_retries_on_request_exception():
    responses = [
        req_lib.exceptions.ConnectionError("connection failed"),
        _mock_response(200, {"ok": True}),
    ]
    with patch("src.http_client.requests.post", side_effect=responses):
        with patch("src.http_client.time.sleep"):
            r = post_with_retry("http://test", {}, "{}")
            assert r.status_code == 200


def test_raises_after_all_exceptions():
    exc = req_lib.exceptions.ConnectionError("always fails")
    with patch("src.http_client.requests.post", side_effect=[exc] * 4):
        with patch("src.http_client.time.sleep"):
            with pytest.raises(req_lib.exceptions.ConnectionError):
                post_with_retry("http://test", {}, "{}")


def test_stream_success_on_first_try():
    with patch("src.http_client.requests.post", return_value=_mock_response(200)) as mock_post:
        r = post_with_retry_stream("http://test", {}, "{}")
        assert r.status_code == 200
        assert mock_post.call_count == 1


def test_stream_retries_on_503():
    responses = [_mock_response(503), _mock_response(200)]
    with patch("src.http_client.requests.post", side_effect=responses):
        with patch("src.http_client.time.sleep"):
            r = post_with_retry_stream("http://test", {}, "{}")
            assert r.status_code == 200


def test_stream_retries_on_exception():
    exc = req_lib.exceptions.ConnectionError("fail")
    with patch("src.http_client.requests.post", side_effect=[exc, _mock_response(200)]):
        with patch("src.http_client.time.sleep"):
            r = post_with_retry_stream("http://test", {}, "{}")
            assert r.status_code == 200


def test_stream_raises_after_all_exceptions():
    exc = req_lib.exceptions.ConnectionError("always")
    with patch("src.http_client.requests.post", side_effect=[exc] * 4):
        with patch("src.http_client.time.sleep"):
            with pytest.raises(req_lib.exceptions.ConnectionError):
                post_with_retry_stream("http://test", {}, "{}")


def test_stream_max_retries_exhausted_5xx():
    responses = [_mock_response(503)] * 4
    with patch("src.http_client.requests.post", side_effect=responses):
        with patch("src.http_client.time.sleep"):
            r = post_with_retry_stream("http://test", {}, "{}")
            assert r.status_code == 503
