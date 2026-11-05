"""safe_get / validate_public_url tests. DNS and HTTP are faked."""
import socket

import pytest

from research_agent.tools import http
from research_agent.tools.http import FetchError, safe_get, validate_public_url


def fake_dns(mapping):
    def getaddrinfo(host, port, proto=0):
        if host not in mapping:
            raise socket.gaierror("unknown host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mapping[host], port))]
    return getaddrinfo


class FakeResponse:
    def __init__(self, status=200, headers=None, chunks=(b"hello",)):
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self._chunks = chunks

    def iter_content(self, chunk_size):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def dns(monkeypatch):
    monkeypatch.setattr(http.socket, "getaddrinfo", fake_dns({
        "public.org": "93.184.216.34", "localhost": "127.0.0.1",
        "internal.corp": "10.0.0.5", "metadata.cloud": "169.254.169.254",
    }))


@pytest.mark.parametrize("url", [
    "file:///C:/Windows/win.ini", "ftp://public.org/x", "http://localhost/admin",
    "http://internal.corp/", "http://metadata.cloud/latest", "http://nowhere.invalid/", "https://",
])
def test_refuses_unsafe_urls(url):
    with pytest.raises(FetchError):
        validate_public_url(url)


def test_allows_public_url():
    validate_public_url("https://public.org/page")


def test_successful_get(monkeypatch):
    monkeypatch.setattr(http.requests, "get", lambda *a, **kw: FakeResponse(chunks=(b"ab", b"cd")))
    response = safe_get("https://public.org/p", timeout=5, max_bytes=100)
    assert response.body == b"abcd"
    assert response.content_type == "text/html"


def test_redirect_to_private_address_is_blocked(monkeypatch):
    monkeypatch.setattr(http.requests, "get",
                        lambda *a, **kw: FakeResponse(302, {"Location": "http://internal.corp/secret"}))
    with pytest.raises(FetchError, match="non-public"):
        safe_get("https://public.org/p", timeout=5, max_bytes=100)


def test_relative_redirect_is_followed(monkeypatch):
    responses = iter([FakeResponse(301, {"Location": "/new"}), FakeResponse()])
    seen = []
    def get(url, **kw):
        seen.append(url)
        return next(responses)
    monkeypatch.setattr(http.requests, "get", get)
    response = safe_get("https://public.org/old", timeout=5, max_bytes=100)
    assert seen == ["https://public.org/old", "https://public.org/new"]
    assert response.url == "https://public.org/new"


def test_redirect_loop_stops(monkeypatch):
    monkeypatch.setattr(http.requests, "get", lambda *a, **kw: FakeResponse(302, {"Location": "/again"}))
    with pytest.raises(FetchError, match="Too many redirects"):
        safe_get("https://public.org/p", timeout=5, max_bytes=100, max_redirects=3)


def test_size_limit(monkeypatch):
    monkeypatch.setattr(http.requests, "get", lambda *a, **kw: FakeResponse(chunks=(b"x" * 60, b"x" * 60)))
    with pytest.raises(FetchError, match="limit"):
        safe_get("https://public.org/p", timeout=5, max_bytes=100)


def test_http_error_status(monkeypatch):
    monkeypatch.setattr(http.requests, "get", lambda *a, **kw: FakeResponse(404))
    with pytest.raises(FetchError, match="HTTP 404"):
        safe_get("https://public.org/p", timeout=5, max_bytes=100)


def test_network_error_becomes_fetch_error(monkeypatch):
    def boom(*a, **kw):
        raise http.requests.ConnectionError("refused")
    monkeypatch.setattr(http.requests, "get", boom)
    with pytest.raises(FetchError, match="refused"):
        safe_get("https://public.org/p", timeout=5, max_bytes=100)
