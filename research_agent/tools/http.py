"""Safe HTTP GET, shared by fetch_page and read_rss_feed.

Once the LLM can choose *any* URL to fetch, the URL is untrusted input. Text
on a web page could trick the model into fetching http://localhost:8080/admin
or a cloud metadata address. That class of attack is called SSRF
(server-side request forgery). So before every request, and before every
redirect, we check:
  * the scheme is http or https (no file://, ftp://, ...);
  * the host resolves only to public IP addresses (no 127.0.0.1, 10.x, 192.168.x, ...).
We also cap the download size and the time, so one huge or slow page can't
stall the run or flood the LLM's context.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests

USER_AGENT = "ResearchAgent/0.2 (educational research bot)"
REDIRECT_CODES = {301, 302, 303, 307, 308}


class FetchError(Exception):
    """Any reason a fetch was refused or failed. The message is safe to show the LLM."""


@dataclass
class FetchedResponse:
    url: str            # final URL, after redirects
    status: int
    content_type: str   # e.g. "text/html" (parameters such as charset removed)
    body: bytes
    headers: dict[str, str]


def validate_public_url(url: str) -> None:
    """Raise FetchError unless `url` is http(s) and points only at public addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError(f"Only http/https URLs are allowed, got '{parsed.scheme or 'none'}'.")
    host = parsed.hostname
    if not host:
        raise FetchError("URL has no host.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(f"Could not resolve host '{host}'.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])  # strip IPv6 zone id
        if not address.is_global:
            raise FetchError(f"Refusing to fetch '{host}': it resolves to a non-public address.")


def safe_get(url: str, *, timeout: float, max_bytes: int, max_redirects: int = 5) -> FetchedResponse:
    """GET a public URL with validated redirects, a time limit and a size limit."""
    current = url
    for _ in range(max_redirects + 1):
        validate_public_url(current)
        try:
            response = requests.get(current, timeout=timeout, stream=True, allow_redirects=False,
                                    headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        except requests.RequestException as exc:
            raise FetchError(f"Request failed: {type(exc).__name__}: {exc}") from exc

        with response:
            # We follow redirects ourselves, so every hop is validated.
            if response.status_code in REDIRECT_CODES and response.headers.get("Location"):
                current = urljoin(current, response.headers["Location"])
                continue
            if response.status_code >= 400:
                raise FetchError(f"HTTP {response.status_code} from {current}")

            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise FetchError(f"Response is {int(declared):,} bytes, over the {max_bytes:,} byte limit.")

            chunks, size = [], 0
            try:
                for chunk in response.iter_content(chunk_size=65_536):
                    size += len(chunk)
                    if size > max_bytes:
                        raise FetchError(f"Response exceeded the {max_bytes:,} byte limit.")
                    chunks.append(chunk)
            except requests.RequestException as exc:
                raise FetchError(f"Download failed: {type(exc).__name__}: {exc}") from exc

            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            return FetchedResponse(url=current, status=response.status_code, content_type=content_type,
                                   body=b"".join(chunks), headers=dict(response.headers))

    raise FetchError(f"Too many redirects (more than {max_redirects}).")
