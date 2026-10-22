"""Web tools: a real search over DuckDuckGo's HTML endpoint, and a page fetcher.

Neither tool invents results. If the network call fails, the agent is told it
failed, which is far more useful to it than a plausible-looking fabrication.

fetch_url only reaches the public internet. Every hop of a redirect chain is
checked, and hostnames are resolved before connecting, so a page cannot steer
the agent to 127.0.0.1, a home router, or a cloud metadata service that hands
out credentials (169.254.169.254).
"""

from __future__ import annotations

import asyncio
import html
import re
import socket
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

from flowai.security.policy import blocked_address_reason, url_problem
from flowai.tools.decorator import tool

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (compatible; FlowAI/0.3; +https://github.com/Ajayramaswamy2005/FlowAI)"
MAX_REDIRECTS = 5
MAX_DOWNLOAD_BYTES = 3_000_000

_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)


def _strip_tags(markup: str) -> str:
    return html.unescape(_TAG_RE.sub("", markup)).strip()


def _clean_link(href: str) -> str:
    """DuckDuckGo wraps results in a redirect; unwrap it back to the real URL."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


async def public_url_problem(url: str) -> str | None:
    """Why this URL must not be fetched, after resolving its host; None if it is public."""
    problem = url_problem(url)
    if problem:
        return problem
    host = urlparse(url).hostname or ""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return None  # let the request itself report that the name does not resolve
    for info in infos:
        reason = blocked_address_reason(info[4][0])
        if reason:
            return f"{host} resolves to {info[4][0]}: {reason}"
    return None


@tool("search_web", description="Search the web with DuckDuckGo and return the top results.")
async def search_web(query: str, max_results: int = 5) -> str:
    """Run a web search and return titles, URLs and snippets."""
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.post(
                SEARCH_URL,
                data={"q": query},
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            body = response.text
    except httpx.HTTPError as exc:
        return f"Web search failed ({type(exc).__name__}: {exc}). No results were retrieved."

    titles = _RESULT_RE.findall(body)
    snippets = _SNIPPET_RE.findall(body)

    if not titles:
        return f"No results found for '{query}'."

    lines = []
    for index, (href, title) in enumerate(titles[:max_results]):
        snippet = _strip_tags(snippets[index]) if index < len(snippets) else ""
        lines.append(f"{index + 1}. {_strip_tags(title)}\n   {_clean_link(href)}\n   {snippet}".rstrip())
    return "\n".join(lines)


@tool("fetch_url", description="Fetch a public web page and return its visible text.")
async def fetch_url(url: str, max_chars: int = 8000) -> str:
    """Download a URL and strip it down to readable text."""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            for _ in range(MAX_REDIRECTS + 1):
                problem = await public_url_problem(url)
                if problem:
                    return f"Error: refusing to fetch {url}: {problem}."
                response = await client.get(url, headers={"User-Agent": USER_AGENT})
                if response.is_redirect and response.headers.get("location"):
                    url = urljoin(url, response.headers["location"])
                    continue
                break
            else:
                return f"Error: more than {MAX_REDIRECTS} redirects."
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Fetch failed ({type(exc).__name__}: {exc})."

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type and "json" not in content_type:
        return f"Error: {url} returned {content_type or 'an unknown content type'}, which is not readable text."

    raw = response.text[:MAX_DOWNLOAD_BYTES]
    text = _strip_tags(_SCRIPT_RE.sub(" ", raw)) if "html" in content_type else raw
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text))
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated at {max_chars} characters)"
    return text or f"{url} returned no readable text."
