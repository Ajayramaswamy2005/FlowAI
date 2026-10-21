"""Web tools: a real search over DuckDuckGo's HTML endpoint, and a page fetcher.

Neither tool invents results. If the network call fails, the agent is told it
failed, which is far more useful to it than a plausible-looking fabrication.
"""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from flowai.tools.decorator import tool

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (compatible; FlowAI/0.1; +https://github.com/Ajayramaswamy2005/FlowAI)"

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


@tool("fetch_url", description="Fetch a web page and return its visible text.")
async def fetch_url(url: str, max_chars: int = 8000) -> str:
    """Download a URL and strip it down to readable text."""
    if not url.startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://."

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Fetch failed ({type(exc).__name__}: {exc})."

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        return f"Error: {url} returned {content_type or 'an unknown content type'}, which is not readable text."

    text = _strip_tags(_SCRIPT_RE.sub(" ", response.text))
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text))
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated at {max_chars} characters)"
    return text or f"{url} returned no readable text."
