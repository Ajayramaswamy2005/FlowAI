"""FlowAI's visual language: deep-water colours, a rail of glyphs, and moving water.

Everything the interface draws comes from here, so the look can be changed
in one place. Each glyph has an ASCII twin for terminals that cannot show
Unicode (old Windows code pages, some CI logs).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from rich.text import Text

# Deep-water palette.
TIDE = "#2dd4bf"      # the agent, the primary accent
FOAM = "#e6fffb"      # bright text
DEPTH = "#64748b"     # quiet text: labels, timings
SHALLOW = "#94a3b8"   # secondary text
KELP = "#a78bfa"      # reasoning (the undertow)
CORAL = "#fb923c"     # tools
SUN = "#fbbf24"       # notices, medium risk
EMBER = "#f43f5e"     # errors, critical risk
MOSS = "#34d399"      # success, low risk
YOU = "#38bdf8"       # the user

#: The wordmark and wave run through these, left to right.
CURRENT = ["#22d3ee", "#2dd4bf", "#34d399", "#60a5fa", "#818cf8", "#c084fc"]

RISK_COLOURS = {"low": MOSS, "medium": SUN, "high": CORAL, "critical": EMBER}

WORDMARK = [
    "█▀▀ █   █▀█ █ █ █ ▄▀█ █",
    "█▀  █▄▄ █▄█ ▀▄▀▄▀ █▀█ █",
]
WORDMARK_ASCII = ["F L O W A I"]


@dataclass(frozen=True)
class Glyphs:
    rail: str
    you: str
    agent: str
    tool: str
    ok: str
    fail: str
    think: str
    gate: str
    end: str
    notice: str
    shield: str
    dot: str
    arrow: str
    levels: str
    unicode: bool


UNICODE = Glyphs(
    rail="│", you="◆", agent="●", tool="◇", ok="✓", fail="✗", think="≋", gate="┃", end="╰─",
    notice="△", shield="◈", dot="·", arrow="›", levels="▁▂▃▄▅▆▇█", unicode=True,
)
ASCII = Glyphs(
    rail="|", you="*", agent="o", tool=">", ok="ok", fail="x", think="~", gate="#", end="`-",
    notice="!", shield="#", dot="-", arrow=">", levels="_.-~^", unicode=False,
)


def glyphs_for(encoding: str | None) -> Glyphs:
    return UNICODE if (encoding or "").lower().replace("-", "").startswith("utf") else ASCII


# ------------------------------------------------------------------- colour


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def blend(a: str, b: str, t: float) -> str:
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return f"#{round(ra + (rb - ra) * t):02x}{round(ga + (gb - ga) * t):02x}{round(ba + (bb - ba) * t):02x}"


def along(colours: list[str], position: float) -> str:
    """The colour at `position` (0..1, wrapping) along a multi-stop gradient."""
    position %= 1.0
    span = position * (len(colours) - 1)
    i = min(int(span), len(colours) - 2)
    return blend(colours[i], colours[i + 1], span - i)


def current_text(text: str, phase: float = 0.0, bold: bool = True) -> Text:
    """Text painted with the current gradient; `phase` moves the colour along it."""
    out = Text()
    width = max(len(text), 1)
    for i, ch in enumerate(text):
        style = along(CURRENT, i / (width * 1.25) + phase)
        out.append(ch, style=f"bold {style}" if bold else style)
    return out


def wave(width: int, glyphs: Glyphs, t: float | None = None, speed: float = 6.0) -> Text:
    """A moving swell drawn with block heights: the 'working' indicator."""
    t = time.monotonic() if t is None else t
    levels = glyphs.levels
    out = Text()
    for i in range(width):
        height = (math.sin(i * 0.7 - t * speed) + 1) / 2
        ch = levels[round(height * (len(levels) - 1))]
        out.append(ch, style=along(CURRENT, i / max(width, 1) * 0.6 + t * 0.15))
    return out


def gauge(fraction: float, glyphs: Glyphs) -> tuple[str, str]:
    """One block showing how full the context window is, and its colour."""
    fraction = max(0.0, min(1.0, fraction))
    levels = glyphs.levels
    ch = levels[min(len(levels) - 1, int(fraction * len(levels)))]
    colour = MOSS if fraction < 0.5 else SUN if fraction < 0.8 else EMBER
    return ch, colour


def human_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def seconds(s: float) -> str:
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{int(s % 60):02d}s"
