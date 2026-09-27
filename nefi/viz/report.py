"""Shareable reports: one self-contained HTML file (figures embedded as base64, metrics and
timing tables, config hash, environment) or a Markdown page with relative image links.

Example::

    from nefi.viz.report import Section, html_report

    html_report(
        "runs/demo",
        "toy1d demo",
        [
            Section("Reconstruction", "GT vs neural field.", images=["runs/demo/compare.png"]),
            Section("Metrics", table=[{"method": "nefi", "psnr": 31.2}]),
        ],
        config_hash=result.config_hash,
    )

The HTML has no external dependencies (no scripts, no web fonts); it follows the reader's light /
dark preference for the page chrome, while figures keep the theme they were rendered in.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import json
import logging
import math
import os
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("nefi")

_MIME = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


@dataclass
class Section:
    """One report section.

    Attributes:
        title: heading.
        text: paragraphs (blank-line separated; ``**bold**`` and ```code``` are rendered).
        images: image paths, ``(path, caption)`` pairs, ``{"path", "caption", "wide"}`` dicts or
            matplotlib figures (rendered to PNG).
        table: list of dict rows (union of keys = columns unless ``columns`` is given).
        columns: column order for ``table``.
        html: trusted raw HTML appended to the section.
        code: preformatted text (e.g. a config or a markdown table).
        subsections: nested sections.
        id: anchor (default: slug of the title).
    """

    title: str
    text: str = ""
    images: list[Any] = field(default_factory=list)
    table: Sequence[Mapping[str, Any]] | None = None
    columns: Sequence[str] | None = None
    html: str = ""
    code: str = ""
    subsections: list[Section] = field(default_factory=list)
    id: str | None = None


def _as_section(s: Section | Mapping[str, Any]) -> Section:
    if isinstance(s, Section):
        return s
    d = dict(s)
    subs = [_as_section(x) for x in d.pop("subsections", []) or []]
    return Section(**d, subsections=subs)


def slug(text: str) -> str:
    """URL-fragment slug of a heading."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(text).strip().lower()).strip("-")
    return s or "section"


# ---------------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------------
def _git_revision(start: Path | None = None) -> str | None:
    """Commit of the enclosing git checkout, read from ``.git`` files (no git command)."""
    try:
        here = (start or Path(__file__)).resolve()
        for d in [here, *here.parents]:
            g = d / ".git"
            if not g.is_dir():
                continue
            head = (g / "HEAD").read_text().strip()
            if not head.startswith("ref:"):
                return head[:12]
            ref = head.split(":", 1)[1].strip()
            branch = ref.rsplit("/", 1)[-1]
            rf = g / ref
            if rf.exists():
                return f"{rf.read_text().strip()[:12]} ({branch})"
            packed = g / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return f"{line.split()[0][:12]} ({branch})"
            return f"({branch}, no commits)"
    except OSError:
        return None
    return None


def environment_info(device: Any = None) -> dict[str, Any]:
    """Versions, platform and accelerator info for reproducibility sections."""
    info: dict[str, Any] = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import nefi

        info["nefi"] = getattr(nefi, "__version__", "?")
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())
        if device is not None:
            from ..utils.device import resolve_device

            info["device"] = str(resolve_device(device))
    except ImportError:  # pragma: no cover
        pass
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            continue
    rev = _git_revision()
    if rev:
        info["git_revision"] = rev
    return info


# ---------------------------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------------------------
def fmt_value(v: Any, precision: int = 4) -> str:
    """Compact cell text: numbers to ``precision`` significant digits, NaN/None → ``—``."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        if math.isinf(v):
            return "∞" if v > 0 else "−∞"
        if v != 0 and (abs(v) >= 1e5 or abs(v) < 1e-3):
            return f"{v:.{max(1, precision - 2)}e}"
        return f"{v:.{precision}g}"
    if isinstance(v, list | tuple):
        if all(isinstance(x, int) for x in v):
            return "×".join(str(x) for x in v)
        return ", ".join(fmt_value(x, precision) for x in v)
    if isinstance(v, Mapping):
        return ", ".join(f"{k}={fmt_value(x, precision)}" for k, x in v.items())
    return str(v)


def _is_num(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _columns(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None) -> list[str]:
    if columns:
        return list(columns)
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def _inline(text: str) -> str:
    """Escape, then render ``**bold**`` and ```code``` spans."""
    t = html.escape(text)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    return t


def _paragraphs(text: str) -> str:
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        if all(ln.lstrip().startswith(("- ", "* ")) for ln in lines):
            items = "".join(f"<li>{_inline(ln.lstrip()[2:])}</li>" for ln in lines)
            out.append(f"<ul>{items}</ul>")
        else:
            out.append(f"<p>{_inline(' '.join(ln.strip() for ln in lines))}</p>")
    return "\n".join(out)


def html_table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None, precision: int = 4
) -> str:
    """HTML table (numbers right-aligned, tabular figures; status cells get a ✓ / ✗ glyph)."""
    rows = list(rows)
    if not rows:
        return "<p class='muted'>(no rows)</p>"
    cols = _columns(rows, columns)
    head = "".join(
        f"<th class='{'num' if any(_is_num(r.get(c)) for r in rows) else ''}'>"
        f"{html.escape(str(c))}</th>"
        for c in cols
    )
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            txt = html.escape(fmt_value(v, precision))
            cls = "num" if _is_num(v) else ""
            if c == "status" and isinstance(v, str):
                glyph = {"ok": "✓", "failed": "✗", "skipped": "–"}.get(v, "")
                txt = f"<span class='status {html.escape(v)}'>{glyph} {html.escape(v)}</span>"
            cells.append(f"<td class='{cls}'>{txt}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def markdown_table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None, precision: int = 4
) -> str:
    """GitHub-flavoured markdown table."""
    rows = list(rows)
    if not rows:
        return "_(no rows)_"
    cols = _columns(rows, columns)
    align = ["---:" if any(_is_num(r.get(c)) for r in rows) else ":---" for c in cols]
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        cells = [fmt_value(r.get(c), precision).replace("|", "\\|") for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _image_items(images: Sequence[Any]) -> list[dict[str, Any]]:
    out = []
    for im in images or []:
        if isinstance(im, Mapping):
            out.append(dict(im))
        elif isinstance(im, tuple | list) and len(im) == 2:
            out.append({"path": im[0], "caption": im[1]})
        else:
            out.append({"path": im, "caption": None})
    return out


def _image_bytes(obj: Any) -> tuple[bytes, str] | None:
    """(bytes, mime) of a path or a matplotlib figure; ``None`` when missing."""
    if hasattr(obj, "savefig"):
        buf = io.BytesIO()
        obj.savefig(buf, format="png", dpi=110)
        return buf.getvalue(), "image/png"
    p = Path(obj)
    if not p.exists():
        return None
    return p.read_bytes(), _MIME.get(p.suffix.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------
_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --hairline: #e1e0d9; --axis: #c3c2b7; --accent: #2a78d6; --good: #006300; --bad: #d03b3b;
  --code: #f0efec;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --hairline: #2c2c2a; --axis: #383835; --accent: #3987e5; --good: #0ca30c; --bad: #e66767;
    --code: #262624;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
  --hairline: #2c2c2a; --axis: #383835; --accent: #3987e5; --good: #0ca30c; --bad: #e66767;
  --code: #262624;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
header h1 { font-size: 26px; margin: 0 0 4px; font-weight: 600; }
header .subtitle { color: var(--ink2); margin: 0 0 12px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 16px; }
.chip {
  font-size: 12px; color: var(--ink2); background: var(--surface);
  border: 1px solid var(--hairline); border-radius: 999px; padding: 2px 10px;
}
nav.toc { background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px;
  padding: 10px 16px; margin: 12px 0 24px; font-size: 14px; }
nav.toc ol { margin: 4px 0; padding-left: 20px; }
nav.toc a { color: var(--accent); text-decoration: none; }
section { margin: 28px 0; }
h2 { font-size: 20px; font-weight: 600; margin: 0 0 8px; padding-top: 8px;
  border-top: 1px solid var(--hairline); }
h3 { font-size: 16px; font-weight: 600; margin: 18px 0 6px; }
p, li { color: var(--ink); max-width: 80ch; }
.muted { color: var(--muted); }
code { background: var(--code); border-radius: 4px; padding: 0 4px; font-size: 13px;
  overflow-wrap: anywhere; }
pre { background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px;
  padding: 10px 12px; overflow-x: auto; font-size: 12.5px; line-height: 1.45; }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr));
  gap: 14px; align-items: start; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--hairline);
  border-radius: 8px; padding: 8px; }
figure.wide { grid-column: 1 / -1; }
figure img { display: block; max-width: 100%; height: auto; margin: 0 auto; }
figcaption { font-size: 13px; color: var(--ink2); margin-top: 6px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--hairline); border-radius: 8px;
  background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13px;
  font-variant-numeric: tabular-nums; }
th, td { padding: 5px 10px; border-bottom: 1px solid var(--hairline); text-align: left;
  white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; background: var(--surface); }
td.num, th.num { text-align: right; }
tbody tr:last-child td { border-bottom: none; }
.status.ok { color: var(--good); }
.status.failed { color: var(--bad); }
footer { margin-top: 40px; font-size: 12px; color: var(--muted); }
@media (max-width: 600px) { header h1 { font-size: 22px; } body { font-size: 14px; } }
"""


def _render_images(items: list[dict[str, Any]], out_dir: Path, embed: bool) -> str:
    figs = []
    for it in items:
        src = it.get("path")
        cap = it.get("caption")
        wide = " wide" if it.get("wide") else ""
        alt = html.escape(str(cap or (Path(src).stem if isinstance(src, str | Path) else "figure")))
        if embed or hasattr(src, "savefig"):
            got = _image_bytes(src)
            if got is None:
                figs.append(
                    f"<figure class='missing{wide}'><p class='muted'>missing image: "
                    f"{html.escape(str(src))}</p></figure>"
                )
                continue
            data, mime = got
            uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        else:
            p = Path(src)
            if not p.exists():
                figs.append(
                    f"<figure class='missing{wide}'><p class='muted'>missing image: "
                    f"{html.escape(str(src))}</p></figure>"
                )
                continue
            uri = html.escape(os.path.relpath(p.resolve(), out_dir.resolve()))
        capt = f"<figcaption>{_inline(str(cap))}</figcaption>" if cap else ""
        img = f"<img src='{uri}' alt='{alt}' loading='lazy'>"
        figs.append(f"<figure class='{wide.strip()}'>{img}{capt}</figure>")
    return f"<div class='figs'>{''.join(figs)}</div>" if figs else ""


def _render_section(s: Section, out_dir: Path, embed: bool, level: int = 2) -> str:
    sid = s.id or slug(s.title)
    parts = [f"<section id='{html.escape(sid)}'><h{level}>{html.escape(s.title)}</h{level}>"]
    if s.text:
        parts.append(_paragraphs(s.text))
    if s.table is not None:
        parts.append(html_table(s.table, s.columns))
    if s.images:
        parts.append(_render_images(_image_items(s.images), out_dir, embed))
    if s.code:
        parts.append(f"<pre>{html.escape(s.code)}</pre>")
    if s.html:
        parts.append(s.html)
    for sub in s.subsections:
        parts.append(_render_section(sub, out_dir, embed, min(level + 1, 4)))
    parts.append("</section>")
    return "\n".join(parts)


def html_report(
    out_dir: str | Path,
    title: str,
    sections: Sequence[Section | Mapping[str, Any]],
    *,
    filename: str = "index.html",
    subtitle: str | None = None,
    config_hash: str | None = None,
    metrics: Sequence[Mapping[str, Any]] | None = None,
    timings: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    environment: bool | Mapping[str, Any] = True,
    meta: Mapping[str, Any] | None = None,
    embed: bool = True,
    toc: bool = True,
) -> Path:
    """Write a self-contained HTML report and return its path.

    Args:
        out_dir: output directory (created).
        title: page title.
        sections: :class:`Section` objects or dicts with the same keys.
        filename: HTML file name.
        subtitle: line under the title.
        config_hash: shown in the header chips (reproducibility).
        metrics: optional metrics rows rendered first (the table view of the figures).
        timings: ``{name: seconds}`` or rows, rendered after the metrics.
        environment: include :func:`environment_info` (or a given mapping) in the footer.
        meta: extra header chips ``{label: value}``.
        embed: embed images as base64 (self-contained); ``False`` links them relatively.
        toc: table of contents.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    secs = [_as_section(s) for s in sections]
    lead: list[Section] = []
    if metrics:
        lead.append(Section("Metrics", table=list(metrics)))
    if timings:
        rows = (
            [{"stage": k, "seconds": v} for k, v in timings.items()]
            if isinstance(timings, Mapping)
            else list(timings)
        )
        lead.append(Section("Timings", table=rows))
    secs = lead + secs
    env = environment_info() if environment is True else (dict(environment) if environment else {})
    chips = []
    if env.get("created_utc"):
        chips.append(f"created {env['created_utc']}")
    if config_hash:
        chips.append(f"config {config_hash}")
    for k, v in (meta or {}).items():
        chips.append(f"{k} {fmt_value(v)}")
    if env.get("nefi"):
        chips.append(f"nefi {env['nefi']}")
    if env.get("git_revision"):
        chips.append(f"rev {env['git_revision']}")
    head = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{_CSS}</style></head>",
        "<body><main>",
        f"<header><h1>{html.escape(title)}</h1>",
    ]
    if subtitle:
        head.append(f"<p class='subtitle'>{_inline(subtitle)}</p>")
    if chips:
        head.append(
            "<div class='chips'>"
            + "".join(f"<span class='chip'>{html.escape(c)}</span>" for c in chips)
            + "</div>"
        )
    head.append("</header>")
    body = []
    if toc and len(secs) > 1:
        items = "".join(
            f"<li><a href='#{html.escape(s.id or slug(s.title))}'>{html.escape(s.title)}</a></li>"
            for s in secs
        )
        body.append(f"<nav class='toc' aria-label='Contents'><ol>{items}</ol></nav>")
    for s in secs:
        body.append(_render_section(s, out, embed))
    foot = ["<footer>"]
    if env:
        foot.append("<h3>Environment</h3>")
        foot.append(html_table([{"key": k, "value": fmt_value(v)} for k, v in env.items()]))
    foot.append("<p>Generated by <code>nefi.viz.report.html_report</code>.</p></footer>")
    doc = "\n".join(head + body + foot + ["</main></body></html>"])
    path = out / filename
    path.write_text(doc, encoding="utf-8")
    log.info("wrote %s (%.0f kB)", path, path.stat().st_size / 1024)
    return path


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------
def _md_section(s: Section, out: Path, level: int, fig_dir: Path, counter: list[int]) -> str:
    lines = [f"{'#' * level} {s.title}", ""]
    if s.text:
        lines += [s.text.strip(), ""]
    if s.table is not None:
        lines += [markdown_table(s.table, s.columns), ""]
    for it in _image_items(s.images):
        src, cap = it.get("path"), it.get("caption") or ""
        if hasattr(src, "savefig"):
            fig_dir.mkdir(parents=True, exist_ok=True)
            counter[0] += 1
            p = fig_dir / f"{slug(s.title)}_{counter[0]}.png"
            src.savefig(p, dpi=110)
            src = p
        rel = os.path.relpath(Path(src).resolve(), out.resolve())
        lines += [f"![{cap}]({rel})", ""]
        if cap:
            lines += [f"*{cap}*", ""]
    if s.code:
        lines += ["```", s.code.rstrip(), "```", ""]
    for sub in s.subsections:
        lines.append(_md_section(sub, out, min(level + 1, 6), fig_dir, counter))
    return "\n".join(lines)


def markdown_report(
    out_dir: str | Path,
    title: str,
    sections: Sequence[Section | Mapping[str, Any]],
    *,
    filename: str = "report.md",
    subtitle: str | None = None,
    config_hash: str | None = None,
    metrics: Sequence[Mapping[str, Any]] | None = None,
    environment: bool | Mapping[str, Any] = True,
) -> Path:
    """Write a Markdown report (relative image links; figures objects are saved next to it)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    secs = [_as_section(s) for s in sections]
    if metrics:
        secs = [Section("Metrics", table=list(metrics))] + secs
    env = environment_info() if environment is True else (dict(environment) if environment else {})
    lines = [f"# {title}", ""]
    if subtitle:
        lines += [subtitle, ""]
    chips = [f"created {env.get('created_utc')}"] if env.get("created_utc") else []
    if config_hash:
        chips.append(f"config `{config_hash}`")
    if chips:
        lines += [" · ".join(chips), ""]
    counter = [0]
    for s in secs:
        lines.append(_md_section(s, out, 2, out / "_figures", counter))
    if env:
        lines += [
            "## Environment",
            "",
            markdown_table([{"key": k, "value": v} for k, v in env.items()]),
            "",
        ]
    path = out / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_json(path: str | Path, obj: Any) -> Path:
    """JSON writer tolerant of tensors, tuples, paths and non-finite floats."""
    try:
        from ..bench.report import to_jsonable
    except ImportError:  # pragma: no cover

        def to_jsonable(x: Any) -> Any:
            return json.loads(json.dumps(x, default=str))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(obj), indent=2) + "\n")
    return p


__all__ = [
    "Section",
    "environment_info",
    "fmt_value",
    "html_report",
    "html_table",
    "markdown_report",
    "markdown_table",
    "slug",
    "write_json",
]
