from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "README.md"
OUTPUT = ROOT / "README_RENDERED.html"


def inline(text: str) -> str:
    placeholders: dict[str, str] = {}

    def hold(fragment: str) -> str:
        key = f"@@HTML{len(placeholders)}@@"
        placeholders[key] = fragment
        return key

    def render_math(expression: str) -> str:
        expression = html.escape(expression.strip(), quote=False)
        expression = expression.replace(r"\cup", "∪").replace(r"\setminus", "∖")
        expression = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", expression)
        expression = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", expression)
        expression = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", expression)
        expression = re.sub(r"\^([A-Za-z0-9]+)", r"<sup>\1</sup>", expression)
        return f'<span class="math" role="math">{expression}</span>'

    text = re.sub(
        r"(?<!\\)\$([^$\n]+)\$",
        lambda m: hold(render_math(m.group(1))),
        text,
    )

    text = re.sub(
        r"\[!\[([^]]*)\]\(([^)]+)\)\]\(([^)]+)\)",
        lambda m: hold(
            f'<a href="{html.escape(m.group(3), quote=True)}">'
            f'<img src="{html.escape(m.group(2), quote=True)}" '
            f'alt="{html.escape(m.group(1), quote=True)}"></a>'
        ),
        text,
    )
    text = re.sub(
        r"!\[([^]]*)\]\(([^)]+)\)",
        lambda m: hold(
            f'<img src="{html.escape(m.group(2), quote=True)}" '
            f'alt="{html.escape(m.group(1), quote=True)}">'
        ),
        text,
    )
    text = re.sub(
        r"\[([^]]+)\]\(([^)]+)\)",
        lambda m: hold(
            f'<a href="{html.escape(m.group(2), quote=True)}">'
            f'{html.escape(m.group(1))}</a>'
        ),
        text,
    )
    text = re.sub(
        r"`([^`]+)`",
        lambda m: hold(f"<code>{html.escape(m.group(1))}</code>"),
        text,
    )
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = text.replace("\\*", "*")
    text = re.sub(r"&lt;(/?(?:sub|sup|kbd|br))\s*/?&gt;", r"<\1>", text)
    for key, fragment in placeholders.items():
        text = text.replace(key, fragment)
    return text


def render(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    i = 0
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{inline(' '.join(part.strip() for part in paragraph))}</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            if not in_code:
                language = stripped[3:].strip()
                cls = f' class="language-{html.escape(language, quote=True)}"' if language else ""
                out.append(f"<pre><code{cls}>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            i += 1
            continue

        if in_code:
            out.append(html.escape(line) + "\n")
            i += 1
            continue

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("<") and stripped.endswith(">"):
            flush_paragraph()
            out.append(line)
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2)
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            out.append(f'<h{level} id="{slug}">{inline(title)}</h{level}>')
            i += 1
            continue

        if stripped in {"---", "***", "___"}:
            flush_paragraph()
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue

        if (
            "|" in stripped
            and i + 1 < len(lines)
            and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1])
        ):
            flush_paragraph()
            header_cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header_cells) + "</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
            continue

        list_match = re.match(r"^(?:[-*+]\s+|(\d+)\.\s+)(.+)$", stripped)
        if list_match:
            flush_paragraph()
            ordered = list_match.group(1) is not None
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while i < len(lines):
                current = re.match(r"^(?:[-*+]\s+|(\d+)\.\s+)(.+)$", lines[i].strip())
                if not current or (current.group(1) is not None) != ordered:
                    break
                items.append(current.group(2))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(item)}</li>" for item in items) + f"</{tag}>")
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph()
    return "\n".join(out)


body = render(SOURCE.read_text(encoding="utf-8"))
document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SharP — README Review</title>
<style>
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: #1f2328; background: #f6f8fa; font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
.page {{ max-width: 1040px; margin: 32px auto; padding: 46px 64px 72px; background: #fff; border: 1px solid #d0d7de; border-radius: 12px; box-shadow: 0 8px 28px rgba(140,149,159,.18); }}
h1, h2, h3 {{ line-height: 1.25; margin-top: 32px; margin-bottom: 16px; font-weight: 650; }}
h1 {{ font-size: 2.25em; }} h2 {{ padding-bottom: .3em; border-bottom: 1px solid #d8dee4; }} h3 {{ font-size: 1.25em; }}
p, blockquote, ul, ol, table, pre {{ margin-top: 0; margin-bottom: 16px; }}
a {{ color: #0969da; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
img {{ display: block; max-width: 100%; height: auto; margin: 24px auto; }}
div[align="center"] img {{ display: inline-block; margin: 2px; }}
p[align="center"] {{ color: #57606a; max-width: 880px; margin: -10px auto 24px; }}
code {{ padding: .2em .4em; background: rgba(175,184,193,.2); border-radius: 6px; font: 85% ui-monospace, SFMono-Regular, Consolas, monospace; }}
.math {{ display: inline-block; padding: 0 .12em; color: #172554; font: italic 1.06em/1.1 "Cambria Math", "STIX Two Math", "Times New Roman", serif; white-space: nowrap; }}
.math sub, .math sup {{ font-size: .72em; line-height: 0; }}
pre {{ overflow: auto; padding: 18px; background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 8px; }}
pre code {{ padding: 0; background: transparent; font-size: 13px; }}
blockquote {{ padding: 8px 16px; color: #57606a; border-left: 4px solid #d0d7de; background: #f6f8fa; }} blockquote p {{ margin: 0; }}
table {{ width: 100%; border-spacing: 0; border-collapse: collapse; display: block; overflow: auto; }} th, td {{ padding: 7px 13px; border: 1px solid #d0d7de; }} th {{ background: #f6f8fa; font-weight: 600; }} tr:nth-child(even) td {{ background: #f6f8fa; }}
hr {{ height: 1px; margin: 30px 0; border: 0; background: #d8dee4; }}
@media (max-width: 720px) {{ .page {{ margin: 0; padding: 24px 18px 48px; border: 0; border-radius: 0; }} }}
</style>
</head>
<body><main class="page markdown-body">{body}</main></body>
</html>
"""
OUTPUT.write_text(document, encoding="utf-8")
print(f"[OK] Rendered {SOURCE.name} -> {OUTPUT.name} ({OUTPUT.stat().st_size:,} bytes)")

local_images = [
    source
    for source in re.findall(r'<img src="([^"]+)"', document)
    if not source.startswith(("http://", "https://"))
]
missing_images = [source for source in local_images if not (ROOT / source).exists()]
if missing_images:
    raise SystemExit("[FAIL] Missing rendered images: " + ", ".join(missing_images))
print("[OK] Rendered local images: " + ", ".join(local_images))
