"""MkDocs hook: show each EDA report as one page per stage on the site.

The reports themselves (docs/projects/*/index.md) are NOT changed: they stay the
single-file deliverable. At build time this hook reads each report, splits it at
its level-2 headings (## 0. Proposal ... ## 5. Synthesis, ## Appendix) and adds
one generated page per stage next to it (stage-0.md ... stage-5.md, appendix.md),
so relative image paths keep working. The one-page report stays published as
"Full report" and is excluded from search, so results point to the stage pages.
"""

import re
from pathlib import Path

from mkdocs.structure.files import File

REPORTS = {
    "projects/eda/index.md": "Classification",
    "projects/eda-regression/index.md": "Regression",
}
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
COMMENT = re.compile(r"<!--.*?-->\n?", re.S)


def split_sections(text):
    """Return (preamble, [(heading_line, body), ...]) split at '## ' outside code fences."""
    sections, current, preamble, in_fence = [], None, [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            current = [line, []]
            sections.append(current)
        elif current is None:
            preamble.append(line)
        else:
            current[1].append(line)
    return "".join(preamble), [(h, "".join(b)) for h, b in sections]


def promote(body):
    """Raise every heading outside code fences by one level (### -> ##)."""
    out, in_fence = [], False
    for line in body.splitlines(keepends=True):
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and re.match(r"^#{3,6} ", line):
            line = line[1:]
        out.append(line)
    return "".join(out)


def stage_name(heading):
    m = re.match(r"## (\d)\.", heading)
    return f"stage-{m.group(1)}.md" if m else "appendix.md"


def build_pages(src_uri, text, label):
    text = COMMENT.sub("", FRONT_MATTER.sub("", text), count=1)
    preamble, sections = split_sections(text)
    folder = src_uri.rsplit("/", 1)[0]
    pages = {}
    for i, (heading, body) in enumerate(sections):
        title = heading[3:].strip()
        # The report's own H1 is replaced by the stage title; its intro note stays on stage 0.
        intro = re.sub(r"^# .*\n", "", preamble, flags=re.M).strip()
        intro = intro + "\n\n" if i == 0 and intro else ""
        tip = (f'!!! tip ""\n    {label} EDA, section {i + 1} of {len(sections)}. '
               f"Use **Next** at the bottom to keep reading, or open the "
               f"[full report on one page](index.md).\n\n")
        pages[f"{folder}/{stage_name(heading)}"] = (
            f"---\ntitle: \"{label}: {title}\"\n---\n\n# {title}\n\n{tip}{intro}{promote(body).strip()}\n"
        )
    return pages


def on_files(files, config):
    docs = Path(config["docs_dir"])
    for src_uri, label in REPORTS.items():
        text = (docs / src_uri).read_text(encoding="utf-8")
        for uri, content in build_pages(src_uri, text, label).items():
            files.append(File.generated(config, uri, content=content))
    return files


def on_page_markdown(markdown, page, config, files):
    # The one-page reports duplicate the stage pages: keep them out of search.
    if page.file.src_uri in REPORTS:
        page.meta["search"] = {"exclude": True}
    return markdown
