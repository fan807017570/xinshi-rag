"""Markdown GFM 表格：从正文中拆出完整表块，并转为可读文本供向量检索。"""

from __future__ import annotations

import re
from typing import List, Literal, Tuple

Segment = Tuple[Literal["text", "table"], str]

# 典型分隔行：| --- | --- | 或 |:---|:---|
_SEP_RE = re.compile(r"^\|[\s\-:|]+\|\s*$")


def _is_table_row_line(line: str) -> bool:
    s = line.strip()
    if not s or not s.startswith("|"):
        return False
    return s.count("|") >= 2


def _is_table_separator_line(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|") or "|" not in s[1:]:
        return False
    return bool(_SEP_RE.match(s)) or re.match(r"^\|[\s\-:|]+\|", s) is not None


def split_markdown_tables(markdown: str) -> List[Segment]:
    """
    将 markdown 拆成交替的纯文本段与完整表格段（不破坏表格行）。
    忽略代码块 ``` 内的内容（不当作表格）。
    """
    lines = markdown.splitlines()
    segments: List[Segment] = []
    text_buf: List[str] = []
    i = 0
    in_fence = False

    def flush_text() -> None:
        nonlocal text_buf
        if text_buf:
            segments.append(("text", "\n".join(text_buf)))
            text_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            text_buf.append(line)
            i += 1
            continue
        if in_fence:
            text_buf.append(line)
            i += 1
            continue

        if _is_table_row_line(line):
            j = i
            table_lines: List[str] = []
            while j < len(lines):
                lj = lines[j]
                sj = lj.strip()
                if sj == "":
                    break
                if _is_table_row_line(lj) or _is_table_separator_line(lj):
                    table_lines.append(lines[j])
                    j += 1
                    continue
                break
            # 至少两行且像 GFM：表头 + 分隔 或 多行 |
            if len(table_lines) >= 2:
                flush_text()
                segments.append(("table", "\n".join(table_lines)))
                i = j
                continue

        text_buf.append(line)
        i += 1

    flush_text()
    return segments


def _parse_table_row(line: str) -> List[str]:
    inner = line.strip().strip("|")
    return [c.strip() for c in inner.split("|")]


def markdown_table_to_readable(md_table: str) -> str:
    """
    将 GFM 表格转为多行「列名: 值」描述，便于中文向量检索。
    解析失败时返回原表字符串。
    """
    raw_lines = [ln for ln in md_table.strip().splitlines() if ln.strip()]
    if len(raw_lines) < 2:
        return md_table.strip()

    rows: List[List[str]] = []
    for ln in raw_lines:
        if _is_table_separator_line(ln):
            continue
        if not _is_table_row_line(ln):
            continue
        rows.append(_parse_table_row(ln))

    if len(rows) < 2:
        return md_table.strip()

    header = rows[0]
    body = rows[1:]
    if not header or not any(c for c in header):
        return md_table.strip()

    lines_out: List[str] = []
    for b in body:
        pairs = []
        for hi, val in zip(header, b):
            if hi == "" and val == "":
                continue
            col = hi if hi else "项"
            pairs.append(f"{col}：{val}")
        if pairs:
            lines_out.append("；".join(pairs))

    if not lines_out:
        return md_table.strip()

    return "\n".join(lines_out) + f"\n（表格共 {len(body)} 行）"
