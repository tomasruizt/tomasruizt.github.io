#!/usr/bin/env python3
"""Post-process Quarto HTML output to normalize non-deterministic ordering.

Quarto serializes JSON with non-deterministic key order and HTML attributes in
non-deterministic order, causing noisy git diffs. This script:
1. Sorts keys in base64-encoded JSON inside <script> tags
2. Sorts keys in inline JSON objects (e.g. GLightbox config)
3. Sorts HTML attributes alphabetically on tags
"""

import base64
import glob
import json
import re


def sort_json_keys(obj):
    """Recursively sort all keys in a JSON-compatible object."""
    if isinstance(obj, dict):
        return {k: sort_json_keys(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [sort_json_keys(item) for item in obj]
    return obj


def normalize_base64_json(match):
    """Decode base64 JSON, sort keys, re-encode."""
    tag = match.group(1)
    b64 = match.group(2).strip()
    closing = match.group(3)
    try:
        decoded = base64.b64decode(b64).decode("utf-8")
        obj = json.loads(decoded)
        normalized = json.dumps(
            sort_json_keys(obj), separators=(",", ":"), ensure_ascii=False
        )
        new_b64 = base64.b64encode(normalized.encode("utf-8")).decode("utf-8")
        return f"{tag}\n{new_b64}\n{closing}"
    except Exception:
        return match.group(0)


BASE64_SCRIPT_RE = re.compile(
    r"(<script[^>]*>)\s*\n?"
    r"((?:[A-Za-z0-9+/\n]+=*\s*)+)"
    r"(</script>)",
)


def normalize_inline_json(match):
    """Sort keys in inline JSON object literals like GLightbox({...})."""
    prefix = match.group(1)
    json_str = match.group(2)
    suffix = match.group(3)
    try:
        obj = json.loads(json_str)
        normalized = json.dumps(
            sort_json_keys(obj), separators=(",", ":"), ensure_ascii=False
        )
        return f"{prefix}{normalized}{suffix}"
    except Exception:
        return match.group(0)


INLINE_JSON_RE = re.compile(r"(GLightbox\()(\{[^}]+\})(\))")

# Regex to find individual HTML attributes (name="value" or name='value' or bare)
ATTR_RE = re.compile(r"""([\w:.-]+)(?:\s*=\s*("[^"]*"|'[^']*'))?""")


def sort_html_attributes(match):
    """Sort attributes within an HTML tag alphabetically."""
    tag_name = match.group(1)
    attrs_str = match.group(2)
    closing = match.group(3)

    attrs = []
    for m in ATTR_RE.finditer(attrs_str):
        name = m.group(1)
        value = m.group(2)
        if value is not None:
            attrs.append((name, f"{name}={value}"))
        else:
            attrs.append((name, name))

    if not attrs:
        return match.group(0)

    sorted_attrs = " ".join(a[1] for a in sorted(attrs, key=lambda x: x[0]))
    return f"<{tag_name} {sorted_attrs}{closing}"


# Match opening HTML tags with at least one attribute.
# Captures: (1) tag name, (2) all attributes, (3) closing > or />
HTML_TAG_RE = re.compile(
    r"<([a-zA-Z][\w-]*)"  # tag name
    r"(\s+(?:[\w:.-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'))?[\s]*)+)"  # attributes
    r"(/?>)",  # closing
)


def normalize_file(path):
    with open(path, "r") as f:
        content = f.read()

    original = content
    content = BASE64_SCRIPT_RE.sub(normalize_base64_json, content)
    content = INLINE_JSON_RE.sub(normalize_inline_json, content)
    content = HTML_TAG_RE.sub(sort_html_attributes, content)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        print(f"  normalized: {path}")


def main():
    files = glob.glob("docs/**/*.html", recursive=True)
    files += glob.glob("docs/**/*.xml", recursive=True)
    print(f"Normalizing {len(files)} files...")
    for path in sorted(files):
        normalize_file(path)
    print("Done.")


if __name__ == "__main__":
    main()
