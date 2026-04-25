"""YAML frontmatter parsing and serialization for markdown files.

Frontmatter is delimited by --- on its own line at the start of the file.
The opening --- must be the very first line. The closing --- ends the
frontmatter block. Everything after is the body.
"""
import yaml


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Returns (metadata_dict, body_without_frontmatter).
    If no frontmatter is present, returns ({}, original_content).
    """
    if not content.startswith("---"):
        return {}, content

    # Find closing --- (must be on its own line after the opening)
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content

    yaml_str = content[4:end]  # skip opening "---\n"
    body = content[end + 4:]   # skip "\n---"

    try:
        meta = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError:
        return {}, content

    return meta, body


def serialize_frontmatter(meta: dict, body: str) -> str:
    """Serialize metadata and body into a markdown string with YAML frontmatter.

    If meta is empty, returns just the body (no frontmatter block).
    """
    if not meta:
        return body

    yaml_str = yaml.dump(meta, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return f"---\n{yaml_str}---\n{body}"
