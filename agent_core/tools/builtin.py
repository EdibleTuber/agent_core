"""Builtin tool list. Populated as tools are added in subsequent tasks."""
from agent_core.tools.base import Tool

BUILTIN_TOOLS: list[type[Tool]] = []
