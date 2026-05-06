"""Builtin tools shipped with agent_core.

Seven read-only shell-style tools (vault-scoped, pure-Python) plus five
tools backed by framework managers already wired by run_daemon. All can be
opted out via the agent's `disabled_builtins` ClassVar.
"""
from agent_core.tools.base import Tool
from agent_core.tools._framework import (
    AddLearning,
    FetchUrl,
    SearchVault,
    SearchWeb,
    UpdateScratch,
)
from agent_core.tools._shell import Cat, Find, Grep, Head, Ls, ReadLines, Tail


BUILTIN_TOOLS: list[type[Tool]] = [
    Cat, Head, Tail, Ls, Grep, Find, ReadLines,
    FetchUrl, SearchVault, SearchWeb, UpdateScratch, AddLearning,
]
