"""run_daemon: the agent_core entry point.

Constructs framework managers from BaseConfig, populates them on the agent,
calls Agent.setup() to let the agent construct domain-specific resources,
then attaches tool_executor / command_registry / prompt_builder (validated
against the agent's full attribute set — framework + domain), then starts
the daemon.
"""
from __future__ import annotations

import asyncio
import logging

from agent_core.agent import Agent
from agent_core.allowlist import AllowlistManager
from agent_core.approval_registry import ApprovalRegistry
from agent_core.workers.tool_approval import ToolApprovalRegistry
from agent_core.channels import ChannelStore
from agent_core.config import BaseConfig, load_config
from agent_core.daemon import Daemon
from agent_core.inference import InferenceClient
from agent_core.learning import LearningManager
from agent_core.profile import ProfileManager
from agent_core.retrieval import RetrievalClient
from agent_core.utils.fetcher import URLFetcher
from agent_core.websearch import WebSearchClient
from agent_core.wisdom import WisdomManager

logger = logging.getLogger(__name__)


def _attach_registries(agent) -> None:
    """Build tool_executor, command_registry, prompt_builder from agent's
    class-level tools/commands/disabled_builtins. Validation of `requires`
    happens here, after Agent.setup() has run — so domain managers constructed
    in setup() (e.g. wiki, reorganizer, compiler) are visible to `requires`
    checks. Misconfiguration still fails fast at boot, before any user message.
    """
    from agent_core.commands.registry import CommandRegistry
    from agent_core.prompts.builder import SystemPromptBuilder
    from agent_core.tools.executor import ToolExecutor

    cls = type(agent)
    # Union declarative cls.tools with dynamic register_tools().
    # cls.tools is the historical PAL pattern; register_tools() is new in
    # v1.2.0 for agents that need runtime construction (post-MCP-discovery).
    declared = list(cls.tools)
    dynamic = list(agent.register_tools())
    all_tools = declared + [t for t in dynamic if t not in declared]
    agent.tool_executor = ToolExecutor.build(
        agent,
        all_tools,
        disabled=cls.disabled_builtins,
    )
    # Pre-assign a sentinel so that commands requiring "command_registry"
    # (e.g. Help) pass the hasattr check at build time. The real registry
    # replaces it immediately below.
    agent.command_registry = None
    agent.command_registry = CommandRegistry.build(
        agent,
        list(cls.commands),
        disabled=cls.disabled_builtins,
    )
    agent.prompt_builder = SystemPromptBuilder(
        profile=agent.profile,
        wisdom=agent.wisdom,
        channels=agent.channels,
        tool_executor=agent.tool_executor,
        command_registry=agent.command_registry,
        agent=agent,
    )


def run_daemon(
    agent: Agent, config_cls: type[BaseConfig] = BaseConfig,
) -> None:
    """Construct managers, wire onto agent, call setup, attach registries, start daemon.

    Order:
    1. Framework managers (config, profile, wisdom, etc.) wired onto agent.
    2. agent.setup() — agent constructs domain-specific resources.
    3. _attach_registries() — validates `requires` against full agent state,
       builds tool_executor / command_registry / prompt_builder.
    4. Daemon starts.
    """
    config = load_config(
        config_cls, agent_name=agent.name, env_prefix=agent.env_prefix,
    )
    agent.config = config
    agent.profile = ProfileManager(
        config.vault_path, agent_name=agent.name, username=config.username,
    )
    agent.wisdom = WisdomManager(config.vault_path, agent_name=agent.name)
    agent.learning = LearningManager(config.vault_path, agent_name=agent.name)
    agent.allowlist = AllowlistManager(config.vault_path, agent_name=agent.name)
    agent.approval_registry = ApprovalRegistry()
    agent.tool_approval_registry = ToolApprovalRegistry()
    agent.channels = ChannelStore(
        vault_path=config.vault_path,
        agent_name=agent.name,
        history_depth=config.history_depth,
    )
    agent.inference = InferenceClient(
        base_url=config.inference_url, model=config.model,
    )
    agent.retrieval = RetrievalClient(
        base_url=config.inference_url, collection_id=config.collection_id,
    )
    agent.websearch = WebSearchClient(base_url=config.searxng_url)
    agent.fetcher = URLFetcher(
        max_bytes=config.fetch_max_bytes,
        timeout=config.fetch_timeout,
    )

    agent.setup()

    _attach_registries(agent)

    logging.basicConfig(level=logging.INFO)
    daemon = Daemon(agent)
    asyncio.run(daemon.serve())
