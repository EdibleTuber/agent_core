"""run_daemon: the agent_core entry point.

Constructs framework managers from BaseConfig, populates them on the agent,
calls Agent.setup() to let the agent construct domain-specific resources,
then starts the daemon.
"""
from __future__ import annotations

import asyncio
import logging

from agent_core.agent import Agent
from agent_core.allowlist import AllowlistManager
from agent_core.approval_registry import ApprovalRegistry
from agent_core.channels import ChannelStore
from agent_core.config import BaseConfig, load_config
from agent_core.daemon import Daemon
from agent_core.inference import InferenceClient
from agent_core.learning import LearningManager
from agent_core.profile import ProfileManager
from agent_core.retrieval import RetrievalClient
from agent_core.websearch import WebSearchClient
from agent_core.wisdom import WisdomManager

logger = logging.getLogger(__name__)


def run_daemon(
    agent: Agent, config_cls: type[BaseConfig] = BaseConfig,
) -> None:
    """Construct managers, wire onto agent, call setup, start daemon."""
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
    agent.setup()

    logging.basicConfig(level=logging.INFO)
    daemon = Daemon(agent)
    asyncio.run(daemon.serve())
