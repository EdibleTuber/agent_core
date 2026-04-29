import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from agent_core.learning import LearningManager
from agent_core.learning_scanner import LearningScanner
from agent_core.protocol import LearningCandidateProposalMessage


def test_take_pending_returns_message_on_match(tmp_path: Path):
    lm = LearningManager(tmp_path, "pal")
    emitted: list = []
    extractor = AsyncMock(return_value={"title": "T", "body": "B"})
    scanner = LearningScanner(
        learning_manager=lm,
        extractor=extractor,
        emit=lambda msg: emitted.append(msg),
    )
    asyncio.run(scanner.maybe_scan(
        recent_turns=[],
        latest_user_message="you always merge",
    ))
    assert len(emitted) == 1
    pid = emitted[0].proposal_id
    popped = scanner.take_pending(pid)
    assert isinstance(popped, LearningCandidateProposalMessage)
    assert popped.title == "T"
    # Pending should now be cleared
    assert scanner._pending_id is None


def test_take_pending_returns_none_for_mismatch(tmp_path: Path):
    lm = LearningManager(tmp_path, "pal")
    emitted: list = []
    extractor = AsyncMock(return_value={"title": "T", "body": "B"})
    scanner = LearningScanner(
        learning_manager=lm,
        extractor=extractor,
        emit=lambda msg: emitted.append(msg),
    )
    asyncio.run(scanner.maybe_scan(
        recent_turns=[],
        latest_user_message="you always",
    ))
    popped = scanner.take_pending("wrong-id")
    assert popped is None
    # Pending still pointing at emitted one
    assert scanner._pending_id == emitted[0].proposal_id


def test_take_pending_drains_queue(tmp_path: Path):
    lm = LearningManager(tmp_path, "pal")
    emitted: list = []
    extractor = AsyncMock(return_value={"title": "A", "body": "x"})
    scanner = LearningScanner(
        learning_manager=lm,
        extractor=extractor,
        emit=lambda msg: emitted.append(msg),
    )
    asyncio.run(scanner.maybe_scan(
        recent_turns=[], latest_user_message="you always a",
    ))
    first_id = emitted[0].proposal_id

    extractor.return_value = {"title": "B", "body": "y"}
    asyncio.run(scanner.maybe_scan(
        recent_turns=[], latest_user_message="you always b",
    ))
    # Second candidate queued
    assert len(emitted) == 1
    assert len(scanner.queued) == 1

    # Approve first -> queue drains, second candidate emitted
    popped = scanner.take_pending(first_id)
    assert popped.title == "A"
    assert len(emitted) == 2
    assert emitted[1].title == "B"
