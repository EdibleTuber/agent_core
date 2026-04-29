import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from agent_core.learning import LearningManager
from agent_core.learning_scanner import LearningScanner


def test_scanner_emits_candidate_on_signal(tmp_path: Path):
    lm = LearningManager(tmp_path, "pal")
    emitted: list = []
    extractor = AsyncMock(return_value={"title": "Granularity", "body": "focused"})
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
    assert emitted[0].title == "Granularity"
    assert emitted[0].body == "focused"
    assert emitted[0].trigger_excerpt.startswith("you always merge")


def test_scanner_silent_on_no_signal(tmp_path: Path):
    lm = LearningManager(tmp_path, "pal")
    emitted: list = []
    extractor = AsyncMock(return_value={"title": "x", "body": "y"})
    scanner = LearningScanner(
        learning_manager=lm,
        extractor=extractor,
        emit=lambda msg: emitted.append(msg),
    )
    asyncio.run(scanner.maybe_scan(
        recent_turns=[],
        latest_user_message="tell me about IoT security",
    ))
    assert emitted == []
    extractor.assert_not_called()


def test_scanner_silent_on_duplicate(tmp_path: Path):
    lm = LearningManager(tmp_path, "pal")
    lm.add("Granularity Over Consolidation", "keep focused", source="conversation")
    emitted: list = []
    extractor = AsyncMock(return_value={"title": "Granularity Over Consolidation", "body": "x"})
    scanner = LearningScanner(
        learning_manager=lm,
        extractor=extractor,
        emit=lambda msg: emitted.append(msg),
    )
    asyncio.run(scanner.maybe_scan(
        recent_turns=[],
        latest_user_message="you always merge",
    ))
    assert emitted == []


def test_scanner_silent_when_extractor_returns_none(tmp_path: Path):
    lm = LearningManager(tmp_path, "pal")
    emitted: list = []
    extractor = AsyncMock(return_value=None)
    scanner = LearningScanner(
        learning_manager=lm,
        extractor=extractor,
        emit=lambda msg: emitted.append(msg),
    )
    asyncio.run(scanner.maybe_scan(
        recent_turns=[],
        latest_user_message="you always merge",
    ))
    assert emitted == []


def test_scanner_queues_while_proposal_pending(tmp_path: Path):
    lm = LearningManager(tmp_path, "pal")
    emitted: list = []
    extractor = AsyncMock(return_value={"title": "Another", "body": "x"})
    scanner = LearningScanner(
        learning_manager=lm,
        extractor=extractor,
        emit=lambda msg: emitted.append(msg),
    )
    scanner.mark_pending("prior-proposal-id")
    asyncio.run(scanner.maybe_scan(
        recent_turns=[],
        latest_user_message="you always merge",
    ))
    # Candidate is queued, not emitted.
    assert emitted == []
    assert len(scanner.queued) == 1


def test_scanner_drains_queue_when_cleared(tmp_path: Path):
    lm = LearningManager(tmp_path, "pal")
    emitted: list = []
    extractor = AsyncMock(return_value={"title": "q", "body": "x"})
    scanner = LearningScanner(
        learning_manager=lm,
        extractor=extractor,
        emit=lambda msg: emitted.append(msg),
    )
    scanner.mark_pending("p1")
    asyncio.run(scanner.maybe_scan(
        recent_turns=[],
        latest_user_message="you always do that",
    ))
    assert emitted == []
    scanner.clear_pending()
    assert len(emitted) == 1
    # Pending id is now the emitted proposal's id
    assert scanner._pending_id == emitted[0].proposal_id


def test_scanner_generates_unique_proposal_ids(tmp_path: Path):
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
    scanner.clear_pending()
    extractor.return_value = {"title": "B", "body": "y"}
    asyncio.run(scanner.maybe_scan(
        recent_turns=[], latest_user_message="you always b",
    ))
    assert len(emitted) == 2
    assert emitted[0].proposal_id != emitted[1].proposal_id
