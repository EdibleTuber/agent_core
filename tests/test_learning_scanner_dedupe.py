from agent_core.learning_scanner import is_duplicate_candidate


def test_exact_title_match_is_duplicate():
    existing = ["granularity-over-consolidation", "strategic-research-sprints"]
    assert is_duplicate_candidate("Granularity Over Consolidation", existing) is True


def test_near_title_match_is_duplicate():
    existing = ["granularity-over-consolidation"]
    # Different wording, same idea -> slug token overlap is high.
    assert is_duplicate_candidate("Granularity vs Consolidation", existing) is True


def test_distinct_title_is_not_duplicate():
    existing = ["granularity-over-consolidation"]
    assert is_duplicate_candidate("Prefer Typed Protobuf Schemas", existing) is False


def test_empty_existing_returns_false():
    assert is_duplicate_candidate("Anything", []) is False


def test_empty_title_returns_false():
    assert is_duplicate_candidate("", ["granularity-over-consolidation"]) is False


def test_short_tokens_ignored():
    # Only tokens > 2 chars count toward similarity; very short words don't
    # trigger false positives.
    existing = ["a-b-c-d"]
    # No real overlap (tokens all under length 3)
    assert is_duplicate_candidate("e f g h", existing) is False
