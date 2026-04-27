from datetime import datetime, timedelta, timezone

from agent_core.approval_registry import ApprovalRegistry, ResearchProposal


def test_create_proposal_returns_pending():
    registry = ApprovalRegistry()
    proposal_id = registry.create_proposal(
        topic="indirect prompt injection",
        depth=3,
        rationale="vault has no sources on this",
    )
    assert proposal_id
    proposal = registry.get(proposal_id)
    assert isinstance(proposal, ResearchProposal)
    assert proposal.topic == "indirect prompt injection"
    assert proposal.depth == 3
    assert proposal.rationale == "vault has no sources on this"
    assert proposal.status == "pending"
    assert proposal.proposal_id == proposal_id


def test_get_unknown_returns_none():
    registry = ApprovalRegistry()
    assert registry.get("nonexistent") is None


def test_create_proposal_generates_unique_ids():
    registry = ApprovalRegistry()
    ids = {
        registry.create_proposal(topic=f"t{i}", depth=3, rationale="r")
        for i in range(10)
    }
    assert len(ids) == 10


def test_approve_sets_event_and_status():
    registry = ApprovalRegistry()
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    registry.approve(pid)
    proposal = registry.get(pid)
    assert proposal.status == "approved"
    assert proposal.event.is_set()


def test_decline_sets_event_and_status():
    registry = ApprovalRegistry()
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    registry.decline(pid)
    proposal = registry.get(pid)
    assert proposal.status == "declined"
    assert proposal.event.is_set()


def test_consume_only_valid_from_approved():
    registry = ApprovalRegistry()
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    # cannot consume a pending proposal
    assert registry.consume(pid) is False
    assert registry.get(pid).status == "pending"
    registry.approve(pid)
    assert registry.consume(pid) is True
    assert registry.get(pid).status == "consumed"
    # cannot consume twice
    assert registry.consume(pid) is False


def test_approve_unknown_id_is_noop():
    registry = ApprovalRegistry()
    registry.approve("nonexistent")  # should not raise


def test_approve_declined_proposal_is_noop():
    registry = ApprovalRegistry()
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    registry.decline(pid)
    registry.approve(pid)
    assert registry.get(pid).status == "declined"


def test_expiry_transitions_pending_to_expired():
    registry = ApprovalRegistry(expiry_minutes=0)  # immediate expiry
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    registry.expire_stale()
    proposal = registry.get(pid)
    assert proposal.status == "expired"
    assert proposal.event.is_set()


def test_expiry_leaves_non_pending_alone():
    registry = ApprovalRegistry(expiry_minutes=0)
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    registry.approve(pid)
    registry.expire_stale()
    assert registry.get(pid).status == "approved"


def test_edit_declines_old_proposal_and_issues_new():
    registry = ApprovalRegistry()
    old_pid = registry.create_proposal(topic="original", depth=3, rationale="r")
    new_pid = registry.edit(old_pid, new_topic="refined", new_depth=5)
    assert new_pid != old_pid
    old = registry.get(old_pid)
    new = registry.get(new_pid)
    assert old.status == "declined"
    assert old.event.is_set()
    # The new proposal is created approved (user has already committed
    # to the edited topic/depth via the CLI edit workflow).
    assert new.status == "approved"
    assert new.topic == "refined"
    assert new.depth == 5
    assert new.event.is_set()


def test_edit_unknown_id_returns_none():
    registry = ApprovalRegistry()
    assert registry.edit("nonexistent", new_topic="x", new_depth=3) is None


def test_edit_non_pending_returns_none():
    registry = ApprovalRegistry()
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    registry.approve(pid)
    assert registry.edit(pid, new_topic="x", new_depth=3) is None


def test_edit_records_successor_id_on_old_proposal():
    registry = ApprovalRegistry()
    old_pid = registry.create_proposal(topic="original", depth=3, rationale="r")
    new_pid = registry.edit(old_pid, new_topic="refined", new_depth=5)
    old = registry.get(old_pid)
    assert old.successor_id == new_pid


def test_get_successor_returns_new_proposal():
    registry = ApprovalRegistry()
    old_pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    new_pid = registry.edit(old_pid, new_topic="u", new_depth=4)
    successor = registry.get_successor(old_pid)
    assert successor is not None
    assert successor.proposal_id == new_pid
    assert successor.topic == "u"


def test_get_successor_returns_none_when_no_edit():
    registry = ApprovalRegistry()
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    assert registry.get_successor(pid) is None


def test_get_successor_unknown_id_returns_none():
    registry = ApprovalRegistry()
    assert registry.get_successor("nonexistent") is None


def test_create_proposal_defaults_to_research_kind():
    registry = ApprovalRegistry()
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    proposal = registry.get(pid)
    assert proposal.kind == "research"


def test_proposal_is_new_dataclass_name():
    from agent_core.approval_registry import Proposal
    registry = ApprovalRegistry()
    pid = registry.create_proposal(topic="t", depth=3, rationale="r")
    assert isinstance(registry.get(pid), Proposal)


def test_create_proposal_with_compile_kind():
    registry = ApprovalRegistry()
    paths = ["raw/summaries/a.md", "raw/summaries/b.md"]
    pid = registry.create_proposal(
        kind="compile",
        summary_paths=paths,
        rationale="promote research findings",
    )
    proposal = registry.get(pid)
    assert proposal.kind == "compile"
    assert proposal.summary_paths == paths
    assert proposal.rationale == "promote research findings"
    assert proposal.status == "pending"


def test_edit_compile_proposal_carries_kind_and_paths():
    registry = ApprovalRegistry()
    old_pid = registry.create_proposal(
        kind="compile",
        summary_paths=["raw/summaries/a.md"],
        rationale="r",
    )
    new_pid = registry.edit(
        old_pid,
        summary_paths=["raw/summaries/a.md", "raw/summaries/b.md"],
    )
    assert new_pid is not None
    new = registry.get(new_pid)
    assert new.kind == "compile"
    assert new.summary_paths == ["raw/summaries/a.md", "raw/summaries/b.md"]
    assert new.status == "approved"


def test_create_compile_proposal_rejects_empty_paths():
    registry = ApprovalRegistry()
    import pytest
    with pytest.raises(ValueError):
        registry.create_proposal(
            kind="compile",
            summary_paths=[],
            rationale="r",
        )


def test_create_research_proposal_without_topic_raises():
    registry = ApprovalRegistry()
    import pytest
    with pytest.raises(ValueError):
        registry.create_proposal(kind="research", rationale="r")


def test_create_proposal_with_reorg_kind():
    registry = ApprovalRegistry()
    ops = [
        {"type": "move", "src": "A.md", "dst": "B.md"},
        {"type": "merge", "src": "C.md", "dst": "D.md"},
    ]
    pid = registry.create_proposal(
        kind="reorg",
        operations=ops,
        rationale="consolidate duplicates",
    )
    proposal = registry.get(pid)
    assert proposal.kind == "reorg"
    assert proposal.operations == ops
    assert proposal.rationale == "consolidate duplicates"


def test_create_reorg_proposal_rejects_empty_operations():
    registry = ApprovalRegistry()
    import pytest
    with pytest.raises(ValueError):
        registry.create_proposal(
            kind="reorg",
            operations=[],
            rationale="r",
        )


def test_create_reorg_proposal_rejects_invalid_op_type():
    registry = ApprovalRegistry()
    import pytest
    with pytest.raises(ValueError):
        registry.create_proposal(
            kind="reorg",
            operations=[{"type": "delete", "src": "A.md", "dst": "B.md"}],
            rationale="r",
        )


def test_create_reorg_proposal_rejects_missing_src_dst():
    registry = ApprovalRegistry()
    import pytest
    with pytest.raises(ValueError):
        registry.create_proposal(
            kind="reorg",
            operations=[{"type": "move", "src": "A.md"}],  # no dst
            rationale="r",
        )


def test_edit_reorg_proposal_carries_kind_and_operations():
    registry = ApprovalRegistry()
    old_ops = [{"type": "move", "src": "A.md", "dst": "B.md"}]
    new_ops = [{"type": "move", "src": "A.md", "dst": "C.md"}]
    old_pid = registry.create_proposal(
        kind="reorg", operations=old_ops, rationale="r",
    )
    new_pid = registry.edit(old_pid, operations=new_ops)
    new = registry.get(new_pid)
    assert new.kind == "reorg"
    assert new.operations == new_ops
    assert new.status == "approved"


def test_create_consolidate_proposal():
    reg = ApprovalRegistry()
    pid = reg.create_proposal(
        kind="consolidate",
        summary_paths=["Security/a.md", "Security/b.md"],
        target_path="Security/Combined.md",
        target_title="Combined",
        rationale="merge overlapping notes",
    )
    p = reg.get(pid)
    assert p.kind == "consolidate"
    assert p.summary_paths == ["Security/a.md", "Security/b.md"]
    assert p.target_path == "Security/Combined.md"
    assert p.target_title == "Combined"
    assert p.status == "pending"


def test_consolidate_requires_two_sources():
    import pytest
    reg = ApprovalRegistry()
    with pytest.raises(ValueError, match="at least two"):
        reg.create_proposal(
            kind="consolidate",
            summary_paths=["Security/a.md"],
            target_path="Security/Combined.md",
            target_title="Combined",
            rationale="r",
        )


def test_consolidate_requires_target():
    import pytest
    reg = ApprovalRegistry()
    with pytest.raises(ValueError, match="target_path"):
        reg.create_proposal(
            kind="consolidate",
            summary_paths=["Security/a.md", "Security/b.md"],
            target_path="",
            target_title="Combined",
            rationale="r",
        )


def test_consolidate_requires_target_title():
    import pytest
    registry = ApprovalRegistry()
    with pytest.raises(ValueError, match="target_title"):
        registry.create_proposal(
            kind="consolidate",
            summary_paths=["Security/a.md", "Security/b.md"],
            target_path="Security/Combined.md",
            target_title="",
            rationale="r",
        )


def test_consolidate_rejects_none_sources():
    import pytest
    registry = ApprovalRegistry()
    with pytest.raises(ValueError, match="at least two"):
        registry.create_proposal(
            kind="consolidate",
            summary_paths=None,
            target_path="Security/Combined.md",
            target_title="Combined",
            rationale="r",
        )


def test_edit_consolidate_proposal_carries_fields():
    registry = ApprovalRegistry()
    old_pid = registry.create_proposal(
        kind="consolidate",
        summary_paths=["Security/a.md", "Security/b.md"],
        target_path="Security/Combined.md",
        target_title="Combined",
        rationale="r",
    )
    new_pid = registry.edit(old_pid)
    assert new_pid is not None
    new = registry.get(new_pid)
    assert new.kind == "consolidate"
    assert new.summary_paths == ["Security/a.md", "Security/b.md"]
    assert new.target_path == "Security/Combined.md"
    assert new.target_title == "Combined"
    assert new.status == "approved"


def test_create_promote_proposal_requires_all_fields():
    import pytest
    from agent_core.approval_registry import ApprovalRegistry
    reg = ApprovalRegistry()

    # Missing slug
    with pytest.raises(ValueError, match="slug"):
        reg.create_proposal(kind="promote", rationale="r", target_title="T", body="b")
    # Missing target_title
    with pytest.raises(ValueError, match="target_title"):
        reg.create_proposal(kind="promote", rationale="r", slug="s", body="b")
    # Missing body
    with pytest.raises(ValueError, match="body"):
        reg.create_proposal(kind="promote", rationale="r", slug="s", target_title="T")


def test_create_promote_proposal_succeeds_with_all_fields():
    from agent_core.approval_registry import ApprovalRegistry
    reg = ApprovalRegistry()
    pid = reg.create_proposal(
        kind="promote",
        rationale="r",
        slug="s",
        target_title="T",
        body="b",
    )
    p = reg.get(pid)
    assert p.kind == "promote"
    assert p.slug == "s"
    assert p.body == "b"
    assert p.target_title == "T"
