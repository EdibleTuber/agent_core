"""ApprovalRegistry — per-session store for research proposal approvals.

Tracks Proposal entries through their lifecycle:
    pending -> approved -> consumed
    pending -> declined
    pending -> expired

Proposals are keyed by proposal_id (uuid4). The registry holds state in
memory for the lifetime of one chat session; it is not persisted.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

ProposalStatus = Literal["pending", "approved", "declined", "consumed", "expired"]
ProposalKind = Literal["research", "compile", "reorg", "consolidate", "promote", "batch_fallback"]

DEFAULT_EXPIRY_MINUTES = 15


@dataclass
class Proposal:
    proposal_id: str
    topic: str
    depth: int
    rationale: str
    status: ProposalStatus
    created_at: datetime
    expires_at: datetime
    kind: ProposalKind = "research"
    successor_id: Optional[str] = None
    summary_paths: Optional[list[str]] = None
    operations: Optional[list[dict]] = None
    target_path: Optional[str] = None
    target_title: Optional[str] = None
    slug: Optional[str] = None
    body: Optional[str] = None
    caller: Optional[str] = None
    context: Optional[str] = None
    approval_choice: Optional[str] = None
    # asyncio.Event is set when the proposal reaches a terminal state
    # (approved, declined, or expired). Not part of the public dataclass
    # fields — carried separately for awaiting.
    event: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)


ResearchProposal = Proposal  # deprecated alias; remove after callers migrate


class ApprovalRegistry:
    def __init__(self, expiry_minutes: int = DEFAULT_EXPIRY_MINUTES) -> None:
        self._proposals: dict[str, Proposal] = {}
        self._expiry_minutes = expiry_minutes

    def create_proposal(
        self,
        *,
        kind: ProposalKind = "research",
        topic: str = "",
        depth: int = 3,
        rationale: str,
        summary_paths: Optional[list[str]] = None,
        operations: Optional[list[dict]] = None,
        target_path: Optional[str] = None,
        target_title: Optional[str] = None,
        slug: Optional[str] = None,
        body: Optional[str] = None,
        caller: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        if kind == "research" and not topic:
            raise ValueError("research proposals require a non-empty topic")
        if kind == "compile":
            if not summary_paths:
                raise ValueError("compile proposals require a non-empty summary_paths list")
        if kind == "reorg":
            if not operations:
                raise ValueError("reorg proposals require a non-empty operations list")
            for op in operations:
                if not isinstance(op, dict):
                    raise ValueError(f"each operation must be a dict, got {type(op).__name__}")
                if op.get("type") not in ("move", "merge"):
                    raise ValueError(f"operation type must be 'move' or 'merge', got {op.get('type')!r}")
                if not op.get("src") or not op.get("dst"):
                    raise ValueError("every operation requires 'src' and 'dst' fields")
        if kind == "consolidate":
            if not summary_paths or len(summary_paths) < 2:
                raise ValueError("consolidate proposals require at least two source paths")
            if not target_path:
                raise ValueError("consolidate proposals require target_path")
            if not target_title:
                raise ValueError("consolidate proposals require target_title")
        if kind == "promote":
            if not slug:
                raise ValueError("promote proposals require slug")
            if not target_title:
                raise ValueError("promote proposals require target_title")
            if not body:
                raise ValueError("promote proposals require body")

        proposal_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        self._proposals[proposal_id] = Proposal(
            proposal_id=proposal_id,
            topic=topic,
            depth=depth,
            rationale=rationale,
            status="pending",
            created_at=now,
            expires_at=now + timedelta(minutes=self._expiry_minutes),
            kind=kind,
            summary_paths=list(summary_paths) if summary_paths else None,
            operations=[dict(op) for op in operations] if operations else None,
            target_path=target_path,
            target_title=target_title,
            slug=slug,
            body=body,
            caller=caller,
            context=context,
        )
        return proposal_id

    def get(self, proposal_id: str) -> Optional[Proposal]:
        return self._proposals.get(proposal_id)

    def approve(self, proposal_id: str, state: Optional[str] = None) -> None:
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.status != "pending":
            return
        proposal.status = "approved"
        if state is not None:
            proposal.approval_choice = state
        proposal.event.set()

    def decline(self, proposal_id: str) -> None:
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.status != "pending":
            return
        proposal.status = "declined"
        proposal.event.set()

    def consume(self, proposal_id: str) -> bool:
        """Mark an approved proposal as consumed. Returns True on success."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.status != "approved":
            return False
        proposal.status = "consumed"
        return True

    def expire_stale(self) -> None:
        """Mark pending proposals past their expiry as expired and signal waiters."""
        now = datetime.now(timezone.utc)
        for proposal in self._proposals.values():
            if proposal.status == "pending" and now >= proposal.expires_at:
                proposal.status = "expired"
                proposal.event.set()

    def get_successor(self, proposal_id: str) -> Optional["Proposal"]:
        """Return the proposal that replaced this one via edit(), or None."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.successor_id is None:
            return None
        return self._proposals.get(proposal.successor_id)

    def edit(
        self,
        proposal_id: str,
        *,
        new_topic: Optional[str] = None,
        new_depth: Optional[int] = None,
        summary_paths: Optional[list[str]] = None,
        operations: Optional[list[dict]] = None,
    ) -> Optional[str]:
        """Replace a pending proposal with a new approved one.

        Returns the new proposal_id, or None if the original is missing
        or not pending.
        """
        old = self._proposals.get(proposal_id)
        if old is None or old.status != "pending":
            return None
        # Decline the old proposal so any waiter gets a terminal state.
        old.status = "declined"
        old.event.set()
        # The user has committed to the edited values via the CLI, so the
        # new proposal is created already-approved.
        new_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        new_proposal = Proposal(
            proposal_id=new_id,
            topic=new_topic if new_topic is not None else old.topic,
            depth=new_depth if new_depth is not None else old.depth,
            rationale=old.rationale,
            status="approved",
            created_at=now,
            expires_at=now + timedelta(minutes=self._expiry_minutes),
            kind=old.kind,
            summary_paths=(
                list(summary_paths) if summary_paths is not None
                else (list(old.summary_paths) if old.summary_paths else None)
            ),
            operations=(
                [dict(op) for op in operations] if operations is not None
                else ([dict(op) for op in old.operations] if old.operations else None)
            ),
            target_path=old.target_path,
            target_title=old.target_title,
        )
        new_proposal.event.set()
        self._proposals[new_id] = new_proposal
        old.successor_id = new_id
        return new_id
