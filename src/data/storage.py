"""Run IDs for collection sessions."""

import uuid


def get_run_id() -> str:
    """Return an 8-character hex ID for one collection run."""
    return uuid.uuid4().hex[:8]
