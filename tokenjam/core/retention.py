"""Storage retention cleanup — deletion bounded by the span the user chose.

Retention does not have an opinion of its own. The cutoff is derived from
``storage.analysis_span`` through ``core/analysis_span.py``, so the job can
never delete history that a claim the product is currently making depends on;
an unbounded span disables the job outright. See that module for the derivation
and the one-directional clamp.

Every run leaves a row in ``retention_events`` — how far back it cut, how much
it removed, and what the oldest surviving row is afterwards. Before that ledger
the job's effect was observable only by measuring the store twice, days apart,
and diffing the two answers, which is how eight weeks of the oldest history came
to be gone before anyone noticed. The job also runs from an apscheduler cron
inside ``tj serve``, so on a machine where the daemon starts ad hoc it fires
only when one happens to be alive: enforcement is irregular by construction and
the configured number is an upper bound on what is kept, never a rolling window.
That is the other half of why the ledger records what a run DID rather than
leaving it to be inferred from the setting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from tokenjam.core.analysis_span import analysis_span_days, retention_days_for
from tokenjam.utils.time_parse import utcnow

if TYPE_CHECKING:
    from tokenjam.core.config import StorageConfig
    from tokenjam.core.db import StorageBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionRun:
    """What one run of the job did.

    ``skipped_reason`` set means the job deleted nothing BY DESIGN — distinct
    from a run that deleted nothing because there was nothing old enough, which
    is a real run with zero counts.
    """
    spans_deleted:    int
    sessions_deleted: int
    cutoff:           datetime | None
    retention_days:   int | None
    skipped_reason:   str | None = None


def run_retention_cleanup(db: StorageBackend, config: StorageConfig) -> RetentionRun:
    """Delete history that has aged out of the configured analysis span.

    Called by the apscheduler background job in `tj serve`. That caller needs
    nothing from the return value, but the shape is the point: a deletion of a
    user's own history reports what it removed, rather than an opaque integer
    nobody attributes to anything.
    """
    days = retention_days_for(config)
    if days is None:
        # An all-available span means every row is still inside what the product
        # offers to analyze, so there is nothing this job may delete.
        return RetentionRun(
            spans_deleted=0, sessions_deleted=0, cutoff=None, retention_days=None,
            skipped_reason="retention is disabled by an all-available analysis span",
        )

    cutoff = utcnow() - timedelta(days=days)
    # The deletes and their ledger row land in ONE transaction — see
    # `delete_spans_before`. Not a detail: this job runs from an apscheduler
    # cron inside an ad-hoc `tj serve`, so being killed mid-run is ordinary, and
    # a completed delete with no trace is the exact failure the ledger exists to
    # make impossible. Two sequential commits would deliver that guarantee only
    # on the happy path, which is the path where nobody needs it.
    spans_deleted, sessions_deleted = db.delete_spans_before(
        cutoff,
        retention_days=days,
        analysis_span_days=analysis_span_days(config),
    )

    if spans_deleted or sessions_deleted:
        logger.info(
            "retention deleted %d span(s) and %d orphaned session(s) older than "
            "%s (%d-day span)",
            spans_deleted, sessions_deleted, cutoff.isoformat(), days,
        )
    return RetentionRun(
        spans_deleted=spans_deleted, sessions_deleted=sessions_deleted,
        cutoff=cutoff, retention_days=days,
    )
