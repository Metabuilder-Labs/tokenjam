"""GET /api/v1/alerts — alert history."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from tj.api.deps import require_api_key
from tj.core.models import AlertFilters, AlertType, Severity
from tj.utils.time_parse import parse_since

router = APIRouter(dependencies=[Depends(require_api_key)])



@router.get("/alerts")
async def get_alerts(
    request: Request,
    agent_id: str | None = None,
    since: str | None = None,
    severity: str | None = None,
    type: str | None = None,
    unread: bool = False,
) -> dict:
    db = request.app.state.db
    try:
        sev = Severity(severity) if severity else None
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid severity: {severity!r}")
    try:
        typ = AlertType(type) if type else None
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid type: {type!r}")
    filters = AlertFilters(
        agent_id=agent_id,
        since=parse_since(since) if since else None,
        severity=sev,
        type=typ,
        unread=unread,
    )
    alerts = db.get_alerts(filters)
    return {
        "alerts": [
            {
                "alert_id": a.alert_id,
                "fired_at": a.fired_at.isoformat(),
                "type": a.type.value,
                "severity": a.severity.value,
                "title": a.title,
                "detail": a.detail,
                "agent_id": a.agent_id,
                "session_id": a.session_id,
                "span_id": a.span_id,
                "acknowledged": a.acknowledged,
                "suppressed": a.suppressed,
            }
            for a in alerts
        ],
        "count": len(alerts),
    }


@router.patch("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, request: Request) -> dict:
    db = request.app.state.db
    conn = getattr(db, "conn", None)
    if conn is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Write operations unavailable in read-only mode")
    result = conn.execute(
        "SELECT alert_id FROM alerts WHERE alert_id = $1", [alert_id]
    ).fetchone()
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    conn.execute(
        "UPDATE alerts SET acknowledged = true WHERE alert_id = $1", [alert_id]
    )
    return {"acknowledged": True, "alert_id": alert_id}
