import logging

from fastapi import HTTPException, status

from app.services.raw_sql.health_queries import health_check

logger = logging.getLogger(__name__)


SAFE_READ_QUERIES = {
    "health_status": health_check,
}


def db_health_check_controller() -> dict:
    try:
        result = health_check()
        return {"status": "ok", "db": result["rows"][0] if result["rows"] else None}
    except Exception as e:
        logger.exception("DB health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "error", "message": "Database unavailable"},
        ) from e


def db_query_controller(query_name: str) -> dict:
    handler = SAFE_READ_QUERIES.get(query_name)
    if handler is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported query_name: {query_name}",
        )
    try:
        result = handler()
        return {
            "status": "success",
            "query_name": query_name,
            "count": result["rowcount"],
            "rows": result["rows"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("DB query failed for query_name=%s", query_name)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "error", "message": "Database unavailable"},
        ) from e
