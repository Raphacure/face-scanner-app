from fastapi import HTTPException, status

from app.services.raw_sql.health_queries import health_check


SAFE_READ_QUERIES = {
    "health_status": "SELECT 1 AS status",
}


def db_health_check_controller() -> dict:
    try:
        result = health_check()
        return {"status": "ok", "db": result["rows"][0] if result["rows"] else None}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "error", "message": str(e)},
        ) from e


def db_query_controller(query_name: str) -> dict:
    if query_name not in SAFE_READ_QUERIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported query_name: {query_name}",
        )
    try:
        if query_name == "health_status":
            result = health_check()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported query_name: {query_name}",
            )
        return {
            "status": "success",
            "query_name": query_name,
            "count": result["rowcount"],
            "rows": result["rows"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "error", "message": str(e)},
        ) from e
