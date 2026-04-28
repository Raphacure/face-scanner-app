from app.services.database_service import execute_query


def db_health_check_controller() -> dict:
    try:
        result = execute_query("SELECT 1 AS status")
        return {"status": "ok", "db": result["rows"][0] if result["rows"] else None}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def db_query_controller(query: str) -> dict:
    try:
        result = execute_query(query)
        return {
            "status": "success",
            "count": result["rowcount"],
            "rows": result["rows"],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
