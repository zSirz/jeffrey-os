from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
import os
from typing import Optional

# Configuration
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("JEFFREY_API_KEY", "jeffrey-dev-key-change-in-prod")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """Verify API key for protected endpoints"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required"
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )

    return True

# Optional: Different permission levels
async def require_write_permission(api_key: str = Security(api_key_header)):
    """Require write permission (for POST/PUT/DELETE)"""
    return await verify_api_key(api_key)

async def require_admin_permission(api_key: str = Security(api_key_header)):
    """Require admin permission (for dream/run, etc)"""
    # For now, same as write, but could check different keys
    return await verify_api_key(api_key)