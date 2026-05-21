"""Shared slowapi limiter.

Lives in its own module so both ``src.main`` and route modules can import it
without creating a circular dependency (main imports routers; routers need
the limiter).
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Keyed on client IP. Behind a reverse proxy you must forward the real client
# IP via X-Forwarded-For — slowapi's get_remote_address reads request.client
# which uvicorn populates from proxy-forwarded headers when --proxy-headers
# is set.
limiter = Limiter(key_func=get_remote_address)
