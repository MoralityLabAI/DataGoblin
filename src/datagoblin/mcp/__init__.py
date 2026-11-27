"""MCP stdio server exposing the DataGoblin research orchestrator as tools."""

from .server import PROTOCOL, VERSION, handle, serve

__all__ = ["PROTOCOL", "VERSION", "handle", "serve"]
