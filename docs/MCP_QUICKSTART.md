# MCP Quickstart Guide

This guide shows how to expose `realbrain_server.tools` through an MCP-compatible local server.

## Prerequisites

- Python 3.10+
- `pip install mcp` (Model Context Protocol SDK)
- `realbrain_server` installed and configured

## Minimal MCP Server

Create a file `mcp_server.py`:

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import realbrain_server.tools as tools

server = Server("realbrain")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="record",
            description="Record a memory or observation",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The content to record"},
                    "category": {"type": "string", "description": "Memory category"}
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="search",
            description="Search recorded memories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "number", "description": "Max results (default 10)"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="activate",
            description="Activate a memory by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory ID to activate"}
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="dream",
            description="Run the dream/consolidation process",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "record":
        result = tools.record(arguments["content"], arguments.get("category", "general"))
    elif name == "search":
        results = tools.search(arguments["query"], arguments.get("limit", 10))
        result = "\n".join(f"- {r.id}: {r.content}" for r in results)
    elif name == "activate":
        result = tools.activate(arguments["memory_id"])
    elif name == "dream":
        result = tools.dream()
    else:
        result = f"Unknown tool: {name}"
    
    return [TextContent(type="text", text=str(result))]

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    
    asyncio.run(main())
```

## Tool Mapping

| MCP Tool | realbrain function | Description |
|----------|-------------------|-------------|
| `record` | `tools.record(content, category)` | Record a memory |
| `search` | `tools.search(query, limit)` | Search memories |
| `activate` | `tools.activate(memory_id)` | Activate a specific memory |
| `dream` | `tools.dream()` | Run consolidation process |

## Safety Boundaries

- The **host** (you) owns all approvals for external actions
- MCP server runs locally — no network exposure
- `dream` is a read-only consolidation — it doesn't modify external state
- `activate` returns memory content but doesn't execute any actions

## Configure Claude Desktop

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "realbrain": {
      "command": "python",
      "args": ["/path/to/realbrain/mcp_server.py"]
    }
  }
}
```

## Test

```bash
python mcp_server.py
# Then connect from Claude Desktop or any MCP client
```

## Security

You can audit this MCP server for free at [MarketNow](https://marketnow.site/submit) — 10-layer security audit with Ed25519 trust card.
