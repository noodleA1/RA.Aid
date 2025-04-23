import asyncio
import threading
from typing import Any, List

from langchain_core.tools import BaseTool
from mcp_use import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter


class MCPUseClientSync:
    """Synchronous wrapper around ``mcp_use.MCPClient``.

    ``mcp_use`` is fully async; RA.Aid’s tool-pipeline expects plain
    synchronous ``langchain_core.tools.BaseTool`` instances.  This wrapper
    starts a background event-loop, initializes the MCP client and converts
    every server tool to a ``StructuredTool`` via the official
    ``LangChainAdapter``.
    """

    def __init__(self, config: str | dict[str, Any]):
        """Create the client.

        Args:
            config: Either a path to a JSON config file **or** a config dict
                    compatible with ``MCPClient``.
        """
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        self._client: MCPClient | None = None
        self._tools: List[BaseTool] = []

        fut = asyncio.run_coroutine_threadsafe(
            self._setup_client(config), self.loop
        )
        fut.result()  # propagate any exceptions synchronously

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    def get_tools_sync(self) -> List[BaseTool]:
        """Return the converted LangChain tools (synchronous)."""
        return self._tools

    def close(self) -> None:
        """Close all MCP sessions and stop the event-loop."""
        if self._client is not None:
            fut = asyncio.run_coroutine_threadsafe(
                self._client.close_all_sessions(), self.loop
            )
            fut.result()

        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _setup_client(self, config: str | dict[str, Any]):
        if isinstance(config, str):
            client = MCPClient.from_config_file(config)
        else:
            client = MCPClient.from_dict(config)

        # create sessions for all declared servers so tools are available
        await client.create_all_sessions(auto_initialize=True)

        adapter = LangChainAdapter()
        self._tools = await adapter.create_tools(client)
        self._client = client
