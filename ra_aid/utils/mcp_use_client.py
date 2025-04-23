import asyncio
import threading
import logging

logger = logging.getLogger(__name__)
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
    Note:
        Using MCP servers often requires external dependencies managed by that
        server. For example, using the Context7 server requires ``npx`` (Node.js)
        to be available in the environment to run ``@upstash/context7-mcp``.

    ``LangChainAdapter``.
    """

    def __init__(self, config: str | dict[str, Any]):
        """Create the client.

        Args:
            config: Either a path to a JSON config file **or** a config dict
                    compatible with ``MCPClient``.
        """
        self.loop = asyncio.new_event_loop()
        self._loop_started = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._loop_started.wait() # Wait for loop to be running

        self._client: MCPClient | None = None
        self._tools: List[BaseTool] = []
        self._closed = False

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
        """Close all MCP sessions and stop the event-loop. Idempotent."""
        if getattr(self, "_closed", False):
            return # Already closed

        logger.debug("Closing MCPUseClientSync...")
        if self._client is not None:
            try:
                # Ensure the loop is running before scheduling
                if self.loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(
                        self._client.close_all_sessions(), self.loop
                    )
                    # Wait with a timeout, but don't block indefinitely if loop is stuck
                    fut.result(timeout=10) 
                else:
                    logger.warning("MCP client loop not running during close.")
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for MCP client sessions to close.")
            except Exception as e:
                logger.error(f"Error closing MCP client sessions: {e}")

        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        self.thread.join() # Wait for thread to finish
        if self.thread.is_alive():
             logger.warning("MCP client background thread did not exit cleanly.")

        self._closed = True
        logger.debug("MCPUseClientSync closed.")

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._loop_started.set() # Signal that the loop is ready
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()
            logger.debug("MCP client event loop closed.")

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
