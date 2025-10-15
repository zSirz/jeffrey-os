"""
CoreClient minimal - Communication with Jeffrey Core via Unix socket
60 lines optimized implementation
"""

import json
import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)


class CoreClient:
    """Minimal client for Core communication via Unix socket"""

    def __init__(self, socket_path: str = "/tmp/jeffrey_core.sock"):
        self.socket_path = socket_path
        self.timeout = 30.0
        self._connected = False

    def is_connected(self) -> bool:
        """Check if client is connected to Core"""
        return self._connected

    async def ask_core(self, query: dict[str, Any]) -> dict[str, Any] | None:
        """Send query to Core and receive response"""
        try:
            # Create Unix socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)

            # Connect to Core
            sock.connect(self.socket_path)
            self._connected = True

            # Send query as JSON
            message = json.dumps(query) + "\n"
            sock.sendall(message.encode("utf-8"))
            logger.debug(f"Sent to Core: {query}")

            # Receive response
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in response_data:
                    break

            # Parse response
            response = json.loads(response_data.decode("utf-8").strip())
            logger.debug(f"Received from Core: {response}")

            sock.close()
            return response

        except Exception as e:
            logger.error(f"CoreClient error: {e}")
            self._connected = False
            return None

    async def health_check(self) -> bool:
        """Check Core health status"""
        response = await self.ask_core({"action": "health"})
        return response is not None and response.get("status") == "healthy"
