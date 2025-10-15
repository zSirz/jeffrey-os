"""Base class for all cognitive modules"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from jeffrey.utils.logger import get_logger


class BaseCognitiveModule(ABC):
    """Abstract base for cognitive modules with standard interface"""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)
        self.active = False
        self.error_count = 0
        self.process_count = 0

    async def initialize(self):
        """Initialize the module"""
        try:
            self.active = True
            await self.on_initialize()
            self.logger.info(f"✅ {self.name} initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self.active = False
            raise

    @abstractmethod
    async def on_initialize(self):
        """Override for custom initialization"""
        pass

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with error tracking"""
        if not self.active:
            return {"error": "Module not active"}

        self.process_count += 1

        try:
            # Validate input
            if not self.validate_input(data):
                return {"error": "Invalid input format"}

            # Process
            result = await self.on_process(data)

            # Validate output
            if not isinstance(result, dict):
                result = {"result": result}

            return result

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Process error: {e}")
            return {"error": str(e)}

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format"""
        return isinstance(data, dict)

    @abstractmethod
    async def on_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Override for custom processing"""
        pass

    async def shutdown(self):
        """Shutdown the module"""
        try:
            await self.on_shutdown()
            self.active = False
            self.logger.info(f"✅ {self.name} shutdown complete")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

    async def on_shutdown(self):
        """Override for custom shutdown"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get module statistics"""
        return {
            "name": self.name,
            "active": self.active,
            "process_count": self.process_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.process_count, 1),
        }
