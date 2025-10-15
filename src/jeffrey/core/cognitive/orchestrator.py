"""Orchestrator for cognitive modules with error isolation"""

import asyncio
from typing import Any

from jeffrey.core.cognitive.base_module import BaseCognitiveModule
from jeffrey.utils.logger import get_logger


class ModuleOrchestrator:
    """Orchestrates cognitive modules with parallel processing and error isolation"""

    def __init__(self, modules: list[BaseCognitiveModule], timeout: float = 1.5):
        self.modules = modules
        self.timeout = timeout
        self.logger = get_logger("ModuleOrchestrator")
        self.stats = {"total_processes": 0, "total_errors": 0, "total_timeouts": 0}

    async def initialize(self):
        """Initialize all modules with error isolation"""
        self.logger.info(f"Initializing {len(self.modules)} modules...")

        init_tasks = []
        for module in self.modules:
            task = self._safe_init(module)
            init_tasks.append(task)

        results = await asyncio.gather(*init_tasks, return_exceptions=True)

        active_count = sum(1 for r in results if r is True)
        self.logger.info(f"✅ Initialized {active_count}/{len(self.modules)} modules")

    async def _safe_init(self, module: BaseCognitiveModule) -> bool:
        """Safely initialize a module"""
        try:
            await asyncio.wait_for(module.initialize(), timeout=5.0)
            return True
        except TimeoutError:
            self.logger.error(f"Timeout initializing {module.name}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize {module.name}: {e}")
            return False

    async def process(self, data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """
        Process data through all modules in parallel
        Returns (merged_results, errors)
        """
        self.stats["total_processes"] += 1

        # Create tasks with timeout
        tasks = []
        for module in self.modules:
            if module.active:
                task = self._process_with_timeout(module, data)
                tasks.append((module.name, task))

        if not tasks:
            return {}, ["No active modules"]

        # Execute in parallel
        names, futures = zip(*tasks)
        results = await asyncio.gather(*futures, return_exceptions=True)

        # Merge results
        merged = {}
        errors = []

        for name, result in zip(names, results):
            if isinstance(result, asyncio.TimeoutError):
                error_msg = f"{name}: Timeout after {self.timeout}s"
                errors.append(error_msg)
                self.stats["total_timeouts"] += 1
                self.logger.warning(error_msg)

            elif isinstance(result, Exception):
                error_msg = f"{name}: {str(result)}"
                errors.append(error_msg)
                self.stats["total_errors"] += 1
                self.logger.error(error_msg)

            elif isinstance(result, dict):
                # Namespace results by module name
                if "error" in result:
                    errors.append(f"{name}: {result['error']}")
                else:
                    merged[name] = result

        return merged, errors

    async def _process_with_timeout(self, module: BaseCognitiveModule, data: dict[str, Any]):
        """Process with timeout protection"""
        return await asyncio.wait_for(module.process(data), timeout=self.timeout)

    async def shutdown(self):
        """Shutdown all modules gracefully"""
        self.logger.info("Shutting down modules...")

        shutdown_tasks = []
        for module in self.modules:
            task = self._safe_shutdown(module)
            shutdown_tasks.append(task)

        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        self.logger.info("✅ All modules shutdown")

    async def _safe_shutdown(self, module: BaseCognitiveModule):
        """Safely shutdown a module"""
        try:
            await asyncio.wait_for(module.shutdown(), timeout=2.0)
        except TimeoutError:
            self.logger.warning(f"Timeout shutting down {module.name}")
        except Exception as e:
            self.logger.error(f"Error shutting down {module.name}: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics"""
        module_stats = [m.get_stats() for m in self.modules]

        return {
            "orchestrator": self.stats,
            "modules": module_stats,
            "active_modules": sum(1 for m in self.modules if m.active),
            "total_modules": len(self.modules),
        }
