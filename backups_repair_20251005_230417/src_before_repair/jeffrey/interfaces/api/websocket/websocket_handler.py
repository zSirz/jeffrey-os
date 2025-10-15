#!/usr/bin/env python3
"""
🚀 WebSocket Handler for Jeffrey OS Dashboard
Real-time communication bridge between Pattern Learner and React frontend

VIVARIUM Protocol Integration:
- Real-time metrics streaming
- Pattern detection events
- Achievement notifications
- System health monitoring
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime

import psutil
import socketio
from aiohttp import web

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.modules.learning.emergence_detector import EmergenceDetector
    from src.modules.learning.metrics_tracker import MetricsTracker
    from src.modules.learning.pattern_learner_ml import PatternLearnerML
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Running in standalone mode without ML components")
    PatternLearnerML = None
    MetricsTracker = None
    EmergenceDetector = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class JeffreyWebSocketHandler:
    """Real-time WebSocket handler for Jeffrey OS Dashboard"""

    def __init__(self, pattern_learner: PatternLearnerML | None = None) -> None:
        self.sio = socketio.AsyncServer(cors_allowed_origins="*", logger=True, engineio_logger=True)
        self.app = web.Application()
        self.sio.attach(self.app)

        # Jeffrey OS components
        self.pattern_learner = pattern_learner
        self.metrics_tracker = MetricsTracker() if MetricsTracker else None
        self.emergence_detector = EmergenceDetector() if EmergenceDetector else None

        # State tracking
        self.connected_clients = set()
        self.last_metrics = {}
        self.achievement_history = []

        # Performance monitoring
        self.start_time = time.time()
        self.event_loop_lag = 0

        self._setup_routes()
        self._setup_event_handlers()

    def _setup_routes(self):
        """Setup HTTP routes for health checks"""

        async def health_check(request):
            return web.json_response(
                {
                    "status": "healthy",
                    "uptime": time.time() - self.start_time,
                    "connected_clients": len(self.connected_clients),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        async def metrics_endpoint(request):
            """REST endpoint for current metrics"""
            metrics = await self._get_current_metrics()
            return web.json_response(metrics)

        self.app.router.add_get("/health", health_check)
        self.app.router.add_get("/api/metrics", metrics_endpoint)

    def _setup_event_handlers(self):
        """Setup Socket.IO event handlers"""

        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"🔗 Client connected: {sid}")
            self.connected_clients.add(sid)

            # Send initial state
            initial_metrics = await self._get_current_metrics()
            await self.sio.emit("metrics_update", initial_metrics, room=sid)

            # Send system health
            system_health = await self._get_system_health()
            await self.sio.emit("system_health", system_health, room=sid)

        @self.sio.event
        async def disconnect(sid):
            logger.info(f"❌ Client disconnected: {sid}")
            self.connected_clients.discard(sid)

        @self.sio.event
        async def request_metrics(sid):
            """Client requests current metrics"""
            metrics = await self._get_current_metrics()
            await self.sio.emit("metrics_update", metrics, room=sid)

        @self.sio.event
        async def request_patterns(sid, data):
            """Client requests pattern history"""
            patterns = await self._get_recent_patterns()
            await self.sio.emit("patterns_update", patterns, room=sid)

        @self.sio.event
        async def simulate_learning(sid, data):
            """Demo mode: simulate learning event"""
            if self.pattern_learner:
                # Simulate pattern learning
                demo_event = {
                    "type": "user_input",
                    "content": data.get("input", "Demo learning event"),
                    "timestamp": time.time(),
                }

                result = await self._process_learning_event(demo_event)
                await self.sio.emit("learning_result", result, room=sid)

    async def _get_current_metrics(self) -> dict:
        """Get current Jeffrey OS metrics"""
        base_metrics = {
            "consciousness_level": 0.15,
            "jeffrey_iq": 102,
            "patterns_detected": 3,
            "surprise_score": 0.23,
            "learning_rate": 0.8,
            "emergence_events": 1,
            "privacy_score": 9.2,
            "timestamp": datetime.now().isoformat(),
        }

        if self.pattern_learner:
            try:
                # Get real metrics from Pattern Learner
                real_metrics = self.pattern_learner.get_metrics()
                base_metrics.update(real_metrics)
            except Exception as e:
                logger.warning(f"Failed to get real metrics: {e}")

        return base_metrics

    async def _get_system_health(self) -> dict:
        """Get current system health metrics"""
        try:
            # Measure event loop lag
            start_time = asyncio.get_event_loop().time()
            await asyncio.sleep(0)
            self.event_loop_lag = (asyncio.get_event_loop().time() - start_time) * 1000

            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            return {
                "eventLoopLag": self.event_loop_lag,
                "memoryUsage": {"used": memory.percent, "total": 100},
                "cpuUsage": cpu_percent,
                "activeConnections": len(self.connected_clients),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "eventLoopLag": 0,
                "memoryUsage": {"used": 0, "total": 100},
                "cpuUsage": 0,
                "activeConnections": 0,
            }

    async def _get_recent_patterns(self) -> list[dict]:
        """Get recently detected patterns"""
        demo_patterns = [
            {
                "id": f"pattern_{int(time.time())}_1",
                "description": "Séquence récurrente détectée",
                "complexity": 0.65,
                "surprise_value": 0.8,
                "timestamp": int(time.time() * 1000),
            },
            {
                "id": f"pattern_{int(time.time())}_2",
                "description": "Pattern émergent identifié",
                "complexity": 0.42,
                "surprise_value": 0.3,
                "timestamp": int(time.time() * 1000) - 5000,
            },
        ]

        if self.pattern_learner and hasattr(self.pattern_learner, "get_recent_patterns"):
            try:
                real_patterns = self.pattern_learner.get_recent_patterns()
                return real_patterns if real_patterns else demo_patterns
            except Exception as e:
                logger.warning(f"Failed to get real patterns: {e}")

        return demo_patterns

    async def _process_learning_event(self, event: dict) -> dict:
        """Process a learning event and return results"""
        if self.pattern_learner:
            try:
                result = await asyncio.create_task(asyncio.to_thread(self.pattern_learner.process_event, event))
                return result
            except Exception as e:
                logger.error(f"Learning event processing failed: {e}")

        # Demo response
        return {
            "success": True,
            "patterns_found": 1,
            "consciousness_delta": 0.001,
            "surprise_score": 0.3,
        }

    async def _check_achievements(self, metrics: dict):
        """Check for new achievements and notify clients"""
        achievements = []

        # IQ milestones
        if metrics.get("jeffrey_iq", 0) >= 130 and "iq_130" not in self.achievement_history:
            achievements.append(
                {
                    "id": "iq_130",
                    "title": "Génie Émergent",
                    "description": "QI de Jeffrey dépasse 130 points",
                    "icon": "🧠",
                    "rarity": "rare",
                }
            )
            self.achievement_history.append("iq_130")

        # Consciousness milestones
        if metrics.get("consciousness_level", 0) >= 0.5 and "consciousness_50" not in self.achievement_history:
            achievements.append(
                {
                    "id": "consciousness_50",
                    "title": "Éveil Numérique",
                    "description": "Niveau de conscience atteint 50%",
                    "icon": "🌟",
                    "rarity": "epic",
                }
            )
            self.achievement_history.append("consciousness_50")

        # Pattern detection milestones
        if metrics.get("patterns_detected", 0) >= 100 and "patterns_100" not in self.achievement_history:
            achievements.append(
                {
                    "id": "patterns_100",
                    "title": "Détecteur Expert",
                    "description": "100 patterns détectés avec succès",
                    "icon": "🎯",
                    "rarity": "uncommon",
                }
            )
            self.achievement_history.append("patterns_100")

        # Notify clients of new achievements
        for achievement in achievements:
            await self.sio.emit("achievement_unlocked", achievement)
            logger.info(f"🏆 Achievement unlocked: {achievement['title']}")

    async def start_metrics_broadcast(self):
        """Start periodic metrics broadcasting"""
        while True:
            try:
                if self.connected_clients:
                    # Get and broadcast current metrics
                    metrics = await self._get_current_metrics()
                    await self.sio.emit("metrics_update", metrics)

                    # Get and broadcast system health
                    system_health = await self._get_system_health()
                    await self.sio.emit("system_health", system_health)

                    # Check for achievements
                    await self._check_achievements(metrics)

                    # Simulate pattern detection (demo mode)
                    if len(self.connected_clients) > 0 and time.time() % 10 < 1:
                        patterns = await self._get_recent_patterns()
                        if patterns:
                            await self.sio.emit("pattern_detected", patterns[0])

                await asyncio.sleep(2)  # Broadcast every 2 seconds

            except Exception as e:
                logger.error(f"Metrics broadcast error: {e}")
                await asyncio.sleep(5)

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the WebSocket server"""
        logger.info(f"🚀 Starting Jeffrey OS WebSocket server on {host}:{port}")

        # Start metrics broadcasting task
        await asyncio.create_task(self.start_metrics_broadcast())

        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        logger.info(f"✅ Jeffrey OS WebSocket server running on http://{host}:{port}")
        logger.info(f"📊 Dashboard available at http://{host}:{port}/health")

        return runner


async def main():
    """Main entry point"""
    # Initialize pattern learner if available
    pattern_learner = None
    if PatternLearnerML:
        try:
            config = {
                "model_name": "microsoft/DialoGPT-medium",
                "device": "cpu",
                "max_length": 512,
                "learning_rate": 2e-5,
            }
            pattern_learner = PatternLearnerML(config)
            logger.info("✅ Pattern Learner initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Pattern Learner: {e}")

    # Create and start WebSocket handler
    handler = JeffreyWebSocketHandler(pattern_learner)
    runner = await handler.start_server()

    try:
        # Keep server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("👋 Shutting down Jeffrey OS WebSocket server...")
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Jeffrey OS WebSocket server stopped")
