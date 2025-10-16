"""
CircadianRunner - Manages biological rhythm as background task
Implements Gemini's vision of internal life independent of requests
"""
import asyncio
import logging
import math
import os
from datetime import datetime, time
from typing import Optional

from jeffrey.core.neuralbus.events import make_event, CIRCADIAN_UPDATE

logger = logging.getLogger(__name__)

class CircadianRunner:
    """
    Manages circadian rhythm as background task
    Gives Jeffrey temporal awareness and biological cycles
    """

    def __init__(self, bus, interval_sec: int = 300):
        self.bus = bus
        # TWEAK 6: Interval configurable par env var
        self.interval = int(os.getenv("JEFFREY_CIRCADIAN_INTERVAL", str(interval_sec)))
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # State tracking
        self.start_time = datetime.now()
        self.current_phase = "dawn"
        self.energy_level = 0.7

        # Phase definitions (simplified 24h cycle)
        self.phases = {
            "night": (0, 6),    # 00:00 - 06:00
            "dawn": (6, 9),     # 06:00 - 09:00
            "morning": (9, 12), # 09:00 - 12:00
            "noon": (12, 15),   # 12:00 - 15:00
            "afternoon": (15, 18), # 15:00 - 18:00
            "dusk": (18, 21),   # 18:00 - 21:00
            "evening": (21, 24) # 21:00 - 00:00
        }

    def get_current_phase(self) -> str:
        """Determine current phase based on time of day"""
        current_hour = datetime.now().hour

        for phase, (start, end) in self.phases.items():
            if start <= current_hour < end:
                return phase

        return "night"  # Default

    def get_energy_level(self) -> float:
        """
        Calculate energy level based on circadian rhythm
        Uses sine wave for natural fluctuation
        """
        current_hour = datetime.now().hour

        # Peak energy at 10am and 4pm, lowest at 3am
        # Using dual-peak circadian model
        primary_peak = 10
        secondary_peak = 16
        lowest_point = 3

        # Calculate based on distance from peaks
        if abs(current_hour - primary_peak) <= 2:
            energy = 0.9 + 0.1 * (1 - abs(current_hour - primary_peak) / 2)
        elif abs(current_hour - secondary_peak) <= 2:
            energy = 0.8 + 0.1 * (1 - abs(current_hour - secondary_peak) / 2)
        elif abs(current_hour - lowest_point) <= 3:
            energy = 0.3 - 0.1 * (1 - abs(current_hour - lowest_point) / 3)
        else:
            # Smooth transition using sine wave
            angle = (current_hour * 15) % 360  # 15 degrees per hour
            energy = 0.5 + 0.3 * math.sin(math.radians(angle))

        return max(0.1, min(1.0, energy))

    async def _loop(self):
        """Main circadian loop - the heartbeat of temporal awareness"""
        self._running = True
        logger.info(f"🌅 Circadian rhythm started - Jeffrey now has temporal awareness (interval: {self.interval}s)")

        while self._running:
            try:
                # Get current state
                phase = self.get_current_phase()
                energy = self.get_energy_level()

                # Update internal state
                phase_changed = (phase != self.current_phase)
                self.current_phase = phase
                self.energy_level = energy

                # Create event data
                event_data = {
                    "phase": phase,
                    "energy_level": round(energy, 2),
                    "phase_changed": phase_changed,
                    "timestamp": datetime.now().isoformat(),
                    "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
                }

                # Special events for phase transitions
                if phase_changed:
                    if phase == "night":
                        event_data["trigger_dreams"] = True
                        logger.info("🌙 Night phase - dream consolidation triggered")
                    elif phase == "dawn":
                        event_data["trigger_awakening"] = True
                        logger.info("🌅 Dawn phase - awakening sequence initiated")

                # Publish circadian update
                event = make_event(
                    CIRCADIAN_UPDATE,
                    event_data,
                    source="jeffrey.biorhythms.circadian"
                )

                await self.bus.publish(event)

                logger.debug(f"⏰ Circadian: {phase} (energy: {energy:.0%})")

            except Exception as e:
                logger.error(f"Circadian tick failed: {e}")

            # Wait for next update
            await asyncio.sleep(self.interval)

    def start(self):
        """Start the circadian rhythm"""
        if not self._task:
            self._running = True  # Set before creating task
            self._task = asyncio.create_task(self._loop())
            return True
        return False

    async def stop(self):
        """Stop the circadian rhythm gracefully"""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("🌑 Circadian rhythm stopped")

    def get_stats(self) -> dict:
        """Get current stats for monitoring"""
        return {
            "running": self._running,
            "current_phase": self.current_phase,
            "energy_level": self.energy_level,
            "interval_sec": self.interval,
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }