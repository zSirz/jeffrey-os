# jeffrey_os/mood/mood_tracker.py
from __future__ import annotations

import math
import random
import time
from collections import deque

import networkx as nx

from ..eventbus import EventBus


class MoodAnomalyDetector:
    """Détection d'anomalies dans les patterns d'humeur"""


def __init__(self) -> None:
    self.threshold_sudden_change = 0.5  # Change > 0.5 in valence
    self.threshold_extreme_negative = -0.8  # Very negative mood
    self.threshold_extreme_positive = 0.9  # Suspiciously positive

    async def check_async(self, mood_history: deque) -> bool:
        """Check for anomalies asynchronously"""

    if len(mood_history) < 2:
        return False

    # Get last two entries
    current = mood_history[-1]['emotion']
    previous = mood_history[-2]['emotion']

    # Check for sudden dramatic changes
    valence_change = abs(current.get('valence', 0) - previous.get('valence', 0))
    if valence_change > self.threshold_sudden_change:
        return True

    # Check for extreme states
    current_valence = current.get('valence', 0)
    if current_valence < self.threshold_extreme_negative or current_valence > self.threshold_extreme_positive:
        return True

    return False


def check(self, mood_history: deque) -> bool:
    """Synchronous version for backward compatibility"""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.check_async(mood_history))
    except RuntimeError:
        # No event loop, run sync version
        return self._check_sync(mood_history)


def _check_sync(self, mood_history: deque) -> bool:
    """Synchronous anomaly check"""
    if len(mood_history) < 2:
        return False

    current = mood_history[-1]['emotion']
    previous = mood_history[-2]['emotion']

    valence_change = abs(current.get('valence', 0) - previous.get('valence', 0))
    current_valence = current.get('valence', 0)

    return (
        valence_change > self.threshold_sudden_change
        or current_valence < self.threshold_extreme_negative
        or current_valence > self.threshold_extreme_positive
    )


class DistributedMoodTracker:
    """Tracking avec anti-echo chamber et differential privacy"""

    def __init__(self, mesh=None, config=None) -> None:
        self.mood_curves = {}  # Per node
        self.global_mood = {'valence': 0, 'energy': 0.5}
        self.anomaly_detector = MoodAnomalyDetector()
        self.quarantine = {}  # Nodes en isolation temporaire
        self.update_counter = 0
        self.contagion_graph = nx.DiGraph()  # Track propagation
        self.privacy_noise_level = 0.05  # Differential privacy
        self.eventbus = EventBus()
        self.mesh = mesh

        # Batch processing et throttling (configuration maintenance)
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 10)
        self.throttle_interval = self.config.get('throttle_interval', 1.0)  # secondes
        self.last_batch_time = time.time()
        self.pending_updates = []  # Queue pour batch processing
        self.high_frequency_threshold = self.config.get('high_frequency_threshold', 5)  # updates/sec
        self.last_update_times = deque(maxlen=50)  # Track frequency

        # Subscribe to mesh events if available
        if self.mesh:
            self.mesh.eventbus.subscribe("node_removed", self.on_node_removed)

    async def track_mood_evolution(self, node_id: str, emotion: dict):
        """Track avec privacy noise, batch processing et throttling avancé"""
        current_time = time.time()
        self.last_update_times.append(current_time)

        # Détection haute fréquence
        if len(self.last_update_times) >= 10:
            recent_times = [t for t in self.last_update_times if current_time - t <= 1.0]
            if len(recent_times) > self.high_frequency_threshold:
                # Mode batch forcé pour gérer haute fréquence
                await self._add_to_batch(node_id, emotion)
                return

        # Batch processing si intervalle pas atteint
        if current_time - self.last_batch_time < self.throttle_interval:
            await self._add_to_batch(node_id, emotion)
            return

        # Process immediately si pas de throttling nécessaire
        await self._process_mood_update(node_id, emotion)

        # Process pending batch si accumulé
        if self.pending_updates:
            await self._process_batch()

    async def _add_to_batch(self, node_id: str, emotion: dict):
        """Ajoute update au batch queue"""
        self.pending_updates.append({'node_id': node_id, 'emotion': emotion, 'timestamp': time.time()})

        # Process batch si taille limite atteinte
        if len(self.pending_updates) >= self.batch_size:
            await self._process_batch()

    async def _process_batch(self):
        """Process tous les updates en batch"""
        if not self.pending_updates:
            return

        # Group par node pour optimisation
        updates_by_node = {}
        for update in self.pending_updates:
            node_id = update['node_id']
            if node_id not in updates_by_node:
                updates_by_node[node_id] = []
            updates_by_node[node_id].append(update)

        # Process chaque node
        for node_id, updates in updates_by_node.items():
            # Utilise le dernier update pour ce node (plus récent)
            latest_update = max(updates, key=lambda x: x['timestamp'])
            await self._process_mood_update(node_id, latest_update['emotion'])

        # Clear batch
        self.pending_updates.clear()
        self.last_batch_time = time.time()

        await self.eventbus.emit(
            "batch_processed",
            {
                "updates_count": sum(len(updates) for updates in updates_by_node.values()),
                "nodes_affected": len(updates_by_node),
                "timestamp": time.time(),
            },
        )

    async def _process_mood_update(self, node_id: str, emotion: dict):
        """Process individual mood update"""
        # Update local curve avec differential privacy
        if node_id not in self.mood_curves:
            self.mood_curves[node_id] = deque(maxlen=288)  # 24h at 5min

        # Add privacy noise
        noisy_emotion = self.add_differential_privacy_noise(emotion)

        self.mood_curves[node_id].append({'timestamp': time.time(), 'emotion': noisy_emotion, 'anonymized': True})

        # Throttling : check anomalies toutes les 5 updates
        self.update_counter += 1
        if self.update_counter % 5 == 0:
            # Check async pour performance
            anomaly = await self.anomaly_detector.check_async(self.mood_curves[node_id])
            if anomaly:
                await self.handle_mood_anomaly(node_id)

        # Quarantine dynamique avec calcul severity corrigé et jitter
        await self._check_quarantine(node_id, emotion)

        # Update contagion graph pour anti-echo
        await self.update_contagion_graph(node_id, emotion)

        # Update global mood avec protection contagion
        await self.update_global_mood_safe()

    def add_differential_privacy_noise(self, emotion: dict) -> dict:
        """Ajoute bruit pour privacy préservation"""
        noisy = emotion.copy()

        # Add Gaussian noise to continuous values
        for key in ['valence', 'arousal', 'dominance']:
            if key in noisy and not math.isnan(noisy[key]):
                noise = random.gauss(0, self.privacy_noise_level)
                noisy[key] = max(-1, min(1, noisy[key] + noise))

        return noisy

    async def update_contagion_graph(self, node_id: str, emotion: dict):
        """Track propagation émotionnelle pour détecter echo chambers"""
        valence = emotion.get('valence', 0)

        # Add node state
        self.contagion_graph.add_node(node_id, **emotion)

        # Check si influence négative se propage
        if valence < -0.3:
            # Track influences from mesh neighbors if available
            neighbors = []
            if self.mesh and hasattr(self.mesh, 'graph'):
                try:
                    neighbors = list(self.mesh.graph.neighbors(node_id))
                except:
                    # Node might not be in mesh graph yet
                    pass

            # If no mesh, check other tracked nodes
            if not neighbors:
                neighbors = [nid for nid in self.mood_curves.keys() if nid != node_id]

            for neighbor in neighbors:
                if neighbor in self.mood_curves and self.mood_curves[neighbor]:
                    neighbor_mood = self.mood_curves[neighbor][-1]['emotion']
                    if neighbor_mood.get('valence', 0) < -0.2:
                        # Add edge with weight based on negative influence
                        weight = abs(valence)
                        self.contagion_graph.add_edge(node_id, neighbor, weight=weight)

        # Détecte echo chamber si trop de connexions négatives
        total_nodes = len(self.mood_curves)
        if total_nodes > 0 and self.contagion_graph.number_of_edges() > total_nodes * 0.3:
            try:
                negative_components = [c for c in nx.strongly_connected_components(self.contagion_graph) if len(c) > 2]
                if negative_components:
                    largest_component = max(negative_components, key=len)
                    severity = len(largest_component) / total_nodes

                    await self.eventbus.emit(
                        "echo_chamber_detected",
                        {
                            "components": [list(c) for c in negative_components],
                            "largest_component_size": len(largest_component),
                            "severity": severity,
                            "total_nodes": total_nodes,
                        },
                    )
            except Exception as e:
                # NetworkX operations can fail, log but continue
                await self.eventbus.emit("contagion_analysis_error", {"error": str(e), "node_id": node_id})

    async def update_global_mood_safe(self):
        """Update global avec protection anti-contagion"""
        # Clean expired quarantine entries
        current_time = time.time()
        expired_quarantine = [node for node, end_time in self.quarantine.items() if current_time > end_time]
        for node in expired_quarantine:
            del self.quarantine[node]
            await self.eventbus.emit(
                "node_released_from_quarantine",
                {"node": node, "duration": current_time - (self.quarantine.get(node, current_time) - 300)},
            )

        # Calcul pondéré excluant quarantined nodes
        active_moods = []

        for node_id, curve in self.mood_curves.items():
            if node_id not in self.quarantine and curve:
                latest = curve[-1]['emotion']
                # Pondération par trust/confidence
                weight = latest.get('confidence', 0.5) * latest.get('trust', 0.8)
                if weight > 0:
                    active_moods.append((latest, weight))

        if not active_moods:
            return

        # Moyenne pondérée
        total_weight = sum(w for _, w in active_moods)
        if total_weight > 0:
            old_valence = self.global_mood['valence']
            old_energy = self.global_mood['energy']

            self.global_mood['valence'] = sum(m.get('valence', 0) * w for m, w in active_moods) / total_weight
            self.global_mood['energy'] = sum(m.get('arousal', 0) * w for m, w in active_moods) / total_weight

            # Emit global mood change if significant
            valence_change = abs(self.global_mood['valence'] - old_valence)
            energy_change = abs(self.global_mood['energy'] - old_energy)

            if valence_change > 0.1 or energy_change > 0.1:
                await self.eventbus.emit(
                    "global_mood_changed",
                    {
                        "new_mood": self.global_mood.copy(),
                        "valence_change": valence_change,
                        "energy_change": energy_change,
                        "active_nodes": len(active_moods),
                        "quarantined_nodes": len(self.quarantine),
                    },
                )

    async def handle_mood_anomaly(self, node_id: str):
        """Handle detected mood anomaly"""
        if node_id in self.mood_curves and self.mood_curves[node_id]:
            latest_mood = self.mood_curves[node_id][-1]['emotion']

            await self.eventbus.emit(
                "mood_anomaly_detected",
                {
                    "node_id": node_id,
                    "mood": latest_mood,
                    "timestamp": time.time(),
                    "severity": self.calculate_anomaly_severity(latest_mood),
                },
            )

    def calculate_anomaly_severity(self, mood: dict) -> str:
        """Calculate severity of mood anomaly"""
        valence = mood.get('valence', 0)
        arousal = mood.get('arousal', 0)

        if valence < -0.8:
            return "critical"
        elif valence < -0.5 or abs(arousal) > 0.8:
            return "high"
        elif valence < -0.3:
            return "medium"
        else:
            return "low"

    async def on_node_removed(self, event):
        """Handle node removal from mesh"""
        node_id = event.data.get('node_id') if hasattr(event, 'data') else event.get('node_id')

        if node_id:
            # Remove from tracking
            if node_id in self.mood_curves:
                del self.mood_curves[node_id]

            # Remove from quarantine
            if node_id in self.quarantine:
                del self.quarantine[node_id]

            # Remove from contagion graph
            if self.contagion_graph.has_node(node_id):
                self.contagion_graph.remove_node(node_id)

    def get_mood_statistics(self) -> dict:
        """Get comprehensive mood statistics"""
        stats = {
            "total_nodes": len(self.mood_curves),
            "active_nodes": len([c for c in self.mood_curves.values() if c]),
            "quarantined_nodes": len(self.quarantine),
            "global_mood": self.global_mood.copy(),
            "contagion_edges": self.contagion_graph.number_of_edges(),
            "mood_variance": 0.0,
        }

        # Calculate mood variance
        if len(self.mood_curves) > 1:
            valences = []
            for curve in self.mood_curves.values():
                if curve:
                    valences.append(curve[-1]['emotion'].get('valence', 0))

            if valences:
                mean_valence = sum(valences) / len(valences)
                variance = sum((v - mean_valence) ** 2 for v in valences) / len(valences)
                stats["mood_variance"] = variance

        return stats

    def get_node_mood_history(self, node_id: str, hours: int = 24) -> list[dict]:
        """Get mood history for a specific node"""
        if node_id not in self.mood_curves:
            return []

        cutoff_time = time.time() - (hours * 3600)
        history = []

        for entry in self.mood_curves[node_id]:
            if entry['timestamp'] > cutoff_time:
                history.append(entry)

        return history

    async def reset_contagion_graph(self):
        """Reset contagion graph to break echo chambers"""
        self.contagion_graph.clear()
        await self.eventbus.emit("contagion_graph_reset", {"timestamp": time.time(), "reason": "manual_reset"})

    def is_node_quarantined(self, node_id: str) -> bool:
        """Check if node is currently quarantined"""
        if node_id not in self.quarantine:
            return False
        return time.time() < self.quarantine[node_id]

    async def force_release_quarantine(self, node_id: str):
        """Force release a node from quarantine"""
        if node_id in self.quarantine:
            del self.quarantine[node_id]
            await self.eventbus.emit("quarantine_force_released", {"node_id": node_id, "timestamp": time.time()})

    def get_batch_statistics(self) -> dict:
        """Get batch processing and throttling statistics"""
        recent_updates = [t for t in self.last_update_times if time.time() - t <= 60.0]  # Last minute

        return {
            "pending_updates": len(self.pending_updates),
            "batch_size": self.batch_size,
            "throttle_interval": self.throttle_interval,
            "last_batch_time": self.last_batch_time,
            "updates_per_minute": len(recent_updates),
            "high_frequency_threshold": self.high_frequency_threshold,
            "time_since_last_batch": time.time() - self.last_batch_time,
            "is_high_frequency": len(recent_updates) > self.high_frequency_threshold * 60,
        }

    async def force_process_batch(self):
        """Force process pending batch (utile pour tests/debugging)"""
        if self.pending_updates:
            await self._process_batch()

    def update_batch_config(self, config: dict) -> None:
        """Update batch processing configuration"""
        if 'batch_size' in config:
            self.batch_size = max(1, config['batch_size'])
        if 'throttle_interval' in config:
            self.throttle_interval = max(0.1, config['throttle_interval'])
        if 'high_frequency_threshold' in config:
            self.high_frequency_threshold = max(1, config['high_frequency_threshold'])

    async def _check_quarantine(self, node_id: str, emotion: dict):
        """Quarantine avec calcul severity corrigé et jitter"""
        valence = emotion.get('valence', 0)

        if valence < -0.5 and node_id not in self.quarantine:
            # Calcul sévérité corrigé pour edge case -0.5
            valence_normalized = max(-1, min(0, valence))

            # Severity: 0 at threshold (-0.5), 1 at extreme (-1)
            if valence_normalized <= -0.5:
                severity = min(1.0, abs(valence_normalized + 0.5) * 2)  # Scale 0-1
            else:
                severity = 0.0  # Below threshold

            confidence = max(0.1, min(1.0, emotion.get('confidence', 0.5)))

            # Duration calculation avec jitter
            base_duration = 300
            severity_factor = 1 + severity  # 1 to 2
            confidence_factor = 2 - confidence  # 1 to 1.9

            calculated_duration = base_duration * severity_factor * confidence_factor / 2

            # Add jitter ±20% for anti-pattern
            jitter = random.uniform(0.8, 1.2)
            duration_with_jitter = calculated_duration * jitter

            # Strict bounds 150-600s
            duration = int(max(150, min(600, duration_with_jitter)))

            self.quarantine[node_id] = time.time() + duration

            # Log COMPLET avec jitter info
            await self.eventbus.emit(
                "node_quarantined",
                {
                    "node": node_id,
                    "reason": "extreme_negative_valence",
                    "valence": float(valence),
                    "valence_normalized": float(valence_normalized),
                    "severity": float(severity),
                    "confidence": float(confidence),
                    "duration": duration,
                    "duration_calculated": float(calculated_duration),
                    "jitter": float(jitter),
                    "timestamp": time.time(),
                    "thresholds": {
                        "valence_threshold": -0.5,
                        "base_duration": base_duration,
                        "min_duration": 150,
                        "max_duration": 600,
                    },
                },
            )
