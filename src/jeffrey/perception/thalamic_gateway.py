"""
Module système pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module système pour jeffrey os.
Il fournit une architecture robuste et évolutive intégrant les composants
nécessaires au fonctionnement optimal du système. L'implémentation suit
les principes de modularité et d'extensibilité pour faciliter l'évolution
future du système.

Le module gère l'initialisation, la configuration, le traitement des données,
la communication inter-composants, et la persistance des états. Il s'intègre
harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
une séparation claire des responsabilités.

L'architecture interne permet une évolution adaptative basée sur les interactions
et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
cohérente et authentique.
"""

from __future__ import annotations

from datetime import datetime

from jeffrey.core.neural_envelope import NeuralEnvelope


class ThalamicGateway:
    """Filtre sensoriel comme le thalamus humain"""

    def __init__(self) -> None:
        self.noise_threshold = 0.1
        self.reflex_patterns = {
            "greeting": ["hello", "hi", "bonjour", "salut"],
            "status": ["status", "health", "alive"],
        }
        self.recall_cache = {}  # Cache pour éviter requêtes répétées
        self.cache_ttl = 0.2  # 200ms

    async def _fetch_recall_cached(self, env: NeuralEnvelope):
        """CORRECTION GPT: Méthode de classe (pas fonction imbriquée)"""
        cache_key = f"{env.ns}:{env.cid}"
        now = datetime.utcnow().timestamp()

        if cache_key in self.recall_cache:
            cached_time, cached_result = self.recall_cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return cached_result

        recall_env = NeuralEnvelope(
            ns=env.ns, topic="mem.recall", payload={"query": env.payload.get("text", ""), "k": 3}
        )
        result = await self.bus.request(f"{env.ns}.mem.recall", recall_env)
        self.recall_cache[cache_key] = (now, result)
        return result

    def _compute_salience(self, env: NeuralEnvelope) -> float:
        """MÉTHODE de classe (pas fonction dans fonction)"""
        factors = {
            "emotional": env.affect.get("intensity", 0.5) if env.affect else 0.5,
            "urgency": env.urgency,
            "confidence": 1.0 - env.confidence,
            "risk": 1.0 if "pii" in env.tags else 0.3,
        }

        weights = {"emotional": 0.3, "urgency": 0.3, "confidence": 0.2, "risk": 0.2}
        salience = sum(factors[k] * weights[k] for k in factors)

        return min(1.0, max(0.0, salience))

    async def start(self, bus, workspace, registry):
        self.bus = bus
        self.workspace = workspace

        async def filter_input(env: NeuralEnvelope):
            """
            ORDRE CRITIQUE :
            1. Guardian check (sécurité)
            2. Émotion (affect)
            3. Salience
            4. Routage S1/S2
            """
            # 1. GUARDIAN AVANT TOUT
            await self.bus.request(f"{env.ns}.guardian.check", env)

            env.add_to_path("thalamus")

            # 2. Évaluation émotionnelle
            affect_env = NeuralEnvelope(ns=env.ns, topic="affect.appraise", payload=env.payload)
            affect = await self.bus.request(f"{env.ns}.affect.appraise", affect_env)
            env.affect = affect

            # 3. Calcul salience
            salience = self._compute_salience(env)
            env.salience = salience

            # 4. Filtrer bruit
            if salience < self.noise_threshold:
                return

            # 5. Check patterns S1 (réflexes)
            text_lower = env.payload.get("text", "").lower()
            for pattern_type, keywords in self.reflex_patterns.items():
                if any(kw in text_lower for kw in keywords):
                    await self.bus.emit(f"{env.ns}.plan.fast", env)
                    return

            # 6. Check mémoire avec CACHE
            recall = await self._fetch_recall_cached(env)
            novelty = 1.0 - recall.get("max_similarity", 0.0)

            # 7. Décision S1 vs S2
            needs_s2 = novelty > 0.6 or env.affect.get("intensity", 0) > 0.6 or env.urgency > 0.7 or "pii" in env.tags

            if needs_s2:
                await self.workspace.propose(env, salience, "thalamus")
            else:
                await self.bus.emit(f"{env.ns}.plan.fast", env)

        bus.subscribe("percept.text", filter_input)

        await registry.register(
            "thalamic_gateway",
            self,
            topics_in=["percept.text"],
            topics_out=["plan.fast", "workspace.propose"],
        )
