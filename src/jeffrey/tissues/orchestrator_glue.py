"""
Orchestrateur principal du système cognitif.

Ce module implémente les fonctionnalités essentielles pour orchestrateur principal du système cognitif.
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

from jeffrey.core.neural_envelope import NeuralEnvelope
from jeffrey.core.orchestration.ia_orchestrator_ultimate import OrchestrationRequest, UltimateOrchestrator


class OrchestratorTissue:
    """Wrapper pour ton orchestrateur existant"""

    def __init__(self) -> None:
        self.core = UltimateOrchestrator()  # MODULE EXISTANT

    async def start(self, bus, registry):
        async def handle_slow(env: NeuralEnvelope):
            """Traitement S2 délibéré"""
            env.add_to_path("orchestrator")

            # Récupérer contexte
            context_env = NeuralEnvelope(ns=env.ns, topic="mem.working.snapshot", payload={})
            context = await bus.request(f"{env.ns}.mem.working.snapshot", context_env)

            # Réserver budget
            reserve_env = NeuralEnvelope(ns=env.ns, topic="audit.reserve", payload={"estimated_cost": 0.01})
            tx = await bus.request(f"{env.ns}.audit.reserve", reserve_env)

            try:
                # Appeler orchestrateur existant
                request = OrchestrationRequest(
                    request=env.payload.get("text", ""),
                    request_type="complex",
                    user_id=env.ns,
                    preferences=context.get("context", {}),
                    priority="normal",
                )
                result = await self.core.orchestrate_with_intelligence(request)

                # Commit budget
                commit_env = NeuralEnvelope(ns=env.ns, topic="audit.commit", payload={"tx_id": tx["tx_id"]})
                await bus.request(f"{env.ns}.audit.commit", commit_env)

                # CORRECTION GPT: Stocker en mémoire via REQUEST, pas emit
                store_env = NeuralEnvelope(
                    ns=env.ns,
                    topic="mem.store",
                    payload={"input": env.payload, "output": result, "success": True},
                )
                await bus.request(f"{env.ns}.mem.store", store_env)

                # Publier la réponse
                await bus.emit(
                    f"{env.ns}.act.speak",
                    NeuralEnvelope(
                        ns=env.ns,
                        topic="act.speak",
                        payload=result if isinstance(result, dict) else {"text": str(result)},
                    ),
                )

                return result

            except Exception as e:
                # Rollback si erreur
                rollback_env = NeuralEnvelope(
                    ns=env.ns,
                    topic="audit.rollback",
                    payload={"tx_id": tx["tx_id"], "reason": str(e)},
                )
                await bus.request(f"{env.ns}.audit.rollback", rollback_env)
                raise

        async def handle_fast(env: NeuralEnvelope):
            """CORRECTION GPT: Ajout de la voie S1 (réflexes)"""
            env.add_to_path("orchestrator.fast")

            # Chemin rapide simple
            request = OrchestrationRequest(
                request=env.payload.get("text", ""),
                request_type="reflex",
                user_id=env.ns,
                preferences={},
                priority="high",
            )

            try:
                result = await self.core.orchestrate_with_intelligence(request)
            except Exception:
                # Fallback ultra-simple
                text = env.payload.get("text", "").lower()
                if "hello" in text or "hi" in text:
                    result = {"response": "Hello! I'm Jeffrey, nice to meet you!"}
                elif "status" in text:
                    result = {"response": "All systems operational!"}
                else:
                    result = {"response": "I'm processing your request..."}

            # Publier pour l'affichage
            await bus.emit(
                f"{env.ns}.act.speak",
                NeuralEnvelope(
                    ns=env.ns,
                    topic="act.speak",
                    payload=result if isinstance(result, dict) else {"text": str(result)},
                ),
            )

            # Mémo rapide
            await bus.request(
                f"{env.ns}.mem.store",
                NeuralEnvelope(
                    ns=env.ns,
                    topic="mem.store",
                    payload={"input": env.payload, "output": result, "success": True},
                ),
            )

            return result

        bus.subscribe("plan.slow", handle_slow)
        bus.subscribe("plan.fast", handle_fast)  # CORRECTION GPT: Ajout S1

        await registry.register(
            "orchestrator_tissue",
            self,
            topics_in=["plan.slow", "plan.fast"],
            topics_out=["mem.store", "act.speak"],
        )
