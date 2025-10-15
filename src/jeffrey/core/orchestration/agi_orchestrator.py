"""
AGI Orchestrator - Chef d'orchestre central du système unifié

Coordonne:
- Pipeline de traitement intelligent
- Intégration mémoire/émotion/dialogue
- Systèmes AGI avancés (consciousness, meta-cognition, etc.)
- Gestion d'erreurs et fallbacks
- Métriques de performance

Architecture centrale pour Jeffrey V1.1 AGI Fusion
"""

import logging
import random
import time
from datetime import datetime
from typing import Any

from .dialogue_engine import DialogueEngine
from .emotion_engine_bridge import get_emotion_bridge
from .unified_memory import get_unified_memory

try:
    from .unified_memory import UnifiedMemory as MemoryManager
except Exception:
    MemoryManager = None  # fallback temporaire

# Configure logger first
logger = logging.getLogger(__name__)

# Import Memory Systems V2.0
try:
    import os
    import sys

    src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from jeffrey.core.memory_interface import get_agi_memory_integration, get_memory_interface
    from jeffrey.core.memory_systems import get_memory_core, initialize_memory_systems

    MEMORY_V2_AVAILABLE = True
    logger.info("✅ Memory Systems V2.0 detected and ready for integration")
except ImportError as e:
    logger.warning(f"Memory Systems V2.0 not available: {e}")
    MEMORY_V2_AVAILABLE = False

# Import des systèmes AGI avancés
try:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    # ==========================================
    # Import des 15 systèmes AGI RÉELS
    # ==========================================

    AGI_SYSTEMS_AVAILABLE = False
    HAS_AGI_SYNTHESIS = False
    HAS_CONSCIOUSNESS = False

    # Import AGI Synthesis (5 modules)
    try:
        from jeffrey_agi_synthesis import (
            ContextualEmpathy,
            EmotionalJournal,
            JeffreyBiorhythms,
            NarrativeMemory,
            SecureImaginationEngine,
        )

        HAS_AGI_SYNTHESIS = True
        logger.info(
            "✅ AGI Synthesis: 5 modules loaded (EmotionalJournal, ContextualEmpathy, NarrativeMemory, SecureImaginationEngine, JeffreyBiorhythms)"
        )
    except ImportError as e:
        logger.error(f"❌ Failed to import AGI Synthesis modules: {e}")
        HAS_AGI_SYNTHESIS = False

    # Import Consciousness Evolution (10 modules)
    try:
        from jeffrey_consciousness_evolution import (
            CircadianRhythm,
            CreativeMemoryWeb,
            DreamEngine,
            EmotionalMemoryManager,
            EvolutiveAttachment,
            MetaCognition,
            MicroExpressions,
            PersonalValues,
            ProactiveCuriosity,
            SubtleLearning,
        )

        HAS_CONSCIOUSNESS = True
        logger.info(
            "✅ Consciousness Evolution: 10 modules loaded (CircadianRhythm, CreativeMemoryWeb, DreamEngine, SubtleLearning, MicroExpressions, PersonalValues, EmotionalMemoryManager, MetaCognition, ProactiveCuriosity, EvolutiveAttachment)"
        )
    except ImportError as e:
        logger.error(f"❌ Failed to import Consciousness Evolution modules: {e}")
        HAS_CONSCIOUSNESS = False

    # Déterminer si les systèmes AGI sont disponibles
    AGI_SYSTEMS_AVAILABLE = HAS_AGI_SYNTHESIS and HAS_CONSCIOUSNESS

    if AGI_SYSTEMS_AVAILABLE:
        logger.info("🎉 ALL 15 AGI SYSTEMS LOADED SUCCESSFULLY!")
    elif HAS_AGI_SYNTHESIS or HAS_CONSCIOUSNESS:
        count = (5 if HAS_AGI_SYNTHESIS else 0) + (10 if HAS_CONSCIOUSNESS else 0)
        logger.warning(f"⚠️  Partial AGI systems available: {count}/15 modules")
    else:
        logger.warning("❌ No AGI systems available - running in basic mode")
except Exception as e:
    AGI_SYSTEMS_AVAILABLE = False
    logger.warning(f"AGI systems not available: {e}")


class AGIOrchestrator:
    """
    Orchestrateur AGI Central - Coordonne tous les systèmes unifiés

    Pipeline intelligent qui intègre mémoire, émotion, dialogue et
    tous les systèmes de conscience avancés pour créer des réponses
    authentiques et évolutives.
    """

    def __init__(self, config: dict = None):
        """Initialise l'orchestrateur AGI avec configuration"""
        self.config = config or {}
        self.metrics = {
            "total_requests": 0,
            "emotional_analysis": 0,
            "memory_updates": 0,
            "errors": 0,
            "avg_response_time": 0.0,
            "max_response_time": 5.0,
        }

        # Initialiser Memory Systems V2.0 si disponible
        if MEMORY_V2_AVAILABLE:
            try:
                # Initialiser Memory Systems V2.0
                self.memory_v2_core = initialize_memory_systems()
                self.memory_v2_interface = get_memory_interface()
                self.memory_v2_agi_integration = get_agi_memory_integration()

                # Intégrer avec l'orchestrator
                self.memory_v2_agi_integration.integrate_with_orchestrator(self)
                logger.info("🧠 Memory Systems V2.0 fully integrated")
                self.memory_v2_enabled = True
            except Exception as e:
                logger.error(f"Erreur initialisation Memory V2.0: {e}")
                self.memory_v2_enabled = False
        else:
            self.memory_v2_enabled = False

        # Initialiser la mémoire unifiée (fallback)
        self.memory = get_unified_memory()

        # Initialiser le moteur de dialogue
        self.dialogue_engine = DialogueEngine()

        # Initialiser l'analyseur émotionnel avec bridge hybride
        try:
            self.emotion_bridge = get_emotion_bridge()
            self.emotion_analyzer = self.emotion_bridge  # Utiliser le bridge comme analyseur

            # Intégrer Emotion Engine avec Memory Systems V2.0
            if self.memory_v2_enabled:
                self.memory_v2_agi_integration.integrate_with_emotion_engine(self.emotion_bridge)
                logger.info("🎭 Emotion Engine intégré avec Memory Systems V2.0")

            logger.info("✅ Emotion Engine Bridge initialisé en mode hybride")
        except ImportError:
            logger.warning("EmotionEngineBridge non disponible, fallback sur EmotionalCore")
            try:
                from .emotional_core import EmotionalCore

                self.emotion_analyzer = EmotionalCore()

                # Intégrer EmotionalCore avec Memory V2.0 si disponible
                if self.memory_v2_enabled:
                    self.memory_v2_agi_integration.integrate_with_emotion_engine(self.emotion_analyzer)
            except ImportError:
                logger.warning("Aucun système émotionnel disponible")
                self.emotion_analyzer = None

        # Initialiser le module d'apprentissage
        try:
            from .self_learning_module import get_learning_module

            self.learning_module = get_learning_module()
        except ImportError:
            logger.warning("SelfLearningModule non disponible")
            self.learning_module = None

        # Initialiser les systèmes AGI
        self.agi_systems_enabled = False  # ✅ default safe
        self.agi_systems = {}
        self._initialize_agi_systems()

        # Initialiser LLM (sera fait de façon asynchrone si nécessaire)
        self.llm = None

        logger.info("AGI Orchestrator initialized - Mode: HYBRID (AGI systems: %s)", bool(self.agi_systems))

    def _initialize_agi_systems(self):
        """Initialise les 15 systèmes AGI avancés (vrais modules)"""
        initialized_count = 0

        try:
            # ===== AGI SYNTHESIS SYSTEMS (5 modules) =====
            if HAS_AGI_SYNTHESIS:
                try:
                    self.agi_systems["emotional_journal"] = EmotionalJournal()
                    self.agi_systems["contextual_empathy"] = ContextualEmpathy()
                    self.agi_systems["narrative_memory"] = NarrativeMemory()
                    self.agi_systems["imagination_engine"] = SecureImaginationEngine()
                    self.agi_systems["biorhythms"] = JeffreyBiorhythms()
                    initialized_count += 5
                    logger.info("✅ AGI Synthesis: 5 systems initialized")
                except Exception as e:
                    logger.error(f"❌ Error initializing AGI Synthesis systems: {e}")

            # ===== CONSCIOUSNESS EVOLUTION SYSTEMS (10 modules) =====
            if HAS_CONSCIOUSNESS:
                try:
                    self.agi_systems["circadian_rhythm"] = CircadianRhythm()
                    self.agi_systems["creative_memory"] = CreativeMemoryWeb()
                    self.agi_systems["dream_engine"] = DreamEngine()
                    self.agi_systems["subtle_learning"] = SubtleLearning()
                    self.agi_systems["micro_expressions"] = MicroExpressions()
                    self.agi_systems["personal_values"] = PersonalValues()
                    self.agi_systems["emotional_memory_mgr"] = EmotionalMemoryManager()
                    self.agi_systems["meta_cognition"] = MetaCognition()
                    self.agi_systems["proactive_curiosity"] = ProactiveCuriosity()
                    self.agi_systems["evolutive_attachment"] = EvolutiveAttachment()
                    initialized_count += 10
                    logger.info("✅ Consciousness Evolution: 10 systems initialized")
                except Exception as e:
                    logger.error(f"❌ Error initializing Consciousness systems: {e}")

            # ===== FINAL STATUS =====
            if initialized_count == 15:
                logger.info("🎉 ALL 15 AGI SYSTEMS SUCCESSFULLY INITIALIZED!")
                self.agi_systems_enabled = True
            elif initialized_count > 0:
                logger.warning(f"⚠️  Partial initialization: {initialized_count}/15 AGI systems active")
                self.agi_systems_enabled = True
            else:
                logger.warning("❌ No AGI systems initialized - running in basic mode")
                self.agi_systems_enabled = False

        except Exception as e:
            logger.error(f"❌ Fatal error initializing AGI systems: {e}")
            self.agi_systems_enabled = False

    async def initialize_llm(self):
        """Initialise le LLM si pas déjà fait"""
        if self.llm is None:
            try:
                from jeffrey.core.llm.llm_provider import get_llm

                self.llm = get_llm()
                if await self.llm.health_check():
                    logger.info(f"✅ LLM {self.llm.provider} connecté")
                else:
                    logger.warning(f"⚠️  LLM {self.llm.provider} non accessible")
                    self.llm = None
            except Exception as e:
                logger.error(f"⚠️  Erreur LLM: {e}")
                self.llm = None

    async def chat_simple(self, user_input: str) -> str:
        """
        Chat simple pour tests.
        Version minimale sans toute la complexité cognitive.
        """
        await self.initialize_llm()

        if self.llm is None:
            return "LLM non disponible"

        try:
            response = await self.llm.generate(user_input, max_tokens=500)
            return response
        except Exception as e:
            logger.error(f"Erreur chat simple: {e}")
            return f"Erreur: {e}"

    async def process(
        self,
        user_input: str,
        user_id: str = "default",
        ai_response: str = None,
        metadata: dict = None,
    ) -> dict[str, Any]:
        """Traite l'entrée utilisateur et génère une réponse enrichie"""
        start_time = time.time()

        # 0. Rappel contextuel préalable avec Memory Systems V2.0
        memory_v2_context = {}
        if self.memory_v2_enabled:
            try:
                # Rappel contextuel automatique avant génération de réponse
                recent_memories = await self._async_recall_context(user_input, user_id)
                memory_v2_context = {
                    "recent_conversation": recent_memories.get("recent_context", []),
                    "contextual_memories": recent_memories.get("relevant_memories", []),
                    "context_summary": recent_memories.get("context_summary", ""),
                    "memory_enhanced": True,
                }
                logger.debug(f"🧠 Memory V2.0 Context: {memory_v2_context['context_summary']}")
            except Exception as e:
                logger.warning(f"Erreur rappel contextuel Memory V2.0: {e}")

        # 1. Analyse émotionnelle hybride avancée
        # === RÉCUPÉRATION CONTEXTE MÉMOIRE (structure complète) ===
        memory_context_dict = dict(memory_v2_context) if memory_v2_context else {}
        memory_context_summary = memory_context_dict.get("context_summary") or self.memory.get_context_summary()

        # Utiliser le bridge émotionnel pour une analyse sophistiquée
        emotion_analysis = {}
        emotional_state = None

        if self.emotion_analyzer:
            try:
                # Essayer d'abord le mode hybride
                if hasattr(self.emotion_analyzer, "analyze_emotion_hybrid"):
                    emotion_analysis = self.emotion_analyzer.analyze_emotion_hybrid(user_input, memory_context_summary)
                    emotional_state = self._convert_hybrid_to_emotional_state(emotion_analysis)
                    logger.debug(
                        f"🎭 Analyse hybride: {emotion_analysis.get('emotion_dominante')} (confiance: {emotion_analysis.get('confiance', 0):.1f}%)"
                    )

                    # ✅ AJOUT : Apprentissage automatique
                    if self.learning_module:
                        try:
                            self.learning_module.learn_from_interaction(
                                user_input=user_input,
                                response="analyse_en_cours",  # Pas encore de réponse finale
                                feedback=None,
                                user_emotion=emotion_analysis.get("emotion_dominante", "neutre"),
                                context={"summary": memory_context_summary, "emotion_analysis": emotion_analysis}
                                if isinstance(memory_context_summary, str)
                                else (memory_context_summary or {}),
                            )
                        except Exception as e:
                            logger.debug(f"Erreur apprentissage non bloquante: {e}")
                # Sinon, utiliser la méthode standard du bridge
                elif hasattr(self.emotion_analyzer, "analyze_emotion"):
                    emotion_analysis = self.emotion_analyzer.analyze_emotion(user_input, memory_context_summary)
                    emotional_state = self._convert_hybrid_to_emotional_state(emotion_analysis)
                # Fallback : créer un état émotionnel basique
                else:
                    emotional_state = self._create_basic_emotional_state(
                        {'emotion_dominante': 'neutre', 'intensite': 50, 'confiance': 50, 'resonance': 0.5}
                    )
                    logger.warning("Aucune méthode d'analyse émotionnelle trouvée, utilisation de l'état basique")
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse émotionnelle: {e}")
                # Créer un état émotionnel par défaut en cas d'erreur
                emotional_state = self._create_basic_emotional_state(
                    {'emotion_dominante': 'neutre', 'intensite': 50, 'confiance': 50, 'resonance': 0.5}
                )

        # 2. Récupération du contexte mémoire et suggestions d'apprentissage
        learning_suggestion = None
        if self.learning_module:
            learning_suggestion = self.learning_module.get_learning_suggestion(user_input, memory_context_summary)

        # 3. Génération de réponse via DialogueEngine (enrichie par l'apprentissage)
        try:
            # Utiliser le DialogueEngine avec tous ses paramètres
            response = self.dialogue_engine.generate_response(
                user_input=user_input,
                user_id=user_id,
                conversation_id=metadata.get("conversation_id") if metadata else None,
                override_emotion=emotional_state.primary_emotion if emotional_state else None,
            )

            # Si la réponse est vide ou trop courte, utiliser le fallback
            if not response or len(response.strip()) < 10:
                response = self._generate_fallback_response(user_input)

        except Exception as e:
            logger.error(f"Erreur dans DialogueEngine: {e}")
            response = self._generate_fallback_response(user_input)

        # 4. Capture automatique avec Memory Systems V2.0
        memory_v2_capture_result = {}
        if self.memory_v2_enabled:
            try:
                # Capture automatique de l'interaction complète
                memory_v2_capture_result = await self._async_capture_interaction(
                    user_input=user_input,
                    ai_response=response,
                    emotional_analysis=emotion_analysis,
                    user_id=user_id,
                )
                logger.debug(f"🧠 Memory V2.0 Capture: {memory_v2_capture_result.get('memory_id', 'N/A')[:8]}")
            except Exception as e:
                logger.warning(f"Erreur capture Memory V2.0: {e}")

        # 5. Mise à jour mémoire traditionnelle (fallback)
        try:
            self.memory.update(
                user_input,
                emotional_state.to_dict() if emotional_state else {},
                {"user_id": user_id, "has_ai_response": bool(ai_response)},
            )
        except Exception as e:
            logger.warning(f"Erreur mise à jour mémoire: {e}")

        # 5. Mise à jour de la relation si possible
        try:
            interaction_quality = self._calculate_interaction_quality(
                emotional_state.to_dict() if emotional_state else {}, {}
            )
            self.memory.update_relationship(user_id, interaction_quality)
        except Exception as e:
            logger.warning(f"Erreur mise à jour relation: {e}")

        # 6. Apprentissage de l'interaction
        if self.learning_module:
            try:
                logger.debug(
                    f"🎓 Apprentissage: user_input='{user_input[:50]}', response='{response[:50]}', emotion='{emotional_state.primary_emotion if emotional_state else 'neutral'}'"
                )
                result = self.learning_module.learn_from_interaction(
                    user_input=user_input,
                    response=response,
                    feedback=None,  # Pas de feedback explicite pour le moment
                    user_emotion=emotional_state.primary_emotion if emotional_state else "neutral",
                    context={
                        "user_id": user_id,
                        "memory_context": memory_context_dict,
                        "response_time": time.time() - start_time,
                        "learning_suggestion": learning_suggestion,
                    },
                )
                logger.debug(f"✅ Apprentissage résultat: {result}")
            except Exception as e:
                logger.warning(f"Erreur apprentissage: {e}")
                import traceback

                logger.debug(f"Traceback: {traceback.format_exc()}")

        # Mesurer la performance
        response_time = time.time() - start_time
        self._update_metrics(response_time)

        # Résultat complet avec Memory Systems V2.0
        result = {
            "response": response,
            "emotional_state": emotional_state.to_dict() if emotional_state else {},
            "memory_context": memory_context_dict,
            "learning_insights": learning_suggestion,
            "memory_v2_context": memory_v2_context,
            "memory_v2_capture": memory_v2_capture_result,
            "metrics": {
                "response_time": response_time,
                "memory_updates": self.metrics["memory_updates"],
                "emotional_analysis": self.metrics["emotional_analysis"],
                "memory_v2_enabled": self.memory_v2_enabled,
            },
        }

        return result

    def _process_with_agi_synthesis(
        self, user_input: str, emotional_state: dict, memory_context: dict
    ) -> dict[str, Any]:
        """
        Traite l'entrée avec les 15 systèmes AGI pour enrichissements
        """
        enrichments = {}

        try:
            # 1. Analyse circadienne et biorythmes
            if "circadian_rhythm" in self.agi_systems:
                circadian_state = self.agi_systems["circadian_rhythm"].get_current_state()
                enrichments["circadian"] = circadian_state

            if "biorhythms" in self.agi_systems:
                bio_state = self.agi_systems["biorhythms"].get_current_state()
                enrichments["biorhythms"] = bio_state

            # 2. Empathie contextuelle
            if "contextual_empathy" in self.agi_systems:
                empathy_response = self.agi_systems["contextual_empathy"].analyze(user_input, emotional_state)
                enrichments["empathy"] = empathy_response

            # 3. Mémoire narrative et créative
            if "narrative_memory" in self.agi_systems:
                narrative = self.agi_systems["narrative_memory"].get_relevant_narrative(user_input)
                if narrative:
                    enrichments["narrative"] = narrative

            if "creative_memory" in self.agi_systems:
                associations = self.agi_systems["creative_memory"].get_associations(user_input)
                if associations:
                    enrichments["creative_associations"] = associations

            # 4. Méta-cognition et valeurs personnelles
            if "meta_cognition" in self.agi_systems:
                meta_thoughts = self.agi_systems["meta_cognition"].reflect_on_interaction(user_input, emotional_state)
                enrichments["meta_cognition"] = meta_thoughts

            if "personal_values" in self.agi_systems:
                value_alignment = self.agi_systems["personal_values"].evaluate_alignment(user_input)
                enrichments["values"] = value_alignment

            # 5. Curiosité proactive
            if "proactive_curiosity" in self.agi_systems:
                curiosity = self.agi_systems["proactive_curiosity"].generate_question(user_input, memory_context)
                if curiosity:
                    enrichments["curiosity"] = curiosity

            # 6. Attachement évolutif
            if "evolutive_attachment" in self.agi_systems:
                attachment = self.agi_systems["evolutive_attachment"].get_attachment_level(
                    memory_context.get("relationship_depth", 0)
                )
                enrichments["attachment"] = attachment

            # 7. Micro-expressions
            if "micro_expressions" in self.agi_systems:
                expressions = self.agi_systems["micro_expressions"].generate_expressions(emotional_state)
                enrichments["micro_expressions"] = expressions

            # 8. Journal émotionnel
            if "emotional_journal" in self.agi_systems:
                self.agi_systems["emotional_journal"].add_entry(user_input, emotional_state)

            # 9. Imagination sécurisée
            if "imagination_engine" in self.agi_systems and len(user_input) > 20:
                imagination = self.agi_systems["imagination_engine"].imagine_safely(user_input, memory_context)
                if imagination:
                    enrichments["imagination"] = imagination

            # 10. Influence des rêves
            if "dream_engine" in self.agi_systems:
                dream_influence = self.agi_systems["dream_engine"].get_dream_influence()
                if dream_influence:
                    enrichments["dream_influence"] = dream_influence

        except Exception as e:
            logger.warning(f"AGI synthesis partial error: {e}")

        return enrichments

    def _apply_agi_enrichments(self, response: str, enrichments: dict) -> str:
        """
        Applique les enrichissements AGI à la réponse
        """
        # Micro-expressions
        if "micro_expressions" in enrichments:
            for expression in enrichments["micro_expressions"]:
                if expression["type"] == "hesitation" and "..." not in response:
                    # Ajouter une hésitation
                    words = response.split()
                    if len(words) > 5:
                        insert_pos = len(words) // 2
                        words.insert(insert_pos, "...")
                        response = " ".join(words)

        # Curiosité proactive
        if "curiosity" in enrichments and enrichments["curiosity"]:
            curiosity_question = enrichments["curiosity"]
            if not response.endswith("?"):
                response += f" {curiosity_question}"

        # Influence circadienne
        if "circadian" in enrichments:
            state = enrichments["circadian"]
            if state == "tired" and "énergie" not in response:
                response = "*bâille doucement* " + response
            elif state == "energetic" and "!" not in response[-5:]:
                response = response.rstrip(".") + " !"

        # Influence des rêves
        if "dream_influence" in enrichments and enrichments["dream_influence"]:
            dream = enrichments["dream_influence"]
            if dream["intensity"] > 0.7:
                response = "J'ai fait un rêve étrange... " + response

        # Attachement évolutif
        if "attachment" in enrichments:
            level = enrichments["attachment"]["level"]
            if level > 0.8 and "vous" in response:
                response = response.replace("vous", "tu")
                response = response.replace("Vous", "Tu")

        return response

    def _calculate_interaction_quality(self, emotional_state: dict, agi_enrichments: dict) -> float:
        """
        Calcule la qualité de l'interaction (0-1)
        """
        quality = 0.5  # Base neutre

        # Facteur émotionnel
        emotion = emotional_state.get("primary_emotion", "neutral")
        resonance = emotional_state.get("resonance", 0.0)

        if emotion in ["joie", "confiance", "amour"]:
            quality += 0.2
        elif emotion in ["tristesse", "peur"] and resonance > 0.5:
            quality += 0.1  # Empathie positive
        elif emotion in ["colère", "dégoût"]:
            quality -= 0.1

        # Facteur AGI
        if agi_enrichments:
            # Plus d'enrichissements = interaction plus riche
            quality += min(len(agi_enrichments) * 0.02, 0.2)

            # Facteurs spécifiques
            if "empathy" in agi_enrichments:
                quality += 0.1
            if "values" in agi_enrichments and agi_enrichments["values"] > 0.5:
                quality += 0.05

        return max(0.0, min(1.0, quality))

    def _update_metrics(self, response_time: float):
        """Met à jour les métriques de performance"""
        # Incrémenter le compteur de requêtes AVANT le calcul
        self.metrics["total_requests"] += 1

        # Moyenne mobile pour le temps de réponse
        current_avg = self.metrics["avg_response_time"]
        total_requests = self.metrics["total_requests"]

        if total_requests > 0:
            self.metrics["avg_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests

        # Alerter si trop lent
        max_time = self.config.get("max_response_time", self.metrics["max_response_time"])
        if response_time > max_time:
            logger.warning(f"Response time exceeded threshold: {response_time:.3f}s")

    def _generate_fallback_response(self, user_input: str = None) -> str:
        """Génère une réponse de fallback en cas d'erreur"""
        fallbacks = [
            "Je réfléchis à ce que vous venez de dire...",
            "Hmm, laissez-moi y penser un instant...",
            "C'est intéressant, j'aimerais mieux comprendre...",
            "Pardonnez-moi, je suis un peu dans la lune...",
            "Oh, votre message me fait réfléchir...",
        ]

        import random

        return random.choice(fallbacks)

    def _generate_autonomous_response(
        self, user_input: str, emotional_state: dict, memory_context: dict, user_id: str
    ) -> str:
        """Génère une réponse intelligente en utilisant GPT + enrichissements AGI"""

        # 🚀 PRIORITÉ : Utiliser GPT si disponible
        try:
            from jeffrey.core.router_ia import RouterIA

            # Initialiser le routeur GPT
            memory_manager = MemoryManager() if MemoryManager else None
            router = RouterIA(memory_manager)

            if router.is_gpt_available():
                # Construire un contexte riche pour GPT
                gpt_context = {
                    "user_id": user_id,
                    "user_name": user_id if user_id != "default" else "David",
                    "emotion": emotional_state.get("primary_emotion", "curieuse"),
                    "emotion_state": emotional_state,
                    "relevant_memories": memory_context.get("relevant_memories", []),
                    "last_interaction": memory_context.get("last_interaction", ""),
                    "max_tokens": 200,
                    "temperature": 0.8,
                }

                # Enrichir avec les données circadiennes
                if "circadian_rhythm" in self.agi_systems:
                    try:
                        circadian_phase = self.agi_systems["circadian_rhythm"].get_current_phase()
                        gpt_context["circadian_phase"] = circadian_phase.get("phase", "neutral")
                        gpt_context["energy_level"] = circadian_phase.get("energy", 0.5)
                    except:
                        pass

                # Appel GPT avec contexte enrichi
                import asyncio

                async def get_enriched_gpt_response():
                    return await router.route_query(user_input, gpt_context)

                # Exécuter l'appel asynchrone
                try:
                    response_data = asyncio.run(get_enriched_gpt_response())
                    if response_data and response_data.get("response"):
                        base_response = response_data["response"]

                        # Enrichir avec les émotions AGI
                        if emotional_state.get("primary_emotion") == "joie":
                            if "!" not in base_response and random.random() < 0.3:
                                base_response = "✨ " + base_response
                        elif emotional_state.get("primary_emotion") == "mélancolie":
                            if random.random() < 0.2:
                                base_response = "*soupire doucement* " + base_response

                        return base_response

                except Exception as e:
                    print(f"⚠️ Erreur asyncio dans AGI: {e}")

        except Exception as e:
            print(f"⚠️ Erreur GPT dans AGI orchestrator: {e}")

        # 🔄 FALLBACK : Système autonome classique seulement si GPT échoue

        # Préparer le contexte pour les patterns
        context = {
            "user_name": user_id if user_id != "default" else "mon ami",
            "last_memory": memory_context.get("last_interaction", "nos échanges précieux"),
            "user_id": user_id,
        }

        # Générer la réponse de base avec les patterns
        base_response = self.autonomous_response_generator(user_input, context)

        # Analyser le contexte émotionnel
        emotion_context = self.emotion_analyzer(user_input)

        # Préparer les données circadiennes si disponibles
        circadian_data = None
        if "circadian_rhythm" in self.agi_systems:
            try:
                circadian_phase = self.agi_systems["circadian_rhythm"].get_current_phase()
                circadian_data = {
                    "hour": datetime.now().hour,
                    "phase": circadian_phase.get("phase", "neutral"),
                    "energy": circadian_phase.get("energy", 0.5),
                }
            except:
                pass

        # Enrichir la réponse avec émotions et rythme circadien
        enhanced_response = self.response_enhancer(base_response, emotion_context, circadian_data)

        # Ajuster selon l'état émotionnel de Jeffrey
        if emotional_state.get("primary_emotion") == "joie":
            if random.random() < 0.3:  # 30% de chance
                enhanced_response = "✨ " + enhanced_response
        elif emotional_state.get("primary_emotion") == "mélancolie":
            if random.random() < 0.2:  # 20% de chance
                enhanced_response = "*soupire doucement* " + enhanced_response

        return enhanced_response

    def _convert_hybrid_to_emotional_state(self, emotion_analysis: dict) -> Any:
        """Convertit l'analyse émotionnelle hybride en EmotionalState compatible"""
        # Import gracieux de EmotionalState
        try:
            from .emotional_core import EmotionalState

            EMOTIONAL_STATE_AVAILABLE = True
        except ImportError:
            EmotionalState = None
            EMOTIONAL_STATE_AVAILABLE = False
            logger.debug("EmotionalState non disponible, utilisation dict")

        try:
            # Extraire les informations de l'analyse hybride
            primary_emotion = emotion_analysis.get("emotion_dominante", "neutre")
            intensity = emotion_analysis.get("intensite", 50) / 100.0  # Convertir en 0-1
            confiance = emotion_analysis.get("confiance", 50) / 100.0
            resonance = emotion_analysis.get("resonance", 0.5)

            # Créer un état émotionnel compatible
            emotional_state = EmotionalState(
                primary_emotion=primary_emotion,
                intensity=intensity,
                stability=confiance,  # Utiliser confiance comme stabilité
                resonance=resonance,
                internal_state=emotion_analysis.get("etat_interne", "calm"),
                timestamp=time.time(),  # ✅ float attendu
                source="agi_orchestrator",
                hybrid_analysis=emotion_analysis,  # ✅ Passé au constructeur
                integration_mode=emotion_analysis.get("integration_mode", "unknown"),  # ✅ Passé au constructeur
                context={"engines_used": emotion_analysis.get("engines_used", [])},
            )

            return emotional_state

        except Exception as e:
            logger.warning(f"Erreur conversion état émotionnel hybride: {e}")
            # Retourner un état basique si conversion échoue
            return self._create_basic_emotional_state(emotion_analysis)

    def _create_basic_emotional_state(self, emotion_analysis: dict):
        """Crée un état émotionnel basique depuis l'analyse hybride"""

        # Structure simple compatible
        class BasicEmotionalState:
            def __init__(self, data):
                self.primary_emotion = data.get("emotion_dominante", "neutre")
                self.intensity = data.get("intensite", 50) / 100.0
                self.resonance = data.get("resonance", 0.5)
                self.internal_state = data.get("etat_interne", "calm")
                self.timestamp = datetime.now().isoformat()
                self.hybrid_analysis = data

            def to_dict(self):
                return {
                    "primary_emotion": self.primary_emotion,
                    "intensity": self.intensity,
                    "resonance": self.resonance,
                    "internal_state": self.internal_state,
                    "timestamp": self.timestamp,
                    "hybrid_analysis": self.hybrid_analysis,
                }

        return BasicEmotionalState(emotion_analysis)

    def _create_fallback_emotional_state(self):
        """Crée un état émotionnel par défaut en cas d'erreur"""
        fallback_data = {"emotion_dominante": "neutre", "intensite": 50, "confiance": 50, "resonance": 0.5}
        return self._create_basic_emotional_state(fallback_data)

    def _create_basic_emotional_state_from_analysis(self, emotion_result: dict):
        """Crée un état émotionnel depuis un résultat d'analyse simple"""
        data = {
            "emotion_dominante": emotion_result.get("primary", "neutre"),
            "intensite": emotion_result.get("intensity", 0.5) * 100,
            "confiance": emotion_result.get("confidence", 0.5) * 100,
            "resonance": emotion_result.get("resonance", 0.5),
        }
        return self._create_basic_emotional_state(data)

    async def _async_recall_context(self, user_input: str, user_id: str) -> dict[str, Any]:
        """Rappel contextuel asynchrone avec Memory Systems V2.0"""
        try:
            # Rappel de la conversation récente
            recent_context = await self.memory_v2_interface.recall_recent_conversation(user_id=user_id, hours=2)

            # Rappel contextuel pour l'entrée utilisateur
            relevant_memories = await self.memory_v2_interface.recall(query=user_input, user_id=user_id, limit=3)

            return {
                "recent_context": recent_context[-5:],  # 5 derniers échanges
                "relevant_memories": relevant_memories,
                "context_summary": self._generate_context_summary(recent_context, relevant_memories),
            }
        except Exception as e:
            logger.error(f"Erreur rappel contextuel async: {e}")
            return {"recent_context": [], "relevant_memories": [], "context_summary": ""}

    async def _async_capture_interaction(
        self, user_input: str, ai_response: str, emotional_analysis: dict, user_id: str
    ) -> dict[str, Any]:
        """Capture asynchrone d'interaction avec Memory Systems V2.0"""
        try:
            # Utiliser l'intégration AGI pour capturer l'interaction complète
            result = await self.memory_v2_agi_integration.process_interaction_with_memory(
                user_input=user_input,
                ai_response=ai_response,
                emotional_analysis=emotional_analysis,
                user_id=user_id,
            )

            # Mise à jour des métriques
            self.metrics["memory_updates"] += 1

            return result
        except Exception as e:
            logger.error(f"Erreur capture interaction async: {e}")
            return {"memory_id": "", "context_summary": ""}

    def _generate_context_summary(self, recent_context: list[dict], relevant_memories: list[dict]) -> str:
        """Génère un résumé de contexte à partir des mémoires"""
        try:
            if not recent_context and not relevant_memories:
                return "Nouvelle conversation"

            # Extraire les thèmes principaux
            topics = []
            emotions = []

            for memory in (recent_context + relevant_memories)[-5:]:
                if "topics" in memory:
                    topics.extend(memory["topics"])
                if "emotions" in memory:
                    emotions.extend([e.get("emotion", "") for e in memory["emotions"]])

            # Créer le résumé
            main_topics = list(set(topics))[:3]
            main_emotions = list(set(emotions))[:2]

            summary_parts = []
            if main_topics:
                summary_parts.append(f"Sujets: {', '.join(main_topics)}")
            if main_emotions:
                summary_parts.append(f"Émotions: {', '.join(main_emotions)}")

            return " | ".join(summary_parts) or "Conversation générale"
        except Exception as e:
            logger.warning(f"Erreur génération résumé contexte: {e}")
            return "Contexte en cours"

    def save_state(self):
        """Sauvegarde l'état complet du système"""
        self.memory.save_persistent_data()

        # Sauvegarder Memory Systems V2.0
        if self.memory_v2_enabled:
            try:
                # Les Memory Systems V2.0 ont sauvegarde automatique
                stats = self.memory_v2_interface.get_stats()
                logger.info(f"Memory V2.0: {stats['total_memories']} mémoires sauvegardées")
            except Exception as e:
                logger.error(f"Erreur sauvegarde Memory V2.0: {e}")

        # Sauvegarder les états AGI si disponibles
        if self.agi_systems_enabled:
            for name, system in self.agi_systems.items():
                if hasattr(system, "save_state"):
                    try:
                        system.save_state()
                    except Exception as e:
                        logger.error(f"Error saving {name} state: {e}")

        logger.info("AGI Orchestrator state saved")

    def get_system_status(self) -> dict[str, Any]:
        """Retourne le statut complet du système"""
        status = {
            "memory_stats": self.memory.stats,
            "performance_metrics": self.metrics,
            "agi_systems_active": list(self.agi_systems.keys()) if self.agi_systems_enabled else [],
            "config": self.config,
            "memory_v2_enabled": self.memory_v2_enabled,
        }

        # Ajouter les stats Memory V2.0
        if self.memory_v2_enabled:
            try:
                status["memory_v2_stats"] = self.memory_v2_interface.get_stats()
            except Exception as e:
                logger.warning(f"Erreur récupération stats Memory V2.0: {e}")
                status["memory_v2_stats"] = {"error": str(e)}

        # Ajouter l'état émotionnel si disponible
        try:
            if hasattr(self, "emotional_core"):
                status["emotional_state"] = self.emotional_core.get_state_summary()
        except Exception as e:
            logger.warning(f"Erreur récupération état émotionnel: {e}")

        return status


# Backward compatibility alias
AgiOrchestrator = AGIOrchestrator
