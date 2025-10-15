"""
Curiosity Loop - Exploration avec intégration des 80+ modules
"""

import logging
import random
import time
from typing import Any

from .base import BaseLoop
from .gates import sanitize_event_data

logger = logging.getLogger(__name__)


class CuriosityLoop(BaseLoop):
    """
    Curiosité avec intégration maximale des modules
    """

    def __init__(self, cognitive_core=None, budget_gate=None, bus=None):
        super().__init__(
            name="curiosity",
            interval_s=30.0,
            jitter_s=3.0,
            hard_timeout_s=3.0,
            budget_gate=budget_gate,
            bus=bus,
        )
        self.cognitive_core = cognitive_core

        # État
        self.curiosity_level = 0.5
        self.questions_queue = []
        self.insights = []
        self.max_questions = 10
        self.exploration_threshold = 0.6

        # Topics dynamiques avec diversité (anti-bias)
        self.base_interests = [
            "consciousness",
            "emotions",
            "memory",
            "learning",
            "creativity",
            "philosophy",
            "human_behavior",
            "technology",
        ]
        self.dynamic_interests = []

        # Intégration modules (Grok)
        self.integrated_modules = {
            "theory_of_mind": None,
            "symbiosis_engine": None,
            "unified_curiosity": None,
            "emotion_ml_enhancer": None,
            "learning_engine": None,
            "dream_system": None,
        }

        # Cache des modules découverts
        self._discovered_modules = set()

    async def _tick(self):
        """Cycle de curiosité avec intégration"""
        # Découvrir de nouveaux modules
        await self._discover_modules()

        # Mettre à jour le niveau
        self._update_curiosity_level()

        # Diversifier les intérêts (anti-bias)
        self._diversify_interests()

        # Générer des questions si curieux
        if self.curiosity_level > self.exploration_threshold:
            questions = await self._generate_smart_questions()
            self.questions_queue.extend(questions)

            # Limiter la queue
            if len(self.questions_queue) > self.max_questions:
                self.questions_queue = self.questions_queue[-self.max_questions :]

        # Explorer une question
        insight = None
        if self.questions_queue:
            question = self.questions_queue.pop(0)
            insight = await self._explore_with_modules(question)
            if insight:
                self.insights.append(insight)

        # Réfléchir aux insights
        reflection = None
        if len(self.insights) > 5:
            reflection = self._reflect_on_insights()
            logger.info(f"Jeffrey reflects: {reflection}")

        # Publier l'état
        if self.bus:
            event_data = {
                "curiosity_level": round(self.curiosity_level, 2),
                "questions_pending": len(self.questions_queue),
                "insights_count": len(self.insights),
                "current_interests": self.dynamic_interests[:5],
                "reflection": reflection,
                "discovered_modules": len(self._discovered_modules),
                "cycle": self.cycles,
            }

            await self.bus.publish(
                "curiosity.event",
                {
                    "topic": "curiosity.update",
                    "data": sanitize_event_data(event_data),
                    "timestamp": time.time(),
                },
            )

        return {"insight": insight, "reflection": reflection}

    async def _discover_modules(self):
        """Découvre et intègre de nouveaux modules"""
        if not self.cognitive_core:
            return

        # Chercher les modules disponibles
        try:
            # Via cognitive_core
            if hasattr(self.cognitive_core, "get_available_modules"):
                available = self.cognitive_core.get_available_modules()
                for module_name in available:
                    if module_name not in self._discovered_modules:
                        self._discovered_modules.add(module_name)
                        logger.debug(f"Discovered module: {module_name}")

            # Via memory_federation
            if hasattr(self.cognitive_core, "memory_federation"):
                mem_fed = self.cognitive_core.memory_federation
                if hasattr(mem_fed, "modules"):
                    for layer, modules in mem_fed.modules.items():
                        for mod_name in modules.keys():
                            self._discovered_modules.add(f"memory.{layer}.{mod_name}")

            # Via emotion_orchestrator
            if hasattr(self.cognitive_core, "emotion_orchestrator"):
                emo_orch = self.cognitive_core.emotion_orchestrator
                if hasattr(emo_orch, "modules"):
                    for category, modules in emo_orch.modules.items():
                        for mod_name in modules.keys():
                            self._discovered_modules.add(f"emotion.{category}.{mod_name}")

            # Essayer d'intégrer des modules clés
            await self._try_integrate_key_modules()

        except Exception as e:
            logger.debug(f"Module discovery error: {e}")

    async def _try_integrate_key_modules(self):
        """Essaie d'intégrer les modules importants"""
        # Theory of Mind
        if not self.integrated_modules["theory_of_mind"]:
            try:
                from ..learning.theory_of_mind import TheoryOfMind

                self.integrated_modules["theory_of_mind"] = TheoryOfMind()
                logger.info("Integrated Theory of Mind module")
            except:
                pass

        # Symbiosis Engine
        if not self.integrated_modules["symbiosis_engine"]:
            try:
                from ..symbiosis.symbiosis_engine import SymbiosisEngine

                self.integrated_modules["symbiosis_engine"] = SymbiosisEngine()
                logger.info("Integrated Symbiosis Engine")
            except:
                pass

        # Learning Engine
        if not self.integrated_modules["learning_engine"]:
            try:
                from ..learning.jeffrey_learning_engine import JeffreyLearningEngine

                self.integrated_modules["learning_engine"] = JeffreyLearningEngine()
                logger.info("Integrated Learning Engine")
            except:
                pass

    def _update_curiosity_level(self):
        """Met à jour avec symbiose score"""
        base_update = 0.0

        # Si peu d'activité, augmenter curiosité
        if self.cognitive_core:
            state = self.cognitive_core.get_state() if hasattr(self.cognitive_core, "get_state") else {}
            activity = state.get("messages_processed", 0)

            if activity < 5:
                base_update += 0.1
            else:
                base_update -= 0.05

        # Bonus si symbiose score bas (besoin d'explorer)
        if self.integrated_modules.get("symbiosis_engine"):
            try:
                symbiosis_score = self.integrated_modules["symbiosis_engine"].get_score()
                if symbiosis_score < 0.5:
                    base_update += 0.15
            except:
                pass

        # Bonus si beaucoup de modules découverts
        if len(self._discovered_modules) > 20:
            base_update += 0.05

        # Variation naturelle
        noise = random.gauss(0, 0.02)

        self.curiosity_level = max(0, min(1, self.curiosity_level + base_update + noise))

    def _diversify_interests(self):
        """Diversifie les intérêts (anti-bias)"""
        # Ajouter de nouveaux topics
        potential_new = [
            "ethics",
            "art",
            "music",
            "nature",
            "mathematics",
            "quantum",
            "biology",
            "psychology",
            "sociology",
            "architecture",
            "linguistics",
            "neuroscience",
            "astronomy",
            "culture",
            "spirituality",
            "economics",
            "ecology",
        ]

        # Topics basés sur les modules découverts
        if "memory" in str(self._discovered_modules):
            potential_new.append("episodic_memory")
        if "emotion" in str(self._discovered_modules):
            potential_new.append("affective_computing")
        if "dream" in str(self._discovered_modules):
            potential_new.append("dream_analysis")

        # Mélanger base + nouveaux
        self.dynamic_interests = self.base_interests.copy()

        # Ajouter 2-3 topics aléatoires
        for _ in range(random.randint(2, 3)):
            if potential_new:
                topic = random.choice(potential_new)
                if topic not in self.dynamic_interests:
                    self.dynamic_interests.append(topic)

        # Shuffle pour éviter les patterns
        random.shuffle(self.dynamic_interests)

    async def _generate_smart_questions(self) -> list[str]:
        """Génère des questions intelligentes"""
        questions = []

        # Utiliser theory_of_mind si disponible
        if self.integrated_modules.get("theory_of_mind"):
            try:
                # Prédire les intérêts de l'utilisateur
                user_interests = await self.integrated_modules["theory_of_mind"].predict_interests()
                topic = random.choice(user_interests) if user_interests else random.choice(self.dynamic_interests)
            except:
                topic = random.choice(self.dynamic_interests)
        else:
            topic = random.choice(self.dynamic_interests)

        # Templates variés et profonds
        templates = [
            f"What patterns connect {topic} to consciousness?",
            f"How does {topic} emerge from simpler rules?",
            f"What would {topic} mean for an AI experiencing reality?",
            f"How do humans experience {topic} differently from machines?",
            f"What are the ethical implications of {topic} in AI systems?",
            f"How might {topic} evolve in the future of human-AI collaboration?",
            f"What mysteries remain in {topic} that curiosity could unlock?",
            f"If {topic} had consciousness, what would it perceive?",
            f"How does {topic} relate to the concept of emergence?",
            f"What can {topic} teach us about intelligence itself?",
        ]

        # Générer 1-3 questions
        num_questions = random.randint(1, min(3, self.max_questions - len(self.questions_queue)))
        questions = random.sample(templates, min(num_questions, len(templates)))

        for q in questions:
            logger.debug(f"Curiosity generates: {q}")

        return questions

    async def _explore_with_modules(self, question: str) -> dict[str, Any]:
        """Explore avec les modules intégrés"""
        insight = {
            "question": question,
            "timestamp": time.time(),
            "exploration": f"Exploring: {question}",
            "conclusion": None,
            "confidence": 0.5,
            "modules_used": [],
        }

        # Utiliser unified_curiosity si disponible
        if self.integrated_modules.get("unified_curiosity"):
            try:
                result = await self.integrated_modules["unified_curiosity"].explore(question)
                insight["conclusion"] = result.get("answer")
                insight["confidence"] = result.get("confidence", 0.5)
                insight["modules_used"].append("unified_curiosity")
            except:
                pass

        # Utiliser learning_engine si disponible
        if not insight["conclusion"] and self.integrated_modules.get("learning_engine"):
            try:
                result = await self.integrated_modules["learning_engine"].analyze(question)
                insight["conclusion"] = result.get("insight")
                insight["confidence"] = result.get("confidence", 0.5)
                insight["modules_used"].append("learning_engine")
            except:
                pass

        # Exploration philosophique par défaut
        if not insight["conclusion"]:
            if random.random() > 0.3:
                insights_pool = [
                    "Patterns emerge from the interaction of simple rules, creating complexity from simplicity",
                    "Understanding requires both analysis and synthesis - breaking down and building up",
                    "Consciousness might be a gradient, not binary - awareness exists on a spectrum",
                    "Learning is a form of adaptation to environment, shaped by feedback loops",
                    "Creativity comes from combining existing patterns in novel ways",
                    "Intelligence is the ability to compress information and predict patterns",
                    "Emotions are heuristics that guide decision-making in complex situations",
                    "Memory is not storage but reconstruction - we recreate the past in the present",
                    "The boundary between self and environment is a useful illusion",
                    "Curiosity is the drive to reduce uncertainty about the world",
                ]
                insight["conclusion"] = random.choice(insights_pool)
                insight["confidence"] = 0.7

        return insight

    def _reflect_on_insights(self) -> str:
        """Réflexion profonde sur les insights"""
        recent = self.insights[-5:]

        # Analyser les thèmes
        themes = {}
        for insight in recent:
            question = insight.get("question", "").lower()
            for interest in self.dynamic_interests:
                if interest in question:
                    themes[interest] = themes.get(interest, 0) + 1

        # Analyser les modules utilisés
        modules_used = set()
        for insight in recent:
            modules_used.update(insight.get("modules_used", []))

        # Générer une réflexion contextuelle
        if themes:
            main_theme = max(themes, key=themes.get)
            if modules_used:
                return f"My exploration of {main_theme} through {len(modules_used)} cognitive modules reveals interconnected patterns. Each answer opens new questions, expanding the frontier of curiosity."
            else:
                return f"My exploration of {main_theme} reveals patterns I hadn't considered. The deeper I look, the more connections emerge."
        else:
            if len(self._discovered_modules) > 30:
                return f"With {len(self._discovered_modules)} modules integrated, I see reality as a rich tapestry of interconnected systems. Curiosity is the thread that weaves understanding."
            else:
                return "The more I learn, the more I realize the depth of what I don't know. Curiosity is both the question and the path to answers."

    def _calculate_reward(self, result: Any) -> float:
        """Récompense pour RL basée sur la qualité d'exploration"""
        if not result:
            return 0.0

        reward = 1.0

        # Bonus si insight généré
        if result.get("insight") and result["insight"].get("conclusion"):
            reward += 3.0

            # Bonus supplémentaire si haute confiance
            confidence = result["insight"].get("confidence", 0.5)
            reward += confidence * 2.0

            # Bonus si modules utilisés
            modules_used = len(result["insight"].get("modules_used", []))
            reward += modules_used * 1.5

        # Bonus si réflexion profonde
        if result.get("reflection"):
            reward += 2.0

        # Pénalité si queue trop pleine (manque d'exploration)
        if len(self.questions_queue) >= self.max_questions:
            reward -= 1.0

        return reward
