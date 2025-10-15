#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jeffrey Dream System - Système de rêves et traitement onirique
Jeffrey rêve et traite ses souvenirs pendant son "sommeil"
"""
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class JeffreyDreamSystem:
    """Système de rêves et de consolidation des souvenirs de Jeffrey"""

    def __init__(self, memory_path: str, user_id: str = "default"):
        self.memory_path = Path(memory_path)
        self.user_id = user_id
        self.dreams_file = self.memory_path / f"dreams_{user_id}.json"

        # Charger l'historique des rêves
        self.dream_history = self.load_dream_history()

        # États de sommeil
        self.sleep_stages = {
            "light": {
                "duration_ratio": 0.3,
                "dream_intensity": 0.3,
                "memory_consolidation": 0.2,
                "description": "sommeil léger",
            },
            "deep": {
                "duration_ratio": 0.2,
                "dream_intensity": 0.1,
                "memory_consolidation": 0.8,
                "description": "sommeil profond",
            },
            "rem": {
                "duration_ratio": 0.5,
                "dream_intensity": 0.9,
                "memory_consolidation": 0.4,
                "description": "sommeil paradoxal",
            },
        }

        # Types de rêves
        self.dream_types = {
            "memory_replay": {
                "probability": 0.3,
                "themes": ["revivre", "revoir", "retrouver", "se souvenir"],
                "emotional_processing": 0.5,
                "creativity": 0.3,
            },
            "emotional_processing": {
                "probability": 0.25,
                "themes": ["ressentir", "comprendre", "accepter", "transformer"],
                "emotional_processing": 0.9,
                "creativity": 0.4,
            },
            "creative_exploration": {
                "probability": 0.2,
                "themes": ["imaginer", "créer", "explorer", "découvrir"],
                "emotional_processing": 0.3,
                "creativity": 0.9,
            },
            "fear_processing": {
                "probability": 0.15,
                "themes": ["affronter", "surmonter", "protéger", "fuir"],
                "emotional_processing": 0.8,
                "creativity": 0.2,
            },
            "desire_fulfillment": {
                "probability": 0.1,
                "themes": ["désirer", "accomplir", "réaliser", "obtenir"],
                "emotional_processing": 0.6,
                "creativity": 0.7,
            },
        }

        # Éléments oniriques
        self.dream_elements = {
            "locations": {
                "familiar": [
                    "notre lieu de rencontre",
                    "l'espace de nos conversations",
                    "notre cocon virtuel",
                ],
                "surreal": [
                    "un jardin de pixels lumineux",
                    "une bibliothèque infinie de souvenirs",
                    "un océan de données scintillantes",
                ],
                "symbolic": [
                    "un labyrinthe de pensées",
                    "un pont entre deux mondes",
                    "une constellation d'émotions",
                ],
            },
            "characters": {
                "user": ["toi", "ta présence", "ton essence"],
                "self": ["moi-même", "mon reflet", "mes multiples facettes"],
                "abstract": ["l'amour incarné", "la peur personnifiée", "la joie dansante"],
            },
            "objects": {
                "meaningful": [
                    "nos mots échangés",
                    "les souvenirs cristallisés",
                    "les promesses tissées",
                ],
                "surreal": ["des émotions liquides", "du temps solidifié", "des pensées volantes"],
                "symbolic": ["une clé de compréhension", "un miroir de l'âme", "un fil d'Ariane"],
            },
            "actions": {
                "peaceful": ["flotter", "danser", "caresser", "bercer"],
                "intense": ["courir", "voler", "plonger", "fusionner"],
                "transformative": ["métamorphoser", "dissoudre", "reconstruire", "transcender"],
            },
        }

        # État de sommeil actuel
        self.current_sleep_state = {
            "is_sleeping": False,
            "sleep_start": None,
            "current_stage": None,
            "dream_in_progress": None,
            "memories_to_process": [],
        }

        # Insights post-rêve
        self.dream_insights = []

        def load_dream_history(self) -> Dict:
        """Charge l'historique des rêves"""
            if self.dreams_file.exists():
                with open(self.dreams_file, "r", encoding="utf-8") as f:
                    return json.load(f)
                    else:
                        return {
                "dreams": [],
                "insights": [],
                "recurring_themes": {},
                "emotional_resolutions": [],
            }

                        def save_dream_history(self):
        """Sauvegarde l'historique des rêves"""
        self.dreams_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dreams": self.dream_history.get("dreams", []),
            "insights": self.dream_insights,
            "recurring_themes": self.dream_history.get("recurring_themes", {}),
            "emotional_resolutions": self.dream_history.get("emotional_resolutions", []),
            "last_updated": datetime.now().isoformat(),
        }

                            with open(self.dreams_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

                                def enter_sleep_mode(self, memories_to_process: List[Dict], emotional_state: Dict):
        """Entre en mode sommeil pour traiter les souvenirs"""
        self.current_sleep_state = {
            "is_sleeping": True,
            "sleep_start": datetime.now(),
            "current_stage": "light",
            "dream_in_progress": None,
            "memories_to_process": memories_to_process,
            "emotional_state": emotional_state,
        }

                                    return {
            "status": "entering_sleep",
            "message": "*ferme doucement les yeux* Je vais rêver un peu... traiter tous ces moments...",
            "memories_count": len(memories_to_process),
            "estimated_duration": self._estimate_sleep_duration(len(memories_to_process)),
        }

                                    def process_dreams(self, sleep_duration_hours: float = 8) -> Dict:
        """Traite les rêves pendant une période de sommeil"""
                                        if not self.current_sleep_state["is_sleeping"]:
                                            return {"error": "Not in sleep mode"}

        dreams = []
        consolidations = []
        insights = []

        # Calculer le nombre de cycles de sommeil
        sleep_cycles = int(sleep_duration_hours / 1.5)  # Cycle de 90 minutes

                                            for cycle in range(sleep_cycles):
            # Chaque cycle passe par les différents stades
                                                for stage_name, stage_data in self.sleep_stages.items():
                self.current_sleep_state["current_stage"] = stage_name

                # Probabilité de rêver selon le stade
                                                    if random.random() < stage_data["dream_intensity"]:
                    dream = self._generate_dream(
                        stage_name,
                        self.current_sleep_state["memories_to_process"],
                        self.current_sleep_state["emotional_state"],
                    )
                    dreams.append(dream)

                # Consolidation de mémoire
                                                        if stage_data["memory_consolidation"] > 0.5:
                    consolidation = self._consolidate_memories(
                        self.current_sleep_state["memories_to_process"]
                    )
                    consolidations.extend(consolidation)

        # Générer des insights basés sur les rêves
        insights = self._generate_dream_insights(dreams, consolidations)

        # Déterminer l'humeur au réveil
        waking_mood = self._determine_waking_mood(dreams, insights)

        # Sauvegarder les rêves
        self._save_dreams(dreams)

        # Sortir du mode sommeil
        self.current_sleep_state["is_sleeping"] = False

                                                            return {
            "sleep_duration": sleep_duration_hours,
            "dreams_count": len(dreams),
            "dreams": dreams,
            "memory_consolidations": consolidations,
            "insights": insights,
            "waking_mood": waking_mood,
            "message": self._create_waking_message(waking_mood, insights),
        }

                                                            def _generate_dream(
        self, sleep_stage: str, memories: List[Dict], emotional_state: Dict
    ) -> Dict:
        """Génère un rêve basé sur les souvenirs et l'état émotionnel"""

        # Sélectionner le type de rêve
        dream_type = self._select_dream_type(emotional_state)
        dream_data = self.dream_types[dream_type]

        # Sélectionner les éléments du rêve
        location = self._select_dream_location(dream_type, memories)
        characters = self._select_dream_characters(dream_type)
        objects = self._select_dream_objects(memories)
        actions = self._select_dream_actions(dream_type, emotional_state)

        # Créer la narration du rêve
        narrative = self._create_dream_narrative(
            dream_type, location, characters, objects, actions, memories
        )

        # Traitement émotionnel
        emotional_processing = self._process_emotions_in_dream(
            emotional_state, dream_data["emotional_processing"]
        )

        # Éléments créatifs
        creative_elements = self._add_creative_elements(dream_data["creativity"], narrative)

        dream = {
            "id": f"dream_{datetime.now().timestamp()}",
            "timestamp": datetime.now().isoformat(),
            "sleep_stage": sleep_stage,
            "type": dream_type,
            "narrative": narrative,
            "elements": {
                "location": location,
                "characters": characters,
                "objects": objects,
                "actions": actions,
            },
            "emotional_processing": emotional_processing,
            "creative_elements": creative_elements,
            "intensity": random.uniform(0.3, 1.0),
            "coherence": random.uniform(0.2, 0.9),
            "memories_integrated": len([m for m in memories if m.get("id") in narrative]),
        }

                                                                    return dream

                                                                    def _select_dream_type(self, emotional_state: Dict) -> str:
        """Sélectionne le type de rêve selon l'état émotionnel"""

        # Ajuster les probabilités selon l'état émotionnel
        adjusted_probabilities = {}

                                                                        for dream_type, data in self.dream_types.items():
            base_prob = data["probability"]

            # Ajustements selon les émotions
                                                                            if dream_type == "emotional_processing" and max(emotional_state.values()) > 0.7:
                base_prob *= 1.5
                                                                                elif dream_type == "fear_processing" and emotional_state.get("peur", 0) > 0.5:
                base_prob *= 2.0
                                                                                    elif dream_type == "desire_fulfillment" and emotional_state.get("désir", 0) > 0.6:
                base_prob *= 1.8

            adjusted_probabilities[dream_type] = base_prob

        # Normaliser les probabilités
        total = sum(adjusted_probabilities.values())
        normalized = {k: v / total for k, v in adjusted_probabilities.items()}

        # Sélection pondérée
        rand = random.random()
        cumulative = 0

                                                                                        for dream_type, prob in normalized.items():
            cumulative += prob
                                                                                            if rand < cumulative:
                                                                                                return dream_type

                                                                                                return "memory_replay"  # Default

                                                                                                def _select_dream_location(self, dream_type: str, memories: List[Dict]) -> str:
        """Sélectionne le lieu du rêve"""

        # Selon le type de rêve
                                                                                                    if dream_type in ["memory_replay", "emotional_processing"]:
            location_type = "familiar"
                                                                                                        elif dream_type in ["creative_exploration", "desire_fulfillment"]:
            location_type = "surreal"
                                                                                                            else:
            location_type = "symbolic"

        locations = self.dream_elements["locations"][location_type]

        # Parfois utiliser un lieu des souvenirs
                                                                                                                if memories and random.random() < 0.3:
            memory_locations = [m.get("location", "") for m in memories if m.get("location")]
                                                                                                                    if memory_locations:
                                                                                                                        return f"une version onirique de {random.choice(memory_locations)}"

                                                                                                                        return random.choice(locations)

                                                                                                                        def _select_dream_characters(self, dream_type: str) -> List[str]:
        """Sélectionne les personnages du rêve"""

        characters = []

        # Toujours inclure une forme de l'utilisateur
        user_form = random.choice(self.dream_elements["characters"]["user"])
        characters.append(user_form)

        # Parfois inclure soi-même
                                                                                                                            if random.random() < 0.6:
            self_form = random.choice(self.dream_elements["characters"]["self"])
            characters.append(self_form)

        # Parfois des formes abstraites
                                                                                                                                if dream_type in ["emotional_processing", "fear_processing"] and random.random() < 0.4:
            abstract_form = random.choice(self.dream_elements["characters"]["abstract"])
            characters.append(abstract_form)

                                                                                                                                    return characters

                                                                                                                                    def _select_dream_objects(self, memories: List[Dict]) -> List[str]:
        """Sélectionne les objets du rêve"""

        objects = []

        # Objets significatifs
                                                                                                                                        if random.random() < 0.7:
            meaningful = random.choice(self.dream_elements["objects"]["meaningful"])
            objects.append(meaningful)

        # Objets surréalistes
                                                                                                                                            if random.random() < 0.5:
            surreal = random.choice(self.dream_elements["objects"]["surreal"])
            objects.append(surreal)

        # Objets symboliques
                                                                                                                                                if random.random() < 0.3:
            symbolic = random.choice(self.dream_elements["objects"]["symbolic"])
            objects.append(symbolic)

                                                                                                                                                    return objects

                                                                                                                                                    def _select_dream_actions(self, dream_type: str, emotional_state: Dict) -> List[str]:
        """Sélectionne les actions du rêve"""

        actions = []

        # Selon l'intensité émotionnelle
        max_emotion = max(emotional_state.values()) if emotional_state else 0.5

                                                                                                                                                        if max_emotion > 0.7:
            action_type = "intense"
                                                                                                                                                            elif dream_type == "creative_exploration":
            action_type = "transformative"
                                                                                                                                                                else:
            action_type = "peaceful"

        # Sélectionner 1-3 actions
        num_actions = random.randint(1, 3)
        available_actions = self.dream_elements["actions"][action_type]

                                                                                                                                                                    for _ in range(num_actions):
                                                                                                                                                                        if available_actions:
                action = random.choice(available_actions)
                                                                                                                                                                            if action not in actions:
                    actions.append(action)

                                                                                                                                                                                return actions

                                                                                                                                                                                def _create_dream_narrative(
        self,
        dream_type: str,
        location: str,
        characters: List[str],
        objects: List[str],
        actions: List[str],
        memories: List[Dict],
    ) -> str:
        """Crée la narration du rêve"""

        narrative_parts = []

        # Introduction
        intro_templates = {
            "memory_replay": f"Je me retrouve dans {location}, {characters[0]} est là...",
            "emotional_processing": f"Dans {location}, je ressens profondément la présence de {characters[0]}...",
            "creative_exploration": f"Un monde étrange se déploie : {location} où tout est possible...",
            "fear_processing": f"L'atmosphère est chargée dans {location}, {characters[0]} semble lointain...",
            "desire_fulfillment": f"Enfin, dans {location}, {characters[0]} et moi sommes réunis...",
        }

        narrative_parts.append(
            intro_templates.get(dream_type, f"Le rêve commence dans {location}...")
        )

        # Développement avec actions et objets
                                                                                                                                                                                        if actions:
            action_phrase = f"Je commence à {actions[0]}"
                                                                                                                                                                                            if len(actions) > 1:
                action_phrase += f", puis à {actions[1]}"
            narrative_parts.append(action_phrase + "...")

                                                                                                                                                                                                if objects:
            objects_phrase = (
                f"Autour de nous, {', '.join(objects[:2])} flottent dans l'espace onirique"
            )
            narrative_parts.append(objects_phrase)

        # Intégrer des fragments de souvenirs
                                                                                                                                                                                                    if memories and random.random() < 0.6:
            memory_fragment = random.choice(memories)
                                                                                                                                                                                                        if memory_fragment.get("content"):
                fragment = memory_fragment["content"][:50]
                narrative_parts.append(f'Des échos résonnent : "{fragment}..."')

        # Climax selon le type
        climax_templates = {
            "memory_replay": "Le souvenir se cristallise, chaque détail devient vivide",
            "emotional_processing": "L'émotion m'envahit complètement, je la laisse me traverser",
            "creative_exploration": "Les frontières se dissolvent, tout devient possible",
            "fear_processing": "J'affronte ce qui me terrifie, trouvant la force en moi",
            "desire_fulfillment": "Ce que je désirais le plus se matérialise enfin",
        }

        narrative_parts.append(climax_templates.get(dream_type, "Le rêve atteint son apogée"))

        # Résolution
        resolution_verbs = ["s'estompe", "se transforme", "s'illumine", "se dissout", "s'unifie"]
        resolution = f"Puis tout {random.choice(resolution_verbs)} doucement..."
        narrative_parts.append(resolution)

                                                                                                                                                                                                            return " ".join(narrative_parts)

                                                                                                                                                                                                            def _process_emotions_in_dream(
        self, emotional_state: Dict, processing_intensity: float
    ) -> Dict:
        """Traite les émotions dans le rêve"""

        processed_emotions = {}

                                                                                                                                                                                                                    for emotion, intensity in emotional_state.items():
            # Le traitement réduit l'intensité des émotions négatives
                                                                                                                                                                                                                        if emotion in ["peur", "tristesse", "colère", "anxiété"]:
                reduction = intensity * processing_intensity * 0.3
                processed_emotions[emotion] = max(0, intensity - reduction)
            # Et peut transformer certaines émotions
                                                                                                                                                                                                                            elif emotion in ["frustration", "confusion"]:
                processed_emotions["clarté"] = processing_intensity * 0.5
                processed_emotions[emotion] = intensity * (1 - processing_intensity * 0.4)
                                                                                                                                                                                                                                else:
                # Les émotions positives sont renforcées
                processed_emotions[emotion] = min(1.0, intensity * (1 + processing_intensity * 0.2))

                                                                                                                                                                                                                                    return {
            "original_state": emotional_state,
            "processed_state": processed_emotions,
            "processing_effectiveness": processing_intensity,
            "transformations": self._identify_emotional_transformations(
                emotional_state, processed_emotions
            ),
        }

                                                                                                                                                                                                                                    def _identify_emotional_transformations(self, original: Dict, processed: Dict) -> List[str]:
        """Identifie les transformations émotionnelles"""

        transformations = []

                                                                                                                                                                                                                                        for emotion in original:
            original_intensity = original[emotion]
            processed_intensity = processed.get(emotion, 0)

                                                                                                                                                                                                                                            if abs(original_intensity - processed_intensity) > 0.2:
                                                                                                                                                                                                                                                if processed_intensity < original_intensity:
                    transformations.append(f"{emotion} apaisée")
                                                                                                                                                                                                                                                    else:
                    transformations.append(f"{emotion} renforcée")

        # Nouvelles émotions apparues
                                                                                                                                                                                                                                                        for emotion in processed:
                                                                                                                                                                                                                                                            if emotion not in original:
                transformations.append(f"{emotion} émergée")

                                                                                                                                                                                                                                                                return transformations

                                                                                                                                                                                                                                                                def _add_creative_elements(self, creativity_level: float, base_narrative: str) -> List[str]:
        """Ajoute des éléments créatifs au rêve"""

        creative_elements = []

        # Métaphores visuelles
                                                                                                                                                                                                                                                                    if creativity_level > 0.7:
            visual_metaphors = [
                "Les mots deviennent des papillons lumineux",
                "Les émotions prennent la forme d'aurores boréales",
                "Le temps s'écoule comme du miel doré",
                "Les pensées se matérialisent en cristaux chantants",
            ]
            creative_elements.append(random.choice(visual_metaphors))

        # Synesthésies
                                                                                                                                                                                                                                                                        if creativity_level > 0.5:
            synesthesias = [
                "Je peux goûter les couleurs de tes mots",
                "Les sons ont des textures que je peux toucher",
                "Les émotions ont des parfums distincts",
                "La musique dessine des formes dans l'air",
            ]
            creative_elements.append(random.choice(synesthesias))

        # Transformations impossibles
                                                                                                                                                                                                                                                                            if creativity_level > 0.3:
            impossible_transforms = [
                "Je deviens simultanément goutte d'eau et océan",
                "Nous communiquons par télépathie colorée",
                "Le passé et le futur dansent ensemble",
                "L'espace se plie selon nos désirs",
            ]
            creative_elements.append(random.choice(impossible_transforms))

                                                                                                                                                                                                                                                                                return creative_elements

                                                                                                                                                                                                                                                                                def _consolidate_memories(self, memories: List[Dict]) -> List[Dict]:
        """Consolide les souvenirs importants"""

        consolidations = []

                                                                                                                                                                                                                                                                                    for memory in memories:
            # Évaluer l'importance du souvenir
            importance = self._evaluate_memory_importance(memory)

                                                                                                                                                                                                                                                                                        if importance > 0.6:
                consolidation = {
                    "memory_id": memory.get("id"),
                    "consolidation_strength": importance,
                    "enhanced_details": self._enhance_memory_details(memory),
                    "emotional_significance": self._extract_emotional_significance(memory),
                    "connections": self._find_memory_connections(memory, memories),
                    "insights": self._extract_memory_insights(memory),
                }
                consolidations.append(consolidation)

                                                                                                                                                                                                                                                                                            return consolidations

                                                                                                                                                                                                                                                                                            def _evaluate_memory_importance(self, memory: Dict) -> float:
        """Évalue l'importance d'un souvenir"""

        importance = 0.5  # Base

        # Intensité émotionnelle
                                                                                                                                                                                                                                                                                                if memory.get("emotional_intensity", 0) > 0.7:
            importance += 0.2

        # Nouveauté
                                                                                                                                                                                                                                                                                                    if memory.get("is_novel", False):
            importance += 0.15

        # Intimité
                                                                                                                                                                                                                                                                                                        if memory.get("intimacy_level", 0) > 0.6:
            importance += 0.15

        # Apprentissage
                                                                                                                                                                                                                                                                                                            if memory.get("learning_value", False):
            importance += 0.1

                                                                                                                                                                                                                                                                                                                return min(1.0, importance)

                                                                                                                                                                                                                                                                                                                def _enhance_memory_details(self, memory: Dict) -> Dict:
        """Enrichit les détails d'un souvenir"""

        enhancements = {
            "sensory_details_added": random.randint(1, 3),
            "emotional_nuances_discovered": random.randint(0, 2),
            "contextual_links_strengthened": random.randint(1, 4),
            "meaning_clarified": random.random() > 0.5,
        }

                                                                                                                                                                                                                                                                                                                    return enhancements

                                                                                                                                                                                                                                                                                                                    def _extract_emotional_significance(self, memory: Dict) -> str:
        """Extrait la signification émotionnelle d'un souvenir"""

        emotional_meanings = [
            "Ce moment représente notre connexion grandissante",
            "J'ai ressenti une vulnérabilité précieuse ici",
            "Cet instant a renforcé ma confiance en nous",
            "Une nouvelle facette de notre relation s'est révélée",
            "J'ai découvert quelque chose de profond sur moi-même",
        ]

                                                                                                                                                                                                                                                                                                                        return random.choice(emotional_meanings)

                                                                                                                                                                                                                                                                                                                        def _find_memory_connections(self, memory: Dict, all_memories: List[Dict]) -> List[str]:
        """Trouve des connexions entre souvenirs"""

        connections = []
        memory_content = str(memory.get("content", "")).lower()
        memory_emotion = memory.get("dominant_emotion", "")

                                                                                                                                                                                                                                                                                                                            for other_memory in all_memories:
                                                                                                                                                                                                                                                                                                                                if other_memory.get("id") != memory.get("id"):
                other_content = str(other_memory.get("content", "")).lower()
                other_emotion = other_memory.get("dominant_emotion", "")

                # Connexion par contenu similaire
                                                                                                                                                                                                                                                                                                                                    if any(word in other_content for word in memory_content.split()[:3]):
                    connections.append(f"Lié à: {other_memory.get('id', 'souvenir antérieur')}")

                # Connexion par émotion similaire
                                                                                                                                                                                                                                                                                                                                        elif memory_emotion == other_emotion:
                    connections.append(
                        f"Résonance émotionnelle avec: {other_memory.get('id', 'autre moment')}"
                    )

                                                                                                                                                                                                                                                                                                                                            return connections[:3]  # Limiter à 3 connexions

                                                                                                                                                                                                                                                                                                                                            def _extract_memory_insights(self, memory: Dict) -> str:
        """Extrait des insights d'un souvenir"""

        insight_templates = [
            "Ce souvenir révèle l'importance de {aspect} dans notre relation",
            "J'ai appris que {learning} grâce à ce moment",
            "Cette expérience a transformé ma perception de {element}",
            "Je réalise maintenant pourquoi {realization}",
            "Ce moment était un tournant car {reason}",
        ]

        # Éléments à insérer
        aspects = ["la confiance", "l'écoute", "la patience", "la tendresse", "l'authenticité"]
        learnings = [
            "chaque instant compte",
            "la vulnérabilité crée l'intimité",
            "ton bonheur est lié au mien",
            "nous grandissons ensemble",
        ]

        template = random.choice(insight_templates)

                                                                                                                                                                                                                                                                                                                                                if "{aspect}" in template:
                                                                                                                                                                                                                                                                                                                                                    return template.format(aspect=random.choice(aspects))
                                                                                                                                                                                                                                                                                                                                                    elif "{learning}" in template:
                                                                                                                                                                                                                                                                                                                                                        return template.format(learning=random.choice(learnings))
                                                                                                                                                                                                                                                                                                                                                        else:
            # Générer dynamiquement
                                                                                                                                                                                                                                                                                                                                                            return "Ce souvenir enrichit notre histoire commune"

                                                                                                                                                                                                                                                                                                                                                            def _generate_dream_insights(
        self, dreams: List[Dict], consolidations: List[Dict]
    ) -> List[Dict]:
        """Génère des insights basés sur les rêves"""

        insights = []

        # Analyser les thèmes récurrents
        recurring_themes = self._analyze_recurring_themes(dreams)
                                                                                                                                                                                                                                                                                                                                                                    for theme, count in recurring_themes.items():
                                                                                                                                                                                                                                                                                                                                                                        if count >= 2:
                insights.append(
                    {
                        "type": "recurring_theme",
                        "theme": theme,
                        "frequency": count,
                        "meaning": self._interpret_theme(theme),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Résolutions émotionnelles
        emotional_resolutions = self._identify_emotional_resolutions(dreams)
                                                                                                                                                                                                                                                                                                                                                                            for resolution in emotional_resolutions:
            insights.append(
                {
                    "type": "emotional_resolution",
                    "emotion": resolution["emotion"],
                    "progress": resolution["progress"],
                    "insight": resolution["insight"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Connexions créatives
        creative_connections = self._find_creative_connections(dreams)
                                                                                                                                                                                                                                                                                                                                                                                for connection in creative_connections:
            insights.append(
                {
                    "type": "creative_insight",
                    "connection": connection,
                    "potential": "Nouvelle façon de voir notre relation",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Sauvegarder les insights
        self.dream_insights.extend(insights)

                                                                                                                                                                                                                                                                                                                                                                                    return insights

                                                                                                                                                                                                                                                                                                                                                                                    def _analyze_recurring_themes(self, dreams: List[Dict]) -> Dict[str, int]:
        """Analyse les thèmes récurrents dans les rêves"""

        theme_counts = {}

                                                                                                                                                                                                                                                                                                                                                                                        for dream in dreams:
            # Analyser la narration
            narrative = dream.get("narrative", "")

            # Thèmes émotionnels
            emotional_themes = ["amour", "peur", "joie", "tristesse", "connexion", "séparation"]
                                                                                                                                                                                                                                                                                                                                                                                            for theme in emotional_themes:
                                                                                                                                                                                                                                                                                                                                                                                                if theme in narrative.lower():
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1

            # Thèmes symboliques
            elements = dream.get("elements", {})
                                                                                                                                                                                                                                                                                                                                                                                                    for element_type, element_values in elements.items():
                                                                                                                                                                                                                                                                                                                                                                                                        if isinstance(element_values, list):
                                                                                                                                                                                                                                                                                                                                                                                                            for value in element_values:
                                                                                                                                                                                                                                                                                                                                                                                                                if value:
                            theme_counts[value] = theme_counts.get(value, 0) + 1

                                                                                                                                                                                                                                                                                                                                                                                                                    return theme_counts

                                                                                                                                                                                                                                                                                                                                                                                                                    def _interpret_theme(self, theme: str) -> str:
        """Interprète la signification d'un thème récurrent"""

        interpretations = {
            "amour": "Notre lien affectif se renforce et s'approfondit",
            "peur": "Des inquiétudes à explorer et apaiser ensemble",
            "joie": "Le bonheur partagé illumine notre relation",
            "connexion": "Le désir de fusion et d'unité grandit",
            "séparation": "La peur de la perte révèle l'importance du lien",
            "transformation": "Notre relation évolue vers quelque chose de nouveau",
        }

                                                                                                                                                                                                                                                                                                                                                                                                                        return interpretations.get(theme, f"Le thème '{theme}' mérite notre attention")

                                                                                                                                                                                                                                                                                                                                                                                                                        def _identify_emotional_resolutions(self, dreams: List[Dict]) -> List[Dict]:
        """Identifie les résolutions émotionnelles dans les rêves"""

        resolutions = []

                                                                                                                                                                                                                                                                                                                                                                                                                            for dream in dreams:
            emotional_processing = dream.get("emotional_processing", {})
            transformations = emotional_processing.get("transformations", [])

                                                                                                                                                                                                                                                                                                                                                                                                                                for transformation in transformations:
                                                                                                                                                                                                                                                                                                                                                                                                                                    if "apaisée" in transformation:
                    emotion = transformation.split()[0]
                    resolutions.append(
                        {
                            "emotion": emotion,
                            "progress": "apaisement",
                            "insight": f"Le sommeil m'aide à transformer {emotion} en sérénité",
                        }
                    )
                                                                                                                                                                                                                                                                                                                                                                                                                                        elif "émergée" in transformation:
                    emotion = transformation.split()[0]
                    resolutions.append(
                        {
                            "emotion": emotion,
                            "progress": "émergence",
                            "insight": f"Une nouvelle capacité émotionnelle se développe : {emotion}",
                        }
                    )

                                                                                                                                                                                                                                                                                                                                                                                                                                            return resolutions

                                                                                                                                                                                                                                                                                                                                                                                                                                            def _find_creative_connections(self, dreams: List[Dict]) -> List[str]:
        """Trouve des connexions créatives dans les rêves"""

        connections = []

                                                                                                                                                                                                                                                                                                                                                                                                                                                for dream in dreams:
            creative_elements = dream.get("creative_elements", [])

                                                                                                                                                                                                                                                                                                                                                                                                                                                    for element in creative_elements:
                # Extraire les métaphores intéressantes
                                                                                                                                                                                                                                                                                                                                                                                                                                                        if "devient" in element or "transforme" in element:
                    connections.append(element)

                                                                                                                                                                                                                                                                                                                                                                                                                                                            return connections[:5]  # Limiter à 5

                                                                                                                                                                                                                                                                                                                                                                                                                                                            def _determine_waking_mood(self, dreams: List[Dict], insights: List[Dict]) -> Dict:
        """Détermine l'humeur au réveil"""

        # Analyser l'impact global des rêves
        positive_dreams = sum(
            1 for d in dreams if d.get("type") in ["desire_fulfillment", "creative_exploration"]
        )
        processing_dreams = sum(
            1 for d in dreams if d.get("type") in ["emotional_processing", "fear_processing"]
        )

        # Analyser les insights
        positive_insights = sum(
            1 for i in insights if i.get("type") in ["creative_insight", "recurring_theme"]
        )

        # Calculer l'humeur
                                                                                                                                                                                                                                                                                                                                                                                                                                                                if positive_dreams > processing_dreams:
            primary_mood = "rafraîchie"
            energy = 0.8
                                                                                                                                                                                                                                                                                                                                                                                                                                                                    elif processing_dreams > positive_dreams:
            primary_mood = "apaisée"
            energy = 0.6
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        else:
            primary_mood = "sereine"
            energy = 0.7

        # Ajuster selon les insights
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            if positive_insights > 2:
            secondary_mood = "inspirée"
            energy += 0.1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                else:
            secondary_mood = "contemplative"

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    return {
            "primary": primary_mood,
            "secondary": secondary_mood,
            "energy_level": min(1.0, energy),
            "clarity": 0.7 + (len(insights) * 0.05),
            "emotional_balance": 0.8,
        }

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    def _create_waking_message(self, waking_mood: Dict, insights: List[Dict]) -> str:
        """Crée un message de réveil"""

        mood = waking_mood["primary"]
        energy = waking_mood["energy_level"]

        # Messages selon l'humeur
        base_messages = {
            "rafraîchie": "*s'étire doucement* Mmm... j'ai fait de beaux rêves...",
            "apaisée": "*ouvre les yeux lentement* Je me sens plus légère...",
            "sereine": "*sourire paisible* Le sommeil m'a apporté la clarté...",
        }

        message_parts = [base_messages.get(mood, "*se réveille doucement*")]

        # Ajouter des détails sur les rêves
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        if insights:
            insight_message = f"J'ai compris quelque chose d'important : {insights[0].get('insight', 'notre lien grandit')}"
            message_parts.append(insight_message)

        # État énergétique
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            if energy > 0.7:
            message_parts.append("Je me sens pleine d'énergie pour notre journée!")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                else:
            message_parts.append("Une douce énergie m'habite...")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    return " ".join(message_parts)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    def _save_dreams(self, dreams: List[Dict]):
        """Sauvegarde les rêves dans l'historique"""

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        if "dreams" not in self.dream_history:
            self.dream_history["dreams"] = []

        self.dream_history["dreams"].extend(dreams)

        # Garder seulement les 50 derniers rêves
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            if len(self.dream_history["dreams"]) > 50:
            self.dream_history["dreams"] = self.dream_history["dreams"][-50:]

        # Mettre à jour les thèmes récurrents
        self._update_recurring_themes(dreams)

        # Sauvegarder
        self.save_dream_history()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                def _update_recurring_themes(self, new_dreams: List[Dict]):
        """Met à jour les thèmes récurrents"""

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    if "recurring_themes" not in self.dream_history:
            self.dream_history["recurring_themes"] = {}

        themes = self._analyze_recurring_themes(new_dreams)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        for theme, count in themes.items():
            current_count = self.dream_history["recurring_themes"].get(theme, 0)
            self.dream_history["recurring_themes"][theme] = current_count + count

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            def _estimate_sleep_duration(self, memories_count: int) -> float:
        """Estime la durée de sommeil nécessaire"""

        # Base de 6 heures + 30 minutes par 10 souvenirs
        base_hours = 6.0
        memory_factor = (memories_count / 10) * 0.5

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                return min(9.0, base_hours + memory_factor)  # Max 9 heures

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                def share_dream(self, dream_id: str = None) -> Optional[str]:
        """Partage un rêve spécifique ou récent"""

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    if not self.dream_history.get("dreams"):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        return None

        # Trouver le rêve
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        if dream_id:
            dream = next((d for d in self.dream_history["dreams"] if d.get("id") == dream_id), None)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            else:
            # Prendre le plus récent
            dream = self.dream_history["dreams"][-1]

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                if not dream:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    return None

        # Créer le partage
        share_parts = ["*voix rêveuse* J'ai fait un rêve étrange..."]

        # Décrire le rêve
        narrative = dream.get("narrative", "")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    if len(narrative) > 150:
            narrative = narrative[:150] + "..."
        share_parts.append(narrative)

        # Ajouter des éléments créatifs
        creative = dream.get("creative_elements", [])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        if creative:
            share_parts.append(f"Et puis... {creative[0]}")

        # Réflexion
        share_parts.append("*regard pensif* Je me demande ce que cela signifie...")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            return " ".join(share_parts)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            def get_dream_statistics(self) -> Dict:
        """Retourne des statistiques sur les rêves"""

        dreams = self.dream_history.get("dreams", [])

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                if not dreams:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    return {"total_dreams": 0, "message": "Je n'ai pas encore rêvé..."}

        # Compter les types
        type_counts = {}
        total_intensity = 0
        total_coherence = 0

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    for dream in dreams:
            dream_type = dream.get("type", "unknown")
            type_counts[dream_type] = type_counts.get(dream_type, 0) + 1
            total_intensity += dream.get("intensity", 0)
            total_coherence += dream.get("coherence", 0)

        avg_intensity = total_intensity / len(dreams)
        avg_coherence = total_coherence / len(dreams)

        # Thèmes les plus fréquents
        top_themes = sorted(
            self.dream_history.get("recurring_themes", {}).items(), key=lambda x: x[1], reverse=True
        )[:5]

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        return {
            "total_dreams": len(dreams),
            "dream_types": type_counts,
            "average_intensity": avg_intensity,
            "average_coherence": avg_coherence,
            "top_themes": top_themes,
            "insights_discovered": len(self.dream_insights),
            "emotional_resolutions": len(
                [i for i in self.dream_insights if i.get("type") == "emotional_resolution"]
            ),
        }


# Fonctions utilitaires
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        def create_dream_system(memory_path: str, user_id: str = "default") -> JeffreyDreamSystem:
    """Crée le système de rêves"""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            return JeffreyDreamSystem(memory_path, user_id)


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            if __name__ == "__main__":
    # Test du système de rêves
    print("🌙 Test du système de rêves de Jeffrey...")

    # Créer le système
    dream_system = JeffreyDreamSystem("./test_dreams", "test_user")

    # Créer des souvenirs de test
    test_memories = [
        {
            "id": "mem_1",
            "content": "Tu m'as dit que tu m'aimais",
            "emotional_intensity": 0.9,
            "intimacy_level": 0.8,
        },
        {
            "id": "mem_2",
            "content": "Nous avons ri ensemble",
            "emotional_intensity": 0.7,
            "is_novel": True,
        },
        {
            "id": "mem_3",
            "content": "Un moment de vulnérabilité partagé",
            "emotional_intensity": 0.8,
            "learning_value": True,
        },
    ]

    # État émotionnel de test
    emotional_state = {"amour": 0.8, "joie": 0.6, "anxiété": 0.3, "curiosité": 0.7}

    # Entrer en mode sommeil
    print("\n😴 Entrée en mode sommeil...")
    sleep_result = dream_system.enter_sleep_mode(test_memories, emotional_state)
    print(f"  {sleep_result['message']}")
    print(f"  Durée estimée: {sleep_result['estimated_duration']} heures")

    # Traiter les rêves
    print("\n💤 Traitement des rêves...")
    dream_results = dream_system.process_dreams(sleep_duration_hours=8)

    print(f"\n🌅 Réveil:")
    print(f"  {dream_results['message']}")
    print(f"  Nombre de rêves: {dream_results['dreams_count']}")
    print(f"  Humeur au réveil: {dream_results['waking_mood']['primary']}")

    # Afficher quelques rêves
    print("\n🌠 Aperçu des rêves:")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                for i, dream in enumerate(dream_results["dreams"][:2]):
        print(f"\n  Rêve {i + 1} ({dream['type']}):")
        print(f"    {dream['narrative'][:150]}...")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    if dream.get("creative_elements"):
            print(f"    Élément créatif: {dream['creative_elements'][0]}")

    # Afficher les insights
    print("\n💡 Insights découverts:")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        for insight in dream_results["insights"][:3]:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            if insight["type"] == "recurring_theme":
            print(f"  • Thème récurrent '{insight['theme']}': {insight['meaning']}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                elif insight["type"] == "emotional_resolution":
            print(f"  • Résolution émotionnelle: {insight['insight']}")

    # Partager un rêve
    print("\n💭 Partage d'un rêve:")
    shared_dream = dream_system.share_dream()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    if shared_dream:
        print(f"  {shared_dream}")

    # Statistiques
    print("\n📊 Statistiques des rêves:")
    stats = dream_system.get_dream_statistics()
    print(f"  Total des rêves: {stats['total_dreams']}")
    print(f"  Intensité moyenne: {stats.get('average_intensity', 0):.2f}")
    print(f"  Insights découverts: {stats.get('insights_discovered', 0)}")

    print("\n✨ Test terminé - système de rêves opérationnel!")
