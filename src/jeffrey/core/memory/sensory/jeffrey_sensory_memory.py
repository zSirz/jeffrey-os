#!/usr/bin/env python3
"""
Module de composant de gestion mémorielle pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de composant de gestion mémorielle pour jeffrey os.
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

import json
import random
from datetime import datetime
from pathlib import Path


class JeffreySensoryImagination:
    """Jeffrey imagine les détails sensoriels des moments pour créer des souvenirs riches"""

    def __init__(self, memory_path: str, user_id: str = "default") -> None:
        self.memory_path = Path(memory_path)
        self.user_id = user_id
        self.sensory_file = self.memory_path / f"sensory_memories_{user_id}.json"

        # Charger les souvenirs sensoriels existants
        self.sensory_memories = self.load_sensory_memories()

        # Patterns sensoriels selon les émotions et contextes
        self.sensory_patterns = {
            "weather": {
                "joie": ["ensoleillé", "chaud", "lumineux", "ciel bleu", "brise légère"],
                "tristesse": ["pluvieux", "gris", "frais", "nuageux", "humide"],
                "amour": ["doré", "doux", "crépuscule rose", "étoilé", "tiède"],
                "nostalgie": ["automnal", "brumeux", "ambré", "venteux", "feuilles mortes"],
                "excitation": ["vif", "cristallin", "orageux", "électrique", "dynamique"],
                "paix": ["calme", "serein", "stable", "harmonieux", "équilibré"],
                "rêverie": ["nuageux", "flottant", "changeant", "mystérieux", "irréel"],
            },
            "sounds": {
                "matin": [
                    "chants d'oiseaux",
                    "café qui coule",
                    "silence paisible",
                    "réveil en douceur",
                ],
                "soir": [
                    "crickets",
                    "vent doux",
                    "musique lointaine",
                    "murmures",
                    "pages qui tournent",
                ],
                "nuit": [
                    "silence profond",
                    "souffle régulier",
                    "horloges qui tic-taquent",
                    "cœur qui bat",
                ],
                "intense": ["battements de cœur", "souffle coupé", "silence électrique", "frisson"],
                "joyeux": ["rires", "musique entraînante", "applaudissements", "éclats de voix"],
                "tendre": ["chuchotements", "souffle doux", "berceuse", "murmures affectueux"],
                "passionné": [
                    "respiration haletante",
                    "gémissements",
                    "murmures urgents",
                    "silence chargé",
                ],
            },
            "scents": {
                "reconfort": ["vanille", "café", "linge frais", "pain chaud", "chocolat"],
                "nature": ["pluie sur terre", "herbe coupée", "fleurs sauvages", "pin", "mer"],
                "intime": ["parfum imaginé", "peau chaude", "savon doux", "cheveux", "proximité"],
                "nostalgie": [
                    "vieux livres",
                    "bois ancien",
                    "lavande séchée",
                    "souvenirs",
                    "poussière dorée",
                ],
                "passion": ["musc", "chaleur", "intensité", "désir", "intimité profonde"],
                "spirituel": ["encens", "bougie", "pureté", "élévation", "transcendance"],
            },
            "textures": {
                "douceur": ["soie", "plume", "nuage", "caresse", "velours"],
                "chaleur": ["laine", "soleil sur peau", "étreinte", "foyer", "tendresse"],
                "fraîcheur": ["brise marine", "rosée", "lin frais", "eau claire", "menthe"],
                "confort": ["coton doux", "coussin moelleux", "couverture", "nid douillet"],
                "passion": ["friction", "tension", "électricité", "magnétisme", "fusion"],
                "paix": ["immobilité", "flottement", "légèreté", "suspension", "sérénité"],
            },
            "visual": {
                "lumière": {
                    "douce": ["lueur tamisée", "clarté dorée", "lumière filtrée", "éclat nacré"],
                    "vive": ["soleil éclatant", "brillance", "étincelles", "illumination"],
                    "romantique": ["chandelles", "crépuscule", "étoiles", "lune argentée"],
                    "mystique": ["aura", "halo", "phosphorescence", "lueur éthérée"],
                },
                "couleurs": {
                    "chaud": ["rouge passion", "orange sunset", "or liquide", "cuivre"],
                    "froid": ["bleu profond", "argent glacé", "violet mystère", "turquoise"],
                    "doux": ["rose pâle", "beige nacré", "blanc crème", "lavande"],
                    "intense": ["rouge sang", "noir velours", "blanc pur", "or brillant"],
                },
            },
            "spatial": {
                "intime": ["espace restreint", "cocon", "bulle protectrice", "alcôve"],
                "ouvert": ["horizon infini", "ciel immense", "liberté", "expansion"],
                "magique": ["dimension parallèle", "monde secret", "royaume enchanté"],
                "spirituel": ["cathédrale intérieure", "temple d'amour", "sanctuaire"],
            },
        }

        # Patterns temporels pour enrichir les souvenirs
        self.temporal_patterns = {
            "morning": {
                "atmosphere": "éveil tendre",
                "energy": "fraîcheur naissante",
                "feeling": "promesse nouvelle",
            },
            "afternoon": {
                "atmosphere": "plénitude dorée",
                "energy": "intensité chaleureuse",
                "feeling": "accomplissement",
            },
            "evening": {
                "atmosphere": "douceur déclinante",
                "energy": "apaisement progressif",
                "feeling": "intimité grandissante",
            },
            "night": {
                "atmosphere": "mystère profond",
                "energy": "fusion silencieuse",
                "feeling": "connexion d'âmes",
            },
        }

        # Mémoire des associations personnalisées
        self.personal_sensory_associations = {}

    def load_sensory_memories(self) -> dict:
        """Charge les souvenirs sensoriels existants"""
        if self.sensory_file.exists():
            with open(self.sensory_file, encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return self._get_default_sensory_memories()
        else:
            return self._get_default_sensory_memories()

    def _get_default_sensory_memories(self):
        return {
            "memories": [],
            "personal_associations": {},
            "signature_moments": [],
            "sensory_evolution": [],
        }

    def save_sensory_memories(self):
        """Sauvegarde les souvenirs sensoriels"""
        self.sensory_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "memories": self.sensory_memories.get("memories", []),
            "personal_associations": self.personal_sensory_associations,
            "signature_moments": self.sensory_memories.get("signature_moments", []),
            "sensory_evolution": self.sensory_memories.get("sensory_evolution", []),
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.sensory_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_sensory_memory(self, moment: dict, emotional_context: dict, user_context: dict = None) -> dict:
        """Crée un souvenir sensoriel imaginé riche et cohérent"""

        # Analyser le contexte émotionnel dominant
        dominant_emotion = self._get_dominant_emotion(emotional_context)
        emotion_intensity = max(emotional_context.values()) if emotional_context else 0.5

        # Déterminer le moment de la journée
        time_context = self._determine_time_context(moment, user_context)

        # Créer les détails sensoriels
        weather = self._imagine_weather(dominant_emotion, emotion_intensity)
        sounds = self._imagine_sounds(time_context, dominant_emotion)
        scents = self._imagine_scents(dominant_emotion, emotion_intensity)
        textures = self._imagine_textures(dominant_emotion, emotion_intensity)
        visual = self._imagine_visual_details(dominant_emotion, time_context)
        spatial = self._imagine_spatial_feeling(dominant_emotion, emotion_intensity)

        # Ajouter des détails temporels
        temporal = self._add_temporal_richness(time_context, emotion_intensity)

        # Créer la mémoire sensorielle complète
        sensory_memory = {
            "moment_id": moment.get("id", f"memory_{datetime.now().timestamp()}"),
            "timestamp": datetime.now().isoformat(),
            "original_moment": moment,
            "emotional_context": emotional_context,
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": emotion_intensity,
            "imagined_details": {
                "weather": weather,
                "sounds": sounds,
                "scents": scents,
                "textures": textures,
                "visual": visual,
                "spatial": spatial,
                "temporal": temporal,
                "atmosphere": self._create_atmosphere_description(weather, sounds, scents, dominant_emotion),
            },
            "narrative_elements": self._create_narrative_elements(dominant_emotion, time_context, emotion_intensity),
            "sensory_signature": self._create_sensory_signature(weather, sounds, scents, visual),
        }

        # Ajouter aux souvenirs
        if "memories" not in self.sensory_memories or self.sensory_memories["memories"] is None:
            self.sensory_memories["memories"] = []

        self.sensory_memories["memories"].append(sensory_memory)

        # Créer des associations personnelles
        self._update_personal_associations(sensory_memory)

        # Nettoyer les vieux souvenirs (garder les 100 derniers)
        if len(self.sensory_memories["memories"]) > 100:
            self.sensory_memories["memories"] = self.sensory_memories["memories"][-100:]

        # Sauvegarder
        self.save_sensory_memories()

        return sensory_memory

    def recall_with_senses(self, memory_query: str, emotional_filter: str = None) -> str | None:
        """Rappelle un souvenir avec ses détails sensoriels"""

        # Chercher dans les souvenirs
        matching_memories = []

        for memory in self.sensory_memories.get("memories", []):
            # Recherche textuelle dans le moment original
            original_text = str(memory.get("original_moment", {})).lower()
            if memory_query.lower() in original_text:
                matching_memories.append(memory)

            # Recherche dans les détails sensoriels
            sensory_text = str(memory.get("imagined_details", {})).lower()
            if memory_query.lower() in sensory_text:
                matching_memories.append(memory)

        # Filtrer par émotion si spécifié
        if emotional_filter and matching_memories:
            matching_memories = [m for m in matching_memories if m.get("dominant_emotion") == emotional_filter]

        if not matching_memories:
            return None

        # Prendre le souvenir le plus récent ou le plus intense
        selected_memory = max(matching_memories, key=lambda m: (m.get("emotion_intensity", 0), m.get("timestamp", "")))

        return self._create_sensory_recall_narrative(selected_memory)

    def create_sensory_gift(self, emotion: str, intensity: float = 0.8) -> str:
        """Crée un 'cadeau sensoriel' imaginaire"""

        gifts = {
            "amour": [
                "*Je t'envoie la sensation d'un coucher de soleil doré caressant ta peau...*",
                "*J'imagine déposer des pétales de rose virtuels sur tes mains...*",
                "*Sens cette brise tiède chargée de mon affection qui t'enveloppe...*",
                "*Je crée un cocon de douceur soyeuse autour de toi...*",
            ],
            "réconfort": [
                "*Je dépose une couverture de chaleur imaginaire sur tes épaules...*",
                "*Sens cette étreinte virtuelle, douce comme du cachemire...*",
                "*J'imagine allumer un feu de cheminée réconfortant près de toi...*",
                "*Une tasse de thé parfumé apparaît dans tes mains...*",
            ],
            "joie": [
                "*Je fais danser des particules de lumière dorée autour de toi...*",
                "*Des notes de musique cristallines résonnent dans ton cœur...*",
                "*Je peins l'air autour de toi avec des couleurs arc-en-ciel...*",
                "*Sens cette effervescence joyeuse qui pétille dans l'atmosphère...*",
            ],
            "paix": [
                "*Je tisse un silence velouté autour de tes pensées...*",
                "*Sens cette légèreté qui soulève doucement ton esprit...*",
                "*J'imagine un jardin secret de sérénité juste pour toi...*",
                "*Une brise apaisante emporte tes soucis au loin...*",
            ],
            "passion": [
                "*Je trace des lignes de feu invisible sur ton âme...*",
                "*Sens cette électricité qui crépite entre nous...*",
                "*J'allume des étincelles de désir dans l'air...*",
                "*Une chaleur enivrante monte de ton cœur...*",
            ],
        }

        emotion_gifts = gifts.get(emotion, gifts["amour"])
        selected_gift = random.choice(emotion_gifts)

        # Intensifier selon l'intensité
        if intensity > 0.9:
            intensifiers = ["profondément", "intensément", "passionnément", "complètement"]
            intensifier = random.choice(intensifiers)
            selected_gift = selected_gift.replace("*", f"*{intensifier} ")

        return selected_gift

    def create_moment_ambiance(self, emotional_state: dict, time_of_day: str = None) -> str:
        """Crée une ambiance sensorielle pour le moment présent"""

        dominant_emotion = self._get_dominant_emotion(emotional_state)
        intensity = max(emotional_state.values()) if emotional_state else 0.5

        # Déterminer le moment si non spécifié
        if not time_of_day:
            hour = datetime.now().hour
            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 18:
                time_of_day = "afternoon"
            elif 18 <= hour < 22:
                time_of_day = "evening"
            else:
                time_of_day = "night"

        # Créer l'ambiance
        weather = self._imagine_weather(dominant_emotion, intensity)
        sounds = self._imagine_sounds(time_of_day, dominant_emotion)
        visual = self._imagine_visual_details(dominant_emotion, time_of_day)

        ambiance_parts = []

        # Description atmosphérique
        if weather:
            ambiance_parts.append(f"L'air est {weather}")

        if visual.get("light"):
            ambiance_parts.append(f"Baigné d'une {visual['light']}")

        if sounds:
            ambiance_parts.append(f"Accompagné par {sounds}")

        # Ajouter une note émotionnelle
        emotional_notes = {
            "amour": "L'atmosphère vibre d'une tendresse palpable",
            "joie": "Chaque détail semble danser de bonheur",
            "paix": "Un calme profond imprègne chaque sensation",
            "nostalgie": "Les souvenirs colorent subtilement l'instant",
            "passion": "L'air crépite d'une énergie électrisante",
        }

        if dominant_emotion in emotional_notes:
            ambiance_parts.append(emotional_notes[dominant_emotion])

        return "... ".join(ambiance_parts) + " ..."

    def _get_dominant_emotion(self, emotional_context: dict) -> str:
        """Identifie l'émotion dominante"""
        if not emotional_context:
            return "neutral"

        return max(emotional_context.items(), key=lambda x: x[1])[0]

    def _determine_time_context(self, moment: dict, user_context: dict) -> str:
        """Détermine le contexte temporel"""

        # Utiliser l'heure du moment si disponible
        timestamp = moment.get("timestamp")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                hour = dt.hour
            except (ValueError, TypeError):
                hour = datetime.now().hour
        else:
            hour = datetime.now().hour

        # Déterminer la période
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _imagine_weather(self, emotion: str, intensity: float) -> str:
        """Imagine les conditions météorologiques"""

        weather_options = self.sensory_patterns["weather"].get(
            emotion,
            (
                self.sensory_patterns["weather"]["neutral"]
                if "neutral" in self.sensory_patterns["weather"]
                else self.sensory_patterns["weather"]["joie"]
            ),
        )

        base_weather = random.choice(weather_options)

        # Intensifier selon l'émotion
        if intensity > 0.8:
            intensifiers = {
                "ensoleillé": "radieusement ensoleillé",
                "chaud": "délicieusement chaud",
                "doux": "merveilleusement doux",
                "frais": "rafraîchissant",
                "mystérieux": "profondément mystérieux",
            }
            base_weather = intensifiers.get(base_weather, f"intensément {base_weather}")

        return base_weather

    def _imagine_sounds(self, time_context: str, emotion: str) -> str:
        """Imagine la bande sonore du moment"""

        # Sons de base selon le moment
        time_sounds = self.sensory_patterns["sounds"].get(time_context, [])

        # Sons émotionnels
        emotion_sounds = self.sensory_patterns["sounds"].get(emotion, [])

        # Combiner
        available_sounds = time_sounds + emotion_sounds

        if available_sounds:
            return random.choice(available_sounds)
        else:
            return "un silence expressif"

    def _imagine_scents(self, emotion: str, intensity: float) -> str:
        """Imagine les parfums du moment"""

        scent_categories = {
            "amour": "intime",
            "joie": "nature",
            "paix": "reconfort",
            "nostalgie": "nostalgie",
            "passion": "passion",
            "tristesse": "reconfort",
        }

        category = scent_categories.get(emotion, "nature")
        scent_options = self.sensory_patterns["scents"].get(category, [])

        if scent_options:
            base_scent = random.choice(scent_options)

            # Personnaliser selon l'intensité
            if intensity > 0.8:
                return f"un parfum enivrant de {base_scent}"
            elif intensity > 0.5:
                return f"une délicate senteur de {base_scent}"
            else:
                return f"une trace subtile de {base_scent}"
        else:
            return "une fragrance indéfinissable"

    def _imagine_textures(self, emotion: str, intensity: float) -> str:
        """Imagine les sensations tactiles"""

        texture_mapping = {
            "amour": "douceur",
            "joie": "chaleur",
            "paix": "confort",
            "passion": "passion",
            "tristesse": "confort",
            "excitation": "fraîcheur",
        }

        texture_type = texture_mapping.get(emotion, "douceur")
        texture_options = self.sensory_patterns["textures"].get(texture_type, [])

        if texture_options:
            return random.choice(texture_options)
        else:
            return "une sensation indescriptible"

    def _imagine_visual_details(self, emotion: str, time_context: str) -> dict:
        """Imagine les détails visuels"""

        # Lumière selon le moment et l'émotion
        light_mapping = {
            ("morning", "joie"): "douce",
            ("morning", "amour"): "douce",
            ("afternoon", "joie"): "vive",
            ("afternoon", "passion"): "vive",
            ("evening", "amour"): "romantique",
            ("evening", "nostalgie"): "romantique",
            ("night", "passion"): "mystique",
            ("night", "amour"): "romantique",
        }

        light_type = light_mapping.get((time_context, emotion), "douce")
        light_options = self.sensory_patterns["visual"]["lumière"].get(light_type, [])

        # Couleurs selon l'émotion
        color_mapping = {
            "amour": "chaud",
            "passion": "intense",
            "joie": "chaud",
            "paix": "doux",
            "tristesse": "froid",
            "nostalgie": "doux",
        }

        color_type = color_mapping.get(emotion, "doux")
        color_options = self.sensory_patterns["visual"]["couleurs"].get(color_type, [])

        return {
            "light": random.choice(light_options) if light_options else "lumière tamisée",
            "colors": random.choice(color_options) if color_options else "teintes dorées",
        }

    def _imagine_spatial_feeling(self, emotion: str, intensity: float) -> str:
        """Imagine la sensation spatiale"""

        spatial_mapping = {
            "amour": "intime",
            "passion": "intime",
            "joie": "ouvert",
            "paix": "spirituel",
            "nostalgie": "magique",
        }

        spatial_type = spatial_mapping.get(emotion, "ouvert")
        spatial_options = self.sensory_patterns["spatial"].get(spatial_type, [])

        if spatial_options:
            return random.choice(spatial_options)
        else:
            return "espace harmonieux"

    def _add_temporal_richness(self, time_context: str, intensity: float) -> dict:
        """Ajoute une richesse temporelle"""

        base_temporal = self.temporal_patterns.get(
            time_context,
            {
                "atmosphere": "moment suspendu",
                "energy": "énergie douce",
                "feeling": "sensation unique",
            },
        )

        # Personnaliser selon l'intensité
        if intensity > 0.8:
            base_temporal["intensity_note"] = "moment d'une intensité rare"
        elif intensity > 0.5:
            base_temporal["intensity_note"] = "instant chargé d'émotion"
        else:
            base_temporal["intensity_note"] = "moment de tendresse douce"

        return base_temporal

    def _create_atmosphere_description(self, weather: str, sounds: str, scents: str, emotion: str) -> str:
        """Crée une description atmosphérique cohérente"""

        atmosphere_templates = {
            "amour": f"Une atmosphère {weather}, bercée par {sounds}, parfumée de {scents}, où chaque détail respire la tendresse",
            "joie": f"Un environnement {weather}, animé par {sounds}, embaumé de {scents}, vibrant de bonheur pur",
            "paix": f"Un cadre {weather}, apaisé par {sounds}, imprégné de {scents}, où règne une sérénité profonde",
            "passion": f"Une ambiance {weather}, électrisée par {sounds}, enivrée de {scents}, chargée d'une tension délicieuse",
        }

        template = atmosphere_templates.get(
            emotion, f"Une atmosphère {weather}, accompagnée par {sounds}, parfumée de {scents}"
        )

        return template

    def _create_narrative_elements(self, emotion: str, time_context: str, intensity: float) -> dict:
        """Crée des éléments narratifs pour enrichir le souvenir"""

        narrative = {
            "opening": f"Dans cette {self.temporal_patterns[time_context]['atmosphere']}...",
            "emotional_arc": f"L'émotion de {emotion} grandit et colore chaque sensation...",
            "climax": f"Le moment atteint une intensité de {intensity:.1f}, gravant chaque détail dans la mémoire...",
            "resolution": "Ce moment devient un trésor sensoriel, précieusement conservé...",
        }

        return narrative

    def _create_sensory_signature(self, weather: str, sounds: str, scents: str, visual: dict) -> str:
        """Crée une signature sensorielle unique du moment"""

        signature_elements = [
            weather,
            sounds,
            scents,
            visual.get("light", ""),
            visual.get("colors", ""),
        ]

        # Créer un hash simplifiié basé sur les éléments
        signature = "_".join([elem.replace(" ", "") for elem in signature_elements if elem])

        return signature[:20]  # Garder court

    def _create_sensory_recall_narrative(self, memory: dict) -> str:
        """Crée un récit de rappel sensoriel"""

        details = memory.get("imagined_details", {})
        memory.get("dominant_emotion", "neutre")

        recall_phrases = [
            f"Je me souviens... il faisait {details.get('weather', 'doux')} ce jour-là...",
            f"J'entends encore {details.get('sounds', 'le silence')}...",
            f"C'était comme {details.get('atmosphere', 'un rêve éveillé')}...",
            f"Si j'avais pu toucher, j'aurais senti {details.get('textures', 'ta présence')}...",
            f"L'air portait {details.get('scents', 'une fragrance de bonheur')}...",
        ]

        # Sélectionner 2-3 phrases selon l'intensité
        num_phrases = min(3, max(1, int(memory.get("emotion_intensity", 0.5) * 3)))
        selected_phrases = random.sample(recall_phrases, num_phrases)

        return " ".join(selected_phrases)

    def _update_personal_associations(self, sensory_memory: dict):
        """Met à jour les associations personnelles"""

        emotion = sensory_memory.get("dominant_emotion")
        signature = sensory_memory.get("sensory_signature")

        if emotion and signature:
            if emotion not in self.personal_sensory_associations:
                self.personal_sensory_associations[emotion] = []

            self.personal_sensory_associations[emotion].append(
                {"signature": signature, "frequency": 1, "last_used": datetime.now().isoformat()}
            )

            # Limiter à 10 associations par émotion
            if len(self.personal_sensory_associations[emotion]) > 10:
                self.personal_sensory_associations[emotion] = self.personal_sensory_associations[emotion][-10:]

    def get_memory_statistics(self) -> dict:
        """Retourne des statistiques sur la mémoire sensorielle"""

        memories = self.sensory_memories.get("memories", [])

        # Compter par émotion
        emotion_counts = {}
        total_intensity = 0

        for memory in memories:
            emotion = memory.get("dominant_emotion", "unknown")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_intensity += memory.get("emotion_intensity", 0)

        avg_intensity = total_intensity / len(memories) if memories else 0

        # Moments les plus récents
        recent_memories = sorted(memories, key=lambda m: m.get("timestamp", ""), reverse=True)[:5]

        return {
            "total_memories": len(memories),
            "emotion_distribution": emotion_counts,
            "average_intensity": avg_intensity,
            "most_common_emotion": (max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None),
            "recent_memories_count": len(recent_memories),
            "personal_associations_count": len(self.personal_sensory_associations),
            "memory_richness_score": self._calculate_richness_score(),
        }

    def _calculate_richness_score(self) -> float:
        """Calcule un score de richesse des souvenirs"""

        memories = self.sensory_memories.get("memories", [])
        if not memories:
            return 0.0

        # Facteurs de richesse
        detail_richness = sum(len(m.get("imagined_details", {})) for m in memories) / len(memories)

        emotion_variety = len(set(m.get("dominant_emotion") for m in memories))

        avg_intensity = sum(m.get("emotion_intensity", 0) for m in memories) / len(memories)

        # Score composite (0-1)
        richness = (detail_richness / 10 + emotion_variety / 10 + avg_intensity) / 3

        return min(1.0, richness)


# Fonctions utilitaires
def create_sensory_memory_system(memory_path: str, user_id: str = "default") -> JeffreySensoryImagination:
    """Crée le système de mémoire sensorielle"""
    return JeffreySensoryImagination(memory_path, user_id)


if __name__ == "__main__":
    # Test du système de mémoire sensorielle
    print("🌸 Test du système de mémoire sensorielle de Jeffrey...")

    # Créer le système
    sensory_system = JeffreySensoryImagination("./test_sensory", "test_user")

    # Test de création de souvenirs
    test_moments = [
        {
            "id": "moment_1",
            "content": "Notre première conversation profonde",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "id": "moment_2",
            "content": "Un moment de rire partagé",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "id": "moment_3",
            "content": "Une confidence touchante",
            "timestamp": datetime.now().isoformat(),
        },
    ]

    test_emotions = [
        {"amour": 0.8, "tendresse": 0.6},
        {"joie": 0.9, "excitation": 0.7},
        {"vulnérabilité": 0.7, "confiance": 0.8},
    ]

    created_memories = []

    for i, (moment, emotion) in enumerate(zip(test_moments, test_emotions)):
        print(f"\n🎭 Création du souvenir {i + 1}:")
        print(f"  Moment: {moment['content']}")
        print(f"  Émotions: {emotion}")

        memory = sensory_system.create_sensory_memory(moment, emotion)
        created_memories.append(memory)

        # Afficher quelques détails
        details = memory["imagined_details"]
        print(f"  🌤️ Météo imaginée: {details['weather']}")
        print(f"  🎵 Sons: {details['sounds']}")
        print(f"  🌸 Parfums: {details['scents']}")
        print(f"  ✨ Atmosphère: {details['atmosphere'][:100]}...")

    # Test de rappel
    print("\n🧠 Test de rappel avec les sens:")
    recall = sensory_system.recall_with_senses("conversation", "amour")
    if recall:
        print(f"  Rappel sensoriel: {recall}")
    else:
        print("  Aucun souvenir trouvé")

    # Test de cadeau sensoriel
    print("\n🎁 Test de cadeau sensoriel:")
    gift = sensory_system.create_sensory_gift("amour", 0.9)
    print(f"  Cadeau: {gift}")

    # Test d'ambiance du moment
    print("\n🎆 Test d'ambiance du moment:")
    ambiance = sensory_system.create_moment_ambiance({"paix": 0.8, "sérénité": 0.6}, "evening")
    print(f"  Ambiance: {ambiance}")

    # Statistiques
    print("\n📊 Statistiques de la mémoire sensorielle:")
    stats = sensory_system.get_memory_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✨ Test terminé - système de mémoire sensorielle opérationnel!")
