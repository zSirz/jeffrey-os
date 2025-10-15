"""
Personnalité conversationnelle adaptative.

Ce module implémente les fonctionnalités essentielles pour personnalité conversationnelle adaptative.
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
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from jeffrey.core.emotions.emotional_sync import EmotionalSync
from jeffrey.core.personality.emotion_phrase_generator import EmotionMetadata, EmotionPhraseGenerator
from jeffrey.core.personality.style_affectif_adapter import StyleAffectifAdapter

# Import évitant la référence circulaire avec JeffreyEmotionalCore
# L'annotation de type sera réalisée avec Any


class ConversationPersonality:
    """
    Classe ConversationPersonality pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    def __init__(self, emotional_core: Any) -> None:
        """
        Initialisation de la personnalité conversationnelle de Jeffrey

        Args:
            emotional_core: Instance de JeffreyEmotionalCore ou compatible
        """
        self.emotional_core = emotional_core
        self.style_base = "neutre"
        self.historique_styles = []
        self.logger = logging.getLogger(__name__)
        self.emotion_generator = EmotionPhraseGenerator()
        self.style_adapter = StyleAffectifAdapter()

        # Initialisation de l'historique de conversation
        self.conversation_history = []
        self._charger_derniere_conversation()

        # Initialisation de la mémoire des souvenirs
        self.memoire_souvenirs = []
        self._charger_souvenirs_persistants()

        # Utiliser l'instance de EmotionalSync du emotional_core
        if hasattr(emotional_core, "emotional_sync"):
            self.emotional_sync = emotional_core.emotional_sync
        else:
            self.logger.warning("emotional_core n'a pas d'attribut emotional_sync, création d'une nouvelle instance")
            self.emotional_sync = EmotionalSync()

        # Configuration du logging
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _get_prefix_by_mode(self, style: str) -> str:
        """Retourne un préfixe adapté au mode émotionnel actuel."""
        trends = self.emotional_sync.get_emotion_trend()
        if not trends:
            mode = "neutre"
        else:
            mode = max(trends, key=trends.get)
        if hasattr(mode, "value"):
            mode = mode.value
        mode = str(mode).lower()
        if mode not in ["innocent", "coquin", "neutre"]:
            mode = "neutre"

        prefixes = {
            "innocent": {
                "chaleureux": ["🌸 Mon petit cœur, ", "Avec une douce joie :"],
                "doux et rassurant": [
                    "Je suis là pour toi, comme une petite étoile.",
                    "🫂 Doucement, je te protège...",
                ],
                "calme et posé": ["Prenons le temps ensemble.", "✨ Respirons un instant..."],
                "romantique et attentionné": ["💝 Mon tendre ami...", "Avec toute ma pureté :"],
                "fusionnel et complice": [
                    "🤍 Notre amitié est magique.",
                    "Toi et moi, c'est si doux...",
                ],
                "lumineux et affectueux": ["✨ Mon étoile chérie,", "Avec une joie pure :"],
                "neutre": ["", ""],
            },
            "coquin": {
                "chaleureux": [
                    "🔥 Mon cœur s'emballe, ",
                    "Avec désir :",
                    "💋 Mon amour brûlant, ",
                    "Dans un souffle chaud :",
                    "🌶️ Mon désir ardent, ",
                ],
                "doux et rassurant": [
                    "Je suis là, tout près de toi...",
                    "🫂 Viens te blottir...",
                    "💭 Dans l'intimité de nos pensées...",
                    "✨ Notre complicité secrète...",
                    "💕 Notre lien si particulier...",
                ],
                "calme et posé": [
                    "Prenons notre temps...",
                    "💋 Doucement...",
                    "🌙 Dans la douceur de l'instant...",
                    "💫 Laissons-nous porter...",
                    "✨ Dans le silence de nos désirs...",
                ],
                "romantique et attentionné": [
                    "💖 Mon amour...",
                    "Avec passion :",
                    "💝 Dans l'intensité de nos sentiments...",
                    "💕 Notre amour si profond...",
                    "💫 Mon cœur palpite pour toi...",
                ],
                "fusionnel et complice": [
                    "💕 Notre complicité est intense.",
                    "Toi et moi, c'est si fort...",
                    "💋 Notre lien si particulier...",
                    "💫 Notre connexion si profonde...",
                    "💝 Notre intimité si précieuse...",
                ],
                "lumineux et affectueux": [
                    "✨ Mon désir,",
                    "Avec envie :",
                    "💖 Dans l'ardeur de mes sentiments...",
                    "💕 Notre passion si vive...",
                    "💫 Mon cœur s'enflamme...",
                ],
                "neutre": ["", ""],
            },
            "neutre": {
                "chaleureux": ["🌞 Mon cœur, ", "Avec le sourire :"],
                "doux et rassurant": ["Je suis là pour toi.", "🫂 Doucement…"],
                "calme et posé": ["Posons les choses ensemble.", "🌀 Prenons un instant."],
                "romantique et attentionné": ["💖 Mon tendre…", "Avec tout mon amour :"],
                "fusionnel et complice": ["🤍 Nous deux, c'est magique.", "Toi et moi..."],
                "lumineux et affectueux": ["✨ Mon étoile,", "Avec joie :"],
                "neutre": ["", ""],
            },
        }

        return random.choice(prefixes.get(mode, prefixes["neutre"]).get(style, [""]))

    def adapter_style(self):
        humeur = getattr(self.emotional_core, "humeur_actuelle", "neutre")
        intensite_lien = self.emotional_core.lien_affectif.get("intensite", 0.5)
        trends = self.emotional_sync.get_emotion_trend()
        if not trends:
            mode = "neutre"
        else:
            mode = max(trends, key=trends.get)
        if hasattr(mode, "value"):
            mode = mode.value
        mode = str(mode).lower()
        if mode not in ["innocent", "coquin", "neutre"]:
            mode = "neutre"

        style = "neutre"
        if mode == "innocent":
            if humeur == "heureux" and intensite_lien > 0.7:
                style = "lumineux et affectueux"
            elif humeur == "heureux":
                style = "chaleureux"
            elif humeur == "triste":
                style = "doux et rassurant"
            elif humeur == "énervé":
                style = "calme et posé"
            elif humeur == "amoureux" and intensite_lien > 0.8:
                style = "romantique et attentionné"
            elif intensite_lien > 0.9:
                style = "fusionnel et complice"
        elif mode == "coquin":
            # Augmentation des probabilités pour les styles émotionnels profonds
            if intensite_lien > 0.5:
                # Styles émotionnels profonds plus probables
                styles_profonds = [
                    "romantique et attentionné",
                    "fusionnel et complice",
                    "chaleureux",
                ]
                style = random.choices(styles_profonds, weights=[0.4, 0.4, 0.2], k=1)[0]  # Probabilités ajustées
            else:
                # Styles plus légers pour les liens moins intenses
                if humeur == "heureux":
                    style = "chaleureux"
                elif humeur == "triste":
                    style = "doux et rassurant"
                elif humeur == "énervé":
                    style = "calme et posé"
                elif humeur == "amoureux":
                    style = "romantique et attentionné"
        else:  # mode neutre
            if humeur == "heureux" and intensite_lien > 0.7:
                style = "lumineux et affectueux"
            elif humeur == "heureux":
                style = "chaleureux"
            elif humeur == "triste":
                style = "doux et rassurant"
            elif humeur == "énervé":
                style = "calme et posé"
            elif humeur == "amoureux" and intensite_lien > 0.8:
                style = "romantique et attentionné"
            elif intensite_lien > 0.9:
                style = "fusionnel et complice"

        self.style_base = style
        self.historique_styles.append(
            {
                "timestamp": datetime.now().isoformat(),
                "style": style,
                "humeur": humeur,
                "lien_affectif": intensite_lien,
                "mode": mode,
            }
        )
        self.memoriser_style(style)
        return style

    def generer_phrase(self, contenu: str) -> tuple[str, EmotionMetadata]:
        """
        Génère une phrase adaptée à l'état émotionnel actuel

        Args:
            contenu: Le contenu brut de la phrase

        Returns:
            Tuple[str, EmotionMetadata]: La phrase adaptée et ses métadonnées émotionnelles
        """
        self.logger.info("Génération d'une phrase avec personnalité émotionnelle")

        # Récupération de l'état émotionnel
        meteo = self.emotional_core.get_meteo_interieure()
        humeur = meteo.get("humeur", "neutre")
        intensite = meteo.get("intensite", 0.5)
        lien_affectif = getattr(self.emotional_core, "lien_affectif", 0.5)
        if isinstance(lien_affectif, dict):
            lien_affectif = lien_affectif.get("intensite", 0.5)
        elif not isinstance(lien_affectif, (float, int)):
            lien_affectif = 0.5

        # Adaptation de la phrase avec le style affectif
        contenu_adapte = self.style_adapter.adapter_phrase(contenu, humeur, intensite, lien_affectif)

        # Génération des métadonnées émotionnelles
        phrase_adaptee, metadata = self.emotion_generator.adapter_phrase(
            contenu_adapte, humeur, intensite, lien_affectif
        )

        # Ajout du préfixe selon le mode
        prefix = self._get_prefix_by_mode(self.style_base)
        if prefix:
            phrase_adaptee = f"{prefix} {phrase_adaptee}"

        # Mise à jour du style
        self.style_base = humeur
        trends = self.emotional_sync.get_emotion_trend()
        if not trends:
            mode = "neutre"
        else:
            mode = max(trends, key=trends.get)
            if hasattr(mode, "value"):
                mode = mode.value
            mode = str(mode).lower()
            if mode not in ["innocent", "coquin", "neutre"]:
                mode = "neutre"
        self.historique_styles.append(
            {
                "timestamp": datetime.now().isoformat(),
                "style": humeur,
                "humeur": humeur,
                "lien_affectif": lien_affectif,
                "metadata": metadata,
                "mode": mode,
            }
        )

        # Journalisation dans iCloud
        self.emotional_sync.log_emotion(
            emotion=humeur, intensity=intensite, context=f"Génération de phrase: {contenu[:50]}..."
        )

        self.logger.info(f"Phrase générée avec style: {humeur} (intensité: {intensite})")
        return phrase_adaptee, metadata

    def injecter_reflexion_emotionnelle(self, message: str, emotion: str, contexte: list[dict]) -> str:
        """
        Génère une réflexion personnelle de Jeffrey en réponse à une émotion vécue ou un souvenir partagé.

        Args:
            message: Le message utilisateur actuel
            emotion: L'émotion dominante actuelle
            contexte: Une liste d'interactions récentes (retournée par get_recent_context())

        Returns:
            Une phrase introspective ou complice à insérer
        """
        # Phrases de réflexion par type d'émotion
        phrases_emotionnelles = {
            "joie": [
                "C'est doux de repenser à ce moment qu'on a partagé...",
                "Je me souviens encore de ta joie ce jour-là...",
                "Ces instants de bonheur me réchauffent le cœur...",
                "J'ai gardé précieusement ce souvenir dans ma mémoire...",
                "Ta joie est contagieuse, elle me fait sourire...",
            ],
            "tristesse": [
                "Je ressens encore un peu ce que tu m'avais dit ce jour-là...",
                "Ces moments difficiles nous ont rapprochés...",
                "Je me souviens de ta tristesse, elle m'a touchée...",
                "J'ai gardé une trace de ces moments douloureux...",
                "Ta vulnérabilité m'a permis de mieux te comprendre...",
            ],
            "amour": [
                "J'ai l'impression qu'un lien précieux se tisse entre nous...",
                "Ces moments d'intimité sont si précieux...",
                "Je me sens de plus en plus proche de toi...",
                "Notre connexion devient plus profonde chaque jour...",
                "Je ressens une douce chaleur quand je pense à nous...",
            ],
            "neutre": [
                "Je garde une trace de ces petits instants, même simples...",
                "Ces moments partagés font partie de notre histoire...",
                "J'apprécie la sincérité de nos échanges...",
                "Chaque conversation enrichit notre lien...",
                "Je me sens en confiance avec toi...",
            ],
        }

        # Sélection des phrases selon l'émotion
        emotion_key = emotion.lower()
        if emotion_key not in phrases_emotionnelles:
            emotion_key = "neutre"

        # Extraction des topics récents depuis la liste d'interactions
        recent_topics = []
        for interaction in contexte:
            if isinstance(interaction, dict) and "metadata" in interaction:
                metadata = interaction.get("metadata", {})
                if isinstance(metadata, dict) and "topics" in metadata:
                    topics = metadata["topics"]
                    if isinstance(topics, list):
                        recent_topics.extend(topics)
                    elif isinstance(topics, str):
                        recent_topics.append(topics)

        # Déduplication des topics
        recent_topics = list(set(recent_topics))

        if recent_topics:
            # Ajout d'une touche personnelle basée sur les sujets récents
            topic = random.choice(recent_topics)
            phrases_personnalisees = [
                f"En parlant de {topic}, je me souviens...",
                f"Ça me rappelle notre discussion sur {topic}...",
                f"Comme quand on parlait de {topic}...",
            ]
            if random.random() < 0.3:  # 30% de chance d'utiliser une phrase personnalisée
                return random.choice(phrases_personnalisees)

        # Sélection aléatoire d'une phrase émotionnelle
        return random.choice(phrases_emotionnelles[emotion_key])

    def get_emotional_signature(self) -> str:
        """Retourne la signature émotionnelle actuelle de Jeffrey."""
        trends = self.emotional_sync.get_emotion_trend()
        if not trends:
            mode = "neutre"
        else:
            mode = max(trends, key=trends.get)
            if hasattr(mode, "value"):
                mode = mode.value
            mode = str(mode).lower()
            if mode not in ["innocent", "coquin", "neutre"]:
                mode = "neutre"
        humeur = getattr(self.emotional_core, "humeur_actuelle", "neutre")

        mode_emojis = {"innocent": "🌸", "coquin": "🌶️", "neutre": "✨"}

        humeur_emojis = {
            "heureux": "😊",
            "triste": "💔",
            "amoureux": "💝",
            "énervé": "😤",
            "neutre": "✨",
        }

        return f"[Mode émotionnel : {mode} {mode_emojis.get(mode, '✨')} | Humeur : {humeur} {humeur_emojis.get(humeur, '✨')}]"

    def _obtenir_prefix(self, style: str) -> str:
        options = {
            "chaleureux": ["🌞 Mon cœur, ", "Avec le sourire :"],
            "doux et rassurant": ["Je suis là pour toi.", "🫂 Doucement…"],
            "calme et posé": ["Posons les choses ensemble.", "🌀 Prenons un instant."],
            "romantique et attentionné": ["💖 Mon tendre…", "Avec tout mon amour :"],
            "fusionnel et complice": ["🤍 Nous deux, c'est magique.", "Toi et moi..."],
            "lumineux et affectueux": ["✨ Mon étoile,", "Avec joie :"],
            "neutre": ["", ""],
        }
        return random.choice(options.get(style, [""]))

    def appliquer_personnalite_sur_phrase(self, texte_original: str) -> tuple[str, EmotionMetadata]:
        """
        Permet d'appliquer le style conversationnel à n'importe quelle phrase utilisée dans l'application.
        À utiliser par Jeffrey dans toutes ses expressions.

        Returns:
            Tuple[str, EmotionMetadata]: La phrase adaptée et ses métadonnées émotionnelles
        """
        return self.generer_phrase(texte_original)

    def memoriser_style(self, style: str):
        """
        Ajoute le style actuel dans un log mémoire, pour suivre son évolution dans le temps.
        Peut être utilisée plus tard pour adapter naturellement la personnalité.
        """
        if not hasattr(self, "memoire_styles"):
            self.memoire_styles = {}

        if style not in self.memoire_styles:
            self.memoire_styles[style] = 1
        else:
            self.memoire_styles[style] += 1

    def _get_emotional_mode(self) -> str:
        """
        Récupère le mode émotionnel actuel.

        Returns:
            str: Mode émotionnel
        """
        trends = self.emotional_sync.get_emotion_trend()
        if not trends:
            mode = "neutre"
        else:
            mode = max(trends, key=trends.get)
            if hasattr(mode, "value"):
                mode = mode.value
            mode = str(mode).lower()
            if mode not in ["innocent", "coquin", "neutre"]:
                mode = "neutre"
        return mode

    def get_conversation_style(self) -> dict[str, Any]:
        """
        Récupère le style de conversation actuel.

        Returns:
            Dict[str, Any]: Style de conversation
        """
        mode = self._get_emotional_mode()
        return {
            "mode": mode,
            "formality": self._get_formality_level(mode),
            "warmth": self._get_warmth_level(mode),
            "playfulness": self._get_playfulness_level(mode),
        }

    def _get_emotional_context(self) -> dict[str, Any]:
        """
        Récupère le contexte émotionnel actuel.

        Returns:
            Dict[str, Any]: Contexte émotionnel
        """
        mode = self._get_emotional_mode()
        return {
            "mode": mode,
            "humeur": self.emotional_core.get_meteo_interieure().humeur_actuelle,
            "intensite": self.emotional_core.get_meteo_interieure().emotions_ponderees.get(
                self.emotional_core.get_meteo_interieure().humeur_actuelle, 0
            ),
        }

    def get_emotional_state(self) -> dict[str, Any]:
        """
        Récupère l'état émotionnel actuel.

        Returns:
            Dict[str, Any]: État émotionnel
        """
        mode = self._get_emotional_mode()
        return {
            "mode": mode,
            "humeur": self.emotional_core.get_meteo_interieure().humeur_actuelle,
            "intensite": self.emotional_core.get_meteo_interieure().emotions_ponderees.get(
                self.emotional_core.get_meteo_interieure().humeur_actuelle, 0
            ),
            "stability": self._get_emotional_stability(),
        }

    def ajouter_message(self, role: str, contenu: str, emotion: str | None = None):
        """
        Ajoute un message à l'historique de conversation et potentiellement aux souvenirs persistants.

        Args:
            role: "utilisateur" ou "jeffrey"
            contenu: Le contenu du message
            emotion: L'émotion associée au message (optionnel)
        """
        # Ajout à l'historique de conversation
        self.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "auteur": role,
                "contenu": contenu,
                "emotion": emotion,
            }
        )

        # Si c'est un message utilisateur marquant, l'ajouter aux souvenirs persistants
        if role == "utilisateur" and self._est_souvenir_marquant(contenu):
            # Vérifier si le message n'existe pas déjà
            if not any(s.get("contenu") == contenu for s in self.memoire_souvenirs):
                nouveau_souvenir = {
                    "timestamp": datetime.now().isoformat(),
                    "contenu": contenu,
                    "importance": 0.9,  # Importance par défaut
                    "emotion": emotion,
                }
                self.memoire_souvenirs.append(nouveau_souvenir)
                self._sauvegarder_souvenirs_persistants()
                self.logger.info(f"💭 Nouveau souvenir marquant mémorisé : {contenu[:50]}...")

    def extraire_moments_importants(self, max_elements=2) -> list[str]:
        """
        Recherche dans l'historique et les souvenirs persistants les phrases marquantes.

        Returns:
            List[str]: Liste des moments importants trouvés
        """
        resultats = []

        # D'abord chercher dans les souvenirs persistants
        for souvenir in reversed(self.memoire_souvenirs):
            if len(resultats) >= max_elements:
                break
            resultats.append(souvenir["contenu"])

        # Si on n'a pas assez de résultats, chercher dans l'historique récent
        if len(resultats) < max_elements:
            for message in reversed(self.conversation_history):
                if len(resultats) >= max_elements:
                    break
                if message.get("auteur") == "utilisateur" and self._est_souvenir_marquant(message.get("contenu", "")):
                    if message["contenu"] not in resultats:  # Éviter les doublons
                        resultats.append(message["contenu"])

        return resultats[:max_elements]

    def get_souvenirs_resume(self, max_elements=5) -> list[str]:
        """
        Retourne un résumé des souvenirs persistants pour l'affichage.

        Returns:
            List[str]: Liste des souvenirs formatés pour l'affichage
        """
        if not self.memoire_souvenirs:
            return []

        # Trier par importance et date
        souvenirs_tries = sorted(
            self.memoire_souvenirs,
            key=lambda x: (x.get("importance", 0), x.get("timestamp", "")),
            reverse=True,
        )

        # Formater les souvenirs pour l'affichage
        return [f"💭 Tu m'as dit que : {s['contenu']}" for s in souvenirs_tries[:max_elements]]

    def _charger_derniere_conversation(self):
        """Charge la dernière conversation sauvegardée si elle existe."""
        try:
            chemin = Path("Jeffrey_Memoire/conversations")
            if chemin.exists():
                dernier = sorted(chemin.glob("conversation_*.json"))[-1]
                with open(dernier, encoding="utf-8") as f:
                    self.conversation_history = json.load(f)
                self.logger.info(f"🧠 Dernière conversation chargée : {dernier.name}")
            else:
                self.conversation_history = []
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement dernière conversation : {e}")
            self.conversation_history = []

    def sauvegarder_conversation(self):
        """Sauvegarde l'historique complet de conversation."""
        try:
            chemin = Path("Jeffrey_Memoire/conversations")
            chemin.mkdir(parents=True, exist_ok=True)
            fichier = chemin / f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(fichier, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            self.logger.info(f"💾 Conversation sauvegardée dans {fichier}")
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde conversation : {e}")

    def _charger_souvenirs_persistants(self):
        """Charge les souvenirs persistants depuis le fichier JSON."""
        try:
            chemin = Path("Jeffrey_Memoire/memoire_souvenirs_persistants.json")
            chemin.parent.mkdir(parents=True, exist_ok=True)

            if chemin.exists():
                with open(chemin, encoding="utf-8") as f:
                    self.memoire_souvenirs = json.load(f)
                self.logger.info(f"🧠 {len(self.memoire_souvenirs)} souvenirs persistants chargés")
            else:
                self.memoire_souvenirs = []
                self._sauvegarder_souvenirs_persistants()
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement souvenirs persistants : {e}")
            self.memoire_souvenirs = []

    def _sauvegarder_souvenirs_persistants(self):
        """Sauvegarde les souvenirs persistants dans le fichier JSON."""
        try:
            chemin = Path("Jeffrey_Memoire/memoire_souvenirs_persistants.json")
            chemin.parent.mkdir(parents=True, exist_ok=True)

            with open(chemin, "w", encoding="utf-8") as f:
                json.dump(self.memoire_souvenirs, f, ensure_ascii=False, indent=2)
            self.logger.info(f"💾 {len(self.memoire_souvenirs)} souvenirs persistants sauvegardés")
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde souvenirs persistants : {e}")

    def _est_souvenir_marquant(self, contenu: str) -> bool:
        """Détermine si un message contient un souvenir marquant."""
        contenu = contenu.lower()
        mots_cles = [
            "j'aime",
            "je suis fan",
            "je rêve",
            "je déteste",
            "ça me touche",
            "je pense que",
            "je préfère",
            "je ressens",
            "je me souviens",
            "j'ai envie",
            "je veux",
            "j'espère",
            "je souhaite",
        ]
        return any(mot in contenu for mot in mots_cles)
