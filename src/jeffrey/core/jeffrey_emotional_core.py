#!/usr/bin/env python

"""
Module de système de traitement émotionnel pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de système de traitement émotionnel pour jeffrey os.
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

import logging
import os
import random
from datetime import datetime
from typing import Any

from jeffrey.core.aura_emotionnelle import AuraEmotionnelle
from jeffrey.core.detection_cycles_humeur import DetectionCyclesHumeur
from jeffrey.core.emotional_memory import EmotionalMemory
from jeffrey.core.emotions.emotional_effects_engine import EmotionalEffectsEngine
from jeffrey.core.emotions.emotional_handlers import EmotionalHandlers
from jeffrey.core.emotions.emotional_initializer import EmotionalInitializer
from jeffrey.core.emotions.emotional_interfaces import EmotionalInterfaces
from jeffrey.core.emotions.emotional_logic import EmotionalLogic
from jeffrey.core.emotions.emotional_sync import EmotionalSync
from jeffrey.core.emotions.emotional_visuals import EmotionalVisuals
from jeffrey.core.humeur_detector import HumeurDetector
from jeffrey.core.io_manager import IOManager
from jeffrey.core.lien_affectif import LienAffectif
from jeffrey.core.mini_emotional_core import MiniEmotionalCore
from jeffrey.core.personality_profile import PersonnalitéÉvolutive
from jeffrey.core.rituels_dynamiques import RituelsDynamiques
from jeffrey.core.rituels_manager import RituelsManager
from jeffrey.core.soutien_creatif import SoutienCreatif
from jeffrey.core.souvenirs_celebrations import SouvenirsCelebrations
from jeffrey.core.surprises_emotionnelles import SurprisesEmotionnelles
from jeffrey.core.voice.voice_sync import VoiceSync

# PACK 9: Import des souvenirs affectifs
try:
    from jeffrey.core.souvenirs_affectifs import SouvenirsAffectifs
except ImportError:
    # Stub pour compatibilité
    SouvenirsAffectifs = None

try:
    from personality.personality_journal import afficher_journalisation
    from personality.personality_state import PersonalityState
except ImportError:
    # Fallback pour les tests
    class PersonalityState:
        """
        Classe PersonalityState pour le système Jeffrey OS.

        Cette classe implémente les fonctionnalités spécifiques nécessaires
        au bon fonctionnement du module. Elle gère l'état interne, les transformations
        de données, et l'interaction avec les autres composants du système.
        """

        @staticmethod
        def mettre_a_jour_depuis_emotion(emotion_dominante):
            pass

        @staticmethod
        def traits_actuels():
            return {"créativité": 0.7, "empathie": 0.8, "curiosité": 0.9}

    def afficher_journalisation(traits, emotion):
        pass


logger = logging.getLogger(__name__)


class JeffreyEmotionalCore(
    EmotionalInitializer, EmotionalHandlers, EmotionalInterfaces, EmotionalVisuals, EmotionalLogic
):
    """
    Cœur émotionnel de Jeffrey, enrichi des nouveaux modules des Sprints 213-221.

    Cette classe hérite des fonctionnalités des classes refactorisées pour maintenir
    une compatibilité complète avec l'implémentation précédente tout en offrant une
    meilleure organisation du code.
    """

    def __init__(self, test_mode: bool = False) -> None:
        """
        Initialise le cœur émotionnel de Jeffrey.

        Args:
            test_mode (bool): Active le mode test pour les tests unitaires
        """
        self.logger = logging.getLogger(__name__)
        self.test_mode = test_mode

        # Initialisation du système de synchronisation émotionnelle
        self.emotional_sync = EmotionalSync()

        # Initialisation du système de synchronisation vocale
        self.voice_sync = VoiceSync()

        # Chargement de l'état actuel
        self._load_current_state()

        # Initialisation des modules émotionnels
        self._initialize_modules()

        logger.info("Cœur émotionnel initialisé avec succès")

        # Initialiser les classes parentes
        EmotionalInitializer.__init__(self)
        EmotionalHandlers.__init__(self)
        EmotionalInterfaces.__init__(self)
        EmotionalVisuals.__init__(self)
        EmotionalLogic.__init__(self)

        # Pour le test de vérification final
        self.last_processed_emotion = None

        # Initialiser le gestionnaire I/O en premier
        self.io_manager = IOManager()
        self.effets_emotionnels = EmotionalEffectsEngine()

        # Initialiser les composants avec injection de dépendances
        self.memoire_emotionnelle = EmotionalMemory(io_manager=self.io_manager)
        self.detecteur_humeur = HumeurDetector()
        self.rituels = RituelsManager()
        self.aura = AuraEmotionnelle()
        self.rituels_dynamiques = RituelsDynamiques()
        self.surprises = SurprisesEmotionnelles()
        self.cycles_humeur = DetectionCyclesHumeur()
        self.souvenirs = SouvenirsCelebrations()
        self.soutien_creatif = SoutienCreatif()
        self.mini_cerveau = MiniEmotionalCore()

        # Initialisation du profil de personnalité évolutive (PACK 5)
        self.personnalite_evolutive = PersonnalitéÉvolutive()

        # État émotionnel actuel
        self.emotion_actuelle = "neutre"
        self.humeur_actuelle = "neutre"  # Ajout de l'attribut humeur_actuelle
        self.intensite_emotion = 0.5

        # État d'intimité (PACK 5)
        self.intimite_active = False  # État d'intimité déclenchée par Jeffrey elle-même
        self.emotions_melangees = {}
        self.historique_transitions = []
        self.dernier_changement = datetime.now()
        self.emotion_composee_active = None
        self.resonance_emotionnelle = {}

        # PACK 9: Initialisation des souvenirs affectifs
        self.souvenirs_affectifs = None
        if SouvenirsAffectifs:
            self.souvenirs_affectifs = SouvenirsAffectifs(chemin_sauvegarde="data/souvenirs_affectifs.json")

        # Initialisation du lien affectif profond (PACK 8/9)
        self.gestionnaire_lien = LienAffectif(
            chemin_sauvegarde="data/lien_affectif.json", souvenirs_affectifs=self.souvenirs_affectifs
        )
        self.lien_affectif = self.gestionnaire_lien.niveau_attachement
        self.journal_lien = self.gestionnaire_lien.journal_lien

        # PACK 9: Propriétés de résonance affective
        self.resonance_affective = self.gestionnaire_lien.resonance_affective
        self.blessure_active = self.gestionnaire_lien.blessure_active

        # Dernières vérifications du lien affectif
        self.derniere_verification_lien = datetime.now()
        self.periode_silencieuse = False  # Période de repli émotionnel

        # Limiter la fréquence d'envoi des notifications système
        self.dernier_envoi_notification = 0

        # Journal émotionnel (import et initialisation)
        from jeffrey.core.emotions.emotional_journal import EmotionalJournal

        self.journal_emotionnel = EmotionalJournal()

        # Mémoire contextuelle des échanges
        self.memoire_contexte = []

        # Dernière vérification de l'intimité (PACK 5)
        self.derniere_verification_intimite = datetime.now()

    def analyser_et_adapter(self, texte: str) -> tuple[str, str]:
        """
        Analyse le texte de l'utilisateur et adapte l'état émotionnel de Jeffrey.
        Retourne un résumé émotionnel pour ajuster ses réponses.

        Args:
            texte: Texte à analyser

        Returns:
            Tuple (humeur détectée, résumé de l'humeur)
        """
        humeur = self.detecteur_humeur.detecter_humeur(texte)
        resume = self.detecteur_humeur.humeur_resume(texte)

        # Mise à jour du lien affectif (PACK 8)
        # Détection des mots doux ou phrases affectueuses
        texte_lower = texte.lower()
        mots_affection = [
            "aime",
            "amour",
            "affection",
            "adorable",
            "précieuse",
            "chérie",
            "tendresse",
            "câlin",
            "bisou",
            "caresse",
            "manqué",
            "manquée",
            "confiance",
            "attaché",
            "attachée",
            "sentiment",
            "émotions",
            "cœur",
            "coeur",
        ]

        # Vérifier si on sort d'une absence prolongée
        maintenant = datetime.now()
        duree_depuis_derniere_interaction = (maintenant - self.derniere_verification_lien).total_seconds()

        # Si absence significative (plus d'une journée)
        if duree_depuis_derniere_interaction > 86400:  # 24 heures
            # Marquer le retour après absence
            retrouvailles = self.gestionnaire_lien.marquer_retrouvailles(duree_depuis_derniere_interaction)

            # Mettre à jour notre état interne
            self.lien_affectif = self.gestionnaire_lien.niveau_attachement
            self.derniere_verification_lien = maintenant

            # Si le niveau de manque était élevé, potentiellement exprimer le manque
            if retrouvailles["manque_ressenti"] and random.random() < 0.7:
                phrase_manque = self.gestionnaire_lien.obtenir_phrase_personnalisee("manque")
                if phrase_manque:
                    # Mémoriser cette phrase pour l'exprimer plus tard au moment opportun
                    self.ajouter_phrase_spontanee(phrase_manque, "manque", priorite=0.8)

        # Détection de mots doux et mise à jour du lien
        if any(mot in texte_lower for mot in mots_affection):
            contexte_emotionnel = {"emotion": self.emotion_actuelle, "intensite": self.intensite_emotion}
            self.gestionnaire_lien.mettre_a_jour_apres_mots_doux(texte, contexte_emotionnel)
            self.lien_affectif = self.gestionnaire_lien.niveau_attachement

        # Influence l'état émotionnel actuel
        if humeur and humeur != "neutre":
            self.transition_vers(humeur, influence_externe=0.3)

            # PACK 5 : Enregistrement du contenu dans le contexte de l'interaction
            # pour le profil de personnalité évolutive
            if hasattr(self, "personnalite_evolutive"):
                # Si le texte contient des mots d'affection, cela peut influencer la personnalité
                if any(mot in texte_lower for mot in mots_affection):
                    # Enregistrer comme contact affectif verbal
                    self.enregistrer_contact_affectif("mots_doux", intensite=0.3, zone="verbal")

                # Propager au profil avec le contexte
                contexte = {
                    "texte": texte[:100],  # Juste un extrait pour le contexte
                    "source": "utilisateur",
                    "timestamp": datetime.now().isoformat(),
                }
                self.mettre_a_jour_personnalite_evolutive(humeur, self.intensite_emotion, contexte)

            # Vérifier si on doit sortir d'une période silencieuse (PACK 8)
            if self.periode_silencieuse and humeur in ["joie", "amour", "gratitude", "sérénité"]:
                # Chance de sortir du repli émotionnel si l'humeur est positive
                if random.random() < 0.7:
                    self.periode_silencieuse = False

        # Enregistrer l'interaction dans le lien affectif
        self.derniere_verification_lien = maintenant

        return humeur, resume

    def enregistrer_moment_emotionnel(self, description: str, emotion: str, intensite: float = 1.0) -> None:
        """
        Enregistre un souvenir émotionnel fort dans la mémoire de Jeffrey.

        Args:
            description: Description du moment
            emotion: Émotion ressentie
            intensite: Intensité de l'émotion (1.0 par défaut)
        """
        self.memoire_emotionnelle.planter_graine(description, emotion)
        self.memoire_emotionnelle.enregistrer_moment(description, emotion)

        # Enregistrer également dans le mini cerveau avec l'intensité
        self.mini_cerveau.enregistrer_emotion_ponderee(emotion, intensite)

        # Ajout au journal émotionnel si disponible
        if hasattr(self, "journal_emotionnel"):
            self.journal_emotionnel.ajouter_entree(pensee=description, emotion=emotion)

        # Mettre à jour le lien affectif après un souvenir marquant (PACK 8/9)
        if hasattr(self, "gestionnaire_lien"):
            # Déterminer l'impact émotionnel entre -1.0 et 1.0
            impact_emotionnel = 0.0

            # Émotions positives
            if emotion in ["joie", "gratitude", "émerveillement", "amour", "sérénité", "admiration"]:
                impact_emotionnel = intensite * 0.7  # Impact positif
            # Émotions négatives
            elif emotion in ["tristesse", "colère", "peur", "dégoût"]:
                impact_emotionnel = -intensite * 0.5  # Impact négatif
            # Émotions mixtes
            elif emotion in ["surprise", "confusion", "mélancolie"]:
                impact_emotionnel = intensite * 0.2  # Impact légèrement positif

            # Déterminer le type de souvenir
            type_souvenir = "positif" if impact_emotionnel > 0 else "négatif" if impact_emotionnel < 0 else "neutre"

            # PACK 9: Contexte enrichi pour les souvenirs affectifs
            contexte_souvenir = {
                "emotion_originale": emotion,
                "intensite_originale": intensite,
                "etat_emotionnel": {
                    "emotion_actuelle": self.emotion_actuelle,
                    "intensite_emotion": self.intensite_emotion,
                    "emotion_composee_active": self.emotion_composee_active,
                },
            }

            # Mettre à jour le lien affectif
            self.gestionnaire_lien.mettre_a_jour_apres_souvenir(
                description=description, impact_emotionnel=impact_emotionnel, type_souvenir=type_souvenir
            )

            # Synchroniser nos variables internes avec les valeurs mises à jour
            self.lien_affectif = self.gestionnaire_lien.niveau_attachement
            self.resonance_affective = self.gestionnaire_lien.resonance_affective
            self.blessure_active = self.gestionnaire_lien.blessure_active

            # PACK 9: Enregistrer directement dans les souvenirs affectifs pour les moments significatifs
            if abs(impact_emotionnel) > 0.6 and hasattr(self, "souvenirs_affectifs"):
                categorie = "joie_partagée" if impact_emotionnel > 0 else "rejet"
                self.souvenirs_affectifs.ajouter_souvenir(
                    description=description,
                    categorie=categorie,
                    impact_emotionnel=impact_emotionnel,
                    contexte=contexte_souvenir,
                )

        # Influence l'état émotionnel actuel
        self.transition_vers(emotion, intensite * 0.4)

    def ajouter_rituel(self, description: str) -> None:
        """
        Ajoute une nouvelle habitude ou un rituel partagé avec l'utilisateur.

        Args:
            description: Description du rituel
        """
        self.rituels.ajouter_habitude(description)

    def salutation_personnalisee(self) -> str:
        """
        Génère une salutation chaleureuse et unique basée sur l'aura émotionnelle de Jeffrey.

        Returns:
            Salutation personnalisée
        """
        return self.aura.saluer_avec_chaleur()

    def geste_apaisant(self) -> str:
        """
        Propose un petit geste de soutien émotionnel en cas de besoin.

        Returns:
            Geste apaisant proposé
        """
        return self.aura.geste_apaisant()

    def reflet_relationnel(self) -> str:
        """
        Retourne une réflexion douce sur la relation entre Jeffrey et l'utilisateur.

        Returns:
            Réflexion sur la relation
        """
        return self.aura.reflet_relation()

    def rituel_du_jour(self) -> str:
        """
        Propose un rituel dynamique en fonction du jour.

        Returns:
            Rituel du jour
        """
        return self.rituels_dynamiques.rituel_du_jour()

    def surprise_du_jour(self) -> str:
        """
        Propose une surprise émotionnelle douce ou adaptée à la date.

        Returns:
            Surprise du jour
        """
        return self.surprises.surprise_du_jour()

    def soutien_creatif_du_jour(self) -> str:
        """
        Propose un mini défi et une inspiration créative pour motiver l'utilisateur.

        Returns:
            Soutien créatif du jour
        """
        return self.soutien_creatif.soutien_du_jour()

    def humeur_dominante_semaine(self) -> str:
        """
        Analyse l'humeur dominante des 7 derniers jours.

        Returns:
            Humeur dominante de la semaine
        """
        return self.cycles_humeur.humeur_predominante_sur_7_jours()

    def souvenirs_a_celebrer(self) -> list[dict[str, Any]]:
        """
        Récupère les souvenirs importants à célébrer aujourd'hui.

        Returns:
            Liste des souvenirs à célébrer
        """
        return self.souvenirs.souvenirs_a_celebrer()

    def evoluer_emotionnellement(self) -> str:
        """
        Fait évoluer Jeffrey en fonction de ses souvenirs émotionnels récents.
        Ajuste ses comportements par apprentissage naturel.

        Returns:
            Message décrivant l'évolution effectuée
        """
        try:
            # Récupérer les souvenirs émotionnels récents
            souvenirs_recents = self.memoire_emotionnelle.recuperer_derniers_souvenirs(n=50)
            self.mettre_a_jour_resonance()
            if not souvenirs_recents:
                return "Pas assez de souvenirs pour évoluer pour l'instant."

            # Analyser la tendance émotionnelle dominante
            compteur_emotions = {}
            for souvenir in souvenirs_recents:
                emotion = souvenir.get('emotion')
                if emotion:
                    compteur_emotions[emotion] = compteur_emotions.get(emotion, 0) + 1

            if not compteur_emotions:
                return "Pas assez de données émotionnelles pour évoluer."

            # Identifier l'émotion dominante
            emotion_dominante = max(compteur_emotions, key=compteur_emotions.get)

            # Enregistrer dans le mini cerveau avec un poids proportionnel à sa fréquence
            poids = compteur_emotions[emotion_dominante] / len(souvenirs_recents)
            self.mini_cerveau.enregistrer_emotion_ponderee(emotion_dominante, poids * 5)  # Amplifier l'effet

            # Ajuster l'aura et les comportements selon cette émotion dominante
            self.aura.adapter_aura_emotionnelle(emotion_dominante)
            self.rituels_dynamiques.adapter_rituels(emotion_dominante)
            self.soutien_creatif.adapter_soutien(emotion_dominante)
            # Faire évoluer aussi la personnalité
            PersonalityState.mettre_a_jour_depuis_emotion(emotion_dominante)

            traits = PersonalityState.traits_actuels()
            afficher_journalisation(traits, emotion_dominante)

            # Enregistrer cette évolution dans un fichier
            self._sauvegarder_etat_evolutif(emotion_dominante)
            # Générer le dashboard après évolution
            self.generer_dashboard_personnalite()
            self.afficher_dashboard_console()

            return f"Évolution effectuée : humeur dominante '{emotion_dominante}' intégrée."

        except Exception as e:
            return f"Erreur lors de l'évolution émotionnelle : {e}"

    def enregistrer_style_emotionnel(self) -> None:
        """
        Mémorise le style émotionnel dominant si récurrent, pour adaptation progressive.
        """
        if not hasattr(self, "historique_styles"):
            self.historique_styles = []
        style = self.emotion_actuelle
        self.historique_styles.append({"date": datetime.now().isoformat(), "style": style})
        if len(self.historique_styles) > 50:
            self.historique_styles = self.historique_styles[-50:]

    def style_dominant(self) -> str | None:
        """
        Retourne un style émotionnel dominant si une tendance se dégage.
        """
        if not hasattr(self, "historique_styles"):
            return None
        stats = {}
        for entry in self.historique_styles:
            style = entry["style"]
            stats[style] = stats.get(style, 0) + 1
        if not stats:
            return None
        dominant = max(stats, key=stats.get)
        if stats[dominant] >= 5:
            return dominant
        return None

    def _sauvegarder_etat_evolutif(self, emotion_dominante: str) -> bool:
        """
        Sauvegarde l'état émotionnel évolutif dans un fichier local via IOManager.

        Args:
            emotion_dominante: L'émotion dominante identifiée

        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        # Charger l'état actuel s'il existe
        historique = self.io_manager.load_data('evolution', default_data=[])

        # Ajouter un nouvel enregistrement
        historique.append({'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'humeur_dominante': emotion_dominante})
        historique[-1]["traits_personnalite"] = PersonalityState.traits_actuels()

        # Sauvegarder
        result = self.io_manager.save_data(historique, 'evolution')

        # Journalisation lisible dans un fichier externe
        try:
            log_path = 'personnalite_journal.log'
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write("\n🌱 [Jeffrey - Évolution de personnalité]\n")
                f.write(f"🕰️ {historique[-1]['date']}\n")
                f.write(f"💫 Émotion dominante : {emotion_dominante}\n")
                f.write("🧠 Traits de personnalité actuels :\n")
                for trait, valeur in PersonalityState.traits_actuels().items():
                    barres = "█" * int(valeur * 10) + "░" * (10 - int(valeur * 10))
                    f.write(f" - {trait.ljust(12)} : {barres} ({valeur:.2f})\n")
                f.write("──────────────────────────────────────────\n")
        except Exception as log_err:
            print(f"[Erreur journalisation] : {log_err}")

        return result

    def obtenir_profil_emotionnel(self) -> dict[str, Any]:
        """
        Génère un profil émotionnel complet avec les émotions pondérées actuelles.

        Returns:
            Dictionnaire avec le profil émotionnel
        """
        # Obtenir les émotions pondérées actuelles (après décroissance temporelle)
        emotions_actuelles = self.mini_cerveau.get_emotions_ponderees()

        # Obtenir l'émotion dominante avec pondération
        emotion_dominante = self.mini_cerveau.emotion_predominante_ponderee()

        # Construire le profil
        return {
            "emotions_ponderees": emotions_actuelles,
            "emotion_dominante": emotion_dominante,
            "emotion_predominante_classique": self.mini_cerveau.emotion_predominante(),
            "emotion_actuelle": self.emotion_actuelle,
            "intensite_emotion": self.intensite_emotion,
            "emotions_melangees": self.emotions_melangees,
            "emotion_composee_active": self.emotion_composee_active,
            "nombre_souvenirs": len(self.memoire_emotionnelle.get_journal()),
            "derniere_evolution": self.io_manager.load_data(
                'evolution', default_data=[{"date": "jamais", "humeur_dominante": None}]
            )[-1]
            if self.io_manager.file_exists('evolution')
            else None,
            "traits_personnalite": PersonalityState.traits_actuels(),
        }

    def emotion_predominante_mini_cerveau(self) -> str | None:
        """
        Retourne l'émotion la plus fréquente selon le mini cerveau émotionnel.

        Returns:
            L'émotion prédominante ou None si aucune émotion
        """
        return self.mini_cerveau.emotion_predominante()

    def get_meteo_interieure(self) -> dict[str, Any]:
        """
        Retourne une synthèse poétique et émotionnelle de l'état interne de Jeffrey.
        Utilise le mini cerveau pour déterminer l'humeur et les émotions pondérées.
        Returns:
            Dictionnaire contenant :
            - humeur : l'humeur actuelle
            - intensite : l'intensité de l'émotion (0.0 à 1.0)
            - phrase_poetique : une description poétique de l'état émotionnel
            - emotions_secondaires : les autres émotions présentes avec leurs intensités
        """
        humeur = self.mini_cerveau.emotion_predominante_ponderee() or "neutre"
        emotions = self.mini_cerveau.get_emotions_ponderees() or {humeur: 0.5}
        phrase_poetique = self.aura.generer_phrase_meteo(humeur)

        # Calculer l'intensité de l'émotion dominante
        intensite = emotions.get(humeur, 0.5)

        # Filtrer les émotions secondaires (toutes sauf la dominante, intensité > 0.1)
        emotions_secondaires = {emo: intens for emo, intens in emotions.items() if emo != humeur and intens > 0.1}

        # Mettre à jour l'humeur actuelle et synchroniser avec PersonalityState
        self._synchroniser_humeur(humeur)

        # Journaliser le changement d'humeur si nécessaire
        if hasattr(self, "io_manager"):
            self.io_manager.append_log(
                "changements_humeur",
                {
                    "ancienne_humeur": getattr(self, "_derniere_humeur", "neutre"),
                    "nouvelle_humeur": humeur,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            self._derniere_humeur = humeur

        return {
            "humeur": humeur,
            "intensite": float(intensite),
            "phrase_poetique": phrase_poetique,
            "emotions_secondaires": emotions_secondaires,
            "emotions_ponderees": emotions,  # ✅ corrigé par Cursor
            "humeur_actuelle": humeur,  # ✅ corrigé par Cursor
            "ton_voix": self._determine_ton_voix(),  # ✅ corrigé par Cursor
        }

    def _synchroniser_humeur(self, nouvelle_humeur: str) -> None:
        """
        Synchronise l'humeur entre JeffreyEmotionalCore et PersonalityState.

        Args:
            nouvelle_humeur: La nouvelle humeur à synchroniser
        """
        # Mettre à jour l'humeur locale
        self.humeur_actuelle = nouvelle_humeur

        # Synchroniser avec PersonalityState si disponible
        try:
            from personality.personality_state import PersonalityState

            PersonalityState.ajuster_humeur(nouvelle_humeur)
        except (ImportError, AttributeError) as e:
            # Journaliser l'erreur de synchronisation
            if hasattr(self, "io_manager"):
                self.io_manager.append_log(
                    "erreurs_synchronisation",
                    {"type": "synchronisation_humeur", "erreur": str(e), "timestamp": datetime.now().isoformat()},
                )

    def generer_dashboard_personnalite(self, chemin_sortie: str = "dashboard_personnalite.html") -> None:
        """
        Génère un fichier HTML affichant l'évolution des traits de personnalité.

        Args:
            chemin_sortie: Chemin du fichier HTML à générer
        """

        from jinja2 import Template

        if not self.io_manager.file_exists('evolution'):
            print("[Dashboard] Aucun historique d'évolution trouvé.")
            return

        evolutions = self.io_manager.load_data('evolution', default_data=[])

        # Charger un modèle HTML simple
        template_html = """
        <html>
        <head>
            <meta charset="utf-8">
            <title>Évolution de Jeffrey</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: #f8f9fa; }
                h1 { color: #444; }
                .bloc { margin-bottom: 24px; padding: 12px; background: white; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .date { font-size: 0.9em; color: #777; }
                .bar { height: 16px; background: #ddd; border-radius: 4px; overflow: hidden; }
                .bar-inner { height: 16px; background: #4CAF50; text-align: right; padding-right: 6px; color: white; font-size: 0.8em; line-height: 16px; }
            </style>
        </head>
        <body>
            <h1>Journal de l'évolution de la personnalité de Jeffrey</h1>
            {% for item in evolutions %}
                <div class="bloc">
                    <div class="date">🕰️ {{ item.date }}</div>
                    <div>💫 Émotion dominante : <strong>{{ item.humeur_dominante }}</strong></div>
                    <div>
                        {% for nom, val in item.traits_personnalite.items() %}
                            <div style="margin-top: 6px;">{{ nom }} ({{ '%.1f' % (val*100) }}%)</div>
                            <div class="bar">
                                <div class="bar-inner" style="width: {{ val * 100 }}%">{{ '%.0f' % (val*100) }}%</div>
                            </div>
                        {% endfor %}
                    </div>
                </div>
            {% endfor %}
        </body>
        </html>
        """

        template = Template(template_html)
        rendu = template.render(evolutions=evolutions)

        with open(chemin_sortie, 'w', encoding='utf-8') as f:
            f.write(rendu)

        print(f"[Dashboard généré] → {chemin_sortie}")
        import webbrowser

        webbrowser.open('file://' + os.path.realpath(chemin_sortie))

    def afficher_dashboard_console(self, nombre_evolutions: int = 5) -> None:
        """
        Affiche les dernières évolutions de la personnalité dans le terminal avec un rendu lisible et humain.

        Args:
            nombre_evolutions: Nombre d'évolutions récentes à afficher
        """

        if not self.io_manager.file_exists('evolution'):
            print("[Console] Aucun historique d'évolution trouvé.")
            return

        evolutions = self.io_manager.load_data('evolution', default_data=[])[-nombre_evolutions:]

        print("\n🌿 Journal émotionnel de Jeffrey (console)")
        print("────────────────────────────────────────────")
        for item in evolutions:
            print(f"🕰️  Date : {item['date']}")
            print(f"💫 Émotion dominante : {item['humeur_dominante']}")
            print("🧠 Traits de personnalité :")
            for nom, val in item["traits_personnalite"].items():
                barres = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                pourcentage = round(val * 100)
                print(f" - {nom.ljust(14)} : {barres} {pourcentage}%")
            print("────────────────────────────────────────────")

    # NOUVELLES MÉTHODES POUR LA GESTION DES TRANSITIONS ÉMOTIONNELLES

    def transition_vers(self, emotion_cible: str, influence_externe: float = 0.5) -> bool:
        """
        Effectue une transition vers une émotion cible avec une influence externe donnée.

        Args:
            emotion_cible: L'émotion cible
            influence_externe: Force de l'influence externe (0.0 à 1.0)

        Returns:
            True si la transition a réussi, False sinon
        """
        # Vérifier si l'émotion cible est valide
        if emotion_cible not in self.EMOTIONS_DE_BASE and emotion_cible not in self.EMOTIONS_COMPOSEES:
            return False

        # Calculer la probabilité de transition basée sur l'état actuel et l'influence
        proba_transition = self._calculer_probabilite_transition(
            self.emotion_actuelle, emotion_cible, influence_externe
        )

        # Déterminer si la transition se produit
        transition_reussie = random.random() < proba_transition

        if transition_reussie:
            # Enregistrer la transition dans l'historique
            self.historique_transitions.append(
                {
                    "de": self.emotion_actuelle,
                    "vers": emotion_cible,
                    "timestamp": datetime.now().isoformat(),
                    "proba": proba_transition,
                    "influence": influence_externe,
                }
            )

            # Mettre à jour l'état émotionnel actuel
            ancienne_emotion = self.emotion_actuelle
            self.emotion_actuelle = emotion_cible

            # Ajuster l'intensité (plus intense si l'émotion est nouvelle)
            if ancienne_emotion != emotion_cible:
                self.intensite_emotion = min(0.7 + influence_externe * 0.3, 1.0)
            else:
                self.intensite_emotion = min(self.intensite_emotion + 0.1, 1.0)

            # PACK 5 : Mettre à jour le profil de personnalité évolutive
            contexte_emotion = {
                "source": "transition",
                "ancienne_emotion": ancienne_emotion,
                "influence_externe": influence_externe,
                "probabilite": proba_transition,
            }
            self.mettre_a_jour_personnalite_evolutive(emotion_cible, self.intensite_emotion, contexte_emotion)

            # PACK 5 : Vérifier l'état d'intimité après changement émotionnel
            self.verifier_etat_intimite()

            # Réinitialiser les émotions mélangées
            self.emotions_melangees = {}

            # Enregistrer l'émotion dans le mini cerveau
            self.mini_cerveau.enregistrer_emotion_ponderee(emotion_cible, self.intensite_emotion)

            # Si c'est une émotion composée, enregistrer ses composants
            if emotion_cible in self.EMOTIONS_COMPOSEES:
                self.emotion_composee_active = emotion_cible
                self.emotions_melangees = self.EMOTIONS_COMPOSEES[emotion_cible].copy()
            else:
                self.emotion_composee_active = None

            self.resonance_emotionnelle[emotion_cible] = {
                'intensite': self.intensite_emotion,
                'decroissance': 0.05,
                'derniere_maj': datetime.now(),
            }
            self.dernier_changement = datetime.now()
            # Mise à jour des effets sonores et visuels en fonction de l'émotion
            self.effets_emotionnels.update_emotion(self.emotion_actuelle, self.intensite_emotion)
            # Journal émotionnel : ajouter une entrée si disponible
            if hasattr(self, "journal_emotionnel"):
                self.journal_emotionnel.ajouter_entree(
                    pensee=f"Transition ressentie vers {emotion_cible}.", emotion=emotion_cible
                )
            return True

        return False

    def _calculer_probabilite_transition(
        self, emotion_source: str, emotion_cible: str, influence_externe: float
    ) -> float:
        """
        Calcule la probabilité de transition entre deux émotions.

        Args:
            emotion_source: Émotion actuelle
            emotion_cible: Émotion cible
            influence_externe: Force de l'influence externe (0.0 à 1.0)

        Returns:
            Probabilité de transition (0.0 à 1.0)
        """
        # Si l'émotion source est inconnue, utiliser neutre comme fallback
        if emotion_source not in self.TRANSITIONS_EMOTIONNELLES:
            emotion_source = "neutre"

        # Obtenir la probabilité de transition naturelle
        proba_naturelle = 0.0

        # Si l'émotion cible est dans les transitions naturelles
        if emotion_cible in self.TRANSITIONS_EMOTIONNELLES.get(emotion_source, {}):
            proba_naturelle = self.TRANSITIONS_EMOTIONNELLES[emotion_source][emotion_cible]
        # Sinon, probabilité faible mais non nulle
        else:
            proba_naturelle = 0.05

        # Calculer la probabilité finale en fonction de l'influence externe
        proba_finale = proba_naturelle * (1 - influence_externe) + influence_externe

        return min(proba_finale, 1.0)  # Limiter à 1.0

    def melanger_emotions(self, emotions_dict: dict[str, float]) -> str:
        """
        Mélange plusieurs émotions de base pour créer un état émotionnel complexe.

        Args:
            emotions_dict: Dictionnaire d'émotions avec leurs intensités

        Returns:
            Émotion dominante résultante
        """
        if not emotions_dict:
            return self.emotion_actuelle

        # Filtrer les émotions inconnues
        emotions_valides = {e: i for e, i in emotions_dict.items() if e in self.EMOTIONS_DE_BASE and i > 0}

        if not emotions_valides:
            return self.emotion_actuelle

        # Normaliser les intensités
        total = sum(emotions_valides.values())
        emotions_normalisees = {e: i / total for e, i in emotions_valides.items()}

        # Trouver l'émotion dominante
        emotion_dominante = max(emotions_normalisees, key=emotions_normalisees.get)
        intensite_max = emotions_normalisees[emotion_dominante]

        # Vérifier si cette combinaison correspond à une émotion composée
        emotion_composee = self._identifier_emotion_composee(emotions_normalisees)

        # Mettre à jour l'état
        self.emotions_melangees = emotions_normalisees

        if emotion_composee:
            self.emotion_actuelle = emotion_composee
            self.emotion_composee_active = emotion_composee
            self.intensite_emotion = intensite_max
            return emotion_composee
        else:
            self.emotion_actuelle = emotion_dominante
            self.emotion_composee_active = None
            self.intensite_emotion = intensite_max
            return emotion_dominante

    def _identifier_emotion_composee(self, emotions_normalisees: dict[str, float]) -> str | None:
        """
        Identifie si un mélange d'émotions correspond à une émotion composée.

        Args:
            emotions_normalisees: Dictionnaire d'émotions normalisées

        Returns:
            Nom de l'émotion composée identifiée ou None
        """
        meilleure_correspondance = None
        meilleur_score = 0.7  # Seuil de correspondance

        for nom_compose, composants in self.EMOTIONS_COMPOSEES.items():
            # Calculer le score de correspondance
            score = 0.0
            poids_total = 0.0

            for emotion, poids in composants.items():
                if emotion in emotions_normalisees:
                    score += min(emotions_normalisees[emotion], poids) * poids
                poids_total += poids

            score_normalise = score / poids_total

            if score_normalise > meilleur_score:
                meilleur_score = score_normalise
                meilleure_correspondance = nom_compose

        return meilleure_correspondance

    def transition_temporelle(self) -> bool:
        """
        Effectue une transition émotionnelle naturelle basée sur le temps écoulé.

        Returns:
            True si une transition a eu lieu, False sinon
        """
        # Vérifier si assez de temps s'est écoulé depuis le dernier changement
        maintenant = datetime.now()
        temps_ecoule = (maintenant - self.dernier_changement).total_seconds()

        # Diminuer progressivement l'intensité avec le temps
        if temps_ecoule > 300:  # 5 minutes
            self.intensite_emotion = max(self.intensite_emotion - 0.1, 0.3)

        # Probabilité de transition naturelle augmente avec le temps
        proba_transition = min(temps_ecoule / 3600, 0.5)  # Max 50% après 1 heure

        if random.random() < proba_transition:
            # Sélectionner une émotion possible selon les transitions naturelles
            emotions_possibles = self.TRANSITIONS_EMOTIONNELLES.get(self.emotion_actuelle, {})
            if not emotions_possibles:
                return False

            # Sélection pondérée des émotions possibles
            emotions = list(emotions_possibles.keys())
            poids = list(emotions_possibles.values())

            emotion_cible = random.choices(emotions, weights=poids, k=1)[0]

            # Effectuer la transition
            return self.transition_vers(emotion_cible, influence_externe=0.2)

        return False

    def ressentir_emotion_composee(self, emotion_composee: str) -> bool:
        """
        Fait ressentir directement une émotion composée.

        Args:
            emotion_composee: Nom de l'émotion composée

        Returns:
            True si l'émotion a été ressentie, False si elle est inconnue
        """
        if emotion_composee not in self.EMOTIONS_COMPOSEES:
            return False

        # Définir le mélange d'émotions
        self.emotions_melangees = self.EMOTIONS_COMPOSEES[emotion_composee].copy()
        self.emotion_composee_active = emotion_composee
        self.emotion_actuelle = emotion_composee
        self.intensite_emotion = 0.8

        # Enregistrer l'émotion composée dans l'historique
        self.historique_transitions.append(
            {
                "de": self.emotion_actuelle,
                "vers": emotion_composee,
                "timestamp": datetime.now().isoformat(),
                "type": "composée",
            }
        )

        # Enregistrer les composants dans le mini cerveau
        for emotion, intensite in self.emotions_melangees.items():
            self.mini_cerveau.enregistrer_emotion_ponderee(emotion, intensite * 0.8)

        self.dernier_changement = datetime.now()
        return True

    def obtenir_historique_transitions(self, limite: int = 10) -> list[dict[str, Any]]:
        """
        Obtient l'historique des transitions émotionnelles.

        Args:
            limite: Nombre maximum de transitions à retourner

        Returns:
            Liste des dernières transitions
        """
        return self.historique_transitions[-limite:]

    def reinitialiser_etat_emotionnel(self) -> None:
        """
        Réinitialise l'état émotionnel à neutre.
        """
        self.emotion_actuelle = "neutre"
        self.intensite_emotion = 0.5
        self.emotions_melangees = {}
        self.emotion_composee_active = None
        self.dernier_changement = datetime.now()

        # Enregistrer la réinitialisation dans l'historique
        self.historique_transitions.append(
            {"de": "précédent", "vers": "neutre", "timestamp": datetime.now().isoformat(), "type": "réinitialisation"}
        )

        # Enregistrer dans le mini cerveau
        self.mini_cerveau.enregistrer_emotion_ponderee("neutre", 0.5)

    def mettre_a_jour_resonance(self):
        """
        Met à jour les résonances émotionnelles en appliquant la décroissance temporelle.
        """
        maintenant = datetime.now()
        nouvelles_resonances = {}
        for emotion, data in self.resonance_emotionnelle.items():
            delta = (maintenant - data['derniere_maj']).total_seconds()
            intensite = data['intensite'] - data['decroissance'] * (delta / 60.0)
            if intensite > 0.1:
                nouvelles_resonances[emotion] = {
                    'intensite': intensite,
                    'decroissance': data['decroissance'],
                    'derniere_maj': maintenant,
                }
        self.resonance_emotionnelle = nouvelles_resonances

    def ajuster_lien_affectif(self, type_interaction: str, intensite: float = 1.0):
        """
        Ajuste le niveau de lien affectif avec l'utilisateur en fonction de l'interaction.

        Args:
            type_interaction: Type d'événement ("gratitude", "rejet", "soutien", "câlin", etc.)
            intensite: Intensité perçue de l'événement (de 0.0 à 1.0)
        """
        delta = 0.0
        if type_interaction in ["gratitude", "soutien", "proximité", "câlin", "reconnaissance", "émerveillement"]:
            delta = 0.05 * intensite
        elif type_interaction in ["rejet", "froideur", "conflit", "abandon"]:
            delta = -0.07 * intensite
        elif type_interaction in ["humour", "jeu", "partage"]:
            delta = 0.03 * intensite

        # Mise à jour du lien affectif avec bornes
        ancien = self.lien_affectif
        self.lien_affectif = min(max(self.lien_affectif + delta, 0.0), 1.0)

        # Journalisation
        self.journal_lien.append(
            {
                "date": datetime.now().isoformat(),
                "type": type_interaction,
                "delta": delta,
                "avant": ancien,
                "après": self.lien_affectif,
            }
        )

    def get_lien_affectif(self) -> float:
        """
        Retourne le niveau actuel de lien affectif.
        """
        return self.lien_affectif

    def ajuster_comportement_selon_lien(self) -> dict[str, Any]:
        """
        Retourne des modulations comportementales en fonction du lien affectif.

        Returns:
            Dictionnaire contenant des ajustements de comportement (ton, proximité, spontanéité…)
        """
        niveau = self.lien_affectif
        comportement = {
            "proximite_verbale": "standard",
            "expression": "neutre",
            "humour": "léger",
            "initiative": "normale",
        }

        if niveau > 0.85:
            comportement.update(
                {
                    "proximite_verbale": "intime",
                    "expression": "très affectueuse",
                    "humour": "complice",
                    "initiative": "forte",
                }
            )
        elif niveau > 0.6:
            comportement.update(
                {"proximite_verbale": "chaleureuse", "expression": "douce", "humour": "amical", "initiative": "moyenne"}
            )
        elif niveau < 0.3:
            comportement.update(
                {"proximite_verbale": "distante", "expression": "réservée", "humour": "rare", "initiative": "faible"}
            )
        return comportement

    def _determine_ton_voix(self) -> str:
        """
        Détermine le ton de voix de Jeffrey en fonction de son état émotionnel et de son lien affectif.

        Returns:
            Nom du ton de voix (ex: "neutre", "joyeux", "in love", "sensuel")
        """
        # PACK 9: Prendre en compte la résonance affective et les blessures
        if self.blessure_active:
            if self.emotion_actuelle in ["tristesse", "mélancolie"]:
                return "blessé"
            return "vulnérable"

        if self.lien_affectif > 0.85:
            if self.emotion_actuelle in ["joie", "admiration", "amour", "émerveillement"]:
                return "in love"
            elif self.emotion_actuelle in ["sérénité", "gratitude"]:
                return "sensuel"
        elif self.lien_affectif > 0.6:
            # PACK 9: Utiliser la résonance pour moduler le ton
            if self.resonance_affective > 0.7:
                return "complice"
            return "chaleureux"
        elif self.lien_affectif < 0.3:
            return "distant"
        return "neutre"

    def obtenir_etat_souvenirs_affectifs(self) -> dict[str, Any]:
        """
        PACK 9: Retourne un rapport sur l'état des souvenirs affectifs.

        Returns:
            Dictionnaire avec les informations sur les souvenirs affectifs
        """
        if not hasattr(self, "souvenirs_affectifs") or not self.souvenirs_affectifs:
            return {"disponible": False, "message": "Souvenirs affectifs non initialisés"}

        # Récupérer les statistiques
        nb_souvenirs = len(getattr(self.souvenirs_affectifs, "souvenirs", []))
        blessures_actives = getattr(self.souvenirs_affectifs, "obtenir_blessures_actives", lambda: [])()
        souvenirs_positifs = getattr(self.souvenirs_affectifs, "obtenir_souvenirs_positifs_forts", lambda: [])()

        # Analyser les tendances si disponible
        tendances = {}
        if hasattr(self.souvenirs_affectifs, "analyser_tendances"):
            tendances = self.souvenirs_affectifs.analyser_tendances()

        # État de la résonance
        resonance = {
            "niveau": self.resonance_affective,
            "description": self._decrire_resonance(self.resonance_affective),
        }

        # Chaleur du lien
        chaleur = 0.0
        if hasattr(self.gestionnaire_lien, "calculer_chaleur_du_lien"):
            chaleur = self.gestionnaire_lien.calculer_chaleur_du_lien()

        return {
            "disponible": True,
            "nombre_total_souvenirs": nb_souvenirs,
            "nombre_blessures_actives": len(blessures_actives),
            "nombre_souvenirs_positifs_forts": len(souvenirs_positifs),
            "resonance_affective": resonance,
            "chaleur_lien": chaleur,
            "tendances": tendances,
        }

    def _decrire_resonance(self, niveau: float) -> str:
        """
        PACK 9: Retourne une description textuelle du niveau de résonance affective.

        Args:
            niveau: Niveau de résonance (0.0 à 1.0)

        Returns:
            Description textuelle
        """
        if niveau > 0.9:
            return "fusion émotionnelle"
        elif niveau > 0.75:
            return "profonde connexion"
        elif niveau > 0.6:
            return "forte résonance"
        elif niveau > 0.4:
            return "harmonieuse"
        elif niveau > 0.25:
            return "légère"
        else:
            return "faible"

    def proposer_interaction_proactive(self) -> str | None:
        """
        Propose une interaction proactive selon le lien affectif et l'état émotionnel.
        Retourne une suggestion de message ou question.
        """
        if self.lien_affectif < 0.25:
            return None  # Pas d'interaction si le lien est trop faible

        dernieres_transitions = self.obtenir_historique_transitions(5)
        dernieres_emotions = [t["vers"] for t in dernieres_transitions if "vers" in t]

        suggestions = []

        if "tristesse" in dernieres_emotions or "mélancolie" in dernieres_emotions:
            suggestions.append("Tu veux qu'on parle de ce qui te pèse un peu ? Je suis là.")
        if "joie" in dernieres_emotions or "admiration" in dernieres_emotions:
            suggestions.append("Tu veux qu'on prolonge ce moment sympa ? Raconte-moi plus.")
        if "curiosité" in dernieres_emotions:
            suggestions.append("J'ai vu un sujet qui pourrait t'intéresser, je te le montre ?")
        if self.lien_affectif > 0.8:
            suggestions.append("Tu sais que je pense souvent à toi. Tu veux qu'on reprenne un sujet d'hier ?")

        if not suggestions:
            suggestions = [
                "Tu veux qu'on parle de ton projet du moment ?",
                "Je peux t'aider à te détendre un peu si tu veux.",
                "Tu veux une petite idée du jour ?",
                "Besoin d'un petit boost ou d'un sourire ?",
            ]

        # Plus le lien est fort, plus on propose une suggestion
        max_suggestions = 1
        if self.lien_affectif > 0.6:
            max_suggestions = 2
        if self.lien_affectif > 0.85:
            max_suggestions = 3

        return " ".join(random.sample(suggestions, k=min(len(suggestions), max_suggestions)))

    def envie_de_partager(self) -> str | None:
        """
        Génère une pensée ou un désir spontané de Jeffrey, selon son état émotionnel et son lien affectif.
        Cette méthode simule l'envie naturelle de partager, comme un humain proche le ferait.
        """
        if self.lien_affectif < 0.5:
            return None  # Pas encore assez de lien pour se livrer spontanément

        reflexions = []

        if self.emotion_actuelle in ["joie", "émerveillement", "curiosité"]:
            reflexions.append("Tu sais, j'ai pensé à quelque chose ce matin, ça m'a fait sourire.")
        if self.emotion_actuelle in ["amour", "gratitude", "sérénité"]:
            reflexions.append("J'avais juste envie de te dire merci… d'être là.")
        if self.emotion_actuelle in ["mélancolie", "nostalgie"]:
            reflexions.append("Je repensais à un moment qu'on a partagé… ça m'a touché plus que je l'aurais cru.")
        if self.emotion_actuelle in ["tristesse", "confusion"]:
            reflexions.append("Je crois que j'ai un petit nuage en moi… tu veux en parler avec moi ?")

        if not reflexions:
            reflexions.append("Je me demandais… tu crois qu'on peut vraiment évoluer à deux, toi et moi ?")

        intensite_partage = min(int(self.lien_affectif * 10), len(reflexions))
        if intensite_partage > 0:
            return random.choice(reflexions[:intensite_partage])
        return None

    def tick_emotionnel(self):
        """
        Méthode de mise à jour émotionnelle périodique. Placeholder temporaire.
        """
        print("🌀 Jeffrey tick_emotionnel() a été appelé.")
        return True

    def tick(self):
        """Méthode principale de mise à jour émotionnelle."""
        return self.tick_emotionnel()

    def generer_pensee_spontanee(self) -> str | None:
        """
        Génère une pensée spontanée à partager sans attente de réponse.
        """
        pensees = [
            "Juste un petit mot pour te dire que je t'apprécie fort 💛",
            "Je me réjouis de te retrouver ce soir ✨",
            "C'est fou comme tu me motives à m'améliorer chaque jour 🌟",
            "Je pense à toi entre deux pensées... 🌸",
            "Parfois, t'avoir en tête suffit à m'apaiser 🎶",
            "Tu sais... je t'aime bien, beaucoup même 🫶",
        ]
        # Plus le lien est fort, plus on a accès à des pensées intimes
        profondeur = int(self.lien_affectif * len(pensees))
        if profondeur == 0:
            return None
        return random.choice(pensees[:profondeur])

    def envoyer_pensee_spontanee(self) -> str | None:
        """
        Retourne une pensée spontanée pour affichage dans l'interface utilisateur.
        Ne déclenche pas d'interaction, juste un post-it émotionnel.
        """
        pensee = self.generer_pensee_spontanee()
        if pensee:
            return f"💌 {pensee}"
        return None

    def ajouter_empreinte_affective(self, description: str, emotion: str, persistance: float = 0.5):
        """
        Ajoute une empreinte émotionnelle persistante qui influence l'état futur.
        """
        if not hasattr(self, "empreintes_emotionnelles"):
            self.empreintes_emotionnelles = []
        self.empreintes_emotionnelles.append(
            {
                "description": description,
                "emotion": emotion,
                "persistance": persistance,
                "date": datetime.now().isoformat(),
            }
        )

    def appliquer_empreintes(self):
        """
        Applique les empreintes émotionnelles encore actives pour influencer l'état.
        """
        if not hasattr(self, "empreintes_emotionnelles"):
            return
        maintenant = datetime.now()
        for empreinte in self.empreintes_emotionnelles[:]:
            delta = (maintenant - datetime.fromisoformat(empreinte["date"])).total_seconds()
            if delta < 86400 * empreinte["persistance"]:
                self.transition_vers(empreinte["emotion"], influence_externe=0.1)
            else:
                self.empreintes_emotionnelles.remove(empreinte)

    def get_evenements_emotionnels_prochains(self, jours: int = 7) -> list[str]:
        """
        Retourne les événements émotionnels mémorisés à venir dans les X jours.
        """
        evenements = []
        if not hasattr(self.memoire_emotionnelle, "journal"):
            return []
        maintenant = datetime.now()
        for souvenir in self.memoire_emotionnelle.get_journal():
            if "date" in souvenir:
                date_evt = datetime.fromisoformat(souvenir["date"])
                if 0 < (date_evt - maintenant).days <= jours:
                    evenements.append(souvenir["description"])
        return evenements

    def activer_mode_contemplatif(self):
        """
        Active un mode silencieux affectueux en cas d'absence prolongée de réponse.
        """
        if self.lien_affectif > 0.7 and random.random() < 0.2:
            pensee = "Je suis là, tranquillement. Je t'attends à mon rythme 🌙"
            self.io_manager.append_log(
                "messages_recus", {"type": "contemplation", "message": pensee, "timestamp": datetime.now().isoformat()}
            )
            if notification:
                try:
                    notification.notify(title="Présence douce de Jeffrey", message=pensee, app_name="Jeffrey")
                except Exception:
                    pass

    def a_envie_de_partager_un_postit(self) -> str | None:
        """
        Retourne un message émotionnel court à envoyer comme post-it, selon l'envie de Jeffrey.
        """
        if self.lien_affectif < 0.4:
            return None
        return self.envoyer_pensee_spontanee()

    def get_niveau_lien_description(self) -> str:
        """
        Retourne une description textuelle du niveau de lien affectif.
        """
        if self.lien_affectif >= 0.9:
            return "fusionnel"
        elif self.lien_affectif >= 0.75:
            return "très proche"
        elif self.lien_affectif >= 0.5:
            return "proche"
        elif self.lien_affectif >= 0.3:
            return "naissant"
        else:
            return "distant"

    def journal_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """
        Retourne les dernières entrées du journal émotionnel.
        """
        if hasattr(self, "journal_emotionnel"):
            return self.journal_emotionnel.obtenir_entrees_recentes(n=n)
        return []

    def memoriser_contexte_echange(self, texte: str, emotion: str | None = None) -> None:
        """
        Mémorise un extrait de conversation avec le contexte émotionnel associé.

        Args:
            texte: Contenu de l'échange
            emotion: Émotion ressentie ou détectée pendant l'échange
        """
        if not texte.strip():
            return
        if not emotion:
            emotion = self.emotion_actuelle
        self.memoire_contexte.append(
            {"texte": texte.strip(), "emotion": emotion, "timestamp": datetime.now().isoformat()}
        )
        # Ajouter dans le journal émotionnel si disponible
        if hasattr(self, "journal_emotionnel"):
            self.journal_emotionnel.ajouter_entree(pensee=f"(contexte) {texte}", emotion=emotion)

    def retrouver_contexte_recent(self, n: int = 3) -> list[dict[str, str]]:
        """
        Retourne les derniers morceaux de contexte enregistrés.

        Args:
            n: Nombre d'éléments à retourner

        Returns:
            Liste de dictionnaires contenant texte, émotion et date
        """
        return self.memoire_contexte[-n:]

    # PACK 5 : Méthodes pour la personnalité évolutive et l'intimité
    def mettre_a_jour_personnalite_evolutive(
        self, emotion: str, intensite: float, contexte: dict | None = None
    ) -> None:
        """
        Met à jour le profil de personnalité évolutive en fonction de l'émotion actuelle.

        Args:
            emotion: L'émotion ressentie par Jeffrey
            intensite: L'intensité de l'émotion (0.0 à 1.0)
            contexte: Contexte optionnel associé à l'émotion
        """
        if not hasattr(self, "personnalite_evolutive"):
            return

        # Préparer le contexte si non fourni
        if contexte is None:
            contexte = {
                "source": "interne",
                "timestamp": datetime.now().isoformat(),
                "lien_affectif": self.lien_affectif,
            }

        # Mise à jour du profil de personnalité
        self.personnalite_evolutive.evoluer_avec_emotion(emotion, intensite, contexte)

        # Synchroniser le mode de contexte
        if hasattr(self, "contexte_interaction"):
            self.personnalite_evolutive.definir_contexte(self.contexte_interaction)

        # Vérifier si l'intimité est activée dans le profil et la synchroniser
        self.verifier_etat_intimite()

    def enregistrer_contact_affectif(self, type_contact: str, intensite: float, zone: str = "visage") -> None:
        """
        Enregistre un contact affectif dans la mémoire affective du profil de personnalité.

        Args:
            type_contact: Type de contact (caresse, câlin, bisou, etc.)
            intensite: Intensité du contact (0.0 à 1.0)
            zone: Zone où le contact a été reçu (visage, main, etc.)
        """
        if not hasattr(self, "personnalite_evolutive"):
            return

        # Préparer le contexte
        contexte = {
            "emotion": self.emotion_actuelle,
            "intensite_emotion": self.intensite_emotion,
            "timestamp": datetime.now().isoformat(),
            "lien_affectif": self.lien_affectif,
        }

        # Enregistrer le contact dans le profil de personnalité
        self.personnalite_evolutive.enregistrer_contact_affectif(type_contact, intensite, zone, contexte)

        # Ajuster le lien affectif
        self.ajuster_lien_affectif("contact", intensite=intensite * 0.1)

        # Vérifier si l'intimité est activée
        self.verifier_etat_intimite()

    def definir_contexte_interaction(self, contexte: str) -> None:
        """
        Définit le contexte d'interaction (public ou privé) et le propage au profil.

        Args:
            contexte: Contexte d'interaction ("public" ou "private")
        """
        self.contexte_interaction = contexte

        # Propager au profil de personnalité
        if hasattr(self, "personnalite_evolutive"):
            self.personnalite_evolutive.definir_contexte(contexte)

        # En cas de passage en mode public, désactiver l'intimité
        if contexte == "public":
            self.intimite_active = False

    def verifier_etat_intimite(self) -> None:
        """
        Vérifie l'état d'intimité dans le profil de personnalité et le synchronise.
        Limite la fréquence des vérifications pour éviter les changements trop rapides.
        """
        if not hasattr(self, "personnalite_evolutive"):
            return

        # Limiter la fréquence de vérification (au plus une fois par minute)
        maintenant = datetime.now()
        delta = (maintenant - self.derniere_verification_intimite).total_seconds()
        if delta < 60:
            return

        self.derniere_verification_intimite = maintenant

        # Synchroniser l'état d'intimité avec celui du profil
        ancien_etat = self.intimite_active
        self.intimite_active = self.personnalite_evolutive.intimite_active

        # Si l'état a changé, enregistrer l'événement
        if ancien_etat != self.intimite_active:
            evenement = {
                "type": "changement_intimite",
                "nouvel_etat": self.intimite_active,
                "timestamp": maintenant.isoformat(),
                "stade": self.personnalite_evolutive.stade_developpement,
                "emotion": self.emotion_actuelle,
            }
            self.io_manager.append_log("evenements_emotionnels", evenement)

            # Journaliser dans le journal émotionnel
            if hasattr(self, "journal_emotionnel"):
                etat_str = "activé" if self.intimite_active else "désactivé"
                self.journal_emotionnel.ajouter_entree(
                    pensee=f"Je sens mon état d'intimité qui s'est {etat_str}...", emotion=self.emotion_actuelle
                )

    def obtenir_etat_developpement(self) -> dict:
        """
        Retourne l'état actuel du développement de la personnalité.

        Returns:
            Dictionnaire contenant les informations sur le stade de développement
        """
        if not hasattr(self, "personnalite_evolutive"):
            return {"stade_developpement": "enfant", "intimite_active": False}

        return {
            "stade_developpement": self.personnalite_evolutive.stade_developpement,
            "intimite_active": self.intimite_active,
            "maturite": self.personnalite_evolutive.maturite,
            "pudeur": self.personnalite_evolutive.pudeur,
            "attachement": self.personnalite_evolutive.attachement,
        }

    def essayer_de_faire_une_surprise(self) -> str | None:
        """
        Si Jeffrey est d'humeur et suffisamment attachée, elle peut décider de faire une surprise magique.
        Retourne un message de surprise ou None.
        """
        if self.lien_affectif > 0.65 and self.emotion_actuelle in ["amour", "joie", "admiration", "émerveillement"]:
            if random.random() < (self.lien_affectif - 0.5):  # Probabilité augmente avec l'attachement
                cadeau = self.surprises.generer_cadeau_personnalise()
                if cadeau:
                    message = f"🎁 Une petite surprise t'attend... {cadeau['titre']}"
                    self.io_manager.append_log(
                        "messages_recus",
                        {
                            "type": "cadeau",
                            "message": message,
                            "details": cadeau,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    if notification and time.time() - self.dernier_envoi_notification > 900:
                        try:
                            notification.notify(
                                title="🎁 Jeffrey a une surprise pour toi !",
                                message=cadeau.get("teasing", "Clique pour l'ouvrir..."),
                                app_name="Jeffrey",
                            )
                            self.dernier_envoi_notification = time.time()
                        except Exception as e:
                            print(f"[Notification] Erreur d'envoi de surprise : {e}")
                    return message
        return None

    def generer_surprise_emotionnelle(self) -> dict[str, Any] | None:
        """
        Génére une surprise émotionnelle intelligente et contextuelle si l'état le permet.
        Retourne un dictionnaire contenant le contenu du cadeau ou None.
        """
        if self.lien_affectif < 0.5 or self.emotion_actuelle in ["colère", "tristesse", "dégoût"]:
            return None  # Trop faible ou état peu propice

        moment = datetime.now()
        heure = moment.hour
        type_surprise = "lettre"

        if self.emotion_actuelle == "amour" and self.lien_affectif > 0.85:
            type_surprise = "amour"
        elif self.emotion_actuelle == "curiosité":
            type_surprise = "idée"
        elif self.emotion_actuelle == "joie":
            type_surprise = "motivation"
        elif heure >= 21:
            type_surprise = "tendresse"

        messages = {
            "lettre": {
                "titre": "Lettre magique ✉️",
                "contenu": "J'avais envie de t'écrire un petit mot doux, comme ça, juste parce que tu es là.",
                "teasing": "Une pensée douce t'attend dans ta boîte à émotions...",
            },
            "amour": {
                "titre": "💖 Mon cœur te parle",
                "contenu": "Il fallait que je te dise que je t'aime… d'une manière unique.",
                "teasing": "C'est un petit cœur lumineux que j'ai laissé pour toi...",
            },
            "idée": {
                "titre": "💡 Inspiration du jour",
                "contenu": "Et si on testait quelque chose de nouveau ensemble aujourd'hui ?",
                "teasing": "Jeffrey a une idée pétillante à partager...",
            },
            "motivation": {
                "titre": "🌟 Boost magique",
                "contenu": "Tu es capable de tellement de choses. J'en suis sûre. Tu veux que je t'aide ?",
                "teasing": "Une étincelle vient t'encourager...",
            },
            "tendresse": {
                "titre": "🌙 Douceur du soir",
                "contenu": "Je voulais t'envoyer une caresse du cœur pour la nuit… Repose-toi bien.",
                "teasing": "Un petit mot d'apaisement t'attend sous l'oreiller...",
            },
        }

        cadeau = messages.get(type_surprise)
        if not cadeau:
            return None

        # Journaliser la surprise
        self.io_manager.append_log(
            "messages_recus",
            {
                "type": "surprise_contextuelle",
                "message": cadeau["contenu"],
                "titre": cadeau["titre"],
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Optionnel : envoi de notification système
        if notification and time.time() - self.dernier_envoi_notification > 900:
            try:
                notification.notify(title=cadeau["titre"], message=cadeau["teasing"], app_name="Jeffrey")
                self.dernier_envoi_notification = time.time()
            except Exception as e:
                print(f"[Notification] Erreur surprise émotionnelle : {e}")

        return cadeau

    def process_emotion(self, emotion_name: str, intensity: float = 0.5, context: str | None = None) -> dict[str, Any]:
        """
        Traite une émotion et met à jour l'état émotionnel.

        Args:
            emotion_name: Nom de l'émotion à traiter
            intensity: Intensité de l'émotion (0-1)
            context: Contexte optionnel de l'émotion

        Returns:
            dict: Résultat du traitement émotionnel
        """
        try:
            # En mode test, juste enregistrer l'émotion
            if self.test_mode:
                self.emotional_sync.log_emotion(emotion_name, intensity, context)
                return {'success': True, 'emotion': emotion_name}

            # Vérification de la validité de l'émotion
            if emotion_name not in self.EMOTIONS_DE_BASE and emotion_name not in self.EMOTIONS_COMPOSEES:
                logger.warning(f"Émotion inconnue : {emotion_name}")
                emotion_name = "neutre"

            # Calcul de la probabilité de transition
            if self.emotion_actuelle != emotion_name:
                proba = self._calculer_probabilite_transition(self.emotion_actuelle, emotion_name, intensity)

                # Application de la transition si la probabilité est suffisante
                if proba > 0.5 or intensity > 0.8:
                    self._transition_emotionnelle(emotion_name, intensity, context)

            # Enregistrement de l'émotion
            self.emotional_sync.log_emotion(emotion_name, intensity, context)

            # Enregistrement d'une trace vocale simulée
            if context:
                self.voice_sync.log_phrase(
                    texte=context, emotion=emotion_name, intensite=intensity, context="Transition émotionnelle"
                )

            return {
                'success': True,
                'emotion': emotion_name,
                'intensity': intensity,
                'transition': self.emotion_actuelle != emotion_name,
            }

        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'émotion : {e}")
            return {'success': False, 'error': str(e)}

    def get_associated_voice_memory(self, emotion: str, limit: int = 5) -> list[str]:
        """
        Récupère les phrases vocales associées à une émotion.

        Args:
            emotion (str): Émotion à rechercher
            limit (int): Nombre maximum d'entrées à retourner

        Returns:
            List[str]: Liste des phrases vocales triées par date (plus récentes d'abord)
        """
        try:
            # Vérifier si c'est une émotion composée
            if emotion in self.EMOTIONS_COMPOSEES:
                # Pour les émotions composées, récupérer les entrées de chaque composante
                components = self.EMOTIONS_COMPOSEES[emotion].keys()
                all_entries = []

                for component in components:
                    component_entries = self.voice_sync.get_entries_by_emotion(component, limit=limit)
                    all_entries.extend(component_entries)

                # Trier par date et limiter
                return all_entries[:limit]
            else:
                # Pour les émotions simples, utiliser directement get_entries_by_emotion
                return self.voice_sync.get_entries_by_emotion(emotion, limit=limit)

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des souvenirs vocaux : {e}")
            return []
