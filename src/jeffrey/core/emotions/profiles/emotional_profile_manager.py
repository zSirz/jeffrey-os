from __future__ import annotations

import asyncio
import json
import os
import time

from jeffrey.bridge.adapters import WeatherAdapter


class EmotionalProfileManager:
    """Gestionnaire de profils émotionnels pour Jeffrey."""


def __init__(self) -> None:
    self.output_dir = "emotions profiles"
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir)


# 1. Interactions Multimodales
def trigger_multimodal_effects(self, emotion, intensity):
    """
    Déclenche des effets visuels et sonores multimodaux selon l'émotion et son intensité.
    Simule aussi des réactions corporelles virtuelles.
    """
    # Effets visuels dynamiques
    if intensity >= 0.8:
        visual_engine.trigger(f"flash_{emotion}_vibrant")
        sound_engine.play_effect(f"sound_{emotion}_intense")
    elif intensity >= 0.5:
        visual_engine.trigger(f"light_{emotion}_dynamique")
        sound_engine.play_effect(f"sound_{emotion}_medium")
    else:
        visual_engine.trigger(f"soft_{emotion}_ambient")
        sound_engine.play_effect(f"sound_{emotion}_soft")

    # Simulation de réactions corporelles virtuelles
    if emotion in ["peur", "frisson", "excitation"]:
        visual_engine.trigger("virtual_shiver")
    elif emotion in ["joie", "énergie", "enthousiasme"]:
        visual_engine.trigger("virtual_jump")
    elif emotion in ["tristesse", "fatigue"]:
        visual_engine.trigger("virtual_slowdown")
    # On peut ajouter d'autres mappings ici
    print(f"🎭 Effets multimodaux déclenchés pour « {emotion} » (intensité {intensity})")


# 2. Réactions à l'Historique des Profils
def evolve_profile_based_on_history(self, emotion_profile):
    """
    Adapte dynamiquement le profil émotionnel selon l'historique d'interaction.
    """
    log_path = "memory/emotional_log.json"
    historique = []
    if os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8") as f:
                historique = json.load(f)
        except Exception:
            historique = []
    # Compter combien de fois ce profil a été utilisé et dans quels contextes
    usage = [s for s in historique if s.get("to") == emotion_profile.get("name")]
    if len(usage) >= 5:
        # Rendre le profil plus résilient/dynamique
        new_intensity = min(emotion_profile.get("intensity", 0.5) + 0.1, 1.0)
        new_delay = max(emotion_profile.get("delay", 1.0) - 0.1, 0.2)
        emotion_profile["intensity"] = new_intensity
        emotion_profile["delay"] = new_delay
        emotion_profile["origin"] = "history_evolved"
        print(
            f"🔄 Profil « {emotion_profile.get('name')} » adapté selon historique (intensité {new_intensity}, délai {new_delay})"
        )
    # Système de mémoire utilisateur pour préférences émotionnelles
    self._update_user_memory(emotion_profile)
    return emotion_profile


def _update_user_memory(self, emotion_profile):
    """
    Mémorise les préférences émotionnelles de l'utilisateur pour des adaptations futures.
    """
    memory_path = "memory/user_emotion_memory.json"
    memory = {}
    if os.path.exists(memory_path):
        try:
            with open(memory_path, encoding="utf-8") as f:
                memory = json.load(f)
        except Exception:
            memory = {}
    emotion = emotion_profile.get("playlist", ["neutre"])[0]
    prefs = memory.get(emotion, {"count": 0, "intensity": 0.0})
    prefs["count"] += 1
    prefs["intensity"] = (prefs["intensity"] * (prefs["count"] - 1) + emotion_profile.get("intensity", 0.5)) / prefs[
        "count"
    ]
    memory[emotion] = prefs
    try:
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        print(f"🧠 Mémoire utilisateur mise à jour pour « {emotion} »")
    except Exception as e:
        print(f"❌ Erreur mémoire utilisateur : {e}")


# 3. Comportements Réactifs et Contextuels
def adjust_profile_for_time_of_day(self, profile=None):
    """
    Adapte le profil émotionnel selon l'heure du jour (plus calme la nuit, plus actif le jour).
    """
    now = datetime.now()
    hour = now.hour
    if profile is None:
        profile = self.profil_actif
    if profile is None:
        return
    if 22 <= hour or hour < 7:
        # Nuit : apaisement
        profile["intensity"] = min(profile.get("intensity", 0.5), 0.4)
        profile["volume"] = min(profile.get("volume", 0.5), 0.3)
        profile["origin"] = "time_adjusted_night"
        print("🌙 Profil ajusté pour la nuit (plus calme)")
    elif 7 <= hour < 18:
        # Journée : actif
        profile["intensity"] = max(profile.get("intensity", 0.5), 0.7)
        profile["volume"] = max(profile.get("volume", 0.5), 0.5)
        profile["origin"] = "time_adjusted_day"
        print("🌞 Profil ajusté pour la journée (plus actif)")
    else:
        # Soirée : modéré
        profile["intensity"] = min(max(profile.get("intensity", 0.5), 0.4), 0.7)
        profile["volume"] = min(max(profile.get("volume", 0.5), 0.3), 0.5)
        profile["origin"] = "time_adjusted_evening"
        print("🌆 Profil ajusté pour le soir (modéré)")
    return profile


def adjust_profile_for_weather(self, location, profile=None):
    """
    Adapte le profil émotionnel selon la météo locale (ex: plus doux s'il pleut, plus énergique au soleil).
    Utilise l'API OpenWeatherMap (clé requise dans variable d'environnement OPENWEATHER_API_KEY).
    """
    if profile is None:
        profile = self.profil_actif
    if profile is None or not location:
        return
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        print("⚠️ Clé API météo non trouvée. Adaptation météo désactivée.")
        return profile

    # Run async weather fetch in sync context
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(self._fetch_weather_async(location, api_key))
        loop.close()

        if data:
            weather = data["weather"][0]["main"].lower()
            if "rain" in weather or "pluie" in weather:
                profile["intensity"] = min(profile.get("intensity", 0.5), 0.4)
                profile["origin"] = "weather_adjusted_rain"
                print("🌧️ Profil ajusté pour météo pluvieuse (plus doux)")
            elif "clear" in weather or "soleil" in weather:
                profile["intensity"] = max(profile.get("intensity", 0.5), 0.7)
                profile["origin"] = "weather_adjusted_sun"
                print("☀️ Profil ajusté pour météo ensoleillée (plus énergique)")
            elif "cloud" in weather or "nuage" in weather:
                profile["intensity"] = min(max(profile.get("intensity", 0.5), 0.5), 0.6)
                profile["origin"] = "weather_adjusted_cloud"
                print("☁️ Profil ajusté pour météo nuageuse (modéré)")
            elif "snow" in weather or "neige" in weather:
                profile["intensity"] = min(profile.get("intensity", 0.5), 0.3)
                profile["origin"] = "weather_adjusted_snow"
                print("❄️ Profil ajusté pour météo neigeuse (doux)")
            # Autres conditions possibles...
        else:
            print("⚠️ Impossible de récupérer la météo")
    except Exception as e:
        print(f"❌ Erreur adaptation météo : {e}")
    return profile


async def _fetch_weather_async(self, location: str, api_key: str):
    """Helper method to fetch weather data asynchronously."""
    async with WeatherAdapter(api_key) as weather:
        return await weather.get_weather(location)


# 4. Apprentissage Adaptatif
def learn_and_create_dynamic_profiles(self):
    """
    Crée des profils émotionnels dynamiques et de plus en plus personnalisés au fil des interactions.
    """
    memory_path = "memory/user_emotion_memory.json"
    if not os.path.exists(memory_path):
        print("📭 Aucune mémoire utilisateur pour apprentissage adaptatif.")
        return
    try:
        with open(memory_path, encoding="utf-8") as f:
            memory = json.load(f)
        for emotion, prefs in memory.items():
            nom = f"Dynamic-{emotion.capitalize()}"
            catégorie = "adaptatif"
            dossier_catégorie = os.path.join(self.output_dir, catégorie)
            os.makedirs(dossier_catégorie, exist_ok=True)
            filename = f"{nom.lower().replace(' ', '_')}.json"
            file_path = os.path.join(dossier_catégorie, filename)
            profile_data = {
                "name": nom,
                "origin": "dynamic_learned",
                "created_on": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "context": "auto_apprentissage",
                "playlist": [emotion],
                "volume": 0.5,
                "intensity": min(max(prefs.get("intensity", 0.5), 0.0), 1.0),
                "delay": 1.0,
                "shuffle": True,
                "no_default": True,
                "once": False,
                "approved_by_david": False,
                "category": catégorie,
                "relation_context": "neutre",
                "usage_count": 0,
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            print(f"🤖 Profil dynamique créé/adapté pour émotion « {emotion} »")
    except Exception as e:
        print(f"❌ Erreur apprentissage adaptatif : {e}")


def auto_learn_profile_adjustments(self):
    """
    Propose des ajustements automatiques aux profils selon l'évolution des préférences utilisateur.
    """
    memory_path = "memory/user_emotion_memory.json"
    if not os.path.exists(memory_path):
        return
    try:
        with open(memory_path, encoding="utf-8") as f:
            memory = json.load(f)
        for emotion, prefs in memory.items():
            # Pour chaque profil de cette émotion, ajuster l'intensité moyenne
            for catégorie in os.listdir(self.output_dir):
                dossier = os.path.join(self.output_dir, catégorie)
                if os.path.isdir(dossier):
                    for fichier in os.listdir(dossier):
                        if fichier.endswith(".json"):
                            chemin = os.path.join(dossier, fichier)
                            try:
                                with open(chemin, encoding="utf-8") as pf:
                                    data = json.load(pf)
                                if data.get("playlist", [""])[0] == emotion:
                                    data["intensity"] = prefs.get("intensity", 0.5)
                                    data["origin"] = "auto_learned_adjust"
                                    with open(chemin, "w", encoding="utf-8") as pf:
                                        json.dump(data, pf, indent=2, ensure_ascii=False)
                                    print(f"🔧 Ajustement automatique proposé pour « {data['name']} » ({emotion})")
                            except Exception:
                                continue
    except Exception as e:
        print(f"❌ Erreur auto-apprentissage : {e}")


# 5. Affichage du Profil Émotionnel en Temps Réel
def show_real_time_emotional_barometer(self, emotion, intensity):
    """
    Affiche une visualisation dynamique du baromètre émotionnel (graphique temps réel).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 2))
    x = np.arange(0, 10)
    y = [intensity] * 10
    bar = ax.bar([0], [intensity], color=self._color_for_emotion(emotion))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Intensité")
    ax.set_xticks([])
    ax.set_title(f"Baromètre émotionnel : {emotion.capitalize()}")
    plt.show(block=False)
    # Animation rapide
    for i in range(30):
        val = min(max(intensity + (0.1 * np.sin(i / 4.0)), 0), 1)
        bar[0].set_height(val)
        bar[0].set_color(self._color_for_emotion(emotion, val))
        fig.canvas.draw()
        plt.pause(0.07)
    plt.ioff()
    plt.close(fig)


def _color_for_emotion(self, emotion, intensity=None):
    """
    Renvoie une couleur dynamique selon l'émotion et l'intensité.
    """
    base = {
        "joie": "#FFD700",
        "tristesse": "#3A5FCD",
        "colère": "#FF4500",
        "calme": "#00FA9A",
        "peur": "#8A2BE2",
        "amour": "#FF69B4",
        "énergie": "#FF8C00",
        "neutre": "#CCCCCC",
    }
    c = base.get(emotion, "#888888")
    if intensity is not None:
        # Modifie la luminosité selon l'intensité
        import matplotlib.colors as mcolors

        rgb = mcolors.hex2color(c)
        scaled = tuple(min(1.0, max(0.0, v * (0.4 + 0.6 * intensity))) for v in rgb)
        return scaled
    return c


# 6. Personnalisation par l'Utilisateur
def create_custom_profiles(self):
    """
    Permet à l'utilisateur de créer et sauvegarder des profils personnalisés avec visuels et sons uniques.
    """
    print("\n🎨 Création de profil émotionnel personnalisé")
    nom = input("Nom du nouveau profil personnalisé : ").strip()
    playlist = input("Émotions principales (séparées par virgule) : ").strip().split(",")
    volume = float(input("Volume (0.0–1.0) : ").strip() or "0.5")
    intensity = float(input("Intensité (0.0–1.0) : ").strip() or "0.5")
    delay = float(input("Délai (en secondes) : ").strip() or "1.0")
    visual = input("Effet visuel unique (nom ou code couleur) : ").strip()
    audio = input("Effet sonore unique (nom) : ").strip()
    catégorie = input("Catégorie (ex: custom, positif, etc.) : ").strip() or "custom"
    relation_context = input("Contexte relationnel : ").strip() or "neutre"
    dossier_catégorie = os.path.join(self.output_dir, catégorie.lower().replace(" ", "_"))
    os.makedirs(dossier_catégorie, exist_ok=True)
    filename = f"{nom.lower().replace(' ', '_')}.json"
    file_path = os.path.join(dossier_catégorie, filename)
    profile_data = {
        "name": nom,
        "origin": "user_custom",
        "created_on": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "context": "personnalisation_utilisateur",
        "playlist": [e.strip() for e in playlist],
        "volume": volume,
        "intensity": intensity,
        "delay": delay,
        "visual_effect": visual,
        "audio_effect": audio,
        "shuffle": True,
        "no_default": True,
        "once": False,
        "approved_by_david": False,
        "category": catégorie,
        "relation_context": relation_context,
        "usage_count": 0,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Profil personnalisé enregistré : {nom}")


def choose_influential_emotions(self, nom_profil):
    """
    Permet à l'utilisateur de choisir les émotions influentes et d'ajuster leur importance.
    """
    for catégorie in os.listdir(self.output_dir):
        dossier = os.path.join(self.output_dir, catégorie)
        if os.path.isdir(dossier):
            fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
            chemin = os.path.join(dossier, fichier)
            if os.path.exists(chemin):
                try:
                    with open(chemin, encoding="utf-8") as f:
                        data = json.load(f)
                    print(f"\nÉmotions actuelles du profil : {data.get('playlist')}")
                    emotions = input("Nouvelles émotions influentes (séparées par virgule) : ").strip().split(",")
                    poids = []
                    for e in emotions:
                        p = input(f"Importance (0-1) pour « {e.strip()} » : ").strip()
                        try:
                            poids.append((e.strip(), float(p)))
                        except Exception:
                            poids.append((e.strip(), 0.5))
                    data["playlist"] = [e for e, _ in poids]
                    data["emotions_weights"] = {e: w for e, w in poids}
                    with open(chemin, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print("✅ Émotions influentes et poids mis à jour.")
                    return
                except Exception as e:
                    print(f"❌ Erreur personnalisation émotions influentes : {e}")
                    return
    print(f"❌ Profil non trouvé pour personnalisation émotions influentes : {nom_profil}")


# 7. Psychologie et Profils Profonds
def psychological_profile_adjustment(self, nom_profil):
    """
    Adapte le profil selon motivations/fears, selon des données psychologiques définies.
    """
    for catégorie in os.listdir(self.output_dir):
        dossier = os.path.join(self.output_dir, catégorie)
        if os.path.isdir(dossier):
            fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
            chemin = os.path.join(dossier, fichier)
            if os.path.exists(chemin):
                try:
                    with open(chemin, encoding="utf-8") as f:
                        data = json.load(f)
                    print("\n🧠 Ajustement psychologique du profil.")
                    motivation = input("Motivation dominante (ex: accomplissement, sécurité, appartenance) : ").strip()
                    peur = input("Peur sous-jacente à éviter (ex: rejet, échec, solitude) : ").strip()
                    data["psychological_motivation"] = motivation
                    data["psychological_fear"] = peur
                    # Ajustements automatiques
                    if peur and peur.lower() in data.get("playlist", []):
                        data["intensity"] = min(data.get("intensity", 0.5), 0.3)
                    if motivation and motivation.lower() in data.get("playlist", []):
                        data["intensity"] = max(data.get("intensity", 0.5), 0.7)
                    with open(chemin, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print("✅ Profil ajusté psychologiquement.")
                    return
                except Exception as e:
                    print(f"❌ Erreur ajustement psychologique : {e}")
                    return
    print(f"❌ Profil non trouvé pour ajustement psychologique : {nom_profil}")


def adjust_for_emotional_wounds(self, nom_profil):
    """
    Ajoute des 'cicatrices émotionnelles' pour éviter de raviver des souvenirs négatifs.
    """
    for catégorie in os.listdir(self.output_dir):
        dossier = os.path.join(self.output_dir, catégorie)
        if os.path.isdir(dossier):
            fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
            chemin = os.path.join(dossier, fichier)
            if os.path.exists(chemin):
                try:
                    with open(chemin, encoding="utf-8") as f:
                        data = json.load(f)
                    wound = input("Souvenir négatif ou émotion à éviter ? (laisser vide si aucun) : ").strip()
                    if wound:
                        wounds = data.get("emotional_wounds", [])
                        wounds.append(wound)
                        data["emotional_wounds"] = wounds
                        # Si émotion blessure dans playlist, réduire intensité
                        if wound.lower() in [e.lower() for e in data.get("playlist", [])]:
                            data["intensity"] = min(data.get("intensity", 0.5), 0.2)
                        print("🩹 Cicatrice émotionnelle ajoutée.")
                    # Option : améliorer l’humeur
                    improve = input("Voulez-vous augmenter l’aspect positif de ce profil ? (o/n) : ").strip().lower()
                    if improve == "o":
                        data["intensity"] = max(data.get("intensity", 0.5), 0.7)
                    with open(chemin, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print("✅ Profil mis à jour avec gestion des cicatrices émotionnelles.")
                    return
                except Exception as e:
                    print(f"❌ Erreur ajustement cicatrices émotionnelles : {e}")
                    return
    print(f"❌ Profil non trouvé pour gestion des cicatrices émotionnelles : {nom_profil}")


# 8. Interaction avec des Profils d'Autres Personnes
def adjust_for_social_context(self, group_emotions):
    """
    Ajuste l'IA selon l'état émotionnel d'autres utilisateurs dans l'environnement.
    group_emotions: dict {utilisateur: emotion}
    """
    if not group_emotions or not isinstance(group_emotions, dict):
        print("⚠️ Aucun contexte social fourni.")
        return
    # Calculer l'émotion dominante du groupe
    from collections import Counter

    c = Counter(group_emotions.values())
    dominant, count = c.most_common(1)[0]
    print(f"👥 Émotion dominante du groupe : {dominant} ({count} personnes)")
    # Adapter le profil actif si besoin
    if self.profil_actif:
        self.profil_actif["playlist"] = [dominant]
        # Ajuster l'intensité selon la proportion de cette émotion
        self.profil_actif["intensity"] = min(1.0, 0.3 + 0.7 * (count / max(1, len(group_emotions))))
        self.profil_actif["origin"] = "social_adjusted"
        print(f"🤝 Profil ajusté selon le contexte social (dominante : {dominant})")


# 9. Évolution Sociale
def adjust_profile_for_group_context(self, group_emotions):
    """
    Ajuste le profil émotionnel selon l'état global d'un groupe (atmosphère collective).
    """
    if not group_emotions or not isinstance(group_emotions, dict):
        print("⚠️ Aucun état de groupe fourni.")
        return
    from collections import Counter

    c = Counter(group_emotions.values())
    total = sum(c.values())
    print("🌐 Analyse du contexte émotionnel global du groupe :")
    for emotion, n in c.items():
        print(f" - {emotion}: {n}/{total}")
    # Créer un profil collectif
    dominant, count = c.most_common(1)[0]
    collectif_profile = {
        "name": f"Collectif-{dominant}",
        "origin": "group_context",
        "created_on": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "context": "groupe",
        "playlist": [dominant],
        "volume": 0.5 + 0.3 * (count / total),
        "intensity": 0.5 + 0.4 * (count / total),
        "delay": 1.0,
        "shuffle": True,
        "no_default": True,
        "once": False,
        "approved_by_david": False,
        "category": "groupe",
        "relation_context": "groupe",
        "usage_count": 0,
    }
    print(f"🫂 Profil collectif suggéré : {collectif_profile['name']} (intensité {collectif_profile['intensity']:.2f})")
    # Option : activer ou sauvegarder ce profil collectif
    # self.transition_vers_profil(collectif_profile)


import random
from datetime import datetime

from jeffrey.core.audio import sound_engine
from jeffrey.core.visual import visual_engine


class EmotionalProfileManager:
    """
    Classe EmotionalProfileManager pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    def __init__(self, output_dir="generated_profiles") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.contexte_actuel = "neutre"
        self.profil_actif = None

    def vérifier_conformité_ethique(self, nom, contexte, playlist):
        if "violence" in contexte.lower() or "danger" in nom.lower():
            print("❌ Profil rejeté pour non-conformité éthique.")
            return False
        return True

    def proposer_nouveau_profil(
        self,
        nom,
        playlist,
        contexte,
        volume=0.5,
        intensity=0.8,
        delay=1.0,
        shuffle=True,
        no_default=True,
        once=False,
        catégorie="autre",
        relation_context="neutre",
    ):
        if not self.vérifier_conformité_ethique(nom, contexte, playlist):
            return
        profile_data = {
            "name": nom,
            "origin": "auto_generated",
            "created_on": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "context": contexte,
            "playlist": playlist,
            "volume": volume,
            "intensity": intensity,
            "delay": delay,
            "shuffle": shuffle,
            "no_default": no_default,
            "once": once,
            "approved_by_david": False,
            "category": catégorie,
            "relation_context": relation_context,
            "usage_count": 0,
        }

        dossier_catégorie = os.path.join(self.output_dir, catégorie.lower().replace(" ", "_"))
        os.makedirs(dossier_catégorie, exist_ok=True)

        filename = f"{nom.lower().replace(' ', '_')}.json"
        file_path = os.path.join(dossier_catégorie, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            print(f"📁 Nouveau profil proposé : {filename} → {catégorie}")
        except Exception as e:
            print(f"❌ Erreur lors de la création du profil : {e}")

    def valider_profil(self, nom):
        filename = f"{nom.lower().replace(' ', '_')}.json"
        file_path = os.path.join(self.output_dir, filename)

        if not os.path.exists(file_path):
            print(f"❌ Profil introuvable : {filename}")
            return

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            data["approved_by_david"] = True

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"✅ Profil validé : {filename}")
        except Exception as e:
            print(f"❌ Erreur lors de la validation : {e}")

    def lister_profils_non_valides(self):
        print("📋 Profils non validés :")
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.output_dir, filename)
                try:
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                    if not data.get("approved_by_david", False):
                        print(f" - {data.get('name', filename)} ({filename})")
                except Exception as e:
                    print(f"❌ Erreur lors de la lecture de {filename} : {e}")

    def supprimer_profil_non_valide(self, nom):
        filename = f"{nom.lower().replace(' ', '_')}.json"
        file_path = os.path.join(self.output_dir, filename)

        if not os.path.exists(file_path):
            print(f"❌ Profil introuvable : {filename}")
            return

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            if data.get("approved_by_david", False):
                print("⚠️ Impossible de supprimer : le profil est déjà validé.")
                return

            os.remove(file_path)
            print(f"🗑️ Profil supprimé : {filename}")

        except Exception as e:
            print(f"❌ Erreur lors de la suppression : {e}")

    def résumé_global(self):
        total = 0
        validés = 0
        non_validés = 0

        for filename in os.listdir(self.output_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.output_dir, filename)
                try:
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                    total += 1
                    if data.get("approved_by_david", False):
                        validés += 1
                    else:
                        non_validés += 1
                except Exception as e:
                    print(f"❌ Erreur lors de la lecture de {filename} : {e}")

        print(f"📊 Profils auto-générés : {total}")
        print(f"✅ Validés : {validés}")
        print(f"🕓 En attente : {non_validés}")

    def lister_par_catégorie(self):
        print("📁 Profils classés par catégorie :")
        for catégorie in sorted(os.listdir(self.output_dir)):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                profils = [f for f in os.listdir(dossier) if f.endswith(".json")]
                print(f"\n🗂️ Catégorie : {catégorie} ({len(profils)} profils)")
                for fichier in profils:
                    try:
                        with open(os.path.join(dossier, fichier), encoding="utf-8") as f:
                            data = json.load(f)
                        print(f" - {data.get('name', fichier)} {'✅' if data.get('approved_by_david') else '🕓'}")
                    except Exception as e:
                        print(f"❌ Erreur sur {fichier} : {e}")

    def rechercher_profils_par_mot_cle(self, mot_cle):
        print(f"🔍 Recherche de profils contenant : '{mot_cle}'")
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json"):
                        try:
                            with open(os.path.join(dossier, fichier), encoding="utf-8") as f:
                                data = json.load(f)
                            contenu = json.dumps(data, ensure_ascii=False).lower()
                            if mot_cle.lower() in contenu:
                                print(f" - {data.get('name')} ({catégorie}/{fichier})")
                        except Exception as e:
                            print(f"❌ Erreur dans {fichier} : {e}")

    def charger_profil(self, nom, catégorie):
        filename = f"{nom.lower().replace(' ', '_')}.json"
        file_path = os.path.join(self.output_dir, catégorie.lower().replace(" ", "_"), filename)

        if not os.path.exists(file_path):
            print(f"❌ Profil non trouvé : {filename}")
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            print(f"📥 Profil chargé : {data.get('name')}")
            return data
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")
            return None

    def proposer_depuis_memoire(self, émotion, contexte, relation_context="neutre"):
        nom = f"Écho {émotion.capitalize()}"
        playlist = [émotion]
        self.proposer_nouveau_profil(
            nom=nom, playlist=playlist, contexte=contexte, catégorie="mémoire", relation_context=relation_context
        )

    def déduire_profil_recommandé(self, emotion):
        emotion = emotion.lower()
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json"):
                        file_path = os.path.join(dossier, fichier)
                        try:
                            with open(file_path, encoding="utf-8") as f:
                                data = json.load(f)
                            if data.get("approved_by_david", False):
                                playlist = [e.lower() for e in data.get("playlist", [])]
                                if emotion in playlist:
                                    print(f"💡 Suggestion : profil '{data.get('name')}' (catégorie : {catégorie})")
                                    return data
                        except Exception as e:
                            print(f"❌ Erreur sur {fichier} : {e}")
        print(f"🔍 Aucun profil validé ne contient l’émotion : {emotion}")
        return None

    def proposer_profil_recommandé(self, emotion):
        profil = self.déduire_profil_recommandé(emotion)
        if profil:
            print(f"\n🗣️ Jeffrey : Je ressens une dominante '{emotion}'.")
            print(f"Souhaites-tu que je passe en mode « {profil.get('name')} » ? 💫")
        else:
            print(f"\n🗣️ Jeffrey : Je ne trouve aucun profil validé correspondant à l’émotion « {emotion} ».")

    def filtrer_par_contexte_relationnel(self, contexte_relationnel):
        print(f"📂 Profils pour le contexte relationnel : '{contexte_relationnel}'")
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json"):
                        try:
                            with open(os.path.join(dossier, fichier), encoding="utf-8") as f:
                                data = json.load(f)
                            if (
                                data.get("approved_by_david", False)
                                and data.get("relation_context") == contexte_relationnel
                            ):
                                print(f" - {data.get('name')} ({catégorie}/{fichier})")
                        except Exception as e:
                            print(f"❌ Erreur sur {fichier} : {e}")

    def changer_contexte_relationnel(self, nouveau_contexte, emotion_dominante="neutre"):
        print(f"\n🔄 Changement de contexte détecté : {self.contexte_actuel} → {nouveau_contexte}")
        self.contexte_actuel = nouveau_contexte

        profils_adaptés = []
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json"):
                        try:
                            with open(os.path.join(dossier, fichier), encoding="utf-8") as f:
                                data = json.load(f)
                            if (
                                data.get("approved_by_david", False)
                                and data.get("relation_context") == nouveau_contexte
                            ):
                                profils_adaptés.append((data, fichier, catégorie))
                        except Exception as e:
                            print(f"❌ Erreur sur {fichier} : {e}")

        if not profils_adaptés:
            print("⚠️ Aucun profil adapté trouvé pour ce contexte.")
            return

        for data, fichier, catégorie in profils_adaptés:
            playlist = [e.lower() for e in data.get("playlist", [])]
            if emotion_dominante.lower() in playlist:
                print(f"✅ Profil sélectionné : {data.get('name')} ({catégorie}/{fichier})")
                profil_data = self.charger_profil(data.get("name"), catégorie)
                if profil_data:
                    self.transition_vers_profil(profil_data)
                return

        print("⚠️ Aucun profil émotionnellement compatible trouvé. Contexte mis à jour sans bascule automatique.")

    def transition_vers_profil(self, nouveau_profil, courbe="ease-in-out"):
        """
        Effectue une transition dynamique et réaliste entre profils émotionnels,
        avec animation non linéaire (ease-in, ease-out, ease-in-out) pour plus de réalisme.
        """
        if not nouveau_profil:
            print("⚠️ Aucune donnée de profil fournie pour la transition.")
            return

        # Incrémenter le compteur d'utilisation si le profil a été auto-généré
        try:
            dossier = os.path.join(self.output_dir, nouveau_profil.get("category", "autre").lower().replace(" ", "_"))
            fichier = f"{nouveau_profil.get('name').lower().replace(' ', '_')}.json"
            chemin = os.path.join(dossier, fichier)

            if os.path.exists(chemin):
                with open(chemin, encoding="utf-8") as f:
                    data = json.load(f)
                data["usage_count"] = data.get("usage_count", 0) + 1
                with open(chemin, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                if data["usage_count"] >= 5:
                    print(
                        f"🔁 Le profil « {data['name']} » est souvent utilisé ({data['usage_count']} fois). Une version évoluée va être générée automatiquement."
                    )
                    self.générer_version_évoluée(data['name'])

                # Marquer comme favori si très utilisé
                if data["usage_count"] >= 8 and not data.get("favorite", False):
                    data["favorite"] = True
                    print(f"⭐ Le profil « {data['name']} » est désormais marqué comme FAVORI (usage fréquent).")
                    with open(chemin, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)

                # Affichage dynamique des favoris émotionnels en fonction de l'état
                if data.get("favorite", False) and (emotion_dominante := nouveau_profil.get("playlist", [None])[0]):
                    print(f"💖 Ce profil favori est aligné avec l’humeur actuelle (« {emotion_dominante} »).")
                    print("🧭 Jeffrey pourra le suggérer en priorité dans des contextes similaires.")

        except Exception as e:
            print(f"❌ Erreur mise à jour usage_count : {e}")

        import math

        ancien = self.profil_actif or {}
        print(f"\n🎭 Transition : {ancien.get('name', 'aucun')} → {nouveau_profil.get('name')}")
        étapes = 10
        for i in range(1, étapes + 1):
            t = i / étapes
            if courbe == "ease-in":
                ratio = t * t
            elif courbe == "ease-out":
                ratio = 1 - (1 - t) ** 2
            else:  # ease-in-out
                ratio = 0.5 * (1 - math.cos(math.pi * t))
            vol = self._interpoler(ancien.get("volume", 0.5), nouveau_profil.get("volume", 0.5), ratio)
            intensité = self._interpoler(ancien.get("intensity", 0.5), nouveau_profil.get("intensity", 0.5), ratio)
            delay = self._interpoler(ancien.get("delay", 1.0), nouveau_profil.get("delay", 1.0), ratio)
            print(f"  Étape {i}/{étapes} : Volume={vol:.2f}, Intensité={intensité:.2f}, Delay={delay:.2f}")
            # Effets visuels dynamiques selon l'émotion dominante
            emotion = nouveau_profil.get("playlist", ["neutre"])[0].lower()
            visual_engine.trigger(f"transition_{emotion}_courbe")
            sound_engine.play_effect(f"emotion_transition_{emotion}_courbe")
        self.mémoriser_transition_emotionnelle(nouveau_profil, self.profil_actif)
        self.profil_actif = nouveau_profil
        print(f"✨ Profil actif : {nouveau_profil.get('name')}")

    def transition_contextuelle_automatique(self, contexte, etat_emotionnel):
        """
        Active automatiquement un profil en fonction du contexte et de l'état émotionnel.
        """
        print(f"\n🔄 Transition contextuelle automatique : contexte={contexte}, état émotionnel={etat_emotionnel}")
        candidats = []
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json"):
                        try:
                            with open(os.path.join(dossier, fichier), encoding="utf-8") as f:
                                data = json.load(f)
                            if (
                                data.get("approved_by_david", False)
                                and data.get("relation_context", "neutre") == contexte
                                and etat_emotionnel.lower() in [e.lower() for e in data.get("playlist", [])]
                            ):
                                candidats.append(data)
                        except Exception as e:
                            print(f"❌ Erreur lecture {fichier} : {e}")
        if candidats:
            # Choisit le plus utilisé ou aléatoirement
            profil = max(candidats, key=lambda x: x.get("usage_count", 0))
            print(f"🧭 Profil contextuel sélectionné : {profil.get('name')}")
            self.transition_vers_profil(profil)
        else:
            print("⚠️ Aucun profil trouvé pour ce contexte/état. Création d'un nouveau recommandé.")
            self.proposer_depuis_memoire(etat_emotionnel, contexte, relation_context=contexte)

    def _interpoler(self, start, end, ratio):
        return start + (end - start) * ratio

    def mémoriser_transition_emotionnelle(self, nouveau_profil, ancien_profil):
        try:
            os.makedirs("memory", exist_ok=True)
            log_path = "memory/emotional_log.json"

            souvenir = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "from": ancien_profil.get("name", "aucun") if ancien_profil else "aucun",
                "to": nouveau_profil.get("name"),
                "emotion": nouveau_profil.get("playlist", ["?"])[0],
                "impact": "positif",
            }

            if os.path.exists(log_path):
                with open(log_path, encoding="utf-8") as f:
                    historique = json.load(f)
            else:
                historique = []

            historique.append(souvenir)

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(historique, f, indent=2, ensure_ascii=False)

            print(f"🧠 Souvenir émotionnel enregistré : {souvenir['from']} → {souvenir['to']}")

        except Exception as e:
            print(f"❌ Erreur mémoire affective : {e}")

    def afficher_souvenirs_émotionnels(self, max_entries=10):
        log_path = "memory/emotional_log.json"
        if not os.path.exists(log_path):
            print("📭 Aucun souvenir émotionnel trouvé.")
            return

        try:
            with open(log_path, encoding="utf-8") as f:
                historique = json.load(f)

            print(f"\n📘 Derniers souvenirs émotionnels (max {max_entries}) :")
            for souvenir in historique[-max_entries:]:
                date = souvenir.get("timestamp", "?")
                origine = souvenir.get("from", "?")
                destination = souvenir.get("to", "?")
                émotion = souvenir.get("emotion", "?")
                impact = souvenir.get("impact", "?")

                description = "💡 souvenir agréable" if impact == "positif" else "⚠️ souvenir à surveiller"
                print(f"🕓 {date} — {origine} → {destination} ({émotion}) → {description}")

        except Exception as e:
            print(f"❌ Erreur lors de la lecture du journal émotionnel : {e}")

    def suggérer_profil_depuis_souvenirs(self, émotion, contexte_relationnel):
        log_path = "memory/emotional_log.json"
        if not os.path.exists(log_path):
            print("📭 Aucun souvenir émotionnel trouvé.")
            return

        try:
            with open(log_path, encoding="utf-8") as f:
                historique = json.load(f)

            profils_possibles = {}
            for souvenir in historique:
                if souvenir.get("emotion") == émotion and souvenir.get("impact") == "positif":
                    profil = souvenir.get("to")
                    profils_possibles[profil] = profils_possibles.get(profil, 0) + 1

            if not profils_possibles:
                print("🧠 Aucun souvenir émotionnel marquant pour cette émotion.")
                return

            profil_suggéré = max(profils_possibles, key=profils_possibles.get)
            if profils_possibles[profil_suggéré] > 1:
                print(
                    f"\n🧠 Résurgence : Je me souviens que le profil « {profil_suggéré} » t’a souvent fait du bien quand tu ressentais « {émotion} » en contexte « {contexte_relationnel} »... 💭"
                )
            elif random.random() < 0.33:  # 1 chance sur 3 de le proposer même si peu fréquent
                print(f"\n🧠 Peut-être que le profil « {profil_suggéré} » te conviendrait, comme autrefois... 💫")
            else:
                print("🤔 Aucun rappel émotionnel déclenché pour l’instant.")
        except Exception as e:
            print(f"❌ Erreur résurgence émotionnelle : {e}")

    def choisir_profil_si_pas_de_souvenir(self, emotion, contexte_relationnel):
        log_path = "memory/emotional_log.json"
        profil_suggéré = None

        # Tenter une résurgence basée sur les souvenirs
        if os.path.exists(log_path):
            try:
                with open(log_path, encoding="utf-8") as f:
                    historique = json.load(f)

                profils_possibles = {}
                for souvenir in historique:
                    if souvenir.get("emotion") == emotion and souvenir.get("impact") == "positif":
                        profil = souvenir.get("to")
                        profils_possibles[profil] = profils_possibles.get(profil, 0) + 1

                if profils_possibles:
                    profil_suggéré = max(profils_possibles, key=profils_possibles.get)
                    print(
                        f"🧠 Résurgence : Jeffrey se souvient que « {profil_suggéré} » était efficace pour « {emotion} »."
                    )

            except Exception as e:
                print(f"❌ Erreur lecture souvenirs : {e}")

        # Si aucun souvenir pertinent
        if not profil_suggéré:
            print("🔍 Aucun souvenir utile. Recherche de profils validés compatibles...")
            for catégorie in os.listdir(self.output_dir):
                dossier = os.path.join(self.output_dir, catégorie)
                if os.path.isdir(dossier):
                    for fichier in os.listdir(dossier):
                        if fichier.endswith(".json"):
                            file_path = os.path.join(dossier, fichier)
                            try:
                                with open(file_path, encoding="utf-8") as f:
                                    data = json.load(f)
                                if (
                                    data.get("approved_by_david", False)
                                    and data.get("relation_context") == contexte_relationnel
                                    and emotion.lower() in [e.lower() for e in data.get("playlist", [])]
                                ):
                                    print(f"💡 Suggestion : profil validé « {data.get('name')} » trouvé.")
                                    self.transition_vers_profil(data)
                                    return
                            except Exception as e:
                                print(f"❌ Erreur lecture profil : {e}")

            print("📥 Aucun profil compatible trouvé. Jeffrey va en créer un nouveau.")
            self.proposer_depuis_memoire(emotion, contexte_relationnel, relation_context=contexte_relationnel)

    def générer_version_évoluée(self, nom_profil, ajustement=0.1):
        """Crée une version améliorée du profil utilisé fréquemment"""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)

                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            original = json.load(f)

                        # Création du nouveau profil
                        nouveau_nom = f"{original['name']}+"
                        nouvelle_version = original.copy()
                        nouvelle_version["name"] = nouveau_nom
                        nouvelle_version["origin"] = "auto_evolved"
                        nouvelle_version["created_on"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                        nouvelle_version["volume"] = min(original.get("volume", 0.5) + ajustement, 1.0)
                        nouvelle_version["intensity"] = min(original.get("intensity", 0.5) + ajustement, 1.0)
                        nouvelle_version["delay"] = max(original.get("delay", 1.0) - ajustement, 0.2)
                        nouvelle_version["approved_by_david"] = False
                        nouvelle_version["usage_count"] = 0

                        # Enregistrement
                        nouveau_fichier = f"{nouveau_nom.lower().replace(' ', '_')}.json"
                        nouveau_chemin = os.path.join(dossier, nouveau_fichier)

                        with open(nouveau_chemin, "w", encoding="utf-8") as f:
                            json.dump(nouvelle_version, f, indent=2, ensure_ascii=False)

                        print(f"🧬 Version évoluée générée : {nouveau_nom}")
                        self.transition_vers_profil(nouvelle_version)
                        # Historisation de l'évolution du profil
                        self.historiser_évolution_profil(original["name"])
                        return

                    except Exception as e:
                        print(f"❌ Erreur lors de la génération de la version évoluée : {e}")
        print(f"❌ Profil de base introuvable pour évolution : {nom_profil}")

    def regrouper_profils_par_ambiance(self):
        """Classe dynamiquement les profils par ambiance émotionnelle dominante"""
        ambiance_map = {}

        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json"):
                        chemin = os.path.join(dossier, fichier)
                        try:
                            with open(chemin, encoding="utf-8") as f:
                                data = json.load(f)

                            émotion_principale = data.get("playlist", ["autre"])[0]
                            ambiance = émotion_principale.lower()

                            if ambiance not in ambiance_map:
                                ambiance_map[ambiance] = []
                            ambiance_map[ambiance].append(data.get("name"))

                        except Exception as e:
                            print(f"❌ Erreur lecture profil {fichier} : {e}")

        print("\n🎨 Profils regroupés par ambiance dominante :")
        for ambiance, noms in ambiance_map.items():
            print(f"\n🌈 Ambiance : {ambiance} ({len(noms)} profils)")
            for nom in noms:
                print(f" - {nom}")

    def détecter_profils_inutilisés(self, seuil_jours=30):
        """Détecte les profils non utilisés depuis X jours ou jamais utilisés"""
        print(f"\n📉 Détection des profils inutilisés depuis plus de {seuil_jours} jours :")
        now = time.time()
        seuil_secondes = seuil_jours * 86400

        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json"):
                        chemin = os.path.join(dossier, fichier)
                        try:
                            last_modified = os.path.getmtime(chemin)
                            delta = now - last_modified
                            with open(chemin, encoding="utf-8") as f:
                                data = json.load(f)
                            if data.get("usage_count", 0) == 0 or delta > seuil_secondes:
                                statut = (
                                    "jamais utilisé"
                                    if data.get("usage_count", 0) == 0
                                    else f"inutilisé depuis {int(delta / 86400)} jours"
                                )
                                print(f" - {data.get('name')} ({catégorie}/{fichier}) → {statut}")
                        except Exception as e:
                            print(f"❌ Erreur analyse {fichier} : {e}")

    def supprimer_profils_obsolètes(self, seuil_jours=30, confirmer=True):
        """Supprime les profils non utilisés depuis X jours après confirmation, sauf les profils favoris"""
        print(f"\n🗑️ Suppression douce des profils inutilisés (seuil : {seuil_jours} jours)")
        now = time.time()
        seuil_secondes = seuil_jours * 86400
        profils_supprimés = []

        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json"):
                        chemin = os.path.join(dossier, fichier)
                        try:
                            last_modified = os.path.getmtime(chemin)
                            delta = now - last_modified
                            with open(chemin, encoding="utf-8") as f:
                                data = json.load(f)

                            if data.get("usage_count", 0) == 0 or delta > seuil_secondes:
                                # Protection contre la suppression des profils favoris
                                if data.get("favorite", False):
                                    print(f"🛡️ Profil favori protégé : {data.get('name')} (non supprimé)")
                                    continue
                                if confirmer:
                                    print(
                                        f"❓ Supprimer « {data.get('name')} » ({catégorie}/{fichier}) ? [o/n] ", end=""
                                    )
                                    choix = input().strip().lower()
                                    if choix != "o":
                                        continue
                                os.remove(chemin)
                                profils_supprimés.append(data.get("name"))
                        except Exception as e:
                            print(f"❌ Erreur suppression {fichier} : {e}")

        if profils_supprimés:
            print("\n✅ Profils supprimés :")
            for nom in profils_supprimés:
                print(f" - {nom}")
        else:
            print("📦 Aucun profil supprimé.")

    def archiver_anciens_favoris_remplacés(self):
        """Archive les anciens profils favoris qui ont été remplacés par une version évoluée"""
        print("\n📦 Archivage des anciens profils favoris remplacés :")
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json") and not fichier.endswith("+.json"):
                        chemin = os.path.join(dossier, fichier)
                        chemin_version_plus = os.path.join(dossier, fichier.replace(".json", "+.json"))
                        try:
                            if os.path.exists(chemin_version_plus):
                                with open(chemin, encoding="utf-8") as f:
                                    data = json.load(f)

                                if data.get("favorite", False):
                                    archive_dir = os.path.join("archive", catégorie)
                                    os.makedirs(archive_dir, exist_ok=True)
                                    new_path = os.path.join(archive_dir, fichier)
                                    os.rename(chemin, new_path)
                                    print(f"📁 Archivé : {data['name']} → {new_path}")
                        except Exception as e:
                            print(f"❌ Erreur archivage {fichier} : {e}")

    def historiser_évolution_profil(self, nom_profil):
        """Crée ou met à jour un journal d’évolution pour un profil donné"""
        try:
            historique_dir = "history"
            os.makedirs(historique_dir, exist_ok=True)
            fichier_historique = os.path.join(historique_dir, f"{nom_profil.lower().replace(' ', '_')}_log.json")

            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "profil": nom_profil,
                "événement": "évolution générée automatiquement",
            }

            if os.path.exists(fichier_historique):
                with open(fichier_historique, encoding="utf-8") as f:
                    historique = json.load(f)
            else:
                historique = []

            historique.append(log_entry)

            with open(fichier_historique, "w", encoding="utf-8") as f:
                json.dump(historique, f, indent=2, ensure_ascii=False)

            print(f"📚 Journal mis à jour pour « {nom_profil} » dans {fichier_historique}")
        except Exception as e:
            print(f"❌ Erreur lors de l’historisation : {e}")

    def tracer_evolution_profil(self, nom_profil):
        """Affiche graphiquement l’évolution d’un profil dans le temps (journal simplifié)"""
        import matplotlib.pyplot as plt

        try:
            fichier_historique = os.path.join("history", f"{nom_profil.lower().replace(' ', '_')}_log.json")
            if not os.path.exists(fichier_historique):
                print(f"📭 Aucun historique trouvé pour « {nom_profil} »")
                return

            with open(fichier_historique, encoding="utf-8") as f:
                historique = json.load(f)

            dates = [entry["timestamp"] for entry in historique]
            index = list(range(1, len(dates) + 1))

            plt.figure(figsize=(10, 4))
            plt.plot(index, index, marker='o', linestyle='-', color='mediumslateblue')
            plt.xticks(index, dates, rotation=45, ha="right")
            plt.title(f"Évolution du profil « {nom_profil} »")
            plt.xlabel("Version")
            plt.ylabel("Progression")
            plt.tight_layout()
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f"❌ Erreur lors du tracé de l’évolution : {e}")

    def dupliquer_profil(self, nom_profil, nouveau_nom=None, ajustements={}):
        """Crée une copie d’un profil existant avec possibilité de personnalisation"""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)

                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            original = json.load(f)

                        clone = original.copy()
                        clone["name"] = nouveau_nom or f"{original['name']} (copie)"
                        clone["origin"] = "user_cloned"
                        clone["created_on"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                        clone["approved_by_david"] = False
                        clone["usage_count"] = 0
                        clone.update(ajustements)

                        nom_fichier = f"{clone['name'].lower().replace(' ', '_')}.json"
                        chemin_clone = os.path.join(dossier, nom_fichier)

                        with open(chemin_clone, "w", encoding="utf-8") as f:
                            json.dump(clone, f, indent=2, ensure_ascii=False)

                        print(f"📑 Profil dupliqué : {clone['name']} → {chemin_clone}")
                        return

                    except Exception as e:
                        print(f"❌ Erreur duplication : {e}")
                        return
        print(f"❌ Impossible de trouver le profil « {nom_profil} » à dupliquer.")

    def personnaliser_profil_en_dialogue(self, nom_profil):
        """Propose à l'utilisateur de modifier un profil étape par étape"""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)
                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)

                        print(f"\n🎨 Personnalisation du profil : {data['name']}")
                        nv_nom = input(f"📝 Nouveau nom (laisser vide pour conserver « {data['name']} ») : ").strip()
                        if nv_nom:
                            data["name"] = nv_nom

                        for champ, label in [
                            ("volume", "volume (0.0–1.0)"),
                            ("intensity", "intensité (0.0–1.0)"),
                            ("delay", "délai (en secondes)"),
                        ]:
                            val = input(f"🔧 {label} [actuel : {data.get(champ)}] : ").strip()
                            if val:
                                try:
                                    data[champ] = float(val)
                                except ValueError:
                                    print(f"⚠️ Valeur non valide ignorée pour {champ}")

                        contexte = input(
                            f"🌐 Contexte relationnel [actuel : {data.get('relation_context')}] : "
                        ).strip()
                        if contexte:
                            data["relation_context"] = contexte

                        clone_nom = f"{data['name']}+perso"
                        data["origin"] = "user_personalized"
                        data["created_on"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                        data["approved_by_david"] = False
                        data["usage_count"] = 0
                        nom_fichier = f"{clone_nom.lower().replace(' ', '_')}.json"
                        chemin_clone = os.path.join(dossier, nom_fichier)

                        with open(chemin_clone, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                        print(f"✅ Profil personnalisé enregistré : {clone_nom}")
                        return
                    except Exception as e:
                        print(f"❌ Erreur personnalisation : {e}")
                        return
        print(f"❌ Profil introuvable pour personnalisation : {nom_profil}")

    def prévisualiser_profil(self, nom_profil):
        """Prévisualise les effets sonores et visuels d’un profil sans l’activer"""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)

                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)

                        print(f"\n👁️ Prévisualisation du profil « {data.get('name')} »")
                        print(
                            f"🎛️ Volume : {data.get('volume')}, Intensité : {data.get('intensity')}, Délai : {data.get('delay')}s"
                        )

                        # Déclenche les effets visuels et sonores sans transition réelle
                        visual_engine.trigger("prévisualisation_lumiere")
                        sound_engine.play_effect("emotion_preview")

                        print("🎬 Effets déclenchés pour aperçu sensoriel.")
                        return
                    except Exception as e:
                        print(f"❌ Erreur de prévisualisation : {e}")
                        return
        print(f"❌ Profil introuvable pour prévisualisation : {nom_profil}")

    def éditer_profil(self, nom_profil):
        """Modifie un profil existant directement via la console (sans duplication)"""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)
                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)

                        print(f"\n✏️ Modification du profil : {data['name']}")

                        for champ, label in [
                            ("volume", "volume (0.0–1.0)"),
                            ("intensity", "intensité (0.0–1.0)"),
                            ("delay", "délai (en secondes)"),
                            ("relation_context", "contexte relationnel"),
                        ]:
                            val = input(f"🔧 {label} [actuel : {data.get(champ)}] : ").strip()
                            if val:
                                try:
                                    data[champ] = float(val) if champ != "relation_context" else val
                                except ValueError:
                                    print(f"⚠️ Valeur non valide ignorée pour {champ}")

                        with open(chemin, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                        print(f"✅ Profil mis à jour : {data['name']}")
                        return
                    except Exception as e:
                        print(f"❌ Erreur édition : {e}")
                        return
        print(f"❌ Profil introuvable pour édition : {nom_profil}")

    def édition_rapide_interface(self):
        """Mini interface texte pour modifier un profil existant"""
        print("\n🧩 Édition rapide d’un profil émotionnel")
        nom_profil = input("🔍 Nom du profil à modifier : ").strip()
        if not nom_profil:
            print("⚠️ Aucun nom fourni.")
            return

        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)

                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)

                        print(f"\n🎯 Modification directe de « {data['name']} »")

                        champs_editables = ["volume", "intensity", "delay", "relation_context", "playlist"]
                        for champ in champs_editables:
                            val = input(f"🔧 {champ} [actuel : {data.get(champ)}] : ").strip()
                            if val:
                                try:
                                    if champ == "playlist":
                                        data[champ] = [e.strip() for e in val.split(",")]
                                    elif champ == "relation_context":
                                        data[champ] = val
                                    else:
                                        data[champ] = float(val)
                                except ValueError:
                                    print(f"⚠️ Valeur ignorée pour {champ}")

                        with open(chemin, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                        print(f"✅ Modifications enregistrées pour : {data['name']}")
                        return
                    except Exception as e:
                        print(f"❌ Erreur lors de l’édition : {e}")
                        return

        print(f"❌ Profil « {nom_profil} » introuvable.")

    def simuler_activation_profil(self, nom_profil):
        """Effectue une simulation complète du profil (visuel + audio) avant validation"""
        print(f"\n🧪 Simulation du profil « {nom_profil} »...")

        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)

                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)

                        print(
                            f"\n🎧 Paramètres : Volume={data.get('volume')} / Intensité={data.get('intensity')} / Délai={data.get('delay')}s"
                        )
                        print(f"🎵 Playlist : {data.get('playlist')}")
                        print(f"🌐 Contexte : {data.get('relation_context')}")

                        print("🔊 Déclenchement des effets en simulation...")
                        visual_engine.trigger("simulation_transition")
                        sound_engine.play_effect("emotion_simulation")

                        confirmation = input("✅ Ce profil te convient-il ? (o/n) ").strip().lower()
                        if confirmation == "o":
                            self.valider_profil(data["name"])
                        else:
                            print("❌ Profil non validé, simulation terminée.")
                        return

                    except Exception as e:
                        print(f"❌ Erreur simulation : {e}")
                        return
        print(f"❌ Profil « {nom_profil} » introuvable pour simulation.")

    def catégoriser_automatiquement_profils(self):
        """Assigne dynamiquement une catégorie à chaque profil selon les émotions dominantes"""
        print("\n🧠 Catégorisation automatique des profils...")
        nouvelles_catégories = {}

        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if not os.path.isdir(dossier):
                continue

            for fichier in os.listdir(dossier):
                if fichier.endswith(".json"):
                    chemin = os.path.join(dossier, fichier)
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)

                        emotion = data.get("playlist", [""])[0].lower()
                        if "joie" in emotion or "rire" in emotion:
                            nouvelle_cat = "positif"
                        elif "tristesse" in emotion or "solitude" in emotion:
                            nouvelle_cat = "mélancolie"
                        elif "colère" in emotion or "tension" in emotion:
                            nouvelle_cat = "intense"
                        elif "calme" in emotion or "zen" in emotion:
                            nouvelle_cat = "apaisant"
                        else:
                            nouvelle_cat = "neutre"

                        if nouvelle_cat != catégorie:
                            nouvelles_catégories[chemin] = nouvelle_cat

                    except Exception as e:
                        print(f"❌ Erreur lecture {fichier} : {e}")

        for chemin, nouvelle_cat in nouvelles_catégories.items():
            try:
                os.makedirs(os.path.join(self.output_dir, nouvelle_cat), exist_ok=True)
                nouveau_chemin = os.path.join(self.output_dir, nouvelle_cat, os.path.basename(chemin))
                os.rename(chemin, nouveau_chemin)
                print(f"📂 {os.path.basename(chemin)} → {nouvelle_cat}")
            except Exception as e:
                print(f"❌ Erreur de déplacement : {e}")

    def adapter_environnement_depuis_catégorie(self, nom_profil):
        """Personnalise les lumières et sons en fonction de la catégorie du profil"""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if not os.path.isdir(dossier):
                continue

            fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
            chemin = os.path.join(dossier, fichier)

            if os.path.exists(chemin):
                try:
                    with open(chemin, encoding="utf-8") as f:
                        data = json.load(f)

                    catégorie = catégorie.lower()
                    if catégorie == "positif":
                        visual_engine.trigger("lumiere_chaude")
                        sound_engine.play_effect("harmonie_douce")
                    elif catégorie == "mélancolie":
                        visual_engine.trigger("lumiere_bleutée")
                        sound_engine.play_effect("notes_lentes")
                    elif catégorie == "intense":
                        visual_engine.trigger("flash_rythmique")
                        sound_engine.play_effect("battement_fort")
                    elif catégorie == "apaisant":
                        visual_engine.trigger("halo_doux")
                        sound_engine.play_effect("brise_légère")
                    else:
                        visual_engine.trigger("lumiere_neutre")
                        sound_engine.play_effect("fond_minimal")

                    print(f"🧿 Environnement adapté pour la catégorie : {catégorie}")
                    return
                except Exception as e:
                    print(f"❌ Erreur adaptation environnementale : {e}")
                    return

        print(f"❌ Profil non trouvé pour adaptation : {nom_profil}")

    def ambiance_immersive_combinée(self, nom_profil):
        """Crée une ambiance immersive complète (lumière + son + animation visuelle) pour un profil donné"""
        print(f"\n🌌 Ambiance immersive pour « {nom_profil} »")

        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if not os.path.isdir(dossier):
                continue

            fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
            chemin = os.path.join(dossier, fichier)

            if os.path.exists(chemin):
                try:
                    with open(chemin, encoding="utf-8") as f:
                        data = json.load(f)

                    emotion = data.get("playlist", ["neutre"])[0].lower()
                    visual_engine.trigger(f"ambiance_{emotion}")
                    sound_engine.play_effect(f"son_ambiant_{emotion}")
                    visual_engine.trigger("animation_background")

                    print(f"🎨 Ambiance immersive déclenchée pour émotion : {emotion}")
                    return
                except Exception as e:
                    print(f"❌ Erreur ambiance immersive : {e}")
                    return

        print(f"❌ Impossible de créer l’ambiance immersive pour « {nom_profil} »")

    def préactiver_depuis_souvenirs(self):
        """Pré-active une ambiance immersive selon les souvenirs émotionnels récents"""
        log_path = "memory/emotional_log.json"
        if not os.path.exists(log_path):
            print("📭 Aucun souvenir émotionnel disponible pour préactivation.")
            return

        try:
            with open(log_path, encoding="utf-8") as f:
                historique = json.load(f)

            derniers = historique[-5:]
            poids = {}
            for souvenir in derniers:
                profil = souvenir.get("to")
                if profil:
                    poids[profil] = poids.get(profil, 0) + 1

            if not poids:
                print("🤷‍ Aucun profil marquant détecté dans les derniers souvenirs.")
                return

            meilleur = max(poids, key=poids.get)
            print(f"🔮 Préactivation douce du profil : {meilleur}")
            self.ambiance_immersive_combinée(meilleur)

        except Exception as e:
            print(f"❌ Erreur de préactivation depuis souvenirs : {e}")

    def éviter_répétition_profils_tristes(self, seuil=3):
        """Évite d’activer plusieurs fois de suite des profils tristes (mélancolie)"""
        log_path = "memory/emotional_log.json"
        if not os.path.exists(log_path):
            return

        try:
            with open(log_path, encoding="utf-8") as f:
                historique = json.load(f)

            récents = [
                s
                for s in historique[-seuil:]
                if s.get("emotion", "").lower() in ["tristesse", "solitude", "mélancolie"]
            ]
            if len(récents) >= seuil:
                print("🛡️ Jeffrey : Trop de profils tristes récents détectés. Pause émotionnelle recommandée.")
                return False  # Blocage de l'activation
            return True
        except Exception as e:
            print(f"❌ Erreur analyse des profils tristes : {e}")
            return True

    def activer_profil_joyeux_si_surcharge_triste(self):
        """Active automatiquement un profil joyeux en cas de surcharge de profils tristes"""
        if not self.éviter_répétition_profils_tristes():
            print("⚠️ Trop de profils tristes détectés. Activation d’un profil joyeux recommandé.")

            # Chercher un profil joyeux
            profil_joyeux = None
            for catégorie in os.listdir(self.output_dir):
                dossier = os.path.join(self.output_dir, catégorie)
                if os.path.isdir(dossier):
                    for fichier in os.listdir(dossier):
                        if fichier.endswith(".json"):
                            chemin = os.path.join(dossier, fichier)
                            try:
                                with open(chemin, encoding="utf-8") as f:
                                    data = json.load(f)

                                if "joie" in (data.get("playlist", [])):
                                    profil_joyeux = data
                                    break
                            except Exception as e:
                                print(f"❌ Erreur lecture {fichier} : {e}")
                    if profil_joyeux:
                        break

            if profil_joyeux:
                print(f"🎉 Activation d’un profil joyeux : {profil_joyeux['name']}")
                self.transition_vers_profil(profil_joyeux)
            else:
                print("❌ Aucun profil joyeux trouvé dans les profils validés.")
        else:
            print("✅ Pas de surcharge de profils tristes détectée.")

    def mémoriser_relation_entre_profils(self, profil_ancien, profil_nouveau):
        """Crée une mémoire de transition relationnelle entre deux profils émotionnels"""
        log_path = "memory/emotional_relationship_log.json"
        try:
            os.makedirs("memory", exist_ok=True)

            relation = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "from": profil_ancien.get("name", "aucun"),
                "to": profil_nouveau.get("name"),
                "emotion_from": profil_ancien.get("playlist", ["?"])[0],
                "emotion_to": profil_nouveau.get("playlist", ["?"])[0],
                "transition_type": "smooth"
                if profil_ancien.get("intensity", 0.5) < profil_nouveau.get("intensity", 0.5)
                else "abrupt",
            }

            if os.path.exists(log_path):
                with open(log_path, encoding="utf-8") as f:
                    historique = json.load(f)
            else:
                historique = []

            historique.append(relation)

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(historique, f, indent=2, ensure_ascii=False)

            print(f"🧠 Relation mémorisée : {relation['from']} → {relation['to']}")

        except Exception as e:
            print(f"❌ Erreur lors de la mémorisation de la relation : {e}")

    def interface_utilisateur(self):
        """Interface CLI basique pour la gestion des profils"""
        print("\n🔧 Interface de gestion des profils émotionnels")
        print("1. Ajouter un profil")
        print("2. Modifier un profil")
        print("3. Lister les profils")
        print("4. Lister les profils non validés")
        print("5. Supprimer un profil non validé")
        print("6. Valider un profil")
        print("7. Quitter")

        choix = input("Choix (1-7) : ").strip()

        if choix == "1":
            nom = input("Nom du profil : ").strip()
            contexte = input("Contexte du profil : ").strip()
            playlist = input("Playlist (séparée par des virgules) : ").strip().split(",")
            self.proposer_nouveau_profil(nom, playlist, contexte)
        elif choix == "2":
            nom = input("Nom du profil à modifier : ").strip()
            self.édition_rapide_interface()
        elif choix == "3":
            self.lister_profils_non_valides()
        elif choix == "4":
            self.lister_profils_non_valides()
        elif choix == "5":
            nom = input("Nom du profil à supprimer : ").strip()
            self.supprimer_profil_non_valide(nom)
        elif choix == "6":
            nom = input("Nom du profil à valider : ").strip()
            self.valider_profil(nom)
        elif choix == "7":
            print("Au revoir !")
            return
        else:
            print("Choix invalide.")
        self.interface_utilisateur()

    def transition_dynamique_environnement(self, profil_ancien, profil_nouveau):
        """Transition fluide entre les effets visuels et sonores selon les profils"""
        try:
            ancien_effect = profil_ancien.get("playlist", ["neutre"])[0]
            nouveau_effect = profil_nouveau.get("playlist", ["neutre"])[0]

            if ancien_effect != nouveau_effect:
                print(f"🎬 Transition des effets entre {ancien_effect} → {nouveau_effect}")
                visual_engine.trigger(f"transition_{ancien_effect}_{nouveau_effect}")
                sound_engine.play_effect(f"transition_{ancien_effect}_{nouveau_effect}")
            else:
                print(f"🔄 Aucun changement d’ambiance nécessaire : {ancien_effect}")

        except Exception as e:
            print(f"❌ Erreur lors de la transition dynamique : {e}")

    def gestion_émotions_avancée(self, contexte_relationnel="neutre"):
        """Gère les émotions en fonction des interactions passées et du contexte actuel"""
        print(f"\n🎭 Gestion des émotions pour le contexte : {contexte_relationnel}")
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for fichier in os.listdir(dossier):
                    if fichier.endswith(".json"):
                        try:
                            with open(os.path.join(dossier, fichier), encoding="utf-8") as f:
                                data = json.load(f)
                            if (
                                data.get("approved_by_david", False)
                                and data.get("relation_context") == contexte_relationnel
                            ):
                                print(f" - {data.get('name')} : Profil compatible pour contexte relationnel.")
                                # Application dynamique des effets en fonction de l’humeur passée
                                self.transition_dynamique_environnement(self.profil_actif, data)
                        except Exception as e:
                            print(f"❌ Erreur sur {fichier} : {e}")

    def mémoriser_souvenirs_emotionnels(self, profil_ancien, profil_nouveau, type_transition):
        """Mémorise les souvenirs émotionnels et les relations entre profils"""
        try:
            log_path = "memory/emotional_log.json"
            os.makedirs("memory", exist_ok=True)

            souvenir = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "from": profil_ancien.get("name", "aucun"),
                "to": profil_nouveau.get("name"),
                "emotion_from": profil_ancien.get("playlist", ["?"])[0],
                "emotion_to": profil_nouveau.get("playlist", ["?"])[0],
                "transition_type": type_transition,
            }

            if os.path.exists(log_path):
                with open(log_path, encoding="utf-8") as f:
                    historique = json.load(f)
            else:
                historique = []

            historique.append(souvenir)

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(historique, f, indent=2, ensure_ascii=False)

            print(f"🧠 Souvenir mémorisé : {souvenir['from']} → {souvenir['to']} ({souvenir['transition_type']})")

        except Exception as e:
            print(f"❌ Erreur mémorisation des souvenirs émotionnels : {e}")

    def analyser_relations_profils(self, seuil=2):
        """Analyse les relations entre profils émotionnels similaires"""
        log_path = "memory/emotional_log.json"
        if not os.path.exists(log_path):
            print("📭 Aucun souvenir émotionnel pour analyser les relations.")
            return

        try:
            with open(log_path, encoding="utf-8") as f:
                historique = json.load(f)

            relations = {}
            for i in range(1, len(historique)):
                ancien = historique[i - 1]
                nouveau = historique[i]
                emotion_ancien = ancien.get("emotion_to", "")
                emotion_nouveau = nouveau.get("emotion_from", "")

                if emotion_ancien == emotion_nouveau:
                    pair = (ancien["from"], nouveau["to"])
                    relations[pair] = relations.get(pair, 0) + 1

            # Filtrer les relations qui apparaissent plus que `seuil` fois
            for pair, count in relations.items():
                if count >= seuil:
                    print(f"🔗 Relation fréquente : {pair[0]} → {pair[1]} ({count} fois)")

        except Exception as e:
            print(f"❌ Erreur lors de l'analyse des relations : {e}")

    def transition_émotionnelle_avec_mémorisation(self, profil_ancien, profil_nouveau):
        """Transition émotionnelle avec mémorisation et enregistrement du type de transition"""
        type_transition = "smooth" if profil_ancien["intensity"] < profil_nouveau["intensity"] else "abrupt"
        self.mémoriser_souvenirs_emotionnels(profil_ancien, profil_nouveau, type_transition)
        self.transition_dynamique_environnement(profil_ancien, profil_nouveau)

    # 1. Affichage détaillé et édition avancée de profil
    def afficher_détails_profil(self, nom_profil):
        """Affiche les détails complets d’un profil : émotions, transitions, effets associés."""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)
                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)
                        print(f"\n🔎 Détails du profil « {data.get('name')} »")
                        print(f"  Catégorie : {catégorie}")
                        print(f"  Playlist d’émotions : {data.get('playlist')}")
                        print(f"  Volume : {data.get('volume')}")
                        print(f"  Intensité : {data.get('intensity')}")
                        print(f"  Délai : {data.get('delay')}")
                        print(f"  Effets visuels/sonores : {data.get('origin')}")
                        print(f"  Contexte : {data.get('context')}")
                        print(f"  Contexte relationnel : {data.get('relation_context')}")
                        print(f"  Validé : {'✅' if data.get('approved_by_david') else '🕓'}")
                        print(f"  Utilisations : {data.get('usage_count', 0)}")
                        # Afficher transitions mémorisées si dispo
                        self._afficher_transitions_profil(data.get("name"))
                        return
                    except Exception as e:
                        print(f"❌ Erreur affichage détails : {e}")
                        return
        print(f"❌ Profil non trouvé pour affichage détaillé : {nom_profil}")

    def _afficher_transitions_profil(self, nom_profil):
        """Affiche les transitions émotionnelles mémorisées pour ce profil"""
        log_path = "memory/emotional_log.json"
        if not os.path.exists(log_path):
            print("  (Aucune transition mémorisée)")
            return
        try:
            with open(log_path, encoding="utf-8") as f:
                historique = json.load(f)
            transitions = [s for s in historique if s.get("to") == nom_profil or s.get("from") == nom_profil]
            if transitions:
                print("  Transitions mémorisées :")
                for t in transitions[-5:]:
                    print(
                        f"    {t.get('from')} → {t.get('to')} ({t.get('emotion') if 'emotion' in t else t.get('emotion_to', '?')}) [{t.get('timestamp')}]"
                    )
            else:
                print("  (Aucune transition mémorisée)")
        except Exception as e:
            print(f"  ❌ Erreur lecture transitions : {e}")

    def éditer_profil_avancé(self, nom_profil):
        """Permet de modifier tous les paramètres avancés d’un profil, y compris effets visuels/sonores."""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)
                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)
                        print(f"\n⚙️ Édition avancée du profil : {data['name']}")
                        champs = [
                            ("name", "Nom"),
                            ("playlist", "Playlist d’émotions (séparées par virgule)"),
                            ("volume", "Volume (0.0–1.0)"),
                            ("intensity", "Intensité (0.0–1.0)"),
                            ("delay", "Délai (en secondes)"),
                            ("relation_context", "Contexte relationnel"),
                            ("origin", "Origine (auto_generated, user_personalized, etc.)"),
                            ("category", "Catégorie"),
                            ("approved_by_david", "Validé (True/False)"),
                        ]
                        for champ, label in champs:
                            val = input(f"🔧 {label} [actuel : {data.get(champ)}] : ").strip()
                            if val:
                                try:
                                    if champ == "playlist":
                                        data[champ] = [e.strip() for e in val.split(",")]
                                    elif champ in ["volume", "intensity", "delay"]:
                                        data[champ] = float(val)
                                    elif champ == "approved_by_david":
                                        data[champ] = val.lower() in ["true", "1", "oui", "o"]
                                    else:
                                        data[champ] = val
                                except Exception as e:
                                    print(f"⚠️ Valeur ignorée pour {champ}: {e}")
                        # Effets visuels/sonores personnalisés ?
                        effet_visuel = input(
                            f"✨ Effet visuel personnalisé (laisser vide pour conserver : {data.get('visual_effect', 'aucun')}) : "
                        ).strip()
                        if effet_visuel:
                            data["visual_effect"] = effet_visuel
                        effet_sonore = input(
                            f"🎵 Effet sonore personnalisé (laisser vide pour conserver : {data.get('audio_effect', 'aucun')}) : "
                        ).strip()
                        if effet_sonore:
                            data["audio_effect"] = effet_sonore
                        with open(chemin, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print("✅ Profil avancé mis à jour.")
                        return
                    except Exception as e:
                        print(f"❌ Erreur édition avancée : {e}")
                        return
        print(f"❌ Profil non trouvé pour édition avancée : {nom_profil}")

    # 2. Tests avancés sur profils
    def effectuer_tests_unitaires(self):
        """Valide la gestion des profils émotionnels et transitions basiques."""
        print("\n🧪 Test unitaire : création, validation, transition, mémorisation.")
        try:
            test_nom = "TestProfil"
            test_playlist = ["joie", "calme"]
            self.proposer_nouveau_profil(test_nom, test_playlist, "test_context")
            self.valider_profil(test_nom)
            profil = None
            for cat in os.listdir(self.output_dir):
                dossier = os.path.join(self.output_dir, cat)
                if os.path.isdir(dossier):
                    fichier = f"{test_nom.lower().replace(' ', '_')}.json"
                    chemin = os.path.join(dossier, fichier)
                    if os.path.exists(chemin):
                        with open(chemin, encoding="utf-8") as f:
                            profil = json.load(f)
                        break
            if profil:
                self.transition_vers_profil(profil)
                print("✅ Test unitaire réussi.")
            else:
                print("❌ Impossible de charger le profil de test.")
        except Exception as e:
            print(f"❌ Erreur test unitaire : {e}")

    def tester_interactions_profils(self):
        """Teste l’interaction entre plusieurs profils et la mémoire des transitions."""
        print("\n🔗 Test des interactions et mémorisation des transitions.")
        try:
            noms = ["TestA", "TestB"]
            for nom in noms:
                self.proposer_nouveau_profil(nom, [nom.lower()], "test_context")
                self.valider_profil(nom)
            profils = []
            for nom in noms:
                for cat in os.listdir(self.output_dir):
                    dossier = os.path.join(self.output_dir, cat)
                    if os.path.isdir(dossier):
                        fichier = f"{nom.lower().replace(' ', '_')}.json"
                        chemin = os.path.join(dossier, fichier)
                        if os.path.exists(chemin):
                            with open(chemin, encoding="utf-8") as f:
                                profils.append(json.load(f))
                            break
            if len(profils) == 2:
                self.transition_vers_profil(profils[0])
                self.transition_vers_profil(profils[1])
                print("✅ Test d’interaction réussi.")
            else:
                print("❌ Profils de test incomplets.")
        except Exception as e:
            print(f"❌ Erreur test interaction : {e}")

    # 3. Optimisation de la gestion des transitions
    def optimiser_transitions(self, profil_ancien, profil_nouveau, contexte=None):
        """Optimise la fluidité des transitions selon le contexte et l’émotion."""
        print("\n🔄 Optimisation de la transition entre profils.")
        try:
            étapes = 7
            for i in range(1, étapes + 1):
                ratio = i / étapes
                vol = self._interpoler(profil_ancien.get("volume", 0.5), profil_nouveau.get("volume", 0.5), ratio)
                intensité = self._interpoler(
                    profil_ancien.get("intensity", 0.5), profil_nouveau.get("intensity", 0.5), ratio
                )
                delay = self._interpoler(profil_ancien.get("delay", 1.0), profil_nouveau.get("delay", 1.0), ratio)
                print(f"  Étape {i}/{étapes} : Volume={vol:.2f}, Intensité={intensité:.2f}, Delay={delay:.2f}")
                # Ajustement dynamique selon contexte ou émotion
                if contexte:
                    print(f"    (Contexte : {contexte})")
                visual_engine.trigger("transition_optimisée")
                sound_engine.play_effect("emotion_transition_optimisée")
            print("✅ Transition optimisée terminée.")
        except Exception as e:
            print(f"❌ Erreur optimisation transition : {e}")

    def transition_non_linéaire(self, profil_ancien, profil_nouveau, courbe="ease-in-out"):
        """Effectue une transition non linéaire (progressive ou avec courbe d’animation)."""
        print(
            f"\n⏩ Transition non linéaire ({courbe}) entre {profil_ancien.get('name', '?')} → {profil_nouveau.get('name', '?')}"
        )
        import math

        étapes = 10
        for i in range(1, étapes + 1):
            t = i / étapes
            if courbe == "ease-in":
                ratio = t * t
            elif courbe == "ease-out":
                ratio = 1 - (1 - t) ** 2
            else:  # ease-in-out
                ratio = 0.5 * (1 - math.cos(math.pi * t))
            vol = self._interpoler(profil_ancien.get("volume", 0.5), profil_nouveau.get("volume", 0.5), ratio)
            intensité = self._interpoler(
                profil_ancien.get("intensity", 0.5), profil_nouveau.get("intensity", 0.5), ratio
            )
            print(f"  Étape {i}/{étapes} : Volume={vol:.2f}, Intensité={intensité:.2f}")
            visual_engine.trigger("transition_nonlineaire")
            sound_engine.play_effect("emotion_transition_nonlineaire")
        print("✅ Transition non linéaire terminée.")

    # 4. Personnalisation par l'utilisateur
    def permettre_personnalisation_utilisateur(self, nom_profil):
        """Permet à l’utilisateur de personnaliser un profil via l’interface (effets, intensité, etc.)."""
        print(f"\n👤 Personnalisation utilisateur du profil « {nom_profil} »")
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)
                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)
                        print(
                            f"Effets actuels : visuel = {data.get('visual_effect', 'aucun')}, audio = {data.get('audio_effect', 'aucun')}"
                        )
                        visual = input("Nouveau nom d’effet visuel (laisser vide pour conserver) : ").strip()
                        audio = input("Nouveau nom d’effet sonore (laisser vide pour conserver) : ").strip()
                        intensité = input(f"Nouvelle intensité (actuel : {data.get('intensity')}) : ").strip()
                        if visual:
                            data["visual_effect"] = visual
                        if audio:
                            data["audio_effect"] = audio
                        if intensité:
                            try:
                                data["intensity"] = float(intensité)
                            except ValueError:
                                print("⚠️ Valeur d’intensité ignorée.")
                        data["origin"] = "user_customized"
                        data["approved_by_david"] = False
                        with open(chemin, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print("✅ Personnalisation enregistrée.")
                        return
                    except Exception as e:
                        print(f"❌ Erreur personnalisation utilisateur : {e}")
                        return
        print("❌ Profil non trouvé pour personnalisation utilisateur.")

    # 5. Interactions avancées entre profils
    def gérer_interactions_profils(self, nom_profil_source, nom_profil_cible):
        """Ajoute des logiques d’interaction entre deux profils (ex: tristesse→joie)."""
        print(f"\n🔀 Gestion avancée interaction : {nom_profil_source} → {nom_profil_cible}")
        profil_source = None
        profil_cible = None
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                for nom, ref in [(nom_profil_source, "source"), (nom_profil_cible, "cible")]:
                    fichier = f"{nom.lower().replace(' ', '_')}.json"
                    chemin = os.path.join(dossier, fichier)
                    if os.path.exists(chemin):
                        with open(chemin, encoding="utf-8") as f:
                            if ref == "source":
                                profil_source = json.load(f)
                            else:
                                profil_cible = json.load(f)
        if profil_source and profil_cible:
            # Exemple: transition spéciale si source tristesse et cible joie
            if ("tristesse" in profil_source.get("playlist", [])) and ("joie" in profil_cible.get("playlist", [])):
                print("✨ Transition spéciale de tristesse vers joie !")
                self.transition_non_linéaire(profil_source, profil_cible, courbe="ease-out")
                self.mémoriser_relation_entre_profils(profil_source, profil_cible)
            else:
                print("Transition standard entre profils.")
                self.transition_vers_profil(profil_cible)
        else:
            print("❌ Impossible de trouver les deux profils pour interaction avancée.")

    # 6. Sauvegarde et chargement de profils personnalisés
    def sauvegarder_profil_personnalisé(self, nom_profil):
        """Permet à l’utilisateur de sauvegarder un profil personnalisé sous un nouveau nom."""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)
                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)
                        nv_nom = input("Nom du profil personnalisé à sauvegarder : ").strip()
                        if not nv_nom:
                            print("⚠️ Aucun nom fourni.")
                            return
                        data["name"] = nv_nom
                        data["origin"] = "user_custom_saved"
                        data["approved_by_david"] = False
                        data["created_on"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                        nv_fichier = f"{nv_nom.lower().replace(' ', '_')}.json"
                        nv_chemin = os.path.join(dossier, nv_fichier)
                        with open(nv_chemin, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"✅ Profil personnalisé sauvegardé sous : {nv_nom}")
                        return
                    except Exception as e:
                        print(f"❌ Erreur sauvegarde profil personnalisé : {e}")
                        return
        print("❌ Profil de base non trouvé pour sauvegarde personnalisée.")

    def charger_profil_personnalisé(self, nom_profil):
        """Charge un profil personnalisé sauvegardé pour utilisation future."""
        for catégorie in os.listdir(self.output_dir):
            dossier = os.path.join(self.output_dir, catégorie)
            if os.path.isdir(dossier):
                fichier = f"{nom_profil.lower().replace(' ', '_')}.json"
                chemin = os.path.join(dossier, fichier)
                if os.path.exists(chemin):
                    try:
                        with open(chemin, encoding="utf-8") as f:
                            data = json.load(f)
                        print(f"📥 Profil personnalisé chargé : {data.get('name')}")
                        return data
                    except Exception as e:
                        print(f"❌ Erreur chargement profil personnalisé : {e}")
                        return None
        print("❌ Profil personnalisé non trouvé.")
        return None
