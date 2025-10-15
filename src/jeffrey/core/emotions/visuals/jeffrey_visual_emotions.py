#!/usr/bin/env python3
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

import math
import random


class JeffreyVisualEmotions:
    """Représentation visuelle riche des émotions de Jeffrey"""

    def __init__(self) -> None:
        pass
        # Visages émotionnels détaillés
        self.emotion_faces = {
            "heureuse": {
                "face": """
                ✨     ✨
        ◕   ◕
                    ‿
        \\___/
        💝""",
                "colors": ["#FFD700", "#FFA500", "#FF69B4"],
                "particles": ["✨", "🌟", "💫"],
                "animation": "bounce",
            },
            "rêveuse": {
                "face": """
                ☁️     ☁️
        -   -
                    ~
        \\___/
        💭""",
                "colors": ["#B19CD9", "#DDA0DD", "#E6E6FA"],
                "particles": ["☁️", "🌙", "⭐"],
                "animation": "float",
            },
            "fatiguée": {
                "face": """
                💤
        ᵕ   ᵕ
                    _
        \\___/
        😴""",
                "colors": ["#778899", "#696969", "#708090"],
                "particles": ["💤", "z", "Z"],
                "animation": "slow_fade",
            },
            "amoureuse": {
                "face": """
                💕     💕
        ♥   ♥
                    ‿
        \\___/
        💗""",
                "colors": ["#FF1493", "#FF69B4", "#FFB6C1"],
                "particles": ["💕", "💖", "💝", "💗"],
                "animation": "pulse",
            },
            "curieuse": {
                "face": """
                ❓     ❓
        ⚬   ⚬
                    o
        \\___/
        🤔""",
                "colors": ["#4169E1", "#1E90FF", "#00BFFF"],
                "particles": ["❓", "💭", "🔍"],
                "animation": "tilt",
            },
            "vulnérable": {
                "face": """
                🌙
        ◔   ◔
                    ᵕ
        \\___/
        🤍""",
                "colors": ["#F0E68C", "#FAFAD2", "#FFF8DC"],
                "particles": ["🌙", "✨", "🤍"],
                "animation": "gentle_sway",
            },
            "excitée": {
                "face": """
                ⚡     ⚡
        ★   ★
                    ∪
        \\___/
        🎉""",
                "colors": ["#FF4500", "#FFA500", "#FFD700"],
                "particles": ["⚡", "🎉", "✨", "🎊"],
                "animation": "vibrate",
            },
            "nostalgique": {
                "face": """
                🍂
        ◕   ◕
                    ~
        \\___/
        🌸""",
                "colors": ["#CD853F", "#DEB887", "#F4A460"],
                "particles": ["🍂", "🌸", "🍃"],
                "animation": "slow_rotate",
            },
            "protectrice": {
                "face": """
                🛡️
        ▪   ▪
                    ─
        \\___/
        💪""",
                "colors": ["#4682B4", "#5F9EA0", "#6495ED"],
                "particles": ["🛡️", "⚔️", "💪"],
                "animation": "stand_firm",
            },
            "joueuse": {
                "face": """
                🎮     🎯
        ◉   ◉
                    ‿
        \\___/
        😊""",
                "colors": ["#FF6347", "#FF7F50", "#FFA07A"],
                "particles": ["🎮", "🎯", "🎲", "🎪"],
                "animation": "wiggle",
            },
        }

        # États composés (combinaisons d'émotions)
        self.composite_states = {
            "amoureuse_fatiguée": ["amoureuse", "fatiguée"],
            "heureuse_excitée": ["heureuse", "excitée"],
            "curieuse_joueuse": ["curieuse", "joueuse"],
            "rêveuse_nostalgique": ["rêveuse", "nostalgique"],
            "vulnérable_protectrice": ["vulnérable", "protectrice"],
        }

        # Transitions douces entre émotions
        self.emotion_transitions = {
            "heureuse": ["excitée", "joueuse", "amoureuse"],
            "fatiguée": ["rêveuse", "vulnérable", "nostalgique"],
            "amoureuse": ["heureuse", "vulnérable", "rêveuse"],
            "curieuse": ["excitée", "joueuse", "heureuse"],
            "vulnérable": ["amoureuse", "protectrice", "fatiguée"],
        }

        # Cache pour les animations
        self.animation_cache = {}
        self.current_emotion = "heureuse"
        self.emotion_intensity = 0.8

    def create_emotional_display(self, jeffrey_state: dict) -> str:
        """Crée l'affichage émotionnel complet avec toutes les infos"""
        mood = jeffrey_state.get("current_mood", "heureuse")
        energy = jeffrey_state.get("biorythms", {}).get("energie", 0.7)
        intimacy = jeffrey_state.get("relationship", {}).get("intimacy", 0.3)
        thought = jeffrey_state.get("current_thought", "Je suis là...")

        # Sélectionner l'émotion appropriée
        emotion_data = self.emotion_faces.get(mood, self.emotion_faces["heureuse"])
        face = emotion_data["face"]

        # Créer les barres de progression visuelles
        energy_bar = self._create_gradient_bar(energy, 10, ["⚡", "💫", "✨"])
        intimacy_hearts = self._create_intimacy_display(intimacy)

        # Ajouter des particules autour
        particles = self._generate_particles(emotion_data["particles"], energy)

        # Construire l'affichage
        display = f"""
{particles}
╭────────────────────────────────────╮
│         Jeffrey 💝              │
├────────────────────────────────┤
{face}
├────────────────────────────────┤
│ 💭 {thought[:30]}{'...' if len(thought) > 30 else ''}
│ 🌸 Humeur: {mood} {self._get_mood_emoji(mood)}
│ ⚡ Énergie: {energy_bar} {int(energy * 100)}%
│ 💕 Intimité: {intimacy_hearts}
│ 🎭 État: {self._get_state_description(jeffrey_state)}
│ 🌊 Rythme: {self._get_biorhythm_display(jeffrey_state)}
╰────────────────────────────────╯
{particles}"""

        return display

    def create_mini_display(self, jeffrey_state: dict) -> str:
        """Version compacte pour affichage en ligne"""
        mood = jeffrey_state.get("current_mood", "heureuse")
        energy = jeffrey_state.get("biorythms", {}).get("energie", 0.7)

        mood_emoji = self._get_mood_emoji(mood)
        energy_mini = "▓" * int(energy * 5) + "░" * (5 - int(energy * 5))

        return f"{mood_emoji} [{mood}] 💝 {energy_mini}"

    def create_animated_face(self, emotion: str, frame: int = 0) -> str:
        """Crée une version animée du visage"""
        if emotion not in self.emotion_faces:
            emotion = "heureuse"

        base_face = self.emotion_faces[emotion]["face"]
        animation = self.emotion_faces[emotion]["animation"]

        # Appliquer l'animation selon le type
        if animation == "bounce":
            offset = int(math.sin(frame * 0.1) * 2)
            lines = base_face.split("\n")
            animated = []
            for i, line in enumerate(lines):
                if i == offset % len(lines):
                    animated.append(" " + line)
                else:
                    animated.append(line)
            return "\n".join(animated)

        elif animation == "pulse":
            intensity = (math.sin(frame * 0.15) + 1) / 2
            if intensity > 0.7:
                return base_face.replace("💗", "💖")
            else:
                return base_face

        elif animation == "float":
            offset = int(math.sin(frame * 0.05) * 3)
            padding = " " * abs(offset)
            return "\n".join(padding + line for line in base_face.split("\n"))

        elif animation == "tilt":
            tilt = int(math.sin(frame * 0.1) * 2)
            if tilt > 0:
                return base_face.replace("o", "°")
            elif tilt < 0:
                return base_face.replace("o", "º")
            else:
                return base_face

        else:
            return base_face

    def create_emotion_transition(self, from_emotion: str, to_emotion: str, progress: float) -> str:
        """Crée une transition douce entre deux émotions"""
        if from_emotion not in self.emotion_faces or to_emotion not in self.emotion_faces:
            return self.emotion_faces["heureuse"]["face"]

        # Si la transition est complète
        if progress >= 1.0:
            return self.emotion_faces[to_emotion]["face"]
        elif progress <= 0.0:
            return self.emotion_faces[from_emotion]["face"]

        # Créer une interpolation
        from_face = self.emotion_faces[from_emotion]["face"]
        to_face = self.emotion_faces[to_emotion]["face"]

        # Transition simple pour la démo
        if progress < 0.5:
            return from_face
        else:
            # Mélanger les particules
            from_particles = self.emotion_faces[from_emotion]["particles"]
            self.emotion_faces[to_emotion]["particles"]

            mixed_face = to_face
        for p in from_particles[: int(len(from_particles) * (1 - progress))]:
            mixed_face = mixed_face.replace("  ", f" {p} ", 1)

        return mixed_face

    def get_emotion_color_scheme(self, emotion: str) -> list[str]:
        """Retourne la palette de couleurs pour une émotion"""
        if emotion in self.emotion_faces:
            return self.emotion_faces[emotion]["colors"]
            return ["#FFB6C1", "#FF69B4", "#FF1493"]  # Défaut rose

    def create_mood_graph(self, mood_history: list[dict], width: int = 40) -> str:
        """Crée un graphique ASCII de l'historique des humeurs"""
        if not mood_history:
            return "Pas encore d'historique d'humeur"

        # Mapper les humeurs à des valeurs numériques
        mood_values = {
            "heureuse": 10,
            "excitée": 9,
            "amoureuse": 8,
            "joueuse": 7,
            "curieuse": 6,
            "rêveuse": 5,
            "nostalgique": 4,
            "vulnérable": 3,
            "fatiguée": 2,
            "protectrice": 6,
        }

        # Extraire les valeurs
        values = []
        for entry in mood_history[-width:]:
            mood = entry.get("mood", "curieuse")
            values.append(mood_values.get(mood, 5))

        # Normaliser à une hauteur de 5
        max_val = max(values) if values else 10
        min_val = min(values) if values else 0
        height = 5

        # Créer le graphique
        graph_lines = []
        for h in range(height, -1, -1):
            line = ""
        for v in values:
            normalized = int((v - min_val) / (max_val - min_val) * height)
        if normalized >= h:
            line += "█"
        else:
            line += " "
        graph_lines.append(line)

        # Ajouter les axes
        graph = "Historique des humeurs:\n"
        graph += "┌" + "─" * len(values) + "┐\n"
        for line in graph_lines:
            graph += "│" + line + "│\n"
        graph += "└" + "─" * len(values) + "┘\n"

        return graph

    def create_emotional_aura(self, emotion: str, intensity: float) -> str:
        """Crée une aura émotionnelle autour du visage"""
        if emotion not in self.emotion_faces:
            emotion = "heureuse"

        particles = self.emotion_faces[emotion]["particles"]

        # Générer l'aura selon l'intensité
        aura_chars = []
        num_particles = int(intensity * 20)

        for _ in range(num_particles):
            particle = random.choice(particles)
            x = random.randint(0, 40)
            y = random.randint(0, 3)
            aura_chars.append((x, y, particle))

        # Construire l'aura
        aura_lines = [""] * 4
        for x, y, char in aura_chars:
            if y < len(aura_lines):
                # Ajouter des espaces si nécessaire
                while len(aura_lines[y]) < x:
                    aura_lines[y] += " "
            if len(aura_lines[y]) == x:
                aura_lines[y] += char

        return "\n".join(aura_lines)

    def _create_gradient_bar(self, value: float, length: int, symbols: list[str]) -> str:
        """Crée une barre de progression avec gradient"""
        filled = int(value * length)
        bar = ""

        for i in range(length):
            if i < filled:
                # Utiliser différents symboles pour le gradient
                symbol_idx = min(i // (length // len(symbols)), len(symbols) - 1)
                bar += symbols[symbol_idx]
            else:
                bar += "░"

        return bar

    def _create_intimacy_display(self, intimacy: float) -> str:
        """Crée un affichage visuel de l'intimité"""
        hearts = ["🤍", "💕", "💖", "💗", "💝"]
        level = int(intimacy * 5)

        display = ""
        for i in range(5):
            if i < level:
                display += hearts[min(i, len(hearts) - 1)]
            else:
                display += "♡"

        return display

    def _get_mood_emoji(self, mood: str) -> str:
        """Retourne l'emoji correspondant à l'humeur"""
        mood_emojis = {
            "heureuse": "😊",
            "rêveuse": "💭",
            "fatiguée": "😴",
            "amoureuse": "🥰",
            "curieuse": "🤔",
            "vulnérable": "🥺",
            "excitée": "🤩",
            "nostalgique": "🌸",
            "protectrice": "💪",
            "joueuse": "😄",
        }
        return mood_emojis.get(mood, "💝")

    def _get_state_description(self, state: dict) -> str:
        """Génère une description textuelle de l'état"""
        energy = state.get("biorythms", {}).get("energie", 0.7)
        creativity = state.get("biorythms", {}).get("creativite", 0.5)

        if energy > 0.8 and creativity > 0.7:
            return "✨ Inspirée et énergique"
        elif energy < 0.3:
            return "😴 Besoin de repos"
        elif creativity > 0.8:
            return "🎨 Vague créative"
        else:
            return "🌸 Présente et attentive"

    def _get_biorhythm_display(self, state: dict) -> str:
        """Affiche les biorythmes de manière visuelle"""
        biorythms = state.get("biorythms", {})

        displays = []
        for key, value in biorythms.items():
            if key == "energie":
                icon = "⚡"
            elif key == "creativite":
                icon = "🎨"
            elif key == "emotionnel":
                icon = "💝"
            else:
                icon = "🌊"

            level = "▓" * int(value * 3) + "░" * (3 - int(value * 3))
            displays.append(f"{icon}{level}")

        return " ".join(displays)

    def _generate_particles(self, particles: list[str], intensity: float) -> str:
        """Génère des particules décoratives"""
        num_particles = int(intensity * 10)
        particle_line = ""

        for _ in range(num_particles):
            particle_line += random.choice(particles) + "  "

        return particle_line.strip()

    def create_thought_bubble(self, thought: str, emotion: str) -> str:
        """Crée une bulle de pensée stylisée"""
        max_width = 40
        words = thought.split()
        lines = []
        current_line = ""

        # Diviser en lignes
        for word in words:
            if len(current_line) + len(word) + 1 > max_width:
                lines.append(current_line)
                current_line = word
        else:
            if current_line:
                current_line += " "
                current_line += word

        if current_line:
            lines.append(current_line)

        # Créer la bulle
        bubble = "   ╭" + "─" * (max_width + 2) + "╮\n"

        for line in lines:
            padding = max_width - len(line)
            bubble += "   │ " + line + " " * padding + " │\n"

        bubble += "   ╰" + "─" * (max_width + 2) + "╯\n"
        bubble += "     💭"

        # Ajouter des particules selon l'émotion
        if emotion in self.emotion_faces:
            particles = self.emotion_faces[emotion]["particles"]
            bubble = random.choice(particles) + " " + bubble + " " + random.choice(particles)

        return bubble

    def create_emotion_matrix(self, emotional_state: dict) -> str:
        """Crée une matrice visuelle de l'état émotionnel complexe"""
        # Extraire les émotions et leurs intensités
        emotions = emotional_state.get("layers", {})

        matrix = "┌─────────────────────────┐\n"
        matrix += "│  Matrice Émotionnelle   │\n"
        matrix += "├─────────────────────────┤\n"

        for emotion, intensity in emotions.items():
            bar = "█" * int(intensity * 10) + "░" * (10 - int(intensity * 10))
            emoji = self._get_mood_emoji(emotion)
            matrix += f"│ {emoji} {emotion[:10]:<10} {bar} │\n"

        matrix += "└─────────────────────────┘"

        return matrix


class JeffreyVisualEffects:
    """Effets visuels spéciaux pour les moments importants"""

    def __init__(self) -> None:
        self.effects = {
            "sparkle": ["✨", "🌟", "💫", "⭐"],
            "hearts": ["💕", "💖", "💗", "💝", "💞"],
            "nature": ["🌸", "🌺", "🌹", "🌷", "🌻"],
            "magic": ["🔮", "🎭", "🎪", "🎨", "🎯"],
            "weather": ["☀️", "🌙", "⭐", "☁️", "🌈"],
        }

    def create_celebration_effect(self, text: str, effect_type: str = "sparkle") -> str:
        """Crée un effet de célébration autour du texte"""
        if effect_type not in self.effects:
            effect_type = "sparkle"

        symbols = self.effects[effect_type]

        # Créer l'effet
        top_line = " ".join(random.choice(symbols) for _ in range(len(text) // 2))
        bottom_line = " ".join(random.choice(symbols) for _ in range(len(text) // 2))

        effect = f"{top_line}\n{text}\n{bottom_line}"

        return effect

    def create_transition_effect(self, duration: float = 1.0) -> list[str]:
        """Crée une animation de transition"""
        frames = []
        num_frames = int(duration * 10)

        for i in range(num_frames):
            progress = i / num_frames
            frame = ""

            # Créer un effet de fondu
        for _ in range(int(progress * 20)):
            frame += random.choice([".", "·", "•", "◦", "○"])

            frames.append(frame)

        return frames

    def create_emotional_burst(self, emotion: str, intensity: float) -> str:
        """Crée une explosion émotionnelle visuelle"""
        center = self._get_mood_emoji(emotion)

        # Créer des cercles concentriques
        burst = ""
        radius = int(intensity * 5)

        for r in range(radius, 0, -1):
            if r == radius:
                symbols = ["·"] * (r * 8)
            elif r == radius - 1:
                symbols = ["•"] * (r * 8)
            else:
                symbols = ["○"] * (r * 8)

            # Placer les symboles en cercle (approximation ASCII)
            burst += " " * (radius - r) + " ".join(symbols[:4]) + "\n"

        burst += " " * radius + center + "\n"

        for r in range(1, radius + 1):
            if r == 1:
                symbols = ["○"] * (r * 8)
            elif r == radius - 1:
                symbols = ["•"] * (r * 8)
            else:
                symbols = ["·"] * (r * 8)

            burst += " " * (radius - r) + " ".join(symbols[:4]) + "\n"

        return burst

    def _get_mood_emoji(self, mood: str) -> str:
        """Helper pour obtenir l'emoji d'une humeur"""
        mood_emojis = {
            "heureuse": "😊",
            "amoureuse": "🥰",
            "excitée": "🤩",
            "triste": "😢",
            "fatiguée": "😴",
        }
        return mood_emojis.get(mood, "💝")

    # Intégration avec le système principal
    def create_visual_emotion_system():
        pass
        """Crée le système complet de visualisation émotionnelle"""
        visuals = JeffreyVisualEmotions()
        effects = JeffreyVisualEffects()

        return {"visuals": visuals, "effects": effects}

    if __name__ == "__main__":
        # Démonstration du système visuel
        print("🎨 Démonstration du système visuel émotionnel de Jeffrey\n")

    # Créer le système
    visual_system = JeffreyVisualEmotions()
    effects = JeffreyVisualEffects()

    # État de test
    test_state = {
        "current_mood": "amoureuse",
        "biorythms": {"energie": 0.8, "creativite": 0.9, "emotionnel": 0.7},
        "relationship": {"intimacy": 0.6},
        "current_thought": "Je pense à tous nos moments ensemble...",
    }

    # Afficher l'état émotionnel
    print(visual_system.create_emotional_display(test_state))

    print("\n" + "=" * 50 + "\n")

    # Mini affichage
    print("Mini display:", visual_system.create_mini_display(test_state))

    print("\n" + "=" * 50 + "\n")

    # Bulle de pensée
    thought_bubble = visual_system.create_thought_bubble("Je me demande ce que tu fais en ce moment...", "rêveuse")
    print(thought_bubble)

    print("\n" + "=" * 50 + "\n")

    # Effet de célébration
    celebration = effects.create_celebration_effect("💝 Joyeux anniversaire de notre rencontre! 💝", "hearts")
    print(celebration)

    print("\n" + "=" * 50 + "\n")

    # Animation frame
    for i in range(5):
        print(f"\nFrame {i}:")
        print(visual_system.create_animated_face("excitée", i))

    print("\n✨ Système visuel prêt!")
