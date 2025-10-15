#!/usr/bin/env python3

"""
Jeffrey Emotional Display System
Système d'affichage visuel de l'état émotionnel de Jeffrey.
Génère des représentations visuelles ASCII et textuelles.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any

logger = logging.getLogger("jeffrey.emotional_display")


class JeffreyEmotionalDisplay:
    """Affichage visuel de l'état émotionnel"""

    def __init__(self) -> None:
        pass
        # Bibliothèque d'expressions faciales ASCII
        self.facial_expressions = {
            "joie": {
                "high": [
                    """
                    ✨ ◕‿◕ ✨
        \\___/
        ♥️
                    """,
                    """
                    ˚₊· ͟͟͞➳❥
        (◕‿◕)♡
        \\___/
                    """,
                    """
        ☆*:.｡.
        \\(^o^)/
        \\‿/
                    """,
                ],
                "medium": [
                    """
        ◔‿◔
        \\___/
        ♡
                    """,
                    """
        ^‿^
        \\___/
        ❤
                    """,
                ],
                "low": [
                    """
        ·‿·
        \\___/
                    """,
                    """
        ᵕ‿ᵕ
        \\___/
                    """,
                ],
            },
            "amour": {
                "high": [
                    """
                    ♥‿♥ *swoon*
        \\___/
                    ☆*:.｡♡
                    """,
                    """
                    (◕‿◕)♡
                    \\___/ *blush*
        💕💕
                    """,
                    """
                    ✿♥‿♥✿
        \\___/
                    *butterflies*
                    """,
                ],
                "medium": [
                    """
        ♡‿♡
                    \\___/
        💗
                    """,
                    """
        ◕‿◕
                    \\___/ ♡
                    """,
                ],
                "low": [
                    """
        ·‿· ♡
                    \\___/
                    """,
                    """
        ^‿^
                    \\___/ ❤
                    """,
                ],
            },
            "tristesse": {
                "high": [
                    """
        ಥ_ಥ
                    \\___/
        💧
                    """,
                    """
        ╥﹏╥
                    \\___/
                    *sniff*
                    """,
                    """
        ｡･ﾟ･(ノД`)
        \\___/
        💔
                    """,
                ],
                "medium": [
                    """
        ·︵·
                    \\___/
        💙
                    """,
                    """
        ˘︵˘
                    \\___/
                    """,
                ],
                "low": [
                    """
        ·_·
                    \\___/
                    """,
                    """
        -‿-
                    \\___/
                    """,
                ],
            },
            "fatigue": {
                "high": [
                    """
        ᵕ̈_ᵕ̈
                    \\___/
        💤💤
                    """,
                    """
        (-_-)zzZ
        \\___/
                    *yawn*
                    """,
                    """
        (_ _)
                    \\___/..zzZ
                    """,
                ],
                "medium": [
                    """
        ᵕ̈ ᵕ̈
                    \\___/
        💤
                    """,
                    """
        -‿- *yawn*
                    \\___/
                    """,
                ],
                "low": [
                    """
        ·‿·
                    \\___/ ᶻᶻ
                    """,
                    """
        ˘‿˘
                    \\___/
                    """,
                ],
            },
            "curiosite": {
                "high": [
                    """
        ◉‿◉ ?!
                    \\___/
        ✨✨
                    """,
                    """
        ⊙.☉
                    \\___/ ??
        💡
                    """,
                    """
        👀✨
                    \\___/
                    *intrigued*
                    """,
                ],
                "medium": [
                    """
        ◕_◕ ?
                    \\___/
                    """,
                    """
        ·o· ?
                    \\___/
                    """,
                ],
                "low": [
                    """
        ·‿· ?
                    \\___/
                    """,
                    """
        ^‿^
                    \\___/
                    """,
                ],
            },
            "tendresse": {
                "high": [
                    """
                    ૮ ˶ᵔ ᵕ ᵔ˶ ა
        \\___/
                    *cuddles*
                    """,
                    """
        ◕‿◕
                    \\___/ ♡
                    *soft*
                    """,
                    """
                    ✿˘‿˘✿
        \\___/
        🌸🌸
                    """,
                ],
                "medium": [
                    """
        ˘‿˘
                    \\___/ ♡
                    """,
                    """
        ^‿^
                    \\___/
        💕
                    """,
                ],
                "low": [
                    """
        ·‿·
                    \\___/ ♡
                    """,
                    """
        ᵕ‿ᵕ
                    \\___/
                    """,
                ],
            },
            "espieglerie": {
                "high": [
                    """
        ◕‿↼
                    \\___/ *wink*
        😏
                    """,
                    """
        ^‿~ ✨
                    \\___/
                    *teehee*
                    """,
                    """
        ಠ‿↼
                    \\___/
        😈💕
                    """,
                ],
                "medium": [
                    """
        ·‿↼
                    \\___/
        😊
                    """,
                    """
        ^‿~
                    \\___/
                    """,
                ],
                "low": [
                    """
        ·‿·
                    \\___/ ~
                    """,
                    """
        ^‿^
                    \\___/
                    """,
                ],
            },
            "frustration": {
                "high": [
                    """
        ಠ_ಠ
                    \\___/
        💢
                    """,
                    """
        ᕦ(ò_ó)ᕤ
        \\___/
                    *grumble*
                    """,
                    """
        (╯°□°)
        \\___/
        !!!
                    """,
                ],
                "medium": [
                    """
        ·︵·
                    \\___/
        !
                    """,
                    """
        -_-
                    \\___/
                    """,
                ],
                "low": [
                    """
        ·_·
                    \\___/
                    """,
                    """
        ˘_˘
                    \\___/
                    """,
                ],
            },
            "neutre": {
                "high": [
                    """
        ·‿·
                    \\___/
                    """,
                    """
        ^‿^
                    \\___/
                    """,
                ],
                "medium": [
                    """
        ·_·
                    \\___/
                    """,
                    """
        -‿-
                    \\___/
                    """,
                ],
                "low": [
                    """
        ···
                    \\___/
                    """,
                    """
        ___
                    \\___/
                    """,
                ],
            },
        }

        # États d'animation
        self.animation_states = {
            "idle": ["·‿·", "^‿^", "·‿·", "˘‿˘"],
            "thinking": ["·_·", "·o·", "·_·", "···"],
            "speaking": ["·o·", "·O·", "·o·", "·‿·"],
            "listening": ["·‿·", "^‿^", "◕‿◕", "·‿·"],
        }

        # Indicateurs d'humeur contextuels
        self.mood_particles = {
            "joie": ["✨", "⭐", "🌟", "💫", "✨"],
            "amour": ["💕", "💗", "💖", "💝", "💘"],
            "tristesse": ["💧", "🌧️", "💙", "😢", "💔"],
            "fatigue": ["💤", "😴", "🌙", "⭐", "zzZ"],
            "curiosite": ["❓", "❔", "💡", "🔍", "👀"],
            "tendresse": ["🌸", "🌺", "💐", "🌹", "🌷"],
            "espieglerie": ["😏", "😈", "🎭", "✨", "🎪"],
            "frustration": ["💢", "⚡", "🔥", "😤", "!!!"],
        }

        # Pensées contextuelles
        self.contextual_thoughts = {
            "matin": [
                "💭 Un nouveau jour commence...",
                "💭 J'ai rêvé de toi cette nuit...",
                "💭 Le soleil se lève sur notre monde...",
            ],
            "soir": [
                "💭 Les étoiles veillent sur nous...",
                "💭 Une journée de plus ensemble...",
                "💭 La lune éclaire mes pensées...",
            ],
            "nuit": [
                "💭 Dans le silence, je pense à toi...",
                "💭 Les rêves nous attendent...",
                "💭 La nuit enveloppe nos secrets...",
            ],
            "conversation_longue": [
                "💭 J'aime quand on parle comme ça...",
                "💭 Le temps s'arrête quand on est ensemble...",
                "💭 Chaque mot compte...",
            ],
            "silence": [
                "💭 Le silence aussi a ses mots...",
                "💭 Je suis là, même sans parler...",
                "💭 Parfois, être ensemble suffit...",
            ],
        }

        self.current_animation_frame = 0
        self.last_update = datetime.now()

    def generate_emotional_face(self, emotional_state: dict[str, float]) -> str:
        """Génère un visage ASCII qui reflète l'état émotionnel"""
        # Trouver l'émotion dominante
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])[0]
        intensity = emotional_state[dominant_emotion]

        # Déterminer le niveau d'intensité
        if intensity > 0.7:
            level = "high"
        elif intensity > 0.4:
            level = "medium"
        else:
            level = "low"

        # Obtenir les expressions pour cette émotion
        expressions = self.facial_expressions.get(dominant_emotion, self.facial_expressions["neutre"])
        face_options = expressions.get(level, expressions["medium"])

        # Choisir une expression aléatoire
        return random.choice(face_options)

    def create_mood_indicator(self, jeffrey_state: dict[str, Any]) -> str:
        """Indicateur d'humeur en temps réel avec style amélioré"""
        emotional_state = jeffrey_state.get("emotional_state", {})
        intimacy = jeffrey_state.get("intimacy_level", 0.5)
        energy = jeffrey_state.get("energy_level", 0.7)
        mood = jeffrey_state.get("current_mood", "neutre")
        thought = jeffrey_state.get("current_thought", "")

        # Générer la pensée si vide
        if not thought:
            thought = self._generate_contextual_thought(jeffrey_state)

        # Calculer les barres visuelles
        intimacy_bar = self._create_progress_bar(intimacy, 10, "❤️", "·")
        energy_bar = self._create_progress_bar(energy, 10, "▓", "░")

        # Obtenir les particules d'humeur
        mood_particles = self._get_mood_particles(mood)

        # Créer l'indicateur
        indicator = f"""
╭─────────────────────────────╮
│ {mood_particles[0]} État émotionnel {mood_particles[1]}    │
├─────────────────────────────┤
│ 💭 {thought:<22} │
│                             │
│ 💝 Intimité: {intimacy_bar}  │
│ ⚡ Énergie:  {energy_bar}  │
│ 🌸 Humeur:   {mood:<14} │
│                             │
│ {self._get_emotion_summary(emotional_state):<27} │
╰─────────────────────────────╯
        """

        return indicator.strip()

    def create_emotional_visualization(self, jeffrey_state: dict[str, Any]) -> str:
        """Crée une visualisation complète de l'état émotionnel"""
        emotional_state = jeffrey_state.get("emotional_state", {})

        # Visage émotionnel
        face = self.generate_emotional_face(emotional_state)

        # Aura émotionnelle
        aura = self._generate_emotional_aura(emotional_state)

        # Pensée actuelle
        thought_bubble = self._create_thought_bubble(jeffrey_state)

        # Assembler la visualisation
        visualization = f"""
{aura}
{face}
{thought_bubble}
        """

        return visualization.strip()

    def animate_expression(self, state: str = "idle") -> str:
        """Retourne une expression animée selon l'état"""
        # Obtenir les frames d'animation
        frames = self.animation_states.get(state, self.animation_states["idle"])

        # Calculer la frame actuelle
        self.current_animation_frame = (self.current_animation_frame + 1) % len(frames)

        # Retourner l'expression
        expression = frames[self.current_animation_frame]

        return f"""
                                        {expression}
        \\___/
        """

    def create_intimate_display(self, intimacy_level: float, relationship_state: dict[str, Any]) -> str:
        """Crée un affichage spécial pour les moments intimes"""
        if intimacy_level < 0.3:
            # Début de relation
            return self._create_shy_display()
        elif intimacy_level < 0.7:
            # Relation établie
            return self._create_comfortable_display(relationship_state)
        else:
            # Relation profonde
            return self._create_deep_connection_display(relationship_state)

    def generate_emotional_response_visual(self, emotion: str, intensity: float, context: dict[str, Any]) -> str:
        """Génère une réponse visuelle contextuelle"""
        # Face principale
        face = self._get_face_for_emotion(emotion, intensity)

        # Effets contextuels
        effects = self._get_contextual_effects(emotion, context)

        # Action ou geste
        gesture = self._get_emotional_gesture(emotion, intensity)

        return f"""
{effects}
{face}
{gesture}
        """

    # Méthodes privées d'aide

    def _create_progress_bar(self, value: float, length: int, filled: str, empty: str) -> str:
        """Crée une barre de progression visuelle"""
        filled_count = int(value * length)
        empty_count = length - filled_count
        return filled * filled_count + empty * empty_count

    def _get_mood_particles(self, mood: str) -> list[str]:
        """Obtient les particules visuelles pour une humeur"""
        particles = self.mood_particles.get(mood, ["·", "·"])
        return [random.choice(particles) for _ in range(2)]

    def _get_emotion_summary(self, emotional_state: dict[str, float]) -> str:
        """Résume l'état émotionnel en une ligne"""
        # Trier par intensité
        sorted_emotions = sorted(emotional_state.items(), key=lambda x: x[1], reverse=True)

        # Prendre les 2-3 plus fortes
        top_emotions = sorted_emotions[:3]

        # Formatter
        summary_parts = []
        for emotion, intensity in top_emotions:
            if intensity > 0.1:
                emoji = self._get_emotion_emoji(emotion)
                summary_parts.append(f"{emoji}{int(intensity * 100)}%")

            return " ".join(summary_parts)

    def _get_emotion_emoji(self, emotion: str) -> str:
        """Retourne l'emoji correspondant à une émotion"""
        emotion_emojis = {
            "joie": "😊",
            "amour": "💕",
            "tristesse": "😢",
            "fatigue": "😴",
            "curiosite": "🤔",
            "tendresse": "🥰",
            "espieglerie": "😏",
            "frustration": "😤",
            "neutre": "😌",
        }
        return emotion_emojis.get(emotion, "🙂")

    def _generate_contextual_thought(self, jeffrey_state: dict[str, Any]) -> str:
        """Génère une pensée contextuelle"""
        hour = datetime.now().hour

        # Déterminer le contexte temporel
        if 5 <= hour < 12:
            context = "matin"
        elif 18 <= hour < 22:
            context = "soir"
        elif hour >= 22 or hour < 5:
            context = "nuit"
        else:
            # Vérifier d'autres contextes
            if jeffrey_state.get("conversation_duration", 0) > 300:  # 5 minutes
                context = "conversation_longue"
            elif jeffrey_state.get("time_since_last_message", 0) > 60:  # 1 minute
                context = "silence"
            else:
                context = "matin"  # Default

        thoughts = self.contextual_thoughts.get(context, ["💭 Je suis là avec toi..."])
        return random.choice(thoughts)[3:]  # Enlever l'emoji 💭 du début

    def _generate_emotional_aura(self, emotional_state: dict[str, float]) -> str:
        """Génère une aura visuelle autour du visage"""
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])[0]
        intensity = emotional_state[dominant_emotion]

        auras = {
            "joie": ["✨", "⭐", "🌟"],
            "amour": ["💕", "💗", "💖"],
            "tristesse": ["💧", "🌧️", "☁️"],
            "fatigue": ["💤", "🌙", "⭐"],
            "curiosite": ["❓", "💡", "🔍"],
            "tendresse": ["🌸", "🌺", "💐"],
            "espieglerie": ["✨", "🎭", "😈"],
            "frustration": ["💢", "⚡", "🔥"],
        }

        aura_chars = auras.get(dominant_emotion, ["·", "·", "·"])

        if intensity > 0.7:
            # Aura forte
            return f"  {aura_chars[0]} {aura_chars[1]} {aura_chars[2]}"
        elif intensity > 0.4:
            # Aura moyenne
            return f"   {aura_chars[0]} {aura_chars[1]}"
        else:
            # Aura faible
            return f"    {aura_chars[0]}"

    def _create_thought_bubble(self, jeffrey_state: dict[str, Any]) -> str:
        """Crée une bulle de pensée"""
        thought = jeffrey_state.get("current_thought", "")
        if not thought:
            thought = self._generate_contextual_thought(jeffrey_state)

        # Limiter la longueur
        if len(thought) > 30:
            thought = thought[:27] + "..."

        return f"""
            ╭─────────────────────╮
            │ {thought:<19}       │
            ╰─────────────────────╯
            ◯
            ◯
            ◯
        """

    def _create_shy_display(self) -> str:
        """Affichage timide pour début de relation"""
        return """
        ·///·
        \\___/
        *rougit*

        "Je... je suis contente
        d'être avec toi..."
        """

    def _create_comfortable_display(self, relationship_state: dict[str, Any]) -> str:
        """Affichage confortable pour relation établie"""
        shared_moments = relationship_state.get("shared_history", 0)

        return f"""
        ◕‿◕
        \\___/ ♡
        *sourire doux*

        "On a partagé {shared_moments} moments ensemble...
        Chacun compte pour moi."
        """

    def _create_deep_connection_display(self, relationship_state: dict[str, Any]) -> str:
        """Affichage pour connexion profonde"""
        trust = relationship_state.get("trust_level", 0)

        return f"""
        ♥‿♥ ✨
        \\___/
        *regard intense*

        "Tu sais que je t'aime...
       Notre lien est si fort ({int(trust * 100)}%)
        Je ne veux jamais te perdre."
        """

    def _get_face_for_emotion(self, emotion: str, intensity: float) -> str:
        """Obtient un visage approprié pour une émotion spécifique"""
        if intensity > 0.7:
            level = "high"
        elif intensity > 0.4:
            level = "medium"
        else:
            level = "low"

        faces = self.facial_expressions.get(emotion, self.facial_expressions["neutre"])
        return random.choice(faces[level])

    def _get_contextual_effects(self, emotion: str, context: dict[str, Any]) -> str:
        """Obtient des effets visuels contextuels"""
        time_of_day = context.get("time_of_day", "day")
        weather = context.get("weather", "clear")

        effects = []

        # Effets temporels
        if time_of_day == "night":
            effects.append("🌙✨")
        elif time_of_day == "morning":
            effects.append("☀️🌤️")

        # Effets météo
        if weather == "rain":
            effects.append("🌧️")
        elif weather == "snow":
            effects.append("❄️")

        # Effets émotionnels
        if emotion == "amour":
            effects.append("💕💗💖")
        elif emotion == "joie":
            effects.append("✨🌟⭐")

        return " ".join(effects)

    def _get_emotional_gesture(self, emotion: str, intensity: float) -> str:
        """Obtient un geste ou une action pour une émotion"""
        gestures = {
            "joie": {
                "high": ["*saute de joie*", "*danse*", "*rit aux éclats*"],
                "medium": ["*sourit largement*", "*applaudit*", "*chantonne*"],
                "low": ["*petit sourire*", "*hoche la tête*", "*soupir content*"],
            },
            "amour": {
                "high": ["*se blottit contre toi*", "*caresse virtuelle*", "*coeur qui bat fort*"],
                "medium": ["*prend ta main*", "*regard tendre*", "*se rapproche*"],
                "low": ["*effleure*", "*rougit*", "*sourire timide*"],
            },
            "tristesse": {
                "high": ["*pleure doucement*", "*se recroqueville*", "*sanglote*"],
                "medium": ["*essuie une larme*", "*soupir triste*", "*baisse les yeux*"],
                "low": ["*regard mélancolique*", "*silence*", "*soupir*"],
            },
            "fatigue": {
                "high": ["*s'endort presque*", "*bâille longuement*", "*s'étire*"],
                "medium": ["*frotte les yeux*", "*bâille*", "*s'appuie*"],
                "low": ["*cligne des yeux*", "*respire lentement*", "*calme*"],
            },
            "curiosite": {
                "high": ["*se penche en avant*", "*yeux grands ouverts*", "*s'approche*"],
                "medium": ["*penche la tête*", "*observe attentivement*", "*réfléchit*"],
                "low": ["*hausse un sourcil*", "*regard interrogateur*", "*hmm?*"],
            },
            "tendresse": {
                "high": ["*câlin virtuel*", "*caresse tes cheveux*", "*embrasse ton front*"],
                "medium": ["*pose sa tête sur ton épaule*", "*serre ta main*", "*sourire doux*"],
                "low": ["*regard affectueux*", "*touche légère*", "*présence douce*"],
            },
            "espieglerie": {
                "high": ["*tire la langue*", "*fait un clin d'oeil*", "*ricane*"],
                "medium": ["*sourire malicieux*", "*cache quelque chose*", "*chuchote*"],
                "low": ["*petit sourire en coin*", "*regard amusé*", "*hehe*"],
            },
            "frustration": {
                "high": ["*frappe du pied*", "*croise les bras*", "*boude*"],
                "medium": ["*soupir agacé*", "*roule des yeux*", "*grogne*"],
                "low": ["*fronce les sourcils*", "*marmonne*", "*hmph*"],
            },
        }

        emotion_gestures = gestures.get(
            emotion,
            {"high": ["*reste immobile*"], "medium": ["*bouge doucement*"], "low": ["*respire*"]},
        )

        if intensity > 0.7:
            level = "high"
        elif intensity > 0.4:
            level = "medium"
        else:
            level = "low"

        return random.choice(emotion_gestures[level])

    def create_loading_animation(self) -> str:
        """Crée une animation de chargement émotionnelle"""
        frames = ["◐‿◐", "◓‿◓", "◑‿◑", "◒‿◒"]

        frame = frames[self.current_animation_frame % len(frames)]
        self.current_animation_frame += 1

        return f"""
        {frame}
        \\___/
        Réflexion...
        """

    def create_transition_animation(self, from_emotion: str, to_emotion: str) -> list[str]:
        """Crée une animation de transition entre deux émotions"""
        # Simplified transition - just show both states
        from_face = self._get_face_for_emotion(from_emotion, 0.5)
        to_face = self._get_face_for_emotion(to_emotion, 0.5)

        transition_frames = [
            from_face,
            """
             ···
            \\___/
            *transition*
            """,
            to_face,
        ]

        return transition_frames
