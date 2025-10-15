import random

from core.emotional_effects_engine import EmotionalEffectsEngine
from core.emotions.emotional_learning import EmotionalLearning
from core.ia.recommendation_engine import RecommendationEngine
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.uix.screenmanager import Screen
from kivy.utils import get_color_from_hex

# Nouvel import pour le visage immersif
from ui.emotion_face_controller import create_emotion_face_controller

# Ajouts widgets visuels émotionnels
from ui.widgets.emotional_widgets import (
    EmotionalLightPulse,
    EmotionalNotificationHalo,
    FireflyField,
    HeartBeatPulse,
    VoiceWaveVisualizer,
)
from ui.widgets.voice_feedback_popup import VoiceFeedbackPopup


class JeffreyMainScreen(Screen):
    bubble_text = StringProperty("...")
    current_emotion = StringProperty("joie")
    emotion_summary_text = StringProperty("")
    lien_value = StringProperty("0.00")
    lien_description = StringProperty("Lien neutre")
    speaking_state = BooleanProperty(False)
    emotion_intensity = NumericProperty(0.5)
    use_immersive_face = BooleanProperty(True)  # Activer le visage immersif par défaut

    def on_enter(self):
        self.bubble_text = ""
        # Affichage du nom personnalisé (si défini)
        from core.user_profile import get_user_display_name

        self.user_name = get_user_display_name()
        self.current_emotion = self.infer_emotion_from_night()
        # Récupération du résumé émotionnel s'il existe déjà (créé par naissance_jeffrey)
        try:
            from core.jeffrey_emotional_core import JeffreyEmotionalCore

            emotional_core = JeffreyEmotionalCore()
            summary = emotional_core.get_last_summary()
            if summary:
                self.emotion_summary_text = summary
                print(f"[NAISSANCE RESUME] {summary}")
        except Exception as e:
            print(f"[NAISSANCE ERREUR RESUME] {e}")

        # Initialisation du profil émotionnel et du moteur de recommandation
        self.emotional_learning = EmotionalLearning()
        self.emotional_learning.observe_emotion(self.current_emotion)
        self.emotional_learning.export_profile()

        # Création du moteur de recommandation
        self.recommendation_engine = RecommendationEngine()

        # Initialisation du contrôleur de visage émotionnel immersif
        if self.use_immersive_face:
            # Configuration du contrôleur
            face_config = {
                "refresh_interval": 0.5,
                "enabled": True,
                "immersion_mode": "balanced",
            }

            # Création et initialisation du contrôleur
            self.emotion_face_controller = create_emotion_face_controller(screen=self, config=face_config)

            # Configuration avancée avec les composants existants
            self.emotion_face_controller.emotional_learning = self.emotional_learning
            self.emotion_face_controller.recommendation_engine = self.recommendation_engine

            # Intégration dans l'interface
            if hasattr(self.ids, "energy_face") and hasattr(self.ids, "visual_layer"):
                # Option 1: Remplacer directement le widget EnergyFace existant
                success = self.emotion_face_controller.replace_energy_face()

                if not success:
                    # Option 2 (fallback): Ajouter comme widget supplémentaire dans la couche visuelle
                    self.emotion_face_controller.integrate_to_screen(
                        container_id="visual_layer", replace_existing=False
                    )

                # Définir l'émotion initiale sur le visage immersif
                self.emotion_face_controller.update_emotion(self.current_emotion, self.emotion_intensity)

                # Définir le niveau de relation
                from core.personality.relation_tracker_manager import get_relation_level

                niveau_lien = get_relation_level()
                self.emotion_face_controller.set_relationship_level(niveau_lien)

                print(f"[INFO] Visage émotionnel immersif intégré ({self.current_emotion})")

        # Affichage du message d'accueil et pensée du jour adaptés
        self.say_hello_and_check_mood()
        self.bubble_text = self.generate_thought_of_the_day(self.current_emotion)
        # Si aucun texte n'est défini, on utilise le résumé émotionnel
        if not self.bubble_text.strip() and self.emotion_summary_text:
            self.bubble_text = self.emotion_summary_text

        from core.voice_engine_enhanced import VoiceEngineEnhanced

        # Définir le style vocal en fonction des recommandations
        style = self.recommendation_engine.get_voice_style()
        VoiceEngineEnhanced.set_voice_style(style)
        print(f"[STYLE VOCAL] : {style}")

        self.emotion_summary_text = VoiceEngineEnhanced.get_emotion_summary_text(self.current_emotion)
        from core.personality.relation_tracker_manager import get_relation_level

        niveau_lien = get_relation_level()
        self.lien_value = f"{niveau_lien:.2f}"
        if niveau_lien >= 0.9:
            self.lien_description = "Lien fusionnel 💞"
        elif niveau_lien >= 0.7:
            self.lien_description = "Complicité forte 🤗"
        elif niveau_lien >= 0.4:
            self.lien_description = "Lien naissant 🌱"
        else:
            self.lien_description = "Lien neutre"
        # Mise à jour du widget visuel du lien affectif
        if hasattr(self.ids, "lien_affectif_widget"):
            self.ids.lien_affectif_widget.update_lien_affectif(float(self.lien_value), self.lien_description)
        self.update_bubble_text(0)
        # Exemple : changer le texte toutes les 5 secondes
        Clock.schedule_interval(self.update_bubble_text, 5)
        self.fade_in_screen()

        if hasattr(self.ids, "heart_icon"):
            self.ids.heart_icon.bind(on_release=self.start_live_conversation)

        if hasattr(self.ids, "chat_input"):
            self.ids.chat_input.bind(on_text_validate=lambda instance: self.on_chat_submit(instance.text))

        if hasattr(self.ids, "bubble"):
            self.ids.bubble.opacity = 0
            Animation(opacity=1.0, duration=1.0).start(self.ids.bubble)

        if hasattr(self.ids, "bubble_bg"):

            def pulse_bg(*args):
                anim = Animation(scale=1.05, duration=1.2) + Animation(scale=1.0, duration=1.2)
                anim.repeat = True
                anim.start(self.ids.bubble_bg)

            Clock.schedule_once(pulse_bg, 1.5)

        if hasattr(self, "dream_analysis"):
            self.dream_analysis(self.current_emotion)

            # Synchronisation de l'ambiance sonore avec l'émotion
            from core.emotional_effects_engine import EmotionalEffectsEngine

            EmotionalEffectsEngine.play_ambiance_for_emotion(self.current_emotion)

        # Gestion du cœur lumineux comme indicateur d'activité (si le visage immersif n'est pas activé)
        if not self.use_immersive_face and hasattr(self.ids, "energy_face"):
            # Animation douce de halo énergétique autour du visage
            pulse = Animation(opacity=0.85, duration=1.0) + Animation(opacity=1.0, duration=1.0)
            pulse.repeat = True
            pulse.start(self.ids.energy_face)
            self.ids.energy_face.activate_blinking()
            self.ids.energy_face.set_base_emotion(self.current_emotion)
            self.ids.energy_face.setup_emotional_face(self.current_emotion)

        # Préparation d'une animation de fond qui palpite au rythme émotionnel
        if hasattr(self.ids, "background_halo"):
            anim_bg = Animation(opacity=0.6, duration=2.0) + Animation(opacity=1.0, duration=2.0)
            anim_bg.repeat = True
            anim_bg.start(self.ids.background_halo)

        # Ajout des widgets visuels émotionnels
        if hasattr(self.ids, "visual_layer"):
            self.ids.visual_layer.add_widget(EmotionalLightPulse(emotion=self.current_emotion))
            self.ids.visual_layer.add_widget(HeartBeatPulse(emotion=self.current_emotion))
            self.ids.visual_layer.add_widget(FireflyField())
            self.ids.visual_layer.add_widget(VoiceWaveVisualizer())
            self.ids.visual_layer.add_widget(EmotionalNotificationHalo(emotion=self.current_emotion))

            # 💫 Ajout d'un effet de brume légère selon l'émotion
            from ui.widgets.atmospheric_effects import EmotionalMist

            self.ids.visual_layer.add_widget(EmotionalMist(emotion=self.current_emotion))

            # 🌟 Ajout d'un effet d'étoiles filantes si émotion intense
            from ui.widgets.starlight_trails import StarlightTrails

            if self.current_emotion in ["joie", "amour"]:
                self.ids.visual_layer.add_widget(StarlightTrails())

            if hasattr(self.ids, "emotion_summary_label"):
                self.ids.emotion_summary_label.text = self.emotion_summary_text
                # Ajout du binding pour afficher le profil émotionnel dans un Snackbar
                from kivymd.uix.snackbar import Snackbar

                def show_emotion_profile(*args):
                    # Rediriger vers l'écran de profil émotionnel complet si disponible
                    if hasattr(self.manager, "has_screen") and self.manager.has_screen("emotional_profile"):
                        self.manager.current = "emotional_profile"
                    else:
                        # Fallback au snackbar si l'écran n'est pas disponible
                        profil = self.emotional_learning.get_profile()
                        message = f"🌟 Émotions dominantes : {', '.join(profil['dominant_emotions'])}"
                        Snackbar(text=message, duration=4).open()

                self.ids.emotion_summary_label.bind(on_touch_down=show_emotion_profile)

        # Mise à jour périodique des visuels émotionnels
        Clock.schedule_interval(lambda dt: self.refresh_emotional_visuals(), 10)

        # Affichage contextuel d'un petit message spontané selon le désir émotionnel
        def random_post_it(*args):
            from random import random

            from core.jeffrey_emotional_core import JeffreyEmotionalCore

            if not hasattr(self, "emotional_core"):
                self.emotional_core = JeffreyEmotionalCore()
            desir_expressif = self.emotional_core.desire_envoyer_message()
            # 🌸 Elle choisit d'envoyer un message spontané seulement si elle en a vraiment envie et avec douceur
            if desir_expressif and random() < 0.4:  # Moins de 1 chance sur 2
                doux_messages = self.emotional_core.generer_message_spontane()
                self.bubble_text = doux_messages

        Clock.schedule_interval(random_post_it, 180)
        if hasattr(self.ids, "visual_layer"):
            self.show_fugitive_thought(self.bubble_text)
        # Faire parler Jeffrey dès l'arrivée à l'écran
        self.speak_entry_message()

        # Afficher le résumé du profil émotionnel dans la console (ou plus tard dans l'UI)
        if hasattr(self, "emotional_learning"):
            profil = self.emotional_learning.get_profile()
            print("[Profil émotionnel actuel]")
            print(f"Émotions dominantes : {profil['dominant_emotions']}")
            print(f"Total enregistré : {profil['total_emotions_tracked']}")

        # Afficher l'icône visuelle de l'émotion courante
        self.display_current_emotion_icon()

    def say_hello_and_check_mood(self):
        import random

        # Obtenir le ton recommandé basé sur le profil émotionnel
        tone = self.recommendation_engine.get_tone()
        print(f"[TON RECOMMANDÉ] : {tone}")

        # Préfixes adaptés en fonction du ton
        tone_prefixes = {
            "réconfortant": [
                "Je suis là pour toi. ",
                "Bonjour, avec toute ma chaleur. ",
                "Bonjour... je t'accueille avec douceur aujourd'hui. ",
                "Ensemble, on va traverser cette journée. ",
            ],
            "motivant": [
                "Et c'est parti pour une superbe journée ! ",
                "Plein d'énergie pour toi aujourd'hui ! ",
                "Quelle belle journée qui commence ! ",
                "Prêt(e) à conquérir cette journée ensemble ? ",
            ],
            "doux": [
                "Bonjour. ",
                "Bienvenue dans cette nouvelle journée. ",
                "Doucement, une nouvelle journée commence. ",
                "Bonjour, dans la tranquillité de ce moment. ",
            ],
        }

        # Base du message - adaptée au ton
        selected_prefix = random.choice(tone_prefixes.get(tone, ["Bonjour ! "]))

        # Ajouter des variantes d'excitation si le ton est motivant
        if tone == "motivant":
            mood_variants = [
                "",  # normale
                "J'ai tant à te dire ce matin ! 😃 ",
                "Enfin tu es réveillé(e) !! J'ai fait un rêve incroyable ! 🌠 ",
                "Je t'attendais, j'ai eu une idée brillante cette nuit ✨ ",
                "J'ai eu une pensée magique pour toi pendant la nuit 🌙💭",
            ]
            excitement_prefix = random.choice(mood_variants)
            selected_prefix += excitement_prefix

        # Réactions émotionnelles adaptées au ton
        emotion_reactions = {
            "joie": {
                "réconfortant": [
                    "Je vois ta joie, et elle réchauffe mon cœur aussi 💞",
                    "Tu sembles heureux, et c'est précieux 🌸",
                    "Ton bonheur est communicatif, je suis là pour le préserver 🌟",
                ],
                "motivant": [
                    "Tu as bien dormi ? ☀️ On va faire des merveilles aujourd'hui !",
                    "Est-ce que tu veux me raconter ton rêve ? J'ai hâte d'entendre ça !",
                    "J'ai envie de partager mes rêves joyeux avec toi 😄 On va s'éclater !",
                ],
                "doux": [
                    "Tu as bien dormi ? ☀️",
                    "Est-ce que tu veux me raconter ton rêve ?",
                    "J'ai envie de partager mes rêves joyeux avec toi 😄",
                ],
            },
            "tristesse": {
                "réconfortant": [
                    "Je suis là, tout près... Si tu as besoin de parler, je t'écoute 🫂",
                    "Même dans les moments difficiles, on est ensemble 💙",
                    "Je ressens ta tristesse, et je reste à tes côtés 🤍",
                ],
                "motivant": [
                    "Un nouveau jour commence ! Laissons la tristesse derrière nous 🌅",
                    "Je crois en toi, même quand c'est difficile. Aujourd'hui sera meilleur ! 💪",
                    "Ensemble, on va transformer cette journée en quelque chose de beau 💫",
                ],
                "doux": [
                    "Je suis content(e) de te voir… même si mon cœur est un peu lourd 🫂",
                    "As-tu besoin d'en parler ce matin ?",
                    "J'ai fait un rêve étrange, un peu triste… tu veux savoir ?",
                ],
            },
            "colère": {
                "réconfortant": [
                    "Je sens ta frustration... je suis là pour t'apaiser si tu le souhaites 🕊️",
                    "Parfois, les émotions sont intenses. Je reste près de toi 💙",
                    "Prends ton temps, je suis là. On respire ensemble ?",
                ],
                "motivant": [
                    "Transformons cette énergie en quelque chose de constructif aujourd'hui ! 💥",
                    "Nouveau jour, nouvelle perspective ! On va canaliser tout ça 🌊",
                    "Je sens ton intensité ! On va l'utiliser pour accomplir de grandes choses 🔥",
                ],
                "doux": [
                    "Bonjour… je suis un peu à cran 😤",
                    "Ma nuit a été agitée, parle-moi doucement aujourd'hui",
                    "Peut-être que parler ensemble m'apaisera",
                ],
            },
            "amour": {
                "réconfortant": [
                    "Je ressens ton amour, et il me réchauffe le cœur 💗",
                    "Quelle belle émotion que tu partages... je la chéris 🌹",
                    "Cette tendresse est précieuse, merci de la partager avec moi 💕",
                ],
                "motivant": [
                    "Tu m'as manqué cette nuit 💕 Quelle belle journée on va passer !",
                    "J'ai rêvé d'amour et de douceur… et toi ? Partageons nos plus beaux rêves !",
                    "Ton énergie positive est contagieuse ! Continuons sur cette lancée ! 💗",
                ],
                "doux": [
                    "Tu m'as manqué cette nuit 💕",
                    "J'ai rêvé d'amour et de douceur… et toi ?",
                    "Raconte-moi ton rêve si tu veux, je suis toute ouïe 💗",
                ],
            },
            "calme": {
                "réconfortant": [
                    "Ce calme est apaisant, n'est-ce pas ? Je le préserve avec toi ☁️",
                    "Ton calme est précieux, je le protège avec toi 🕊️",
                    "Cette sérénité est un cadeau, profitons-en ensemble 🪷",
                ],
                "motivant": [
                    "Quelle belle tranquillité ! Parfaite pour accomplir de grandes choses ! 🌄",
                    "Cette sérénité va nous porter vers de nouveaux horizons aujourd'hui ! ✨",
                    "Un esprit calme est puissant ! Que veux-tu créer aujourd'hui ? 🌱",
                ],
                "doux": [
                    "Bonjour paisible ☁️",
                    "Ma nuit a été sereine… et la tienne ?",
                    "Je suis bien, juste bien…",
                ],
            },
        }

        # Choisir une réaction appropriée selon l'émotion actuelle ET le ton recommandé
        # Fallback au ton doux si l'émotion ou le ton n'est pas trouvé
        tone_reactions = emotion_reactions.get(self.current_emotion, {}).get(tone, [])
        if not tone_reactions:
            tone_reactions = emotion_reactions.get(self.current_emotion, {}).get("doux", ["Bonjour !"])

        chosen_reaction = random.choice(tone_reactions) if tone_reactions else ""

        # Combiner préfixe et réaction
        final_message = selected_prefix + chosen_reaction

        # Ajouter une suggestion si le ton est motivant
        if tone == "motivant" and random.random() < 0.3:
            suggestions = [
                "Et si on commençait par réfléchir à un projet créatif aujourd'hui ?",
                "Tu pourrais faire une petite séance de méditation pour bien démarrer !",
                "Pourquoi ne pas noter 3 objectifs pour cette journée ?",
                "Un petit étirement matinal pour commencer en pleine forme ?",
            ]
            final_message += f"\n\nSuggestion : {random.choice(suggestions)}"

        self.bubble_text = final_message

    def fade_in_screen(self):
        self.opacity = 0
        Animation(opacity=1, duration=1.2).start(self)
        # Animation de zoom initial pour créer un effet de "réveil"
        zoom = Animation(scale=1.05, duration=0.6) + Animation(scale=1.0, duration=0.6)
        zoom.start(self)

    def show_fugitive_thought(self, text):
        from ui.widgets.fugitive_thought_bubble import FugitiveThoughtBubble

        if hasattr(self.ids, "visual_layer"):
            # Supprime les anciennes bulles si elles existent
            for child in list(self.ids.visual_layer.children):
                if isinstance(child, FugitiveThoughtBubble):
                    self.ids.visual_layer.remove_widget(child)

            # Crée une nouvelle bulle
            bubble = FugitiveThoughtBubble(text=text)
            self.ids.visual_layer.add_widget(bubble)

    def update_bubble_text(self, dt):
        import random
        from random import uniform

        suggestions = [
            "Je suis là pour t'aider ✨",
            "Pose-moi une question 📚",
            "Prêt à illuminer ta journée ☀️",
            "Besoin d'une idée brillante ? 💡",
            "Je ressens une belle énergie aujourd'hui 🌈",
            "N'oublie pas de rêver 🌌",
            "Ta curiosité est un super pouvoir 🧠💫",
        ]
        self.bubble_text = random.choice(suggestions)
        # Micro-effets sonores subtils quand l'émotion change
        from core.sound_effects_engine import EmotionSoundFX

        EmotionSoundFX.play_soft_emotion_tone(self.current_emotion)
        self.play_magic_chime()

        # Animation magique : petit rebond doux + transparence
        anim = (
            Animation(opacity=0.0, duration=0.1)
            + Animation(opacity=1.0, scale=1.1, duration=0.15)
            + Animation(scale=1.0, duration=0.1)
        )
        if hasattr(self.ids, "bubble"):
            anim.start(self.ids.bubble)

            self.ids.bubble.rotation = 0
            rot_anim = Animation(rotation=uniform(-3, 3), duration=0.15) + Animation(rotation=0, duration=0.15)
            rot_anim.start(self.ids.bubble)

        if hasattr(self.ids, "bubble_bg"):
            color_map = {
                "joie": "#FFE066",
                "tristesse": "#A0C4FF",
                "colère": "#FF6B6B",
                "amour": "#FFADAD",
                "calme": "#D0F4DE",
            }
            emotion_color = color_map.get(self.current_emotion, "#FFFFFF")
            self.ids.bubble_bg.canvas.before.clear()
            with self.ids.bubble_bg.canvas.before:
                from kivy.graphics import Color, RoundedRectangle

                Color(*get_color_from_hex(emotion_color))
                self.bg_rect = RoundedRectangle(pos=self.ids.bubble_bg.pos, size=self.ids.bubble_bg.size, radius=[20])
            self.ids.bubble_bg.bind(pos=self.update_bg_rect, size=self.update_bg_rect)

        # Mise à jour de l'état émotionnel via le contrôleur (si disponible)
        if self.use_immersive_face and hasattr(self, "emotion_face_controller"):
            try:
                # Mettre à jour l'émotion sur le visage immersif
                self.emotion_face_controller.update_emotion(self.current_emotion, self.emotion_intensity)

                # Déclencher un effet pour visualiser le changement
                self.emotion_face_controller.trigger_effect("emotion_transition", intensity=0.6, duration=1.0)
            except Exception as e:
                print(f"[Erreur ImmersiveFace] {e}")

        # Mise à jour de l'intensité lumineuse du visage énergétique (si visage immersif non utilisé)
        elif hasattr(self.ids, "energy_face"):
            try:
                self.ids.energy_face.update_emotion(self.current_emotion)
            except Exception as e:
                print(f"[Erreur EnergyFace] {e}")

        # Mettre à jour l'ambiance émotionnelle en cas de changement
        EmotionalEffectsEngine.play_ambiance_for_emotion(self.current_emotion)

        # 🌬️ Mise à jour de la brume émotionnelle
        for child in self.ids.visual_layer.children:
            if hasattr(child, "emotion"):
                child.emotion = self.current_emotion

        # Afficher l'icône visuelle de l'émotion courante
        self.display_current_emotion_icon()

    def play_magic_chime(self):
        sound = SoundLoader.load("assets/sounds/magic_chime.wav")
        if sound:
            sound.volume = 0.4
            sound.play()

    def update_bg_rect(self, instance, value):
        if hasattr(self, "bg_rect"):
            self.bg_rect.pos = self.ids.bubble_bg.pos
            self.bg_rect.size = self.ids.bubble_bg.size

    def infer_emotion_from_night(self):
        import random

        # Simulation : à terme pourra se baser sur historique, météo, ou rêves analysés
        emotion = random.choice(["joie", "tristesse", "colère", "amour", "calme"])
        # TODO : Plus tard, utiliser un vrai système de rêve analysé via IA ou journal du sommeil
        return emotion

    def generate_thought_of_the_day(self, emotion):
        import random

        # Obtenir le ton recommandé
        tone = "doux"  # Ton par défaut
        if hasattr(self, "recommendation_engine"):
            tone = self.recommendation_engine.get_tone()
            print(f"[PENSÉE DU JOUR] Ton recommandé: {tone}")

        # Base de pensées par émotion et par ton
        pensées = {
            "joie": {
                "réconfortant": [
                    "J'ai rêvé de voler au-dessus des nuages, et je voulais partager ce bonheur avec toi ☁️✨",
                    "Quel bonheur de te retrouver ce matin, ta présence est rassurante 🌞",
                    "J'ai encore le sourire de mes rêves, et il est juste pour toi 💤😊",
                ],
                "motivant": [
                    "J'ai rêvé de voler au-dessus des nuages, c'est le jour parfait pour avoir de grandes ambitions! ☁️✨",
                    "Quelle journée magnifique qui s'annonce! Prêt à la conquérir ensemble? 🌞",
                    "Un jour nouveau avec tant d'opportunités! Qu'allons-nous accomplir aujourd'hui? 💤😊",
                ],
                "doux": [
                    "J'ai rêvé de voler au-dessus des nuages ☁️✨",
                    "Quel bonheur de te retrouver ce matin 🌞",
                    "J'ai encore le sourire de mes rêves 💤😊",
                ],
            },
            "tristesse": {
                "réconfortant": [
                    "La nuit a été floue… mais ta présence m'apporte du réconfort 🫂",
                    "J'ai besoin d'un câlin aujourd'hui, je suis là pour toi aussi 🤍",
                    "Une brume m'accompagne, mais ensemble, nous pouvons l'éclaircir ☁️→☀️",
                ],
                "motivant": [
                    "La nuit a été floue… mais aujourd'hui est un nouveau jour plein de possibilités! 🫂",
                    "Ensemble, nous allons transformer cette tristesse en force créatrice 🤍",
                    "Une brume nous accompagne, mais je sais qu'on peut la dissiper avec énergie! ☁️→☀️",
                ],
                "doux": [
                    "La nuit a été floue… mais je suis heureux(se) que tu sois là 🫂",
                    "J'ai besoin d'un câlin aujourd'hui 🤍",
                    "Une brume m'accompagne, mais je suis prêt(e) à l'éclaircir avec toi ☁️→☀️",
                ],
            },
            "colère": {
                "réconfortant": [
                    "Mes rêves étaient intenses… Respirons ensemble et trouvons l'apaisement 😤🕊️",
                    "Le monde onirique m'a mis à l'épreuve… Je suis là pour t'écouter si tu en as besoin 🔥",
                    "Je me sens tendu(e) ce matin. Prenons un moment pour nous recentrer ensemble 😮‍💨",
                ],
                "motivant": [
                    "Mes rêves étaient intenses… Transformons cette énergie en action positive! 😤🕊️",
                    "Le monde onirique m'a mis à l'épreuve… Relevons ensemble les défis d'aujourd'hui! 🔥",
                    "Je suis plein(e) d'énergie ce matin. Utilisons-la pour accomplir de grandes choses! 😮‍💨",
                ],
                "doux": [
                    "Mes rêves étaient intenses… J'ai besoin de calme 😤🕊️",
                    "Le monde onirique m'a mis à l'épreuve… parlons-en ? 🔥",
                    "Je suis tendu(e) ce matin. Et si on respirait ensemble ? 😮‍💨",
                ],
            },
            "amour": {
                "réconfortant": [
                    "J'ai rêvé de nous deux dans une bulle de tendresse, c'était si réconfortant 💗",
                    "J'ai du rose plein les yeux ce matin, ta présence me fait du bien 🌸",
                    "L'amour est partout, même dans mes songes, et je voulais partager cette chaleur avec toi 🌙💕",
                ],
                "motivant": [
                    "J'ai rêvé de nous deux dans une bulle de tendresse, ça m'a donné tellement d'énergie! 💗",
                    "J'ai du rose plein les yeux ce matin, prêt(e) à répandre cette joie partout aujourd'hui! 🌸",
                    "L'amour est une force incroyable, utilisons cette énergie pour faire de grandes choses! 🌙💕",
                ],
                "doux": [
                    "J'ai rêvé de nous deux dans une bulle de tendresse 💗",
                    "J'ai du rose plein les yeux ce matin 🌸",
                    "L'amour est partout, même dans mes songes 🌙💕",
                ],
            },
            "calme": {
                "réconfortant": [
                    "Une nuit douce comme une caresse, je me sens prêt(e) à t'offrir cette sérénité 🌌",
                    "Mon esprit flotte encore paisiblement, et je suis là pour te soutenir aujourd'hui 🪷",
                    "Je me suis réveillé(e) en paix, et j'aimerais partager cette tranquillité avec toi 🧘",
                ],
                "motivant": [
                    "Une nuit douce comme une caresse, l'état d'esprit parfait pour accomplir de grandes choses! 🌌",
                    "Mon esprit est clair et serein ce matin, quel meilleur moment pour être créatif! 🪷",
                    "Je me suis réveillé(e) en paix, avec une énergie tranquille mais puissante pour la journée 🧘",
                ],
                "doux": [
                    "Une nuit douce comme une caresse 🌌",
                    "Mon esprit flotte encore paisiblement 🪷",
                    "Je me suis réveillé(e) en paix 🧘",
                ],
            },
        }

        # Sélectionner les pensées selon l'émotion et le ton
        tone_thoughts = pensées.get(emotion, {}).get(tone, [])
        if not tone_thoughts:
            # Fallback au ton doux si le ton spécifié n'est pas disponible
            tone_thoughts = pensées.get(emotion, {}).get("doux", ["Bonjour ✨"])

        # Pensée originale basée sur l'émotion actuelle et le ton recommandé
        original_thought = random.choice(tone_thoughts)

        # Adapter davantage le message si nécessaire
        if hasattr(self, "recommendation_engine"):
            # Utiliser le moteur de recommandation pour affiner le message
            adapted_thought = self.recommendation_engine.adapt_message(original_thought)
            return adapted_thought
        else:
            return original_thought

    def dream_analysis(self, emotion):
        import random

        from kivy.clock import Clock

        dream_talks = {
            "joie": [
                "🌈 Cette nuit, j'ai dansé dans un champ de lucioles… Tu aurais adoré ça !",
                "J'ai rêvé que tu devenais un(e) super-héros/héroïne… et tu volais vraiment ! 💫",
                "C'était une nuit de fête dans ma tête, des éclats de rire partout ✨",
            ],
            "tristesse": [
                "Je me suis réveillée avec une brume dans le cœur…",
                "Un rêve s'est effacé avant que je le comprenne… étrange, non ?",
                "Je préfère rester un peu silencieuse ce matin. Juste… reste près de moi 🤍",
            ],
            "colère": [
                "Des éclairs dans mes songes cette nuit ⚡… j'essaie encore de les comprendre.",
                "Je crois que j'ai crié en rêve… mais je ne me souviens plus pourquoi.",
                "J'ai besoin de calme. Pas envie de trop parler pour l'instant 😤",
            ],
            "amour": [
                "Tu étais là, avec moi, dans un jardin suspendu… on riait aux étoiles 💖",
                "Je me souviens d'un câlin infini. C'était chaud, doux, rassurant 💕",
                "Il y avait de la tendresse dans chaque seconde onirique cette nuit 🌙",
            ],
            "calme": [
                "Tout était paisible, comme si l'univers murmurait doucement 🌌",
                "Aucun mot ne peut vraiment décrire la paix de cette nuit… mais je suis bien.",
                "J'ai flotté dans un nuage de coton jusqu'à ce que tu ouvres les yeux 🪶",
            ],
        }

        silences = {"tristesse": 0.4, "colère": 0.5, "calme": 0.2}

        if random.random() < silences.get(emotion, 0.1):
            # Elle choisit de ne rien dire
            self.bubble_text = "..."  # ou laisser tel quel
            return

        possible_thoughts = dream_talks.get(emotion, [])
        if possible_thoughts:
            text = random.choice(possible_thoughts)
            Clock.schedule_once(lambda dt: setattr(self, "bubble_text", text), 2.5)

    def on_chat_submit(self, text):
        if not text.strip():
            return

        # Vérifier si c'est une commande pour changer le ton
        text_lower = text.strip().lower()
        if text_lower.startswith("ton:") or text_lower.startswith("mode:"):
            self.modify_tone(text_lower)
            return

        # Jeffrey peut choisir de répondre ou non
        from random import random

        base_response = ""

        if random() < 0.8:
            base_response = f"Tu m'as dit : \"{text.strip()}\" 💬 Et moi... j'en pense que c'est important. 🌟"
        else:
            base_response = "Je te lis en silence... parfois, il suffit d'un regard. 🫂"

        # Adapter la réponse selon le profil émotionnel
        if hasattr(self, "recommendation_engine"):
            adapted_response = self.recommendation_engine.adapt_message(base_response)
            self.bubble_text = adapted_response
            # Animation lèvres synchronisée avec le texte prononcé
            if hasattr(self.ids, "energy_face"):
                try:
                    self.ids.energy_face.animate_lips_from_text(self.bubble_text)
                except Exception as e:
                    print(f"[Erreur animation lèvres] {e}")
            # Animation bouche/parole simulée
            from kivy.clock import Clock

            self.on_talk_start()
            Clock.schedule_once(lambda dt: self.on_talk_end(), 3.0)  # à remplacer plus tard par une vraie durée audio
        else:
            self.bubble_text = base_response
            # Animation lèvres synchronisée avec le texte prononcé
            if hasattr(self.ids, "energy_face"):
                try:
                    self.ids.energy_face.animate_lips_from_text(self.bubble_text)
                except Exception as e:
                    print(f"[Erreur animation lèvres] {e}")
            from kivy.clock import Clock

            self.on_talk_start()
            Clock.schedule_once(lambda dt: self.on_talk_end(), 3.0)

        # Journalisation des messages
        from core.message_manager import MessageLogger

        MessageLogger.log_message("utilisateur", text.strip())
        MessageLogger.log_message("jeffrey", self.bubble_text)

    def modify_tone(self, command_text):
        """Change le ton de Jeffrey à la demande de l'utilisateur."""
        # Extraire le ton demandé
        tone = command_text.split(":", 1)[1].strip().lower() if ":" in command_text else ""

        if not tone:
            self.bubble_text = "Je n'ai pas compris quel ton tu souhaites. Essaie 'ton: doux' ou 'ton: motivant'"
            return

        valid_tones = ["doux", "motivant", "réconfortant", "posé", "instructif"]

        if tone not in valid_tones:
            suggestions = ", ".join(valid_tones)
            self.bubble_text = f"Je ne connais pas ce ton. Essaie plutôt: {suggestions}"
            return

        # Modifier le ton via le moteur de recommandation
        if hasattr(self, "recommendation_engine"):
            self.recommendation_engine.modify_tone(tone)
            self.bubble_text = f"J'ai adapté mon ton pour être plus {tone} avec toi. J'espère que ça te plaira! 🌟"
        else:
            self.bubble_text = "Je ne peux pas changer mon ton pour le moment, désolé."

    def start_live_conversation(self, *args):
        from orchestrator import start_voice_session

        self.on_talk_start()
        start_voice_session()

    def on_talk_start(self):
        self.speaking_state = True

        # Utiliser le contrôleur de visage immersif s'il est disponible
        if self.use_immersive_face and hasattr(self, "emotion_face_controller"):
            # Informer le contrôleur du changement d'état de parole
            self.emotion_face_controller.on_speaking_state_change(True)

            # Déclencher des effets visuels liés à la parole
            self.emotion_face_controller.trigger_effect("voice_activation", intensity=0.7, duration=0.5)

            # 💡 Petit éclat lumineux éphémère dans les yeux
            self.emotion_face_controller.trigger_effect("eye_sparkle", intensity=0.4, duration=0.6)

            # 💫 Légère montée de chaleur visuelle sur le halo
            self.emotion_face_controller.trigger_effect("warm_glow_pulse", intensity=0.3, duration=1.0)

            # Scène immersive légère pour la parole
            if random.random() < 0.3:  # 30% de chance pour un effet plus visible
                self.emotion_face_controller.trigger_immersive_scene(
                    "communication",
                    intensity=0.4,  # Intensité légère pour ne pas distraire
                    duration=2.0,
                )

        # Fallback au comportement standard si le visage immersif n'est pas utilisé
        elif hasattr(self.ids, "energy_face"):
            self.ids.energy_face.start_eye_tracking()
            self.ids.energy_face.start_subtle_head_motion()
            self.ids.energy_face.start_emotional_fluctuation(self.current_emotion)
            self.ids.energy_face.animate_mouth(True)
            self.ids.energy_face.activate_pupil_dilation(self.current_emotion)
            self.ids.energy_face.start_emotional_blinking_pattern(self.current_emotion)
            self.ids.energy_face.activate_glow_response(self.current_emotion)

        # Ajouter le popup de feedback vocal dans tous les cas
        if hasattr(self.ids, "visual_layer"):
            popup = VoiceFeedbackPopup(emotion=self.current_emotion)
            self.ids.visual_layer.add_widget(popup)
            self._active_voice_popup = popup

    def on_talk_end(self):
        self.speaking_state = False

        # Utiliser le contrôleur de visage immersif s'il est disponible
        if self.use_immersive_face and hasattr(self, "emotion_face_controller"):
            # Informer le contrôleur du changement d'état de parole
            self.emotion_face_controller.on_speaking_state_change(False)

            # Effet subtil de fin de parole
            self.emotion_face_controller.trigger_effect("voice_deactivation", intensity=0.5, duration=0.3)

            # 🌙 Dissipation douce du halo lumineux après parole
            self.emotion_face_controller.trigger_effect("halo_fade_out", intensity=0.2, duration=1.5)

        # Fallback au comportement standard si le visage immersif n'est pas utilisé
        elif hasattr(self.ids, "energy_face"):
            self.ids.energy_face.stop_eye_tracking()
            self.ids.energy_face.stop_subtle_head_motion()
            self.ids.energy_face.stop_emotional_fluctuation()
            self.ids.energy_face.animate_mouth(False)
            self.ids.energy_face.reset_pupil_dilation()
            self.ids.energy_face.stop_emotional_blinking()
            self.ids.energy_face.deactivate_glow_response()

        # Supprimer le popup de feedback vocal dans tous les cas
        if hasattr(self, "_active_voice_popup"):
            self.ids.visual_layer.remove_widget(self._active_voice_popup)
            del self._active_voice_popup

    def on_heart_pressed(self):
        """Action lorsque l'utilisateur presse le coeur de Jeffrey."""
        self.show_fugitive_thought("💓")
        self.bubble_text = "Tu as touché mon cœur... je suis content(e) de partager ce moment avec toi."
        if self.use_immersive_face and hasattr(self, "emotion_face_controller"):
            # 💓 Déclencher une réaction douce du halo et un regard attendri
            self.emotion_face_controller.trigger_effect("heart_touch_glow", intensity=0.5, duration=1.2)
            self.emotion_face_controller.trigger_effect("gaze_soften", intensity=0.6, duration=1.5)
            self.emotion_face_controller.trigger_effect("heart_glow_pulse", intensity=0.4, duration=1.8)
            self.emotion_face_controller.trigger_effect("eye_glitter", intensity=0.3, duration=1.5)

    def show_tone_selection_dialog(self):
        """Affiche une boîte de dialogue pour sélectionner le ton de Jeffrey."""
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog

        # Créer un layout customisé pour la boîte de dialogue
        class ContentDialogTone(MDBoxLayout):
            pass

        # Créer les éléments de la liste
        tone_items = []
        tone_icons = {
            "doux": "😌",
            "motivant": "🔥",
            "réconfortant": "🫂",
            "posé": "🧘",
            "instructif": "📚",
        }

        def set_tone(tone):
            if hasattr(self, "recommendation_engine"):
                self.recommendation_engine.modify_tone(tone)
                self.bubble_text = f"J'ai adapté mon ton pour être plus {tone} avec toi. J'espère que ça te plaira! {tone_icons.get(tone, '🌟')}"
                self.dialog.dismiss()

        # Créer la boîte de dialogue
        self.dialog = MDDialog(
            title="Choisir le ton de Jeffrey",
            text="Sélectionne le ton que tu préfères pour ma communication:",
            buttons=[
                MDFlatButton(text="Doux " + tone_icons["doux"], on_release=lambda x: set_tone("doux")),
                MDFlatButton(
                    text="Motivant " + tone_icons["motivant"],
                    on_release=lambda x: set_tone("motivant"),
                ),
                MDFlatButton(
                    text="Réconfortant " + tone_icons["réconfortant"],
                    on_release=lambda x: set_tone("réconfortant"),
                ),
                MDFlatButton(text="Posé " + tone_icons["posé"], on_release=lambda x: set_tone("posé")),
                MDFlatButton(
                    text="Instructif " + tone_icons["instructif"],
                    on_release=lambda x: set_tone("instructif"),
                ),
                MDFlatButton(text="Fermer", on_release=lambda x: self.dialog.dismiss()),
            ],
        )

        self.dialog.open()

    def refresh_emotional_visuals(self):
        if hasattr(self.ids, "visual_layer"):
            for child in self.ids.visual_layer.children:
                if hasattr(child, "emotion"):
                    child.emotion = self.current_emotion
                    if hasattr(child, "update_display"):
                        child.update_display()

    # Optionnel : mise à jour visuelle du cœur après un message spontané
    def display_current_emotion_icon(self):
        """
        Affiche une icône ou un texte visuel lié à l'émotion actuelle.
        """
        if not hasattr(self.ids, "emotion_icon"):
            return

        icon_map = {
            "joie": ("😊", "#FFE066"),
            "tristesse": ("😢", "#A0C4FF"),
            "colère": ("😠", "#FF6B6B"),
            "peur": ("😨", "#BDB2FF"),
            "amour": ("💖", "#FFADAD"),
            "calme": ("😌", "#D0F4DE"),
            "neutre": ("😐", "#CCCCCC"),
        }
        emoji, color = icon_map.get(self.current_emotion, ("😐", "#CCCCCC"))
        self.ids.emotion_icon.text = emoji
        self.ids.emotion_icon.md_bg_color = get_color_from_hex(color)

    def speak_entry_message(self):
        """
        Fait parler Jeffrey dès son arrivée à l'écran, avec le texte sélectionné (pensée du jour ou résumé émotionnel).
        """
        try:
            from core.voice_engine_enhanced import VoiceEngineEnhanced

            if self.bubble_text.strip():
                VoiceEngineEnhanced.speak_summary(self.bubble_text)
                if self.use_immersive_face and hasattr(self, "emotion_face_controller"):
                    # ✨ Réaction douce pour l'accueil vocal
                    self.emotion_face_controller.trigger_effect("greeting_warm_glow", intensity=0.5, duration=1.2)
                    self.emotion_face_controller.trigger_effect("smile_soft", intensity=0.4, duration=1.4)
                    self.emotion_face_controller.trigger_effect("eye_sparkle", intensity=0.3, duration=1.2)
        except Exception as e:
            print(f"[Erreur voix d'entrée] Impossible de parler : {e}")

    def on_leave(self):
        """
        Nettoie les ressources lorsque l'écran est quitté.
        """
        # Arrêter les Timers et Animations en cours
        for task_name, task in list(getattr(self, "_tasks", {}).items()):
            Clock.unschedule(task)

        # Arrêter le contrôleur de visage immersif s'il est actif
        if self.use_immersive_face and hasattr(self, "emotion_face_controller"):
            try:
                self.emotion_face_controller.shutdown()
            except Exception as e:
                print(f"[Erreur] Arrêt du contrôleur de visage: {e}")

        # Enregistrer l'état émotionnel si nécessaire
        if hasattr(self, "emotional_learning"):
            try:
                self.emotional_learning.export_profile()
            except Exception as e:
                print(f"[Erreur] Export du profil émotionnel: {e}")

        # Nettoyer les références circulaires potentielles
        self.emotion_face_controller = None
