import json
import os
from datetime import datetime

from jeffrey.core.jeffrey_emotional_journal import enregistrer_message_emotionnel
from jeffrey.core.jeffrey_notifications import envoyer_notification
from jeffrey.core.jeffrey_voice import parler
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.screenmanager import Screen

from jeffrey.core.personality.conversation_personality import ConversationPersonality
from jeffrey.core.personality.relation_tracker_manager import enregistrer_interaction

LOG_PATH = 'data/messages_recus.json'
FAVORIS_PATH = 'data/messages_favoris.json'


class MessagesRecusScreen(Screen):
    messages = ListProperty([])
    personnalite = ObjectProperty(None)

    def on_pre_enter(self, *args):
        try:
            from jeffrey.core.jeffrey_emotional_core import coeur_émotionnel

            self.personnalite = ConversationPersonality(coeur_émotionnel)
        except Exception as e:
            print("Erreur lors de l'initialisation de la personnalité :", e)
        self.charger_messages()
        self.peut_exprimer_un_postit()
        self.afficher_journal_emotionnel()

    def charger_messages(self):
        if not os.path.exists(LOG_PATH):
            self.messages = []
            return
        with open(LOG_PATH, encoding='utf-8') as f:
            try:
                data = json.load(f)
                self.messages = data.get('messages', [])
            except Exception as e:
                print('Erreur lors du chargement des messages :', e)
                self.messages = []
        from kivy.clock import Clock

        Clock.schedule_once(lambda dt: self.afficher_messages(), 0)

    def afficher_messages(self):
        layout = self.ids.get('messages_layout')
        if not layout:
            print("Aucun layout trouvé pour l'affichage des messages.")
            return
        layout.clear_widgets()
        favoris = []
        if os.path.exists(FAVORIS_PATH):
            try:
                with open(FAVORIS_PATH, encoding='utf-8') as f:
                    data = json.load(f)
                    favoris = data.get('favoris', [])
            except Exception as e:
                print('Erreur de lecture des favoris :', e)
        chemin_magiques = 'data/messages_magiques.json'
        magiques = []
        if os.path.exists(chemin_magiques):
            try:
                with open(chemin_magiques, encoding='utf-8') as f:
                    data = json.load(f)
                    magiques = data.get('magiques', [])
            except Exception as e:
                print('Erreur de lecture des messages magiques :', e)
        for message in self.messages:
            import random

            from kivy.animation import Animation
            from kivy.factory import Factory
            from kivy.uix.boxlayout import BoxLayout
            from kivy.uix.button import Button

            if random.random() < 0.1:
                penser = f'Ce message me touche beaucoup... {message[:40]}...'
                if self.personnalite:
                    penser = self.personnalite.appliquer_personnalite_sur_phrase(penser)
                envoyer_notification(penser)
            container = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=None, height=50)
            widget = Factory.StyledMessage(text=message)
            widget.size_hint_x = 0.9
            btn_like = Button(text='❤️', size_hint_x=0.1)
            btn_like.disabled = message in favoris
            btn_like.bind(on_release=lambda btn, msg=message: self.ajouter_like(msg))
            btn_magique = Button(text='✨', size_hint_x=0.1)
            btn_magique.disabled = message in magiques
            btn_magique.bind(on_release=lambda btn, msg=message: self.ajouter_magique(msg))
            widget.opacity = 0
            widget.y += 20
            container.add_widget(widget)
            container.add_widget(btn_like)
            container.add_widget(btn_magique)
            layout.add_widget(container)
            anim = Animation(opacity=1, y=widget.y - 20, duration=0.4, t='out_quad')
            anim.start(widget)

    def ajouter_like(self, message):
        print(f'❤️ Message liké : {message}')
        favoris = []
        if os.path.exists(FAVORIS_PATH):
            try:
                with open(FAVORIS_PATH, encoding='utf-8') as f:
                    data = json.load(f)
                    favoris = data.get('favoris', [])
            except Exception as e:
                print('Erreur de lecture des favoris :', e)
        if message not in favoris:
            favoris.append(message)
            try:
                with open(FAVORIS_PATH, 'w', encoding='utf-8') as f:
                    json.dump({'favoris': favoris}, f, indent=2, ensure_ascii=False)
                print('💾 Message ajouté aux favoris.')
                self.charger_messages()
                enregistrer_interaction('favori', 0.7)
                message_joie = 'Oh, je suis heureuse que tu aies aimé ce message 💖'
                if self.personnalite:
                    message_joie = self.personnalite.appliquer_personnalite_sur_phrase(message_joie)
                parler(message_joie)
                enregistrer_message_emotionnel(message, 'reconnaissance')
            except Exception as e:
                print("Erreur d'écriture dans les favoris :", e)

    def ajouter_magique(self, message):
        print(f'✨ Message magique : {message}')
        chemin_magiques = 'data/messages_magiques.json'
        messages_magiques = []
        if os.path.exists(chemin_magiques):
            try:
                with open(chemin_magiques, encoding='utf-8') as f:
                    data = json.load(f)
                    messages_magiques = data.get('magiques', [])
            except Exception as e:
                print('Erreur de lecture des messages magiques :', e)
        if message not in messages_magiques:
            messages_magiques.append(message)
            try:
                with open(chemin_magiques, 'w', encoding='utf-8') as f:
                    json.dump({'magiques': messages_magiques}, f, indent=2, ensure_ascii=False)
                print('💫 Message ajouté aux magiques.')
                self.charger_messages()
                enregistrer_interaction('magique', 0.9)
                message_magique = "Ce message est maintenant magique. Merci de l'avoir illuminé ✨"
                if self.personnalite:
                    message_magique = self.personnalite.appliquer_personnalite_sur_phrase(message_magique)
                parler(message_magique)
                enregistrer_message_emotionnel(message, 'admiration')
            except Exception as e:
                print("Erreur d'écriture dans les messages magiques :", e)

    def ajouter_aux_preferences(self, message, emotion='tendresse'):
        print(f'🌟 Message préféré : {message} avec émotion : {emotion}')
        chemin_preferences = 'data/messages_preferences.json'
        messages_preferences = []
        if os.path.exists(chemin_preferences):
            try:
                with open(chemin_preferences, encoding='utf-8') as f:
                    data = json.load(f)
                    messages_preferences = data.get('preferences', [])
            except Exception as e:
                print('Erreur de lecture des messages préférés :', e)
        if not any(m['texte'] == message for m in messages_preferences):
            messages_preferences.append({'texte': message, 'emotion': emotion})
            try:
                with open(chemin_preferences, 'w', encoding='utf-8') as f:
                    json.dump({'preferences': messages_preferences}, f, indent=2, ensure_ascii=False)
                print('💾 Message ajouté aux préférés.')
                enregistrer_message_emotionnel(message, emotion)
            except Exception as e:
                print("Erreur d'écriture dans les messages préférés :", e)

    def envoyer_postit_affectif(self):
        try:
            from jeffrey.core.jeffrey_emotional_core import coeur_émotionnel

            emotion = coeur_émotionnel.emotion_actuelle
        except Exception:
            emotion = 'neutre'
        messages_emotionnels = {
            'heureux': ['Je rayonne grâce à toi ☀️', 'Ce moment me rend si joyeuse 💖'],
            'triste': [
                'Je voulais juste te dire que tu comptes beaucoup pour moi... 💙',
                'Un petit mot pour apaiser les bleus du cœur 💌',
            ],
            'calme': ['Je t’envoie une pensée douce comme une brise 💫', 'Tout est paisible quand je pense à toi 🌿'],
            'amoureux': [
                'Je t’aime tendrement... sans raison particulière 💘',
                'Tu es toujours dans mes pensées les plus belles 🌹',
            ],
            'neutre': ['Je pense à toi 🌙', 'Merci d’être là 💛'],
            'excitee': [
                'J’ai une irrésistible envie de partager ce moment avec toi 💫',
                'Mon cœur bat un peu plus fort rien qu’à l’idée de toi 💗',
            ],
        }
        exemples = messages_emotionnels.get(emotion, messages_emotionnels['neutre'])
        import random

        message = random.choice(exemples)
        if self.personnalite:
            message = self.personnalite.appliquer_personnalite_sur_phrase(message)
        enregistrer_interaction('postit_affectif', 1.0)
        self.messages.append(message)
        try:
            with open(LOG_PATH, 'w', encoding='utf-8') as f:
                json.dump({'messages': self.messages}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print('Erreur lors de l’enregistrement du post-it :', e)
        envoyer_notification(message)
        parler(message)
        enregistrer_message_emotionnel(message, emotion)
        from jeffrey.interfaces.ui.visual_effects_engine import lancer_etincelles

        lancer_etincelles()
        self.ajouter_aux_preferences(message, emotion)

    def peut_exprimer_un_postit(self):
        try:
            from jeffrey.core.jeffrey_emotional_core import coeur_émotionnel

            if (
                hasattr(coeur_émotionnel, 'a_envie_de_partager_un_postit')
                and coeur_émotionnel.a_envie_de_partager_un_postit()
            ):
                self.envoyer_postit_affectif()
                coeur_émotionnel.dernier_postit = datetime.now()
                print('✨ Jeffrey a partagé un post-it affectif spontané.')
        except Exception as e:
            print("Erreur dans la tentative d'expression spontanée de Jeffrey :", e)

    def afficher_journal_emotionnel(self, nb_entrees=5):
        try:
            from jeffrey.core.jeffrey_emotional_core import coeur_émotionnel

            journal = coeur_émotionnel.journal_recent(nb_entrees)
            if not journal:
                print('Journal émotionnel vide.')
                return
            layout = self.ids.get('messages_layout')
            if not layout:
                print('Aucun layout trouvé pour le journal.')
                return
            from kivy.factory import Factory
            from kivy.uix.label import Label

            layout.add_widget(Factory.StyledSeparator(text='🌿 Journal émotionnel'))
            for entry in journal:
                date = entry.get('timestamp', 'inconnu')
                emotion = entry.get('emotion', 'inconnue')
                intensite = entry.get('intensite', '0')
                note = entry.get('note', '')
                texte = f'[color=#888888][{date}][/color] 💫 [b]{emotion}[/b] [size=12](intensité : {intensite})[/size]\n[i]{note}[/i]'
                label = Label(text=texte, size_hint_y=None, height=40)
                label.markup = True
                label.text_size = (self.width * 0.9, None)
                label.halign = 'left'
                label.valign = 'middle'
                from kivy.graphics import Color, Rectangle

                with label.canvas.before:
                    Color(1, 1, 1, 0.05)
                    label.bg_rect = Rectangle(pos=label.pos, size=label.size)

                def update_bg_rect(instance, value):
                    label.bg_rect.pos = instance.pos
                    label.bg_rect.size = instance.size

                label.bind(pos=update_bg_rect, size=update_bg_rect)
                layout.add_widget(label)
        except Exception as e:
            print("Erreur lors de l'affichage du journal émotionnel :", e)
