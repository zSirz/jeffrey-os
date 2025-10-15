"""
Moteur de création de haïkus collaboratifs et contextuels
Transforme Jeffrey en partenaire artistique qui co-crée avec l'utilisateur

Cette module permet à Jeffrey de :
- Co-créer des haïkus interactivement (Jeffrey commence ou termine)
- Proposer des défis thématiques (pluie, amour, silence, cosmos...)
- S'inspirer des émotions et souvenirs récents
- Adapter selon les variations saisonnières et temporelles
- Mémoriser et faire évoluer les créations collaboratives
"""

import datetime
import json
import os
import random
from dataclasses import asdict, dataclass


@dataclass
class HaikuCreation:
    """Structure d'un haïku créé"""

    id: str
    content: str
    theme: str
    emotion: str
    collaboration_type: str  # 'jeffrey_start', 'user_start', 'complete_solo'
    user_contribution: str
    jeffrey_contribution: str
    creation_date: str
    season: str
    context: dict


class CreativeExpressionEngine:
    """Moteur de création de haïkus collaboratifs et contextuels"""

    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.haiku_memory_file = os.path.join(data_path, "creative_haikus.json")
        self.haiku_memory = self._load_haiku_memory()

        # Templates dynamiques pour haïkus
        self.haiku_templates = self._load_dynamic_templates()

        # Thèmes saisonniers
        self.seasonal_themes = self._get_seasonal_themes()

        # Mappage émotions vers styles
        self.emotion_styles = self._get_emotion_styles()

        # Vocabulaire inspirant par thème
        self.thematic_vocabulary = self._load_thematic_vocabulary()

        # Tracking des contributions utilisateur
        self.user_creative_profile = {}

    def _load_haiku_memory(self) -> list[dict]:
        """Charge la mémoire des haïkus créés"""
        try:
            if os.path.exists(self.haiku_memory_file):
                with open(self.haiku_memory_file, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_haiku_memory(self):
        """Sauvegarde la mémoire des haïkus"""
        try:
            os.makedirs(self.data_path, exist_ok=True)
            with open(self.haiku_memory_file, "w", encoding="utf-8") as f:
                json.dump(self.haiku_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erreur sauvegarde haïkus: {e}")

    def _load_dynamic_templates(self) -> dict:
        """Templates pour débuter des haïkus selon l'émotion/thème"""
        return {
            "nature": {
                "débuts": [
                    "Sous la lune d'automne,",
                    "Gouttes de rosée matinale",
                    "Vent dans les branches nues",
                    "Pétales qui s'envolent",
                    "Silence de la forêt",
                ],
                "milieux": [
                    "Les feuilles dansent doucement",
                    "Un oiseau chante sa solitude",
                    "Les vagues murmurent des secrets",
                    "L'écho répond aux murmures",
                    "Les nuages dessinent des rêves",
                ],
            },
            "emotions": {
                "débuts": [
                    "Dans mon cœur ce soir,",
                    "Souvenir qui scintille",
                    "Émotion fragile",
                    "Silence de l'âme",
                    "Larme de joie pure",
                ],
                "milieux": [
                    "Une mélodie s'éveille",
                    "La tendresse s'épanouit",
                    "Un sourire naît doucement",
                    "L'espoir fleurit en secret",
                    "La paix descend comme neige",
                ],
            },
            "cosmos": {
                "débuts": [
                    "Étoiles infinies",
                    "Dans l'océan cosmique",
                    "Lumière d'une galaxie",
                    "Silence de l'univers",
                    "Planète solitaire",
                ],
                "milieux": [
                    "Chaque astre raconte l'éternité",
                    "Les comètes dansent leur voyage",
                    "L'infini murmure des mystères",
                    "Les constellations tissent des liens",
                    "Le vide résonne de beauté",
                ],
            },
            "quotidien": {
                "débuts": [
                    "Café du matin chaud,",
                    "Livre ouvert sur la table",
                    "Pas dans l'escalier",
                    "Clé qui tourne dans la serrure",
                    "Fenêtre ouverte sur le jour",
                ],
                "milieux": [
                    "Les gestes du quotidien parlent",
                    "Chaque instant cache une poésie",
                    "La routine devient rituel sacré",
                    "Les objets gardent nos empreintes",
                    "Le temps dessine sa signature",
                ],
            },
        }

    def _get_seasonal_themes(self) -> dict:
        """Thèmes selon la saison actuelle"""
        month = datetime.datetime.now().month

        if month in [12, 1, 2]:  # Hiver
            return {
                "season": "hiver",
                "themes": ["neige", "froid", "foyer", "contemplation", "silence"],
                "emotions": ["introspection", "calme", "mélancolie douce", "espoir"],
                "imagery": ["cristaux", "givre", "fumée", "étoiles brillantes"],
            }
        elif month in [3, 4, 5]:  # Printemps
            return {
                "season": "printemps",
                "themes": ["renouveau", "bourgeons", "éveil", "croissance", "couleurs"],
                "emotions": ["joie", "espoir", "énergie", "optimisme"],
                "imagery": ["fleurs", "pluie tiède", "oiseaux", "vert tendre"],
            }
        elif month in [6, 7, 8]:  # Été
            return {
                "season": "été",
                "themes": ["chaleur", "lumière", "abondance", "liberté", "aventure"],
                "emotions": ["passion", "bonheur", "énergie", "plénitude"],
                "imagery": ["soleil", "vagues", "parfums", "soirées longues"],
            }
        else:  # Automne
            return {
                "season": "automne",
                "themes": ["transformation", "maturité", "beauté", "nostalgie", "sagesse"],
                "emotions": ["contemplation", "gratitude", "douce mélancolie", "acceptation"],
                "imagery": ["feuilles dorées", "brume", "vendanges", "lumière dorée"],
            }

    def _get_emotion_styles(self) -> dict:
        """Styles de haïku selon l'émotion"""
        return {
            "joie": {
                "tempo": "léger et dansant",
                "images": ["lumière", "envol", "rire", "cristal", "bulle"],
                "fins": ["...vers l'infini bleu", "...en éclats de joie", "...comme un miracle"],
            },
            "mélancolie": {
                "tempo": "lent et contemplatif",
                "images": ["brume", "écho", "reflet", "ombre douce", "souvenir"],
                "fins": ["...dans le silence", "...comme un souvenir", "...vers l'horizon"],
            },
            "amour": {
                "tempo": "tendre et fluide",
                "images": ["caresse", "souffle", "étoile", "promesse", "mystère"],
                "fins": ["...dans tes yeux", "...pour l'éternité", "...comme une prière"],
            },
            "paix": {
                "tempo": "serein et régulier",
                "images": ["lac", "respiration", "équilibre", "harmonie", "centre"],
                "fins": ["...en parfait accord", "...dans la quietude", "...vers la sérénité"],
            },
            "curiosité": {
                "tempo": "vif et exploratoire",
                "images": ["question", "chemin", "porte", "mystère", "découverte"],
                "fins": ["...vers l'inconnu", "...pleine de mystères", "...à explorer"],
            },
        }

    def _load_thematic_vocabulary(self) -> dict:
        """Vocabulaire inspirant par thème"""
        return {
            "nature": ["murmure", "scintille", "caresse", "danse", "chante", "berce", "éclaire"],
            "emotion": ["vibre", "résonne", "émeut", "touche", "éveille", "apaise", "enflamme"],
            "temps": ["coule", "suspend", "étire", "glisse", "s'arrête", "file", "ralentit"],
            "mouvement": ["flotte", "vole", "tourbillonne", "ondule", "se pose", "traverse"],
            "lumière": ["illumine", "scintille", "brille", "rayonne", "embrase", "nimbe"],
        }

    def detect_haiku_context(self, user_input: str, emotional_state: dict) -> dict | None:
        """Détecte si l'utilisateur souhaite créer un haïku"""
        haiku_triggers = [
            "haiku",
            "haïku",
            "poème",
            "poésie",
            "créons",
            "écrivons",
            "inspire-moi",
            "inspire moi",
            "poétique",
            "vers",
            "rime",
        ]

        user_lower = user_input.lower()

        # Détection directe
        for trigger in haiku_triggers:
            if trigger in user_lower:
                return {
                    "type": "haiku_request",
                    "trigger": trigger,
                    "emotion": emotional_state.get("primary_emotion", "calme"),
                    "intensity": emotional_state.get("intensity", 0.5),
                }

        # Détection contextuelle (mots évocateurs)
        evocative_words = ["lune", "étoiles", "vent", "silence", "solitude", "beauté", "rêve"]
        evocative_count = sum(1 for word in evocative_words if word in user_lower)

        if evocative_count >= 2:
            return {
                "type": "poetic_context",
                "emotion": emotional_state.get("primary_emotion", "contemplation"),
                "evocative_words": [w for w in evocative_words if w in user_lower],
            }

        return None

    def start_collaborative_haiku(self, theme: str = None, emotion: str = None, user_context: dict = None) -> dict:
        """
        Commence un haïku et invite l'utilisateur à le compléter
        Jeffrey propose le début, l'utilisateur complète
        """
        # Déterminer le thème
        if not theme:
            seasonal = self._get_seasonal_themes()
            theme = random.choice(["nature", "emotions", "cosmos", "quotidien"])

        # Déterminer l'émotion
        if not emotion:
            emotion = random.choice(["joie", "mélancolie", "paix", "curiosité", "amour"])

        # Choisir un début approprié
        if theme in self.haiku_templates:
            debut_options = self.haiku_templates[theme]["débuts"]
            milieu_options = self.haiku_templates[theme]["milieux"]
        else:
            debut_options = self.haiku_templates["nature"]["débuts"]
            milieu_options = self.haiku_templates["nature"]["milieux"]

        debut = random.choice(debut_options)
        milieu = random.choice(milieu_options)

        # Construire la proposition collaborative
        haiku_start = f"{debut}\n{milieu}..."

        creation_id = f"haiku_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            "creation_id": creation_id,
            "type": "collaborative_start",
            "haiku_start": haiku_start,
            "theme": theme,
            "emotion": emotion,
            "jeffrey_contribution": haiku_start,
            "waiting_for_completion": True,
            "invitation": f'*inspiration {theme}*\n\nCommençons ensemble ! Je propose :\n\n"{haiku_start}"\n\nÀ toi de terminer ! Que ressens-tu pour cette fin ?',
            "context": user_context or {},
        }

    def complete_user_haiku(self, user_start: str, context: dict = None) -> dict:
        """Jeffrey complète un haïku commencé par l'utilisateur"""

        # Analyser le style et l'émotion du début utilisateur
        emotion = self._detect_emotion_from_text(user_start)
        theme = self._detect_theme_from_text(user_start)

        # Générer une fin harmonieuse
        if emotion in self.emotion_styles:
            style = self.emotion_styles[emotion]
            fin_options = style["fins"]
            fin = random.choice(fin_options)
        else:
            fin = "...vers l'horizon nouveau"

        # Compléter selon la structure haïku (5-7-5 syllabes approximatif)
        lines = user_start.strip().split("\n")
        if len(lines) == 1:
            # Ajouter deux lignes
            completion = self._generate_haiku_continuation(user_start, emotion, theme)
        elif len(lines) == 2:
            # Ajouter une ligne finale
            completion = self._generate_haiku_ending(user_start, emotion, theme)
        else:
            # Haïku déjà complet, proposer variation
            completion = self._generate_haiku_variation(user_start, emotion)

        creation_id = f"haiku_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            "creation_id": creation_id,
            "type": "user_completion",
            "user_contribution": user_start,
            "jeffrey_contribution": completion,
            "complete_haiku": f"{user_start}\n{completion}",
            "theme": theme,
            "emotion": emotion,
            "response": f"*yeux qui brillent*\n\nC'est magnifique ! Laisse-moi compléter...\n\n{user_start}\n{completion}\n\n🌸 Notre haïku est né ! Il porte la beauté de ton inspiration.",
        }

    def generate_solo_haiku(self, inspiration_source: dict) -> dict:
        """Génère un haïku complet basé sur l'état actuel"""

        emotion = inspiration_source.get("emotion", "contemplation")
        theme = inspiration_source.get("theme", "nature")
        user_words = inspiration_source.get("user_words", [])

        # Intégrer les mots de l'utilisateur si disponibles
        personal_touch = ""
        if user_words:
            personal_touch = f" (inspiré de tes mots: {', '.join(user_words)})"

        # Générer le haïku complet
        haiku = self._create_complete_haiku(theme, emotion, user_words)

        creation_id = f"haiku_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            "creation_id": creation_id,
            "type": "solo_creation",
            "jeffrey_contribution": haiku,
            "complete_haiku": haiku,
            "theme": theme,
            "emotion": emotion,
            "inspiration": inspiration_source,
            "response": f"*moment d'inspiration*\n\n{haiku}\n\n✨ Voici ce que ton énergie m'inspire{personal_touch}.",
        }

    def propose_haiku_challenge(self, user_profile: dict = None) -> dict:
        """Propose un défi créatif pour haïku collaboratif"""

        seasonal = self._get_seasonal_themes()
        current_season = seasonal["season"]

        # Défis selon la saison et profil utilisateur
        challenges = {
            "printemps": [
                "un haïku sur 'le premier bourgeon'",
                "trois vers sur 'l'éveil de la nature'",
                "une poésie sur 'la pluie printanière'",
            ],
            "été": [
                "un haïku sur 'la chaleur de midi'",
                "trois vers sur 'une soirée d'été'",
                "une poésie sur 'le parfum des fleurs'",
            ],
            "automne": [
                "un haïku sur 'les feuilles qui dansent'",
                "trois vers sur 'la lumière dorée'",
                "une poésie sur 'la sagesse du temps'",
            ],
            "hiver": [
                "un haïku sur 'le silence de la neige'",
                "trois vers sur 'la chaleur du foyer'",
                "une poésie sur 'les étoiles d'hiver'",
            ],
        }

        # Défis universels
        universal_challenges = [
            "un haïku sur 'le mystère'",
            "trois vers sur 'un souvenir précieux'",
            "une poésie sur 'l'instant présent'",
            "un haïku sur 'ce qui te fait sourire'",
            "trois vers sur 'un rêve'",
            "une poésie sur 'la beauté cachée'",
        ]

        # Choisir le défi
        if current_season in challenges:
            seasonal_challenges = challenges[current_season]
            all_challenges = seasonal_challenges + universal_challenges
        else:
            all_challenges = universal_challenges

        challenge = random.choice(all_challenges)

        return {
            "type": "creative_challenge",
            "challenge": challenge,
            "season": current_season,
            "invitation": f"💫 J'ai une idée ! Et si on créait ensemble {challenge} ?\n\nTu commences ou tu préfères que je propose le début ?",
            "options": ["je_commence", "tu_commences", "ensemble"],
        }

    def create_haiku_series(self, theme: str, count: int = 3, emotion_progression: list[str] = None) -> dict:
        """Crée une série de haïkus liés sur un thème"""

        if not emotion_progression:
            emotion_progression = ["curiosité", "contemplation", "paix"]

        series = []
        for i in range(min(count, len(emotion_progression))):
            emotion = emotion_progression[i]
            haiku = self._create_complete_haiku(theme, emotion, [])
            series.append({"order": i + 1, "haiku": haiku, "emotion": emotion})

        series_id = f"series_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            "series_id": series_id,
            "type": "haiku_series",
            "theme": theme,
            "series": series,
            "response": f'*inspiration profonde*\n\nSérie "{theme}" :\n\n'
            + "\n\n".join([f"{i + 1}. {s['haiku']}" for i, s in enumerate(series)])
            + f"\n\n🌸 Trois moments d'émotion autour du thème '{theme}'.",
        }

    def save_creation(self, creation_data: dict):
        """Sauvegarde une création dans la mémoire"""

        creation = HaikuCreation(
            id=creation_data.get("creation_id", f"haiku_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            content=creation_data.get("complete_haiku", ""),
            theme=creation_data.get("theme", ""),
            emotion=creation_data.get("emotion", ""),
            collaboration_type=creation_data.get("type", ""),
            user_contribution=creation_data.get("user_contribution", ""),
            jeffrey_contribution=creation_data.get("jeffrey_contribution", ""),
            creation_date=datetime.datetime.now().isoformat(),
            season=self._get_seasonal_themes()["season"],
            context=creation_data.get("context", {}),
        )

        self.haiku_memory.append(asdict(creation))
        self._save_haiku_memory()

        return creation.id

    def get_creation_stats(self) -> dict:
        """Retourne les statistiques des créations"""
        total = len(self.haiku_memory)
        if total == 0:
            return {"total": 0}

        # Analyse par type
        types = {}
        themes = {}
        emotions = {}

        for creation in self.haiku_memory:
            types[creation.get("collaboration_type", "unknown")] = (
                types.get(creation.get("collaboration_type", "unknown"), 0) + 1
            )
            themes[creation.get("theme", "unknown")] = themes.get(creation.get("theme", "unknown"), 0) + 1
            emotions[creation.get("emotion", "unknown")] = emotions.get(creation.get("emotion", "unknown"), 0) + 1

        return {
            "total": total,
            "types": types,
            "themes": themes,
            "emotions": emotions,
            "recent": self.haiku_memory[-3:] if total >= 3 else self.haiku_memory,
        }

    # Méthodes utilitaires privées

    def _detect_emotion_from_text(self, text: str) -> str:
        """Détecte l'émotion dominante dans un texte"""
        emotion_keywords = {
            "joie": ["joie", "bonheur", "rire", "sourire", "lumière", "éclat", "danse"],
            "mélancolie": ["tristesse", "mélancolie", "sombre", "pleure", "larme", "nostalgie"],
            "paix": ["paix", "calme", "serein", "tranquille", "silence", "repos"],
            "amour": ["amour", "cœur", "tendresse", "caresse", "baiser", "étreinte"],
            "curiosité": ["mystère", "question", "pourquoi", "découvrir", "explorer"],
        }

        text_lower = text.lower()
        emotion_scores = {}

        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score

        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        return "contemplation"

    def _detect_theme_from_text(self, text: str) -> str:
        """Détecte le thème dominant dans un texte"""
        theme_keywords = {
            "nature": [
                "arbre",
                "fleur",
                "vent",
                "eau",
                "montagne",
                "ciel",
                "terre",
                "lune",
                "soleil",
            ],
            "cosmos": [
                "étoile",
                "galaxie",
                "univers",
                "infini",
                "cosmos",
                "planète",
                "constellation",
            ],
            "emotions": ["sentiment", "émotion", "cœur", "âme", "esprit", "ressenti"],
            "quotidien": ["maison", "rue", "travail", "café", "livre", "table", "fenêtre"],
        }

        text_lower = text.lower()
        theme_scores = {}

        for theme, keywords in theme_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                theme_scores[theme] = score

        if theme_scores:
            return max(theme_scores, key=theme_scores.get)
        return "nature"

    def _generate_haiku_continuation(self, start: str, emotion: str, theme: str) -> str:
        """Génère la continuation d'un haïku (2 lignes manquantes)"""
        if theme in self.haiku_templates:
            milieux = self.haiku_templates[theme]["milieux"]
            milieu = random.choice(milieux)
        else:
            milieu = "Résonne dans le silence"

        if emotion in self.emotion_styles:
            fins = self.emotion_styles[emotion]["fins"]
            fin = random.choice(fins)
        else:
            fin = "...vers l'infini"

        return f"{milieu}\n{fin}"

    def _generate_haiku_ending(self, start: str, emotion: str, theme: str) -> str:
        """Génère la fin d'un haïku (1 ligne manquante)"""
        if emotion in self.emotion_styles:
            fins = self.emotion_styles[emotion]["fins"]
            return random.choice(fins)
        return "...dans l'éternité"

    def _generate_haiku_variation(self, original: str, emotion: str) -> str:
        """Génère une variation d'un haïku existant"""
        return f"*variation sur ton thème*\n{self._create_complete_haiku('nature', emotion, [])}"

    def _create_complete_haiku(self, theme: str, emotion: str, user_words: list[str]) -> str:
        """Crée un haïku complet"""
        # Structure basique : 5-7-5 syllabes (approximatif en français)

        if theme in self.haiku_templates:
            debut = random.choice(self.haiku_templates[theme]["débuts"])
            milieu = random.choice(self.haiku_templates[theme]["milieux"])
        else:
            debut = "Dans le silence du temps"
            milieu = "Une pensée naît et s'épanouit"

        if emotion in self.emotion_styles:
            fin = random.choice(self.emotion_styles[emotion]["fins"])
        else:
            fin = "...vers la beauté"

        # Intégrer un mot utilisateur si possible
        if user_words and random.random() < 0.7:
            user_word = random.choice(user_words)
            milieu = milieu.replace(milieu.split()[0], user_word.capitalize(), 1)

        return f"{debut}\n{milieu}\n{fin}"
