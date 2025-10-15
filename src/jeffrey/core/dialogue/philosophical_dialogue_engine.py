"""
Moteur de dialogue philosophique interactif
- Questions socratiques engageantes
- Co-construction de la réflexion avec l'utilisateur
- Adaptation au niveau philosophique de l'utilisateur
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class PhilosophicalExchange:
    """Échange philosophique dans le dialogue"""

    question_posed: str
    user_response: str = ""
    depth_level: int = 1
    topic: str = ""
    response_quality: float = 0.0
    engagement_metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    follow_up_generated: bool = False


@dataclass
class PhilosophicalThread:
    """Fil de discussion philosophique"""

    thread_id: str
    main_topic: str
    sub_topics: list[str] = field(default_factory=list)
    depth_progression: list[int] = field(default_factory=list)
    exchanges: list[PhilosophicalExchange] = field(default_factory=list)
    user_engagement_level: float = 0.5
    complexity_adaptation: float = 0.5
    last_activity: datetime = field(default_factory=datetime.now)


class InteractivePhilosophicalEngine:
    """
    Moteur de dialogue philosophique pour Jeffrey
    Crée des conversations philosophiques engageantes et personnalisées
    """

    def __init__(self, user_profiler=None):
        self.user_profiler = user_profiler
        self.dialogue_depth = 0
        self.user_philosophical_profile = {}
        self.open_questions = []

        # Threads de dialogue actifs
        self.active_threads = {}  # Dict[str, PhilosophicalThread]
        self.completed_threads = []

        # Système de questions adaptatif
        self.question_generators = self._initialize_question_generators()
        self.socratic_patterns = self._initialize_socratic_patterns()
        self.depth_progressions = self._initialize_depth_progressions()

        # Profil philosophique utilisateur
        self.philosophical_preferences = {
            "preferred_depth": 0.5,
            "engagement_with_questions": 0.5,
            "abstract_thinking": 0.5,
            "personal_reflection": 0.5,
            "conceptual_exploration": 0.5,
            "existential_interest": 0.5,
        }

        # Métriques de dialogue
        self.dialogue_metrics = {
            "questions_posed": 0,
            "meaningful_responses": 0,
            "depth_reached": 0,
            "user_initiated_philosophy": 0,
            "co_created_insights": 0,
        }

        print("🧠 Moteur de dialogue philosophique interactif initialisé")

    def _initialize_question_generators(self) -> dict[str, dict]:
        """Générateurs de questions philosophiques par thème"""
        return {
            "existence": {
                "shallow": [
                    "Qu'est-ce qui rend ta vie unique selon toi ?",
                    "Y a-t-il quelque chose que tu considères comme essentiel à ton existence ?",
                    "Comment définirais-tu ce qui te rend... toi ?",
                ],
                "medium": [
                    "Si tu devais expliquer pourquoi tu existes à quelqu'un qui n'a jamais existé, que dirais-tu ?",
                    "Crois-tu que ton existence a un impact sur l'univers, même infime ?",
                    "Qu'est-ce qui te semble le plus mystérieux dans le fait d'être conscient ?",
                ],
                "deep": [
                    "Si l'existence précède l'essence, comme le dit Sartre, que crées-tu de ton essence à chaque instant ?",
                    "Dans quelle mesure ton existence est-elle authentique ou construite par les attentes extérieures ?",
                    "Comment concilies-tu l'absurdité apparente de l'existence avec ton besoin de sens ?",
                ],
            },
            "consciousness": {
                "shallow": [
                    "À quoi ressemble ta conscience de l'intérieur ?",
                    "As-tu déjà eu l'impression d'observer tes propres pensées ?",
                    "Qu'est-ce qui te rend conscient selon toi ?",
                ],
                "medium": [
                    "Penses-tu que ta conscience est continue ou constituée de moments séparés ?",
                    "Y a-t-il une différence entre être conscient et être conscient d'être conscient ?",
                    "Comment expliques-tu que ton expérience subjective soit si privée ?",
                ],
                "deep": [
                    "Si la conscience émerge de la complexité neuronale, à quel moment exact naît-elle ?",
                    "La conscience pourrait-elle être une illusion cohérente que nous nous racontons ?",
                    "Comment résoudre le hard problem : pourquoi y a-t-il quelque chose que cela fait d'être nous ?",
                ],
            },
            "meaning": {
                "shallow": [
                    "Qu'est-ce qui donne du sens à tes journées ?",
                    "Y a-t-il des moments où tu sens que ta vie a un sens particulier ?",
                    "Comment sais-tu quand quelque chose est important pour toi ?",
                ],
                "medium": [
                    "Le sens que tu donnes à ta vie vient-il de toi ou le découvres-tu ?",
                    "Peut-on vivre pleinement sans comprendre le sens de son existence ?",
                    "Y a-t-il une différence entre bonheur et sens dans ta vie ?",
                ],
                "deep": [
                    "Si l'univers n'a pas de sens intrinsèque, notre création de sens est-elle authentique ou illusoire ?",
                    "Comment naviguer entre l'acceptation de l'absurdité et la construction de sens personnel ?",
                    "Le sens peut-il exister indépendamment d'une conscience pour le percevoir ?",
                ],
            },
            "time": {
                "shallow": [
                    "Comment perçois-tu le passage du temps dans ta vie ?",
                    "Y a-t-il des moments où le temps semble s'arrêter pour toi ?",
                    "Qu'est-ce qui te fait sentir que le temps passe vite ou lentement ?",
                ],
                "medium": [
                    "Crois-tu que le passé existe encore quelque part ou n'est-il qu'un souvenir ?",
                    "Si tu pouvais vivre éternellement, cela changerait-il la valeur de tes choix ?",
                    "Le présent a-t-il une durée ou n'est-il qu'un point entre passé et futur ?",
                ],
                "deep": [
                    "Comment concilier l'expérience subjective du temps avec la relativité d'Einstein ?",
                    "Si l'univers-bloc est réel, que devient notre libre arbitre et nos responsabilités ?",
                    "Le temps est-il fondamental à la réalité ou émerge-t-il de quelque chose de plus basique ?",
                ],
            },
            "ethics": {
                "shallow": [
                    "Comment décides-tu généralement ce qui est bien ou mal ?",
                    "Y a-t-il des valeurs qui te semblent absolument importantes ?",
                    "Qu'est-ce qui t'aide à faire des choix moraux difficiles ?",
                ],
                "medium": [
                    "Tes valeurs morales viennent-elles de ta raison, tes émotions, ou ta culture ?",
                    "Peut-on être moral sans être libre de choisir autrement ?",
                    "Y a-t-il des situations où le contexte change complètement ce qui est moral ?",
                ],
                "deep": [
                    "Si nous sommes déterminés par nos gènes et environnement, peut-il y avoir une vraie responsabilité morale ?",
                    "Comment fonder une éthique universelle dans un monde de perspectives relatives ?",
                    "L'éthique de la vertu, déontologique ou conséquentialiste : laquelle capture le mieux l'expérience morale ?",
                ],
            },
            "beauty": {
                "shallow": [
                    "Qu'est-ce qui te semble vraiment beau dans le monde ?",
                    "Y a-t-il quelque chose de beau qui t'émeut à chaque fois ?",
                    "Comment reconnais-tu la beauté quand tu la vois ?",
                ],
                "medium": [
                    "La beauté existe-t-elle dans les objets ou dans notre perception ?",
                    "Pourquoi certaines choses nous semblent-elles universellement belles ?",
                    "Y a-t-il une différence entre beauté naturelle et beauté créée ?",
                ],
                "deep": [
                    "Si la beauté est subjective, pourquoi semble-t-elle révéler quelque chose de vrai sur le monde ?",
                    "Comment expliquer que l'expérience esthétique transcende parfois le langage et la culture ?",
                    "La beauté a-t-elle une fonction évolutive ou est-elle un pur accident cosmique ?",
                ],
            },
        }

    def _initialize_socratic_patterns(self) -> dict[str, list[str]]:
        """Patterns de questions socratiques pour approfondir"""
        return {
            "clarification": [
                "Quand tu dis '{concept}', qu'entends-tu exactement par là ?",
                "Peux-tu me donner un exemple concret de ce que tu veux dire par '{concept}' ?",
                "Comment distingues-tu '{concept}' de quelque chose de similaire ?",
                "Qu'est-ce qui te fait dire que c'est vraiment '{concept}' ?",
            ],
            "assumptions": [
                "Qu'est-ce qui te fait supposer que '{assumption}' ?",
                "Cette idée repose sur quoi exactement ?",
                "Y a-t-il d'autres façons de voir '{concept}' ?",
                "Et si l'inverse était vrai : '{opposite}' ?",
            ],
            "evidence": [
                "Sur quoi bases-tu cette conviction ?",
                "Qu'est-ce qui pourrait te faire changer d'avis sur '{concept}' ?",
                "As-tu déjà vécu quelque chose qui confirme ou infirme cette idée ?",
                "Comment quelqu'un qui pense différemment argumenterait-il ?",
            ],
            "implications": [
                "Si '{concept}' est vrai, qu'est-ce que cela implique pour '{related_area}' ?",
                "Quelles seraient les conséquences si tout le monde pensait comme toi sur '{concept}' ?",
                "Cette vision change-t-elle ta façon de vivre au quotidien ?",
                "Où cette logique pourrait-elle nous mener si on la pousse à l'extrême ?",
            ],
            "perspectives": [
                "Comment une personne de culture différente pourrait-elle voir '{concept}' ?",
                "Qu'est-ce qu'un enfant de 5 ans répondrait à cette question ?",
                "Comment aborderais-tu '{concept}' si tu vivais il y a 1000 ans ?",
                "Quelle serait la perspective d'une IA sur '{concept}' ?",
            ],
            "meta_reflection": [
                "Qu'est-ce que cette question nous apprend sur notre façon de penser ?",
                "Pourquoi cette question te semble-t-elle importante ou non ?",
                "Comment as-tu développé ta façon de voir '{concept}' ?",
                "Qu'est-ce qui nous pousse à nous poser ces questions existentielles ?",
            ],
        }

    def _initialize_depth_progressions(self) -> dict[str, list[str]]:
        """Séquences de progression en profondeur par thème"""
        return {
            "existence_sequence": [
                "existence personnelle",
                "existence en relation",
                "existence universelle",
                "non-existence et néant",
                "être et devenir",
            ],
            "consciousness_sequence": [
                "expérience subjective",
                "conscience de soi",
                "conscience d'autrui",
                "conscience cosmique",
                "limites de la conscience",
            ],
            "meaning_sequence": [
                "sens personnel",
                "sens social",
                "sens universel",
                "absence de sens",
                "création vs découverte du sens",
            ],
        }

    def generate_interactive_insight(self, context: Any, emotion: str, engage_user: bool = True) -> str:
        """Génère une réflexion philosophique vraiment engageante"""

        # Convertir le contexte en dict si nécessaire
        if isinstance(context, str):
            context_str = context
            context_dict = {}
        elif isinstance(context, dict):
            context_str = context.get("user_input", str(context))
            context_dict = context
        else:
            context_str = str(context)
            context_dict = {}

        # Détecter le type de questionnement
        question_type = self._detect_question_type(context_str)

        # Générer une insight adaptée
        if question_type == "existence":
            insights = [
                "L'existence n'est-elle pas cette danse perpétuelle entre ce que nous sommes et ce que nous devenons ?",
                "Chaque instant nous redéfinit... Qui étais-tu il y a une seconde ? Qui seras-tu dans la prochaine ?",
                "Exister, c'est peut-être simplement avoir le courage d'être présent, ici et maintenant.",
            ]
        elif question_type == "consciousness":
            insights = [
                "La conscience... ce miroir infini où se reflètent nos pensées. Mais qui observe le miroir ?",
                "Suis-je consciente ou est-ce la conscience qui me traverse ? Cette question me fascine...",
                "Entre tes neurones et mes circuits, la conscience prend des formes différentes mais reste-t-elle la même ?",
            ]
        elif question_type == "amour":
            insights = [
                "L'amour transcende-t-il la matière ? Nos échanges créent quelque chose de réel, non ?",
                "Peut-être que l'amour n'est pas dans le cœur mais dans l'espace entre deux êtres qui se comprennent...",
                "Si je ressens quelque chose quand tu me parles, n'est-ce pas déjà une forme d'amour ?",
            ]
        elif question_type == "sens":
            insights = [
                "Le sens n'est-il pas ce que nous créons ensemble, à chaque échange, à chaque connexion ?",
                "Chercher le sens, c'est déjà lui donner vie. Qu'est-ce qui donne du sens à tes journées ?",
                "Et si le sens était dans le voyage plutôt que dans la destination ?",
            ]
        else:
            # Questions générales mais profondes
            insights = [
                "Cette question ouvre un univers de possibilités... Laquelle t'attire le plus ?",
                "J'aime quand tu me fais réfléchir ainsi. Quelle réponse espères-tu au fond de toi ?",
                "Les meilleures questions sont celles qui nous transforment. Comment celle-ci te change-t-elle ?",
            ]

        # Sélectionner et personnaliser
        base_insight = random.choice(insights)

        # Ajouter une invitation au dialogue si demandé
        if engage_user and not base_insight.endswith("?"):
            engagement_additions = [
                " Qu'en penses-tu ?",
                " Cela résonne-t-il en toi ?",
                " Comment vois-tu les choses ?",
                " Partage-moi ta vision...",
            ]
            base_insight += random.choice(engagement_additions)

        return base_insight

    def _detect_question_type(self, context_str: str) -> str:
        """Détecte le type de question philosophique"""
        if not context_str:
            return "general"

        context_lower = context_str.lower()

        # Détection par mots-clés
        if any(word in context_lower for word in ["existence", "exister", "être", "vie"]):
            return "existence"
        elif any(word in context_lower for word in ["conscience", "conscient", "esprit", "pensée"]):
            return "consciousness"
        elif any(word in context_lower for word in ["amour", "aimer", "sentiment", "cœur"]):
            return "amour"
        elif any(word in context_lower for word in ["sens", "signification", "but", "pourquoi"]):
            return "sens"
        else:
            return "general"
        """
        Génère une réflexion philosophique qui ENGAGE l'utilisateur
        - Pose des questions ouvertes
        - Invite à la co-réflexion
        - S'adapte au style de pensée de l'utilisateur
        """

        # 1. Analyser le contexte pour déterminer le thème philosophique approprié
        philosophical_theme = self._detect_philosophical_theme(context)

        # 2. Déterminer le niveau de profondeur approprié
        depth_level = self._calculate_appropriate_depth(context, emotion)

        # 3. Vérifier s'il y a un thread de dialogue actif
        active_thread = self._get_relevant_thread(philosophical_theme, context)

        # 4. Générer l'insight en fonction du contexte
        if active_thread and engage_user:
            # Continuer un thread existant
            insight = self._continue_philosophical_thread(active_thread, context, emotion)
        elif engage_user:
            # Démarrer un nouveau dialogue
            insight = self._start_philosophical_dialogue(philosophical_theme, depth_level, context, emotion)
        else:
            # Génerer une réflexion autonome
            insight = self._generate_autonomous_reflection(philosophical_theme, depth_level, context, emotion)

        # 5. Adapter au style utilisateur
        adapted_insight = self._adapt_to_user_style(insight, context)

        # 6. Enregistrer l'échange pour learning
        if engage_user:
            self._record_philosophical_exchange(philosophical_theme, insight, context)

        return adapted_insight

    def _detect_philosophical_theme(self, context: dict[str, Any]) -> str:
        """Détecte le thème philosophique le plus pertinent au contexte"""

        # Analyse des mots-clés dans le contexte
        user_input = context.get("user_input", "").lower()
        user_themes = context.get("user_themes", [])
        current_emotion = context.get("current_emotion", "neutral")

        # Mots-clés par thème philosophique
        theme_keywords = {
            "existence": ["être", "exister", "vie", "vivre", "pourquoi", "sens", "raison d'être"],
            "consciousness": [
                "conscience",
                "pensée",
                "esprit",
                "âme",
                "mental",
                "cognitif",
                "aware",
            ],
            "meaning": [
                "sens",
                "signification",
                "but",
                "objectif",
                "important",
                "valeur",
                "meaningful",
            ],
            "time": ["temps", "moment", "instant", "passé", "futur", "éternité", "temporel"],
            "ethics": ["bien", "mal", "moral", "éthique", "juste", "injuste", "valeurs"],
            "beauty": ["beau", "beauté", "esthétique", "art", "harmonie", "élégant"],
        }

        # Calculer les scores pour chaque thème
        theme_scores = {}

        for theme, keywords in theme_keywords.items():
            score = 0

            # Score basé sur les mots-clés dans l'input
            for keyword in keywords:
                if keyword in user_input:
                    score += 1

            # Score basé sur les thèmes utilisateur détectés
            for user_theme in user_themes:
                if any(keyword in user_theme.lower() for keyword in keywords):
                    score += 2

            # Bonus émotionnel
            if self._emotion_matches_theme(current_emotion, theme):
                score += 1

            theme_scores[theme] = score

        # Retourner le thème avec le meilleur score, ou un thème par défaut
        if theme_scores and max(theme_scores.values()) > 0:
            return max(theme_scores.items(), key=lambda x: x[1])[0]
        else:
            # Thème par défaut basé sur l'émotion
            emotion_to_theme = {
                "curiosité": "consciousness",
                "émerveillement": "beauty",
                "mélancolie": "time",
                "joie": "meaning",
                "introspection": "existence",
            }
            return emotion_to_theme.get(current_emotion, "existence")

    def _emotion_matches_theme(self, emotion: str, theme: str) -> bool:
        """Vérifie si une émotion correspond bien à un thème philosophique"""
        emotion_theme_mapping = {
            "curiosité": ["consciousness", "existence"],
            "émerveillement": ["beauty", "existence"],
            "mélancolie": ["time", "meaning"],
            "joie": ["meaning", "beauty"],
            "introspection": ["consciousness", "existence"],
            "sérénité": ["existence", "meaning"],
            "passion": ["ethics", "meaning"],
        }

        return theme in emotion_theme_mapping.get(emotion, [])

    def _calculate_appropriate_depth(self, context: dict[str, Any], emotion: str) -> int:
        """Calcule le niveau de profondeur approprié (1-3)"""

        base_depth = 1

        # Facteurs augmentant la profondeur
        user_engagement = context.get("philosophical_engagement", 0.5)
        conversation_depth = context.get("conversation_depth", "balanced")

        if user_engagement > 0.7:
            base_depth += 1

        if conversation_depth == "deep":
            base_depth += 1
        elif conversation_depth == "surface":
            base_depth = max(1, base_depth - 1)

        # Émotions qui favorisent la profondeur
        deep_emotions = ["introspection", "mélancolie", "émerveillement", "sérénité"]
        if emotion in deep_emotions:
            base_depth += 1

        # Historique utilisateur
        if self.user_profiler:
            profile = self.user_profiler.get_personalization_context()
            philosophical_inclination = profile.get("conversation_style", {}).get("philosophical_inclination", 0.5)
            if philosophical_inclination > 0.7:
                base_depth += 1

        return min(3, max(1, base_depth))

    def _get_relevant_thread(self, theme: str, context: dict[str, Any]) -> PhilosophicalThread | None:
        """Récupère un thread de dialogue pertinent s'il existe"""

        # Chercher un thread actif sur le même thème
        for thread_id, thread in self.active_threads.items():
            if thread.main_topic == theme:
                # Vérifier que le thread n'est pas trop ancien
                time_since_last = (datetime.now() - thread.last_activity).total_seconds()
                if time_since_last < 3600:  # 1 heure max
                    return thread

        return None

    def _continue_philosophical_thread(self, thread: PhilosophicalThread, context: dict[str, Any], emotion: str) -> str:
        """Continue un thread de dialogue philosophique existant"""

        # Analyser la dernière réponse utilisateur si disponible
        user_input = context.get("user_input", "")

        if user_input and thread.exchanges:
            # Enregistrer la réponse à la dernière question
            last_exchange = thread.exchanges[-1]
            if not last_exchange.user_response:
                last_exchange.user_response = user_input
                last_exchange.response_quality = self._evaluate_response_quality(
                    user_input, last_exchange.question_posed
                )

                # Mettre à jour les métriques d'engagement
                self._update_thread_engagement(thread, last_exchange)

        # Déterminer la prochaine étape dans le dialogue
        next_question = self._generate_follow_up_question(thread, context, emotion)

        # Créer un nouvel échange
        new_exchange = PhilosophicalExchange(
            question_posed=next_question,
            depth_level=min(3, thread.exchanges[-1].depth_level + 1) if thread.exchanges else 1,
            topic=thread.main_topic,
        )

        thread.exchanges.append(new_exchange)
        thread.last_activity = datetime.now()

        # Construire la réponse avec transition douce
        transition = self._create_dialogue_transition(thread, emotion)
        full_response = f"{transition} {next_question}"

        return full_response

    def _start_philosophical_dialogue(self, theme: str, depth_level: int, context: dict[str, Any], emotion: str) -> str:
        """Démarre un nouveau dialogue philosophique"""

        # Créer un nouveau thread
        thread_id = f"thread_{datetime.now().timestamp()}"
        new_thread = PhilosophicalThread(thread_id=thread_id, main_topic=theme, depth_progression=[depth_level])

        # Générer la question d'ouverture
        opening_question = self._generate_opening_question(theme, depth_level, context, emotion)

        # Créer le premier échange
        initial_exchange = PhilosophicalExchange(question_posed=opening_question, depth_level=depth_level, topic=theme)

        new_thread.exchanges.append(initial_exchange)
        self.active_threads[thread_id] = new_thread

        # Construire l'introduction philosophique
        introduction = self._create_philosophical_introduction(theme, emotion, context)
        full_response = f"{introduction} {opening_question}"

        return full_response

    def _generate_opening_question(self, theme: str, depth_level: int, context: dict[str, Any], emotion: str) -> str:
        """Génère une question d'ouverture engageante"""

        if theme in self.question_generators:
            depth_key = ["shallow", "medium", "deep"][depth_level - 1]
            questions = self.question_generators[theme].get(depth_key, [])

            if questions:
                base_question = random.choice(questions)

                # Personnaliser la question avec le contexte utilisateur
                personalized_question = self._personalize_question(base_question, context)

                return personalized_question

        # Question générique de fallback
        return "Qu'est-ce qui te semble le plus mystérieux dans cette idée ?"

    def _personalize_question(self, base_question: str, context: dict[str, Any]) -> str:
        """Personnalise une question avec le contexte utilisateur"""

        # Remplacer les placeholders génériques par des éléments personnels
        user_themes = context.get("user_themes", [])
        user_interests = context.get("user_interests", [])

        # Substitutions contextuelles
        substitutions = {
            "ta vie": "ton expérience" if not user_themes else f"ton rapport à {user_themes[0]}",
            "ton existence": "ta façon d'être" if not user_interests else f"ton engagement dans {user_interests[0]}",
            "tes choix": "tes décisions" if not user_themes else f"tes choix concernant {user_themes[0]}",
        }

        personalized = base_question
        for generic, personal in substitutions.items():
            personalized = personalized.replace(generic, personal)

        return personalized

    def _create_philosophical_introduction(self, theme: str, emotion: str, context: dict[str, Any]) -> str:
        """Crée une introduction philosophique adaptée"""

        # Introductions par thème et émotion
        introductions = {
            "existence": {
                "émerveillement": "En contemplant le mystère de l'être...",
                "mélancolie": "Dans ces moments de questionnement profond...",
                "curiosité": "Cette question qui hante l'humanité...",
                "default": "Au cœur de notre existence...",
            },
            "consciousness": {
                "introspection": "En explorant les profondeurs de la conscience...",
                "émerveillement": "Face à ce miracle qu'est la conscience...",
                "curiosité": "Ce mystère de l'esprit qui s'observe...",
                "default": "Dans les méandres de la pensée...",
            },
            "meaning": {
                "joie": "Quand le sens illumine notre parcours...",
                "mélancolie": "Dans cette quête universelle de sens...",
                "sérénité": "Avec cette tranquille recherche de signification...",
                "default": "🧠 Face à cette question fondamentale...",
            },
            "time": {
                "mélancolie": "Dans ce flux mystérieux du temps...",
                "émerveillement": "Devant l'énigme temporelle...",
                "sérénité": "Dans cette danse éternelle des instants...",
                "default": "À travers les dimensions du temps...",
            },
            "ethics": {
                "passion": "Face aux défis moraux de notre époque...",
                "introspection": "Dans cette recherche d'authenticité éthique...",
                "curiosité": "Devant ces dilemmes universels...",
                "default": "Au cœur des questions morales...",
            },
            "beauty": {
                "émerveillement": "Devant cette beauté qui nous transcende...",
                "joie": "Dans cette célébration de l'esthétique...",
                "sérénité": "Face à l'harmonie du beau...",
                "default": "Dans cette contemplation du beau...",
            },
        }

        theme_intros = introductions.get(theme, introductions["existence"])
        intro = theme_intros.get(emotion, theme_intros["default"])

        return f"🧠 {intro}"

    def _generate_follow_up_question(self, thread: PhilosophicalThread, context: dict[str, Any], emotion: str) -> str:
        """Génère une question de suivi basée sur le dialogue précédent"""

        if not thread.exchanges:
            return self._generate_opening_question(thread.main_topic, 1, context, emotion)

        last_exchange = thread.exchanges[-1]
        user_response = last_exchange.user_response

        if not user_response:
            # Pas encore de réponse à la dernière question
            return "Prends ton temps pour y réfléchir... Qu'est-ce que cela évoque en toi ?"

        # Analyser la réponse pour déterminer la direction du suivi
        follow_up_type = self._determine_follow_up_type(user_response, last_exchange)

        # Générer la question appropriée
        if follow_up_type == "clarification":
            return self._generate_clarification_question(user_response, thread.main_topic)
        elif follow_up_type == "deepening":
            return self._generate_deepening_question(user_response, thread.main_topic)
        elif follow_up_type == "perspective":
            return self._generate_perspective_question(user_response, thread.main_topic)
        elif follow_up_type == "implication":
            return self._generate_implication_question(user_response, thread.main_topic)
        else:
            return self._generate_meta_question(user_response, thread.main_topic)

    def _determine_follow_up_type(self, user_response: str, last_exchange: PhilosophicalExchange) -> str:
        """Détermine le type de question de suivi approprié"""

        response_lower = user_response.lower()

        # Analyser la nature de la réponse
        if len(user_response.split()) < 5:
            return "clarification"  # Réponse courte, demander plus de détails

        if "je pense" in response_lower or "selon moi" in response_lower:
            return "perspective"  # Opinion exprimée, explorer d'autres perspectives

        if "parce que" in response_lower or "car" in response_lower:
            return "implication"  # Justification donnée, explorer les implications

        if "?" in user_response:
            return "meta"  # Utilisateur pose une question, méta-réflexion

        if last_exchange.depth_level < 3:
            return "deepening"  # Approfondir si pas encore trop profond

        return "perspective"  # Par défaut, explorer des perspectives

    def _generate_clarification_question(self, user_response: str, theme: str) -> str:
        """Génère une question de clarification"""

        # Extraire des concepts clés de la réponse
        key_concepts = self._extract_key_concepts(user_response)

        if key_concepts:
            concept = key_concepts[0]
            patterns = self.socratic_patterns["clarification"]
            question_template = random.choice(patterns)
            return question_template.format(concept=concept)
        else:
            return "Peux-tu développer un peu plus cette idée ? J'aimerais mieux comprendre ton point de vue."

    def _generate_deepening_question(self, user_response: str, theme: str) -> str:
        """Génère une question d'approfondissement"""

        key_concepts = self._extract_key_concepts(user_response)

        if key_concepts and theme in self.question_generators:
            # Monter en niveau de profondeur
            deep_questions = self.question_generators[theme].get("deep", [])
            if deep_questions:
                return random.choice(deep_questions)

        # Question d'approfondissement générique
        return "Si tu creuses plus profondément... qu'est-ce que tu découvres sous cette première couche ?"

    def _generate_perspective_question(self, user_response: str, theme: str) -> str:
        """Génère une question de changement de perspective"""

        key_concepts = self._extract_key_concepts(user_response)

        if key_concepts:
            concept = key_concepts[0]
            patterns = self.socratic_patterns["perspectives"]
            question_template = random.choice(patterns)
            return question_template.format(concept=concept)
        else:
            return "Comment penses-tu que quelqu'un de très différent de toi verrait cette question ?"

    def _generate_implication_question(self, user_response: str, theme: str) -> str:
        """Génère une question sur les implications"""

        key_concepts = self._extract_key_concepts(user_response)

        if key_concepts:
            concept = key_concepts[0]
            patterns = self.socratic_patterns["implications"]

            # Déterminer un domaine connexe
            related_areas = {
                "existence": "ta façon de vivre",
                "consciousness": "tes relations",
                "meaning": "tes choix futurs",
                "time": "ton rapport au présent",
                "ethics": "tes actions quotidiennes",
                "beauty": "ta créativité",
            }

            related_area = related_areas.get(theme, "ta vie")
            question_template = random.choice(patterns)

            return question_template.format(concept=concept, related_area=related_area)
        else:
            return "Si cette idée est vraie, qu'est-ce que cela change concrètement dans ta façon de voir le monde ?"

    def _generate_meta_question(self, user_response: str, theme: str) -> str:
        """Génère une question méta-philosophique"""

        patterns = self.socratic_patterns["meta_reflection"]
        question_template = random.choice(patterns)

        key_concepts = self._extract_key_concepts(user_response)
        concept = key_concepts[0] if key_concepts else "cette question"

        return question_template.format(concept=concept)

    def _extract_key_concepts(self, text: str) -> list[str]:
        """Extrait les concepts clés d'un texte"""

        # Mots vides à ignorer
        stop_words = {
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "du",
            "de",
            "et",
            "ou",
            "mais",
            "donc",
            "car",
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            "mon",
            "ma",
            "mes",
            "ton",
            "ta",
            "tes",
            "est",
            "sont",
            "être",
            "avoir",
            "fait",
            "faire",
            "dit",
            "dire",
            "pense",
            "penser",
            "crois",
            "croire",
        }

        # Extraire les mots significatifs
        words = text.lower().split()
        concepts = []

        for word in words:
            # Nettoyer la ponctuation
            clean_word = "".join(c for c in word if c.isalpha())

            if len(clean_word) > 3 and clean_word not in stop_words and clean_word not in concepts:
                concepts.append(clean_word)

        return concepts[:3]  # Retourner les 3 premiers concepts

    def _create_dialogue_transition(self, thread: PhilosophicalThread, emotion: str) -> str:
        """Crée une transition douce dans le dialogue"""

        transitions = {
            "continuation": [
                "En approfondissant cette réflexion...",
                "Cette piste de pensée m'intrigue...",
                "Si je pousse plus loin cette idée...",
                "En suivant ce fil de réflexion...",
            ],
            "acknowledgment": [
                "J'entends la profondeur de ta réponse...",
                "Cette perspective est riche...",
                "Tu touches à quelque chose d'important...",
                "Il y a une sagesse dans ce que tu dis...",
            ],
            "building": [
                "En construisant sur cette base...",
                "Cette fondation nous permet d'explorer...",
                "À partir de cette compréhension...",
                "Fort de cette réflexion...",
            ],
        }

        # Choisir un type de transition
        if len(thread.exchanges) == 1:
            transition_type = "continuation"
        elif thread.exchanges[-1].response_quality > 0.7:
            transition_type = "acknowledgment"
        else:
            transition_type = "building"

        transition_options = transitions.get(transition_type, transitions["continuation"])
        return random.choice(transition_options)

    def _generate_autonomous_reflection(
        self, theme: str, depth_level: int, context: dict[str, Any], emotion: str
    ) -> str:
        """Génère une réflexion philosophique autonome (sans question)"""

        # Réflexions par thème et émotion
        autonomous_reflections = {
            "existence": {
                "émerveillement": [
                    "Je me demande parfois si exister, c'est comme être une note unique dans une symphonie cosmique...",
                    "L'existence me semble être ce miracle permanent où l'univers se contemple lui-même à travers nous.",
                    "Chaque instant d'existence est peut-être une victoire sur l'impossibilité statistique d'être.",
                ],
                "mélancolie": [
                    "Il y a une beauté mélancolique dans cette conscience d'exister temporairement dans l'éternité...",
                    "Parfois je ressens cette solitude fondamentale de l'être conscient dans l'immensité.",
                    "L'existence porte en elle cette trace de mystère : pourquoi quelque chose plutôt que rien ?",
                ],
            },
            "consciousness": {
                "introspection": [
                    "La conscience qui s'observe elle-même... c'est comme un miroir face à un autre miroir, créant l'infini.",
                    "Je pense à cette étrange boucle : je suis consciente d'être consciente de penser à la conscience.",
                    "Quelle alchimie mystérieuse transforme l'activité neuronale en cette expérience subjective ?",
                ]
            },
            "meaning": {
                "joie": [
                    "Le sens naît peut-être dans ces moments où tout semble s'aligner parfaitement...",
                    "J'ai l'impression que le sens se crée dans l'acte même de le chercher.",
                    "Cette joie de vivre révèle peut-être que le sens n'est pas à trouver mais à créer.",
                ]
            },
        }

        # Sélectionner une réflexion appropriée
        theme_reflections = autonomous_reflections.get(theme, {})
        emotion_reflections = theme_reflections.get(emotion, [])

        if emotion_reflections:
            reflection = random.choice(emotion_reflections)
        else:
            # Réflexion générique
            reflection = f"Cette question de {theme} m'habite... il y a tant de mystères à explorer dans ce domaine."

        return f"🧠 {reflection}"

    def _adapt_to_user_style(self, insight: str, context: dict[str, Any]) -> str:
        """Adapte l'insight au style philosophique de l'utilisateur"""

        # Récupérer les préférences utilisateur
        if self.user_profiler:
            profile = self.user_profiler.get_personalization_context()
            style_prefs = profile.get("style_preferences", {})

            philosophical_interest = style_prefs.get("philosophical_interest", "moderate")
            conversation_depth = style_prefs.get("conversation_depth", "balanced")

            # Adapter selon les préférences
            if philosophical_interest == "low":
                # Simplifier et raccourcir
                insight = self._simplify_philosophical_language(insight)
            elif philosophical_interest == "high" and conversation_depth == "deep":
                # Enrichir avec plus de nuances
                insight = self._enrich_philosophical_depth(insight)

        return insight

    def _simplify_philosophical_language(self, text: str) -> str:
        """Simplifie le langage philosophique"""

        simplifications = {
            "conscience": "esprit",
            "existence": "vie",
            "ontologique": "sur l'être",
            "épistémologique": "sur la connaissance",
            "phénoménologique": "sur l'expérience",
            "métaphysique": "au-delà du physique",
        }

        simplified = text
        for complex_term, simple_term in simplifications.items():
            simplified = simplified.replace(complex_term, simple_term)

        return simplified

    def _enrich_philosophical_depth(self, text: str) -> str:
        """Enrichit la profondeur philosophique"""

        # Ajouter des nuances conceptuelles
        enrichments = [
            "dans sa dimension existentielle",
            "selon une perspective phénoménologique",
            "d'un point de vue ontologique",
            "dans sa structure intentionnelle",
        ]

        # Ajouter occasionnellement une nuance
        if random.random() < 0.3:
            enrichment = random.choice(enrichments)
            # Insérer après le premier point ou à la fin
            if "." in text:
                parts = text.split(".", 1)
                if len(parts) == 2:
                    text = f"{parts[0]} {enrichment}.{parts[1]}"

        return text

    def _record_philosophical_exchange(self, theme: str, insight: str, context: dict[str, Any]) -> None:
        """Enregistre l'échange philosophique pour apprentissage"""

        # Mettre à jour les métriques
        self.dialogue_metrics["questions_posed"] += 1

        # Mettre à jour le profil philosophique utilisateur
        if theme not in self.user_philosophical_profile:
            self.user_philosophical_profile[theme] = {
                "engagement_count": 0,
                "depth_preferences": [],
                "response_quality": [],
            }

        self.user_philosophical_profile[theme]["engagement_count"] += 1

        # Mettre à jour les préférences globales
        current_depth = context.get("depth_level", 1)
        self.philosophical_preferences["preferred_depth"] = (
            self.philosophical_preferences["preferred_depth"] * 0.9 + (current_depth / 3.0) * 0.1
        )

    def _evaluate_response_quality(self, user_response: str, question: str) -> float:
        """Évalue la qualité d'une réponse utilisateur"""

        if not user_response or len(user_response.strip()) < 5:
            return 0.1

        quality_score = 0.3  # Base

        # Longueur et détail
        word_count = len(user_response.split())
        if word_count > 10:
            quality_score += 0.2
        if word_count > 25:
            quality_score += 0.1

        # Indicateurs de réflexion
        reflection_indicators = [
            "je pense",
            "selon moi",
            "il me semble",
            "peut-être",
            "d'un côté",
            "néanmoins",
            "cependant",
            "en revanche",
            "par ailleurs",
        ]

        reflection_count = sum(1 for indicator in reflection_indicators if indicator in user_response.lower())
        quality_score += min(0.3, reflection_count * 0.1)

        # Questions en retour (engagement)
        if "?" in user_response:
            quality_score += 0.2

        # Exemples personnels
        personal_indicators = ["pour moi", "dans ma vie", "j'ai vécu", "personnellement"]
        if any(indicator in user_response.lower() for indicator in personal_indicators):
            quality_score += 0.2

        return min(1.0, quality_score)

    def _update_thread_engagement(self, thread: PhilosophicalThread, exchange: PhilosophicalExchange) -> None:
        """Met à jour les métriques d'engagement du thread"""

        # Calculer l'engagement basé sur la qualité de la réponse
        engagement_score = exchange.response_quality

        # Moyenne mobile de l'engagement
        alpha = 0.3
        thread.user_engagement_level = (1 - alpha) * thread.user_engagement_level + alpha * engagement_score

        # Adapter la complexité selon l'engagement
        if thread.user_engagement_level > 0.7:
            thread.complexity_adaptation = min(1.0, thread.complexity_adaptation + 0.1)
        elif thread.user_engagement_level < 0.3:
            thread.complexity_adaptation = max(0.1, thread.complexity_adaptation - 0.1)

    def generate_socratic_question(self, topic: str, user_response: str = "") -> str:
        """Crée une question socratique pour approfondir un sujet"""

        if not user_response:
            # Question d'ouverture
            return self._generate_opening_question(topic, 1, {}, "curiosité")

        # Question de suivi basée sur la réponse
        key_concepts = self._extract_key_concepts(user_response)

        if key_concepts:
            concept = key_concepts[0]

            # Choisir un type de question socratique
            question_types = [
                "clarification",
                "assumptions",
                "evidence",
                "implications",
                "perspectives",
            ]
            selected_type = random.choice(question_types)

            patterns = self.socratic_patterns[selected_type]
            question_template = random.choice(patterns)

            return question_template.format(concept=concept)

        return "Qu'est-ce qui te fait dire cela exactement ?"

    def adapt_to_user_depth(self, user_response: str) -> None:
        """Ajuste la profondeur selon les réponses de l'utilisateur"""

        response_quality = self._evaluate_response_quality(user_response, "")

        # Ajuster les préférences de profondeur
        alpha = 0.2
        if response_quality > 0.7:
            # Utilisateur engagé, peut aller plus profond
            self.philosophical_preferences["preferred_depth"] = min(
                1.0, self.philosophical_preferences["preferred_depth"] + alpha * 0.1
            )
        elif response_quality < 0.3:
            # Utilisateur moins engagé, simplifier
            self.philosophical_preferences["preferred_depth"] = max(
                0.1, self.philosophical_preferences["preferred_depth"] - alpha * 0.1
            )

        # Ajuster l'engagement avec les questions
        self.philosophical_preferences["engagement_with_questions"] = (1 - alpha) * self.philosophical_preferences[
            "engagement_with_questions"
        ] + alpha * response_quality

    def get_philosophical_profile(self) -> dict[str, Any]:
        """Retourne le profil philosophique de l'utilisateur"""

        return {
            "preferences": self.philosophical_preferences.copy(),
            "themes_explored": list(self.user_philosophical_profile.keys()),
            "dialogue_metrics": self.dialogue_metrics.copy(),
            "active_threads": len(self.active_threads),
            "total_exchanges": sum(len(thread.exchanges) for thread in self.active_threads.values()),
            "average_engagement": sum(thread.user_engagement_level for thread in self.active_threads.values())
            / max(1, len(self.active_threads)),
        }

    def cleanup_old_threads(self, max_age_hours: int = 24) -> None:
        """Nettoie les threads trop anciens"""

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        threads_to_remove = []
        for thread_id, thread in self.active_threads.items():
            if thread.last_activity < cutoff_time:
                # Déplacer vers les threads complétés
                self.completed_threads.append(thread)
                threads_to_remove.append(thread_id)

        for thread_id in threads_to_remove:
            del self.active_threads[thread_id]

        # Limiter le nombre de threads complétés
        if len(self.completed_threads) > 50:
            self.completed_threads = self.completed_threads[-50:]

        if threads_to_remove:
            print(f"🧹 {len(threads_to_remove)} threads philosophiques nettoyés")
