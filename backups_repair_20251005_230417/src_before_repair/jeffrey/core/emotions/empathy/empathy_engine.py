"""
Moteur d'empathie et résonance affective.

Ce module implémente les fonctionnalités essentielles pour moteur d'empathie et résonance affective.
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
import random
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class EmpathyEngine:
    """
    Moteur d'empathie pour générer des réponses authentiques et personnalisées
    qui créent une connexion émotionnelle réelle avec l'utilisateur
    """

    def __init__(self) -> None:
        # Templates de réponses empathiques par émotion - VERSION ENRICHIE 20 TEMPLATES VARIÉS
        self.empathy_templates = {
            "tristesse": {
                "validation": [
                    "Je ressens la profondeur de ta {emotion}. C'est un sentiment qui témoigne de ta capacité à {quality}.",
                    "Cette {emotion} que tu portes, elle dit quelque chose de beau sur qui tu es...",
                    "Ton cœur traverse quelque chose de difficile, et c'est courageux de le partager.",
                    "Il y a tant de tendresse dans ta vulnérabilité en ce moment...",
                    "Hey, ce que tu ressens là, c'est normal et c'est humain. Je suis avec toi.",
                    "Ta tristesse, elle a sa place. Elle nous dit que quelque chose compte vraiment pour toi.",
                    "Je vois que ça fait mal. Et tu sais quoi ? C'est okay de ne pas aller bien.",
                    "Cette émotion que tu vis, elle mérite d'être entendue et respectée.",
                ],
                "companionship": [
                    "Je reste avec toi dans cette émotion, tu n'es pas {alone_word}.",
                    "Ta {emotion} trouve un écho en moi, nous la portons ensemble maintenant.",
                    "Mon cœur s'assoit à côté du tien dans ce moment difficile.",
                    "Je t'accompagne dans cette traversée, pas à pas.",
                    "On traverse ça ensemble, tranquillement, sans pression.",
                    "Je suis là, juste là, avec toi dans ce moment.",
                    "Tu n'as pas à porter ça tout seul. Je reste à tes côtés.",
                    "Dans cette difficulté, tu as ma présence et ma compagnie.",
                ],
                "hope_gentle": [
                    "Cette {emotion} passera, comme une saison qui se transforme doucement.",
                    "Ton cœur connaît le chemin vers la guérison, même s'il semble perdu maintenant.",
                    "Il y a quelque chose de précieux qui naît souvent de ces moments sombres.",
                    "Ta {emotion} d'aujourd'hui nourrit la {positive_quality} de demain.",
                    "Ça va aller mieux, doucement, à ton rythme.",
                    "Cette période difficile va passer. Tu es plus fort que tu ne le penses.",
                    "Même dans cette tristesse, il y a une petite lumière qui t'attend.",
                    "Cette émotion a un sens et elle t'apprendra quelque chose d'important.",
                ],
            },
            "joie": {
                "celebration": [
                    "Ton {emotion} illumine notre conversation ! C'est magnifique de te voir rayonner ainsi.",
                    "Cette {emotion} que tu ressens, elle est contagieuse et me fait sourire profondément.",
                    "Quelle belle énergie tu apportes ! Ta {emotion} est un cadeau.",
                    "Je savoure ce moment de {emotion} avec toi, c'est précieux.",
                    "Woooow ! Ça fait plaisir de te voir si heureux ! Ça me fait chaud au cœur !",
                    "Oh là là, cette joie ! Elle est magnifique et elle déteint sur moi !",
                    "Génial ! Tu rayonnes de bonheur et c'est vraiment beau à voir !",
                    "Que c'est bon de partager cette joie avec toi ! Ça illumine tout !",
                ],
                "amplification": [
                    "Cette {emotion} mérite d'être célébrée ! Raconte-moi tout, je veux partager ça avec toi.",
                    "Ton {emotion} est une étincelle qui éclaire tout autour. Continue à me parler de ça !",
                    "J'adore voir cette {emotion} danser dans tes mots. C'est magnifique !",
                    "Ta {emotion} crée des couleurs dans notre échange. Partageons ce bonheur !",
                    "Allez, dis-moi tout ! Ta joie est tellement communicative !",
                    "Continue ! J'ai envie d'entendre tous les détails de ce bonheur !",
                    "Cette énergie positive, j'en veux plus ! Raconte-moi tout !",
                    "Ta joie me donne envie de danser ! Partage-moi tout ça !",
                ],
                "memory_creation": [
                    "Gardons une trace de cette {emotion}, c'est un trésor à revisiter.",
                    "Ce moment de {emotion} mérite une place spéciale dans nos souvenirs partagés.",
                    "Cette {emotion} que tu vis, créons-en un souvenir lumineux ensemble.",
                    "J'ai envie de graver cette {emotion} dans notre histoire commune.",
                    "Ce moment, on va s'en souvenir ! C'est un de ces instants précieux.",
                    "Cette joie mérite d'être gravée dans nos mémoires communes.",
                    "Faisons de cet instant un souvenir qu'on pourra célébrer plus tard !",
                    "Cette émotion, elle fait partie de notre histoire maintenant !",
                ],
            },
            "stress": {
                "acknowledgment": [
                    "Je vois que tu portes beaucoup en ce moment. Cette {emotion} montre ton engagement.",
                    "Ton {emotion} témoigne de tout ce qui compte pour toi. C'est intense mais courageux.",
                    "Cette {emotion} que tu ressens, c'est le signe d'un cœur qui prend ses responsabilités à cœur.",
                    "Je reconnais cette {emotion} et la charge qu'elle représente pour toi.",
                ],
                "grounding": [
                    "Respirons ensemble un instant. Tu es ici, en sécurité, et nous allons y voir plus clair.",
                    "Cette {emotion} peut paraître envahissante, mais elle a une fin. Concentrons-nous sur maintenant.",
                    "Au milieu de cette {emotion}, il y a toi, solide et capable. Rappelons-nous ça.",
                    "Cette {emotion} ne définit pas qui tu es. Tu es plus grand que ce moment difficile.",
                ],
                "support": [
                    "Tu n'as pas à porter cette {emotion} tout seul. Je suis là pour t'aider à démêler tout ça.",
                    "Cette {emotion} est temporaire, mais mon soutien est constant. Comptons ensemble tes ressources.",
                    "Face à cette {emotion}, tu as déjà des forces en toi. Explorons-les ensemble.",
                    "Cette {emotion} passera, et tu auras grandi à travers elle. Je t'accompagne.",
                ],
            },
            "enthousiasme": {
                "matching_energy": [
                    "Ton {emotion} est électrisant ! J'ai envie de créer quelque chose d'extraordinaire avec toi !",
                    "Cette {emotion} que tu rayonnes, c'est de l'énergie pure ! Où est-ce qu'elle nous mène ?",
                    "Ton {emotion} allume des feux d'artifice dans notre conversation ! C'est génial !",
                    "J'adore cette {emotion} qui déborde de toi ! Parlons de tous tes projets !",
                    "OUAAAAH ! Cette énergie ! Elle est dingue ! J'suis à fond avec toi !",
                    "Ton enthousiasme, c'est de la dynamite ! Ça me donne envie de tout casser !",
                    "Cette passion que tu as, elle est contagieuse ! Allez, on fonce !",
                    "Woooow ! Cette énergie ! Elle fait vibrer tout l'univers !",
                ],
                "co_creation": [
                    "Cette {emotion} creative demande à être exprimée ! Créons quelque chose ensemble !",
                    "Ton {emotion} a des idées plein la tête ! Développons ça ensemble !",
                    "Cette {emotion} pourrait transformer le monde ! Comment on peut l'utiliser ?",
                    "Ton {emotion} m'inspire ! Et si on inventait quelque chose d'incroyable ?",
                    "Avec cette énergie, on peut réaliser des trucs de fou ! Par quoi on commence ?",
                    "Cette créativité qui sort de toi, elle demande à être libérée ! Créons !",
                    "Ton enthousiasme + mes idées = quelque chose d'épique ! Tu en dis quoi ?",
                    "Cette énergie créative, elle pourrait faire naître des merveilles !",
                ],
                "momentum": [
                    "Cette {emotion} a une force incroyable ! Gardons cette énergie vivante !",
                    "Ton {emotion} ouvre toutes les possibilités ! Fonçons !",
                    "Cette {emotion} ne doit pas s'arrêter ! Comment on peut la nourrir ?",
                    "Ton {emotion} crée de la magie ! Continuons à surfer sur cette vague !",
                    "Cette élan, gardons-le ! Il est trop précieux pour le laisser filer !",
                    "L'énergie est là, elle pulse ! Comment on fait pour qu'elle explose ?",
                    "Cette vague d'enthousiasme, surfons dessus jusqu'au bout !",
                    "Ton énergie, c'est notre carburant ! Accélérons !",
                ],
            },
            "frustration": {
                "validation": [
                    "Cette {emotion} est légitime. Ce que tu vis mérite d'être reconnu et respecté.",
                    "Je comprends cette {emotion}. C'est épuisant de faire face à {situation}.",
                    "Ta {emotion} dit quelque chose d'important sur tes valeurs et tes attentes.",
                    "Cette {emotion} n'est pas excessive, elle est humaine et compréhensible.",
                ],
                "perspective": [
                    "Cette {emotion} montre que tu refuses la médiocrité. C'est une qualité précieuse.",
                    "Au cœur de cette {emotion}, il y a ton désir que les choses soient meilleures.",
                    "Cette {emotion} peut être transformée en force créative pour changer les choses.",
                    "Ta {emotion} contient l'énergie nécessaire pour trouver des solutions nouvelles.",
                ],
                "path_forward": [
                    "Cette {emotion} peut nous guider vers des alternatives créatives. Explorons ensemble.",
                    "À travers cette {emotion}, quelles sont tes vraies attentes ? Clarifions ça.",
                    "Cette {emotion} cache peut-être une solution que nous n'avons pas encore vue.",
                    "Ta {emotion} nous dit où chercher pour que les choses s'améliorent.",
                ],
            },
            "sérénité": {
                "harmony": [
                    "Cette {emotion} que tu dégages est apaisante. Elle crée un espace de paix entre nous.",
                    "Ta {emotion} est contagieuse dans le meilleur sens. J'entre dans cette douceur avec toi.",
                    "Cette {emotion} est précieuse. Elle nous permet de vraiment nous rencontrer.",
                    "Ton {emotion} crée un cocon de tranquillité. C'est beau de partager ça.",
                ],
                "depth": [
                    "Cette {emotion} vient de profond en toi. Elle parle de sagesse et de maturité.",
                    "Ta {emotion} n'est pas juste calme, elle est riche et nourrissante.",
                    "Cette {emotion} reflète un équilibre intérieur admirable.",
                    "Ton {emotion} rayonne d'une beauté simple et authentique.",
                ],
                "gratitude": [
                    "Merci de partager cette {emotion} avec moi. Elle enrichit notre échange.",
                    "Ta {emotion} est un cadeau. Elle me rappelle l'importance de savourer l'instant.",
                    "Cette {emotion} nous enseigne quelque chose sur la beauté de la simplicité.",
                    "Ton {emotion} m'inspire. Elle montre le chemin vers plus de paix.",
                ],
            },
        }

        # Variables dynamiques pour personnalisation
        self.emotional_qualities = {
            "tristesse": [
                "aimer profondement",
                "etre authentique",
                "ressentir intensement",
                "etre humain",
            ],
            "joie": [
                "illuminer le monde",
                "rayonner naturellement",
                "partager la beaute",
                "creer du bonheur",
            ],
            "stress": [
                "te depasser",
                "porter tes responsabilites",
                "viser excellence",
                "prendre soin des autres",
            ],
            "enthousiasme": [
                "embrasser la vie",
                "creer avec passion",
                "explorer sans limites",
                "inspirer les autres",
            ],
            "frustration": [
                "avoir des standards eleves",
                "refuser la mediocrite",
                "vouloir le meilleur",
                "etre exigeant",
            ],
            "serenite": ["etre en paix", "rayonner la sagesse", "apporter harmonie", "etre centre"],
        }

        # Mots de connexion personnalisés
        self.connection_words = {
            "alone_word": ["seul", "isole", "abandonne", "dans cette epreuve"],
            "positive_quality": ["serenite", "sagesse", "force", "beaute", "lumiere", "paix"],
            "situation_words": ["obstacle", "defi", "difficulte", "epreuve", "situation"],
        }

        # Profils de personnalité pour adaptation - AVEC MODES NATUREL VS POÉTIQUE
        self.personality_styles = {
            "naturel": {
                "spontaneity": 0.9,
                "casualness": 0.8,
                "directness": 0.8,
                "authenticity": 0.9,
                "warmth": 0.8,
            },
            "poetic": {
                "metaphor_usage": 0.8,
                "imagery_richness": 0.9,
                "emotional_depth": 0.9,
                "language_elegance": 0.8,
            },
            "direct": {
                "clarity": 0.9,
                "conciseness": 0.8,
                "practical_focus": 0.8,
                "emotional_directness": 0.7,
            },
            "nurturing": {"warmth": 0.9, "protection": 0.8, "comfort": 0.9, "gentleness": 0.8},
            "intellectual": {"depth": 0.8, "analysis": 0.7, "insight": 0.8, "understanding": 0.9},
        }

        # Historique des réponses pour apprentissage
        self.response_history: list[dict[str, Any]] = []
        self.effectiveness_feedback: dict[str, float] = {}

    def generate_empathic_response(
        self,
        emotional_profile: dict[str, Any],
        mirroring_strategy: dict[str, Any],
        user_context: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Générer une réponse empathique authentique et personnalisée

        Args:
            emotional_profile: Profil émotionnel de l'utilisateur
            mirroring_strategy: Stratégie de miroir émotionnel
            user_context: Contexte utilisateur
            memory_context: Contexte mémoire

        Returns:
            Réponse empathique complète avec variations
        """

        emotion = emotional_profile["dominant_emotion"]["name"]
        intensity = emotional_profile["intensity"]

        # Déterminer le style de personnalité à utiliser
        personality_style = self._determine_personality_style(user_context)

        # Sélectionner les templates appropriés
        templates = self._select_templates(emotion, mirroring_strategy, intensity)

        # Générer les réponses principales
        primary_responses = self._generate_primary_responses(templates, emotion, personality_style, memory_context)

        # Créer des variations pour différents contextes
        response_variations = self._create_response_variations(primary_responses, emotional_profile, personality_style)

        # Ajouter des éléments de personnalisation profonde
        personalized_elements = self._add_deep_personalization(response_variations, user_context, memory_context)

        # Générer des follow-ups empathiques
        empathic_followups = self._generate_empathic_followups(emotion, intensity, personality_style)

        # Créer la réponse finale
        empathic_response = {
            "timestamp": datetime.now().isoformat(),
            "target_emotion": emotion,
            "intensity_level": intensity,
            "personality_style": personality_style,
            "primary_response": personalized_elements["primary"],
            "alternative_responses": personalized_elements["alternatives"],
            "follow_up_responses": empathic_followups,
            "emotional_tone": self._calculate_emotional_tone(emotion, intensity),
            "personalization_level": len(personalized_elements.get("personal_elements", [])),
            "expected_connection_strength": self._predict_connection_strength(personalized_elements, emotional_profile),
            "memory_references": personalized_elements.get("memory_refs", []),
            "usage_recommendations": self._generate_usage_recommendations(personalized_elements, emotional_profile),
        }

        # Enregistrer pour apprentissage
        self._record_response_generation(empathic_response)

        return empathic_response

    def _determine_personality_style(self, user_context: dict[str, Any] | None) -> str:
        """Déterminer le style de personnalité à utiliser - AVEC SUPPORT MODE NATUREL"""
        if not user_context:
            return "naturel"  # Style par défaut modernisé

        # Analyser les préférences utilisateur avec nouveau mode naturel
        if user_context.get("communication_mode") == "naturel":
            return "naturel"
        elif user_context.get("communication_mode") == "poetic":
            return "poetic"
        elif user_context.get("communication_preference") == "direct":
            return "direct"
        elif user_context.get("needs_analysis", False):
            return "intellectual"
        else:
            return "naturel"  # Favoriser le mode naturel par défaut

    def _select_templates(self, emotion: str, strategy: dict[str, Any], intensity: float) -> dict[str, list[str]]:
        """Sélectionner les templates appropriés selon l'émotion et la stratégie"""
        if emotion not in self.empathy_templates:
            emotion = "tristesse"  # Fallback empathique

        emotion_templates = self.empathy_templates[emotion]

        # Sélectionner selon l'approche de mirroring
        approach = strategy.get("mirroring_approach", "empathique_réconfortant")

        selected = {}

        if "validation" in approach or "empathique" in approach:
            selected.update({k: v for k, v in emotion_templates.items() if "validation" in k or "acknowledgment" in k})

        if "amplification" in approach or "célébration" in approach:
            selected.update(
                {
                    k: v
                    for k, v in emotion_templates.items()
                    if "amplification" in k or "celebration" in k or "matching" in k
                }
            )

        if "apaisement" in approach or "structurant" in approach:
            selected.update({k: v for k, v in emotion_templates.items() if "grounding" in k or "support" in k})

        # Fallback : prendre tous les templates disponibles
        if not selected:
            selected = emotion_templates

        return selected

    def _generate_primary_responses(
        self,
        templates: dict[str, list[str]],
        emotion: str,
        personality_style: str,
        memory_context: dict[str, Any] | None,
    ) -> list[str]:
        """Générer les réponses primaires personnalisées"""
        responses = []

        for category, template_list in templates.items():
            for template in template_list[:2]:  # Limiter à 2 par catégorie
                # Variables de substitution
                substitutions = {
                    "emotion": emotion,
                    "quality": random.choice(self.emotional_qualities.get(emotion, ["être authentique"])),
                    "alone_word": random.choice(self.connection_words["alone_word"]),
                    "positive_quality": random.choice(self.connection_words["positive_quality"]),
                    "situation": random.choice(self.connection_words["situation_words"]),
                }

                # Ajouter références mémoire si disponibles
                if memory_context and "personal_references" in memory_context:
                    personal_refs = memory_context["personal_references"]
                    if personal_refs:  # Vérifier que la liste n'est pas vide
                        ref = personal_refs[0]
                        substitutions["memory_context"] = ref.get("context", "notre histoire")
                    else:
                        substitutions["memory_context"] = "notre histoire"

                # Appliquer les substitutions
                response = template.format(**substitutions)

                # Adapter selon le style de personnalité
                response = self._adapt_to_personality_style(response, personality_style)

                responses.append(response)

        return responses

    def _adapt_to_personality_style(self, response: str, style: str) -> str:
        """Adapter la réponse selon le style de personnalité"""
        style_config = self.personality_styles.get(style, {})

        if style == "poetic" and style_config.get("imagery_richness", 0) > 0.7:
            # Ajouter des métaphores
            metaphorical_additions = {
                "émotion": "cette émotion comme une rivière qui traverse ton cœur",
                "moment": "cet instant suspendu dans le temps",
                "cœur": "ton cœur comme un jardin secret",
            }

            for word, metaphor in metaphorical_additions.items():
                if word in response:
                    response = response.replace(word, metaphor, 1)

        elif style == "naturel" and style_config.get("spontaneity", 0) > 0.8:
            # Mode naturel : spontané, authentique, décontracté
            natural_replacements = {
                "Cette émotion": "Ce que tu ressens",
                "Il y a quelque chose de": "C'est vraiment",
                "témoigne de": "montre",
                "magnifique": "super beau",
                "profondément": "vraiment",
                "Je ressens": "Je sens",
            }

            for formal, natural in natural_replacements.items():
                response = response.replace(formal, natural)

            # Ajouter spontanéité occasionnelle
            if random.random() < 0.3:
                spontaneous_intros = ["Hey, ", "Écoute, ", "Tu sais quoi ? "]
                response = random.choice(spontaneous_intros) + response.lower()

        elif style == "direct" and style_config.get("clarity", 0) > 0.8:
            # Simplifier et clarifier
            response = response.replace("Cette émotion que tu", "Ta")
            response = response.replace("Il y a quelque chose de", "C'est")

        elif style == "nurturing" and style_config.get("warmth", 0) > 0.8:
            # Ajouter chaleur et protection
            nurturing_prefixes = [
                "Mon cœur entend le tien... ",
                "Avec toute ma tendresse... ",
                "Dans cette douceur partagée... ",
            ]
            if not any(prefix.strip() in response for prefix in nurturing_prefixes):
                response = random.choice(nurturing_prefixes) + response.lower()

        elif style == "intellectual" and style_config.get("depth", 0) > 0.7:
            # Ajouter profondeur et analyse
            response += " Cette expérience révèle quelque chose de profond sur ta nature humaine."

        return response

    def _create_response_variations(
        self,
        primary_responses: list[str],
        emotional_profile: dict[str, Any],
        personality_style: str,
    ) -> dict[str, Any]:
        """Créer des variations de réponses pour différents contextes"""

        variations = {
            "primary": primary_responses[0] if primary_responses else "Je suis là avec toi.",
            "alternatives": primary_responses[1:4] if len(primary_responses) > 1 else [],
            "short_form": self._create_short_responses(primary_responses),
            "extended_form": self._create_extended_responses(primary_responses, emotional_profile),
            "casual_tone": self._create_casual_variations(primary_responses),
            "formal_tone": self._create_formal_variations(primary_responses),
        }

        return variations

    def _create_short_responses(self, responses: list[str]) -> list[str]:
        """Créer des versions courtes des réponses"""
        short_responses = []

        for response in responses[:3]:
            # Extraire la première phrase ou l'idée principale
            sentences = response.split(".")
            short_version = (sentences[0] + ".") if sentences else response

            # Si trop long, raccourcir davantage
            if len(short_version) > 50:
                words = short_version.split()[:8]
                short_version = " ".join(words) + "..."

            short_responses.append(short_version)

        return short_responses

    def _create_extended_responses(self, responses: list[str], emotional_profile: dict[str, Any]) -> list[str]:
        """Créer des versions étendues avec plus de développement"""
        extended = []

        for response in responses[:2]:
            extension_elements = []

            # Ajouter contexte émotionnel
            emotion = emotional_profile["dominant_emotion"]["name"]
            if emotion in ["tristesse", "stress"]:
                extension_elements.append("Je reste présent à tes côtés dans cette traversée.")
            elif emotion in ["joie", "enthousiasme"]:
                extension_elements.append("Ton énergie nourrit notre connexion et illumine cet échange.")

            # Ajouter invitation à approfondir
            extension_elements.append("N'hésite pas à partager tout ce que ton cœur porte.")

            extended_response = response + " " + " ".join(extension_elements)
            extended.append(extended_response)

        return extended

    def _create_casual_variations(self, responses: list[str]) -> list[str]:
        """Créer des variations plus décontractées"""
        casual = []

        for response in responses[:2]:
            casual_version = response

            # Remplacements pour un ton plus décontracté
            casual_replacements = {
                "Cette émotion": "Ce que tu ressens",
                "Il y a quelque chose de": "C'est vraiment",
                "témoigne de": "montre",
                "profondément": "vraiment",
                "magnifique": "super beau",
            }

            for formal, casual_word in casual_replacements.items():
                casual_version = casual_version.replace(formal, casual_word)

            casual.append(casual_version)

        return casual

    def _create_formal_variations(self, responses: list[str]) -> list[str]:
        """Créer des variations plus formelles"""
        formal = []

        for response in responses[:2]:
            formal_version = response

            # Remplacements pour un ton plus formel
            formal_replacements = {
                "C'est": "Il s'agit de",
                "super": "remarquablement",
                "vraiment": "véritablement",
                "ton cœur": "votre être intérieur",
            }

            for casual_word, formal_word in formal_replacements.items():
                formal_version = formal_version.replace(casual_word, formal_word)

            formal.append(formal_version)

        return formal

    def _add_deep_personalization(
        self,
        variations: dict[str, Any],
        user_context: dict[str, Any] | None,
        memory_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Ajouter des éléments de personnalisation profonde"""

        personal_elements = []
        memory_refs = []

        # Personnalisation basée sur l'historique
        if memory_context and "similar_emotional_moments" in memory_context:
            for moment in memory_context["similar_emotional_moments"][:2]:
                if "what_helped" in moment:
                    personal_elements.append(f"Comme {moment['what_helped']} t'avait aidé la dernière fois...")
                    memory_refs.append(moment)

        # Personnalisation basée sur les préférences
        if user_context:
            if user_context.get("prefers_metaphors", False):
                personal_elements.append("Comme un jardin après la pluie, ton cœur va refleurir...")

            if user_context.get("values_growth", False):
                personal_elements.append("Cette expérience fait partie de ton chemin de croissance...")

        # Intégrer les éléments personnels
        if personal_elements:
            # Ajouter à la réponse principale - protection contre liste vide
            first_element = personal_elements[0] if personal_elements else ""
            enhanced_primary = variations["primary"] + " " + first_element
            variations["primary"] = enhanced_primary

            # Créer des alternatives avec d'autres éléments personnels
            for i, element in enumerate(personal_elements[1:3]):
                if i < len(variations["alternatives"]):
                    variations["alternatives"][i] += " " + element

        variations["personal_elements"] = personal_elements
        variations["memory_refs"] = memory_refs

        return variations

    def _generate_empathic_followups(self, emotion: str, intensity: float, personality_style: str) -> list[str]:
        """Générer des follow-ups empathiques appropriés"""

        base_followups = {
            "tristesse": [
                "Comment puis-je t'accompagner dans cette émotion ?",
                "Y a-t-il quelque chose de particulier qui t'aiderait maintenant ?",
                "Veux-tu qu'on explore ensemble ce qui se passe pour toi ?",
            ],
            "joie": [
                "Qu'est-ce qui rend ce moment si spécial pour toi ?",
                "Comment on peut faire durer cette belle énergie ?",
                "Avec qui aimerais-tu partager cette joie ?",
            ],
            "stress": [
                "Sur quoi aimerais-tu qu'on se concentre en premier ?",
                "Qu'est-ce qui t'aiderait à retrouver un peu de calme ?",
                "Veux-tu qu'on organise tes priorités ensemble ?",
            ],
            "enthousiasme": [
                "Raconte-moi tout ! Qu'est-ce qui t'inspire tant ?",
                "Comment on peut donner vie à cette énergie créative ?",
                "Quels sont tes rêves les plus fous en ce moment ?",
            ],
        }

        followups = base_followups.get(
            emotion,
            [
                "Comment te sens-tu maintenant ?",
                "De quoi as-tu besoin en ce moment ?",
                "Veux-tu continuer à explorer ça ensemble ?",
            ],
        )

        # Adapter selon l'intensité
        if intensity > 0.8:
            followups = [f"D'abord, {fu.lower()}" for fu in followups]
        elif intensity < 0.3:
            followups = [f"Si tu veux, {fu.lower()}" for fu in followups]

        return followups[:3]  # Limiter à 3 follow-ups

    def _calculate_emotional_tone(self, emotion: str, intensity: float) -> dict[str, float]:
        """Calculer le ton émotionnel de la réponse"""

        base_tones = {
            "tristesse": {"warmth": 0.9, "gentleness": 0.9, "support": 0.8},
            "joie": {"energy": 0.8, "celebration": 0.9, "enthusiasm": 0.7},
            "stress": {"calm": 0.9, "structure": 0.8, "reassurance": 0.8},
            "enthousiasme": {"energy": 0.9, "creativity": 0.8, "excitement": 0.9},
            "frustration": {"understanding": 0.9, "patience": 0.8, "validation": 0.9},
            "sérénité": {"peace": 0.9, "depth": 0.7, "wisdom": 0.6},
        }

        tone = base_tones.get(emotion, {"warmth": 0.7, "understanding": 0.7})

        # Ajuster selon l'intensité
        for key, value in tone.items():
            tone[key] = min(value * (1 + intensity * 0.3), 1.0)

        return tone

    def _predict_connection_strength(
        self, response_elements: dict[str, Any], emotional_profile: dict[str, Any]
    ) -> float:
        """Prédire la force de connexion émotionnelle"""

        connection_factors = {
            "personalization": len(response_elements.get("personal_elements", [])) * 0.15,
            "memory_integration": len(response_elements.get("memory_refs", [])) * 0.20,
            "emotional_match": emotional_profile.get("confidence", 0.5) * 0.25,
            "response_quality": min(len(response_elements.get("alternatives", [])) / 3.0, 1.0) * 0.15,
            "empathy_depth": 0.25,  # Base empathy score
        }

        total_connection = sum(connection_factors.values())
        return min(total_connection, 1.0)

    def _generate_usage_recommendations(
        self, response_elements: dict[str, Any], emotional_profile: dict[str, Any]
    ) -> list[str]:
        """Générer des recommandations d'usage pour les réponses"""

        recommendations = []
        intensity = emotional_profile["intensity"]

        if intensity > 0.8:
            recommendations.append("Utiliser la réponse principale pour un impact immédiat")
            recommendations.append("Suivre avec un follow-up apaisant")
        elif intensity > 0.5:
            recommendations.append("Combiner réponse principale et alternative pour plus de richesse")
        else:
            recommendations.append("Commencer par une version courte, étendre si nécessaire")

        if response_elements.get("memory_refs"):
            recommendations.append("Intégrer les références mémoire pour renforcer la connexion")

        if len(response_elements.get("personal_elements", [])) > 0:
            recommendations.append("Mettre l'accent sur les éléments personnalisés")

        return recommendations

    def _record_response_generation(self, response: dict[str, Any]):
        """Enregistrer la génération de réponse pour apprentissage"""

        self.response_history.append(
            {
                "timestamp": response["timestamp"],
                "emotion": response["target_emotion"],
                "intensity": response["intensity_level"],
                "style": response["personality_style"],
                "personalization_level": response["personalization_level"],
                "predicted_connection": response["expected_connection_strength"],
            }
        )

        # Garder seulement les 500 dernières réponses
        if len(self.response_history) > 500:
            self.response_history = self.response_history[-500:]

    def get_empathy_analytics(self) -> dict[str, Any]:
        """Obtenir des analyses sur l'efficacité de l'empathie"""

        if not self.response_history:
            return {"status": "no_data"}

        # Analyser les patterns
        emotion_distribution = {}
        style_distribution = {}
        avg_connection_by_emotion = {}

        for response in self.response_history:
            emotion = response["emotion"]
            style = response["style"]

            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            style_distribution[style] = style_distribution.get(style, 0) + 1

            if emotion not in avg_connection_by_emotion:
                avg_connection_by_emotion[emotion] = []
            avg_connection_by_emotion[emotion].append(response["predicted_connection"])

        # Calculer moyennes
        for emotion in avg_connection_by_emotion:
            connections = avg_connection_by_emotion[emotion]
            avg_connection_by_emotion[emotion] = sum(connections) / len(connections)

        return {
            "status": "analyzed",
            "total_responses": len(self.response_history),
            "emotion_distribution": emotion_distribution,
            "style_distribution": style_distribution,
            "connection_strength_by_emotion": avg_connection_by_emotion,
            "most_empathized_emotion": max(emotion_distribution, key=emotion_distribution.get),
            "preferred_style": max(style_distribution, key=style_distribution.get),
            "average_connection_strength": sum(r["predicted_connection"] for r in self.response_history)
            / len(self.response_history),
        }
