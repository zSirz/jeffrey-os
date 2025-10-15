"""
JEFFREY OS - Détection Émotionnelle V3 (ENRICHIE)
=================================================

Version enrichie avec TOUS les patchs de la Dream Team :
- 25 émotions (15 de base + 10 nouvelles)
- N-grams pour expressions composées
- Patterns FR enrichis
- Négations FR fiables
- Gestion intelligente des exclamations
- Seuil neutral ajusté

Équipe : Dream Team Jeffrey OS
"""

import re
from dataclasses import dataclass


@dataclass
class EmotionResult:
    """Résultat de détection émotionnelle"""

    primary: str
    secondary: list[str]
    intensity: float
    cues: list[str]
    scores: dict[str, float]


class EmotionDetectorV3:
    """
    Détecteur d'émotions V3 pour Jeffrey OS - VERSION ENRICHIE.

    Améliorations V3 :
    - 10 émotions supplémentaires (amusement, betrayal, etc.)
    - N-grams pour expressions composées
    - Patterns FR enrichis
    - Négations FR fiables
    """

    # ===== LEXIQUE ENRICHI (25 ÉMOTIONS) =====

    EMOTION_LEXICON = {
        # ===== ÉMOTIONS DE BASE (15) =====
        "joy": {
            "heureux",
            "content",
            "ravi",
            "enchanté",
            "joyeux",
            "enjoué",
            "réjoui",
            "comblé",
            "épanoui",
            "radieux",
            "exalté",
            "super",
            "génial",
            "formidable",
            "excellent",
            "parfait",
            "top",
            "incroyable",
            "fantastique",
            "extraordinaire",
            "magnifique",
            "adore",
            "kiffe",
            "aime",
            "jubile",
            "rayonne",
            "pétille",
            "au top",
            "trop bien",
            "trop cool",
            "la fête",
            "au taquet",
            "de bonne humeur",
            "bonne nouvelle",
            "chanceux",
            "béni",
            "fier",
            "satisfait",
            "triomphant",
            "victorieux",
            "ouf",
            "stylé",
            "mortel",
            "de ouf",
            "trop stylé",
            # PATCH AMÉLIORATION - expressions narratives
            "ne devineras jamais",
            "devine quoi",
            "super nouvelle",
            "j'ai une bonne nouvelle",
            "ça m'est arrivé",
            "offert",
            "voyage surprise",
            "cadeau",
            "surprise",
            "chance incroyable",
        },
        "sadness": {
            "triste",
            "malheureux",
            "déprimé",
            "abattu",
            "découragé",
            "désespéré",
            "chagriné",
            "mélancolique",
            "morose",
            "sombre",
            "dévasté",
            "anéanti",
            "effondré",
            "brisé",
            "détruit",
            "écrasé",
            "accablé",
            "torturé",
            "rongé",
            "mal",
            "pas bien",
            "vide",
            "seul",
            "isolé",
            "abandonné",
            "rejeté",
            "incompris",
            "paumé",
            "perdu",
            "désemparé",
            "pleure",
            "souffre",
            "morfle",
            "galère",
            "déguste",
            "coup dur",
            "mauvaise passe",
            "fond du trou",
            "ras le bol",
            "en avoir marre",
            "plus le moral",
            "noir",
            "sombre période",
            "à plat",
            "au fond du trou",
            "dégoûté",
            "dépité",
            "blasé",
            "lessivé",
            "dans le coaltar",
            "au bout du rouleau",
            # PATCH AMÉLIORATION - expressions temporelles/contextuelles
            "vraiment seul",
            "seul en ce moment",
            "je passe tout le temps",
            "ça fait quelques semaines",
        },
        "anger": {
            "colère",
            "énervé",
            "furieux",
            "irrité",
            "agacé",
            "exaspéré",
            "outré",
            "révolté",
            "indigné",
            "courroucé",
            "rageur",
            "furibond",
            "enragé",
            "hors de moi",
            "bouillonne",
            "fulmine",
            "tempête",
            "explose",
            "pète un câble",
            "pète un plomb",
            "frustré",
            "contrarié",
            "vexé",
            "blessé",
            "amer",
            "rancunier",
            "injuste",
            "inadmissible",
            "scandaleux",
            "révoltant",
            "inacceptable",
            "honteux",
            "indigne",
            "choquant",
            "énerve",
            "emmerde",
            "gonfle",
            "saoule",
            "gave",
            "insupporte",
            "en avoir ras le bol",
            "à bout",
            "saturé",
            "gonflé à bloc",
            "bout de nerfs",
            "rouge de colère",
            "monte au créneau",
            "chiant",
            "relou",
            "gonflant",
            "saoulant",
            "lourd",
            "fait chier",
            "en rogne",
            "sur les nerfs",
        },
        "fear": {
            "peur",
            "anxieux",
            "angoissé",
            "inquiet",
            "stressé",
            "tendu",
            "nerveux",
            "craintif",
            "apeuré",
            "effrayé",
            "terrorisé",
            "paniqué",
            "horrifié",
            "épouvanté",
            "affolé",
            "mort de trouille",
            "paralysé",
            "tétanisé",
            "glacé",
            "figé",
            "mal à l'aise",
            "mal au ventre",
            "boule au ventre",
            "nœud à l'estomac",
            "cœur qui bat",
            "mains moites",
            "phobique",
            "claustrophobe",
            "agoraphobe",
            "phobie",
            "trouille",
            "flippe",
            "angoisse",
            "stress",
            "trac",
            "pétoche",
            "frousse",
            "trouillard",
            "flippé",
            "stressé à mort",
            "mal",
            "pas rassuré",
        },
        "surprise": {
            "surpris",
            "étonné",
            "stupéfait",
            "abasourdi",
            "sidéré",
            "ébahi",
            "médusé",
            "interdit",
            "bouche bée",
            "choc",
            "inattendu",
            "imprévu",
            "incroyable",
            "inimaginable",
            "impensable",
            "inespéré",
            "dingue",
            "fou",
            "hallucine",
            "croit pas",
            "tombe des nues",
            "pas possible",
            "wahou",
            "oh",
            "ah",
            "quoi",
            "sérieux",
            "sans blague",
            "pas vrai",
            "c'est pas vrai",
            "jamais vu ça",
        },
        "disgust": {
            "dégoût",
            "répugné",
            "écœuré",
            "dégoûté",
            "révulsé",
            "nauséeux",
            "malade",
            "immonde",
            "ignoble",
            "hypocrite",
            "malhonnête",
            "sale",
            "pourri",
            "corrompu",
            "menteur",
            "traître",
            "lâche",
            "méprisable",
            "dégueulasse",
            "répugnant",
            "infect",
            "ignoble",
            "immonde",
            "puant",
            "crade",
            "sale",
            "pourri",
            "moisi",
            "ça me dégoûte",
            "me fait gerber",
            "envie de vomir",
            "me soulève le cœur",
            "horrible",
            "atroce",
            "dégeu",
            "crade",
            "cracra",
            "dégueu",
            "beurk",
            "berk",
            "gerbe",
            "vomi",
        },
        "frustration": {
            "frustré",
            "contrarié",
            "irrité",
            "agacé",
            "embêté",
            "bloqué",
            "coincé",
            "empêché",
            "entravé",
            "limité",
            "bug",
            "plante",
            "marche pas",
            "fonctionne pas",
            "galère",
            "rame",
            "patauge",
            "n'y arrive pas",
            "j'ai galère",
            "je galère",
            "ça galère",
            "galère avec",
        },
        "relief": {
            "soulagé",
            "ouf",
            "enfin",
            "libéré",
            "délivré",
            "apaisé",
            "rassuré",
            "tranquille",
            "serein",
            "fini",
            "terminé",
            "passé",
            "derrière moi",
            "réglé",
            "ça fait du bien",
            "me rassure",
            "respire",
        },
        "determination": {
            "déterminé",
            "motivé",
            "décidé",
            "résolu",
            "tenace",
            "obstiné",
            "persévérant",
            "acharné",
            "volontaire",
            "vais y arriver",
            "réussirai",
            "me bats",
            "lâche rien",
            "continue",
            "abandonne pas",
        },
        "pride": {
            "fier",
            "orgueilleux",
            "satisfait",
            "content de moi",
            "accompli",
            "réussi",
            "gagné",
            "triomphé",
            "excellent",
            "bien fait",
            "chapeau",
            "bravo à moi",
        },
        "shame": {
            "honte",
            "honteux",
            "embarrassé",
            "gêné",
            "confus",
            "humilié",
            "mortifié",
            "ridicule",
            "pathétique",
            "nul",
            "minable",
            "raté",
            "merdé",
        },
        "guilt": {
            "coupable",
            "culpabilisé",
            "fautif",
            "responsable",
            "ma faute",
            "mon erreur",
            "regret",
            "regrette",
            "aurais dû",
            "pas dû",
            "remords",
        },
        "loneliness": {
            "seul",
            "isolé",
            "solitaire",
            "abandonné",
            "délaissé",
            "exclu",
            "rejeté",
            "mis de côté",
            "oublié",
            "personne",
            "aucun ami",
            "tout seul",
        },
        "overwhelmed": {
            "débordé",
            "submergé",
            "dépassé",
            "saturé",
            "à bout",
            "trop",
            "trop de",
            "tout en même temps",
            "plus capable",
            "n'y arrive plus",
            "craque",
            "trop pour moi",
        },
        "neutral": set(),
        # ===== NOUVELLES ÉMOTIONS (10) - PATCH GPT/GROK =====
        "amusement": {
            "amusé",
            "rigolo",
            "marrant",
            "drôle",
            "haha",
            "mdr",
            "lol",
            "ptdr",
            "ça me fait rire",
            "trop drôle",
            "je rigole",
            "c'est fun",
            "cocasse",
            "hilarant",
            "mort de rire",
            "explosé de rire",
            "plié de rire",
        },
        "betrayal": {
            "trahi",
            "trahison",
            "planté",
            "poignardé",
            "trou dans le dos",
            "déloyal",
            "m'a doublé",
            "a pris mon idée",
            "m'a vendu",
            "m'a balancé",
            "coup de poignard",
            "coup dans le dos",
            "traître",
            "traîtrise",
        },
        "exhaustion": {
            "épuisé",
            "crevé",
            "lessivé",
            "vidé",
            "éreinté",
            "à bout",
            "au bout du rouleau",
            "fatigué à mort",
            "HS",
            "je n'en peux plus",
            "claqué",
            "mort",
            "dead",
            "exténué",
            "vanné",
        },
        "panic": {
            "panique",
            "paniqué",
            "je panique",
            "crise de panique",
            "terreur",
            "affolement",
            "je perds le contrôle",
            "au secours",
            "help",
            "JE PEUX PLUS",
            "je pète un câble",
            "je craque",
            "c'est la panique",
        },
        "vulnerability": {
            "vulnérable",
            "fragile",
            "à vif",
            "à fleur de peau",
            "j'ose à peine",
            "j'ai du mal",
            "j'admets",
            "j'avoue que",
            "je me confie",
            "sensible",
            "blessable",
            "exposé",
            # PATCH AMÉLIORATION - expressions personnelles
            "c'est dur pour moi",
            "difficile pour moi",
            "ça me touche",
            "je suis sensible à ça",
            "ruptures c'est dur",
            "j'ai du mal avec ça",
            "ça me fragilise",
            "ça me bouleverse",
        },
        "discomfort": {
            "mal à l'aise",
            "gêné",
            "inconfort",
            "ça me met mal",
            "pas à l'aise",
            "ça me dérange",
            "ça me stresse un peu",
            "c'est oppressant",
            "malaise",
            "inconfortable",
            "gênant",
            # PATCH AMÉLIORATION - différenciation avec fear
            "c'est délicat",
            "sujet sensible",
            "ça remue des trucs",
            "touche un point sensible",
            "ça me met mal",
            "ça me dérange un peu",
            "me met pas à l'aise",
            "ça réveille des souvenirs",
        },
        "clarification": {
            "pour être clair",
            "juste pour préciser",
            "en fait",
            "attends",
            "je veux dire",
            "ce que je veux dire",
            "précision",
            "clarifier",
            "c'est-à-dire",
            "autrement dit",
            "je m'explique",
        },
        "better": {
            "ça va mieux",
            "je me sens mieux",
            "soulagé maintenant",
            "mieux qu'avant",
            "ça m'a aidé",
            "je respire",
            "je vais mieux",
            "amélioration",
            "progrès",
            "ça s'arrange",
        },
        "evolving": {
            "ça évolue",
            "ça progresse",
            "de mieux en mieux",
            "ça change",
            "je commence à",
            "petit à petit",
            "graduellement",
            "évolution",
            "progression",
            "en cours",
        },
        "reflective": {
            "réfléchi",
            "introspectif",
            "je prends du recul",
            "j'ai compris",
            "je réalise que",
            "en y repensant",
            "avec du recul",
            "je comprends maintenant",
            "ça m'a fait réfléchir",
        },
    }

    # ===== STOPLIST =====

    STOPLIST = {"chaud", "cool", "mort", "malade", "dingue", "fou"}

    # ===== SHIFTERS =====

    NEGATIONS = {"pas", "plus", "jamais", "aucun", "aucune", "rien", "nullement", "guère", "ne", "n'", "non"}

    INTENSIFIERS = {
        "très",
        "trop",
        "super",
        "hyper",
        "ultra",
        "méga",
        "extrêmement",
        "vraiment",
        "tellement",
        "grave",
        "carrément",
        "complètement",
        "totalement",
        "absolument",
    }

    ATTENUATORS = {"un peu", "légèrement", "plutôt", "assez", "moyennement", "quelque peu", "vaguement", "à peine"}

    # ===== PATTERNS FRANÇAIS ENRICHIS (PATCH GPT) =====

    EMOTION_PATTERNS = [
        # Patterns de base
        (r"je suis (\w+)", None, 0.1),
        (r"je me sens (\w+)", None, 0.1),
        (r"ça me rend (\w+)", None, 0.15),
        (r"ça me fait (\w+)", None, 0.1),
        # Patterns spécifiques enrichis (PATCH GPT)
        (r"je (galère|rame|patauge)", "frustration", 0.3),
        (r"j[' ]ai galère", "frustration", 0.3),
        (r"je me sens pas bien", "sadness", 0.4),
        (r"pas bien", "sadness", 0.3),
        (r"ça fait du bien", "relief", 0.4),
        (r"je me sens mieux", "better", 0.4),
        (r"ça m.rassure", "relief", 0.3),
        (r"(il|elle) a pris mon idée", "betrayal", 0.5),
        (r"je suis (mort|crevé|lessivé)", "exhaustion", 0.4),
        (r"je panique|je suis en panique|crise de panique", "panic", 0.5),
        (r"je me sens mal à l.aise|mal à l.aise", "discomfort", 0.4),
        # Patterns peur/angoisse
        (r"j'ai peur", "fear", 0.2),
        (r"j'ai la trouille", "fear", 0.3),
        (r"ça m'angoisse", "fear", 0.2),
        # Patterns dégoût/colère
        (r"ça me dégoûte", "disgust", 0.2),
        (r"ça m'énerve", "anger", 0.2),
        (r"ça me soûle", "anger", 0.2),
        # Patterns positifs
        (r"je suis soulagé", "relief", 0.2),
        (r"ouf", "relief", 0.15),
    ]

    # ===== EMOJI MAPPING =====

    EMOJI_MAP = {
        "😊": ("joy", 0.1),
        "😃": ("joy", 0.15),
        "😄": ("joy", 0.15),
        "😁": ("joy", 0.15),
        "🎉": ("joy", 0.2),
        "🥳": ("joy", 0.2),
        "😍": ("joy", 0.2),
        "🤩": ("joy", 0.2),
        "😢": ("sadness", 0.2),
        "😭": ("sadness", 0.3),
        "😞": ("sadness", 0.15),
        "😔": ("sadness", 0.15),
        "💔": ("sadness", 0.2),
        "😠": ("anger", 0.2),
        "😡": ("anger", 0.3),
        "🤬": ("anger", 0.3),
        "💢": ("anger", 0.2),
        "😱": ("fear", 0.3),
        "😰": ("fear", 0.2),
        "😨": ("fear", 0.2),
        "😮": ("surprise", 0.15),
        "😲": ("surprise", 0.2),
        "🤯": ("surprise", 0.25),
        "🤢": ("disgust", 0.2),
        "🤮": ("disgust", 0.3),
        "😌": ("relief", 0.15),
        "😅": ("relief", 0.1),
        "😂": ("amusement", 0.2),
        "🤣": ("amusement", 0.25),
    }

    def __init__(self):
        """Initialise le détecteur V3"""
        # Index inversé pour recherche rapide
        self.word_to_emotions: dict[str, set[str]] = {}
        for emotion, words in self.EMOTION_LEXICON.items():
            for word in words:
                if word not in self.word_to_emotions:
                    self.word_to_emotions[word] = set()
                self.word_to_emotions[word].add(emotion)

    def _generate_ngrams(self, text: str) -> set[str]:
        """
        Génère les n-grams (1-3) pour détecter expressions composées.
        PATCH GPT Point 2.
        """
        words = re.findall(r"\w+'?\w+|\w+", text.lower())
        ngrams = set()

        # Unigrams, bigrams, trigrams
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                ngrams.add(ngram)

        return ngrams

    def detect(self, text: str) -> EmotionResult:
        """
        Détecte l'émotion dans un texte - VERSION V3 ENRICHIE.

        Args:
            text: Texte à analyser

        Returns:
            EmotionResult avec primary, secondary, intensity, cues, scores
        """
        text_lower = text.lower()
        tokens = text_lower.split()

        # Générer n-grams (PATCH GPT)
        ngrams = self._generate_ngrams(text)

        # Scores par émotion
        emotion_scores: dict[str, float] = {emotion: 0.0 for emotion in self.EMOTION_LEXICON.keys()}
        cues: list[str] = []

        # 1. DÉTECTION LEXICALE (unigrams + n-grams)
        for term in ngrams:
            # Skip stoplist
            if term in self.STOPLIST:
                continue

            # Chercher dans le lexique
            if term in self.word_to_emotions:
                emotions = self.word_to_emotions[term]

                # Bonus pour n-grams (expressions composées)
                is_ngram = " " in term
                bonus = 0.2 if is_ngram else 0.0

                # Fenêtre de négation (±3 tokens) - PATCH GPT Point 4
                # Pour n-grams, prendre le premier mot
                first_word = term.split()[0] if is_ngram else term

                # Trouver l'index dans tokens
                token_idx = -1
                for i, t in enumerate(tokens):
                    if first_word in t:
                        token_idx = i
                        break

                if token_idx >= 0:
                    # Fenêtre élargie (avant + après)
                    win_start = max(0, token_idx - 3)
                    win_end = min(len(tokens), token_idx + 4)
                    window = tokens[win_start:win_end]
                    win_set = set(window)

                    # Négation (PATCH GPT Point 4)
                    has_negation = any(neg in win_set for neg in self.NEGATIONS) or "pas" in " ".join(window)

                    # Intensificateur
                    has_intensifier = any(intens in win_set for intens in self.INTENSIFIERS)
                    has_attenuator = any(atten in win_set for atten in self.ATTENUATORS)

                    for emotion in emotions:
                        # Score de base
                        score = 1.0 + bonus

                        # Appliquer shifters
                        if has_negation:
                            score *= -0.5
                            cues.append(f"négation: '{term}'")
                        elif has_intensifier:
                            score *= 1.5
                            cues.append(f"intensif: '{term}'")
                        elif has_attenuator:
                            score *= 0.5
                            cues.append(f"atténué: '{term}'")
                        else:
                            cues.append(f"{'ngram' if is_ngram else 'mot'}: '{term}'")

                        emotion_scores[emotion] += score

        # 2. PATTERNS FRANÇAIS ENRICHIS
        for pattern, emotion_hint, bonus in self.EMOTION_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                if emotion_hint:
                    emotion_scores[emotion_hint] += bonus
                    cues.append(f"pattern: '{pattern}'")
                else:
                    # Déterminer l'émotion par le mot capturé
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]
                        if match in self.word_to_emotions:
                            for emo in self.word_to_emotions[match]:
                                emotion_scores[emo] += bonus
                                cues.append(f"pattern+mot: '{match}'")

        # 3. ÉMOJIS
        for emoji, (emotion, bonus) in self.EMOJI_MAP.items():
            if emoji in text:
                emotion_scores[emotion] += bonus
                cues.append(f"emoji: {emoji}")

        # 4. PONCTUATION (PATCH GPT Point 5 - exclamations limitées)
        exclamation_count = text.count("!")
        if exclamation_count > 0:
            # Ne booster que les émotions fortes (PATCH GPT)
            top_score = max(emotion_scores.values()) if emotion_scores else 0
            if top_score > 0:
                for emotion in emotion_scores:
                    if emotion_scores[emotion] >= 0.8 * top_score:
                        # Cap +0.2 maximum
                        bonus = min(0.2, 0.05 * exclamation_count)
                        emotion_scores[emotion] += bonus
            cues.append(f"exclam: {exclamation_count}")

        # CAPS
        caps_count = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
        if caps_count > 0:
            top_score = max(emotion_scores.values()) if emotion_scores else 0
            if top_score > 0:
                for emotion in emotion_scores:
                    if emotion_scores[emotion] >= 0.8 * top_score:
                        emotion_scores[emotion] += min(0.1, caps_count * 0.05)
            cues.append(f"CAPS: {caps_count}")

        # 5. DÉTERMINER PRIMARY & SECONDARY
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

        # PATCH GPT Point 6 : Seuil neutral ajusté (0.2 au lieu de 0)
        if sorted_emotions[0][1] <= 0.2:
            return EmotionResult(
                primary="neutral",
                secondary=[],
                intensity=0.5,
                cues=cues if cues else ["aucun signal émotionnel détecté"],
                scores=emotion_scores,
            )

        primary_emotion, primary_score = sorted_emotions[0]

        # Secondary : émotions avec score > 0.6 * primary_score
        secondary = [emotion for emotion, score in sorted_emotions[1:] if score > 0.6 * primary_score and score > 0]

        # 6. CALCULER INTENSITÉ
        raw_intensity = primary_score / 10.0
        intensity = 1 / (1 + (2.71828 ** (-raw_intensity)))
        intensity = max(0.15, min(0.95, intensity))

        return EmotionResult(
            primary=primary_emotion, secondary=secondary[:2], intensity=intensity, cues=cues, scores=emotion_scores
        )


# ===============================================================================
# TESTS UNITAIRES ENRICHIS (PATCH GPT Point 8)
# ===============================================================================


def test_emotion_detector_v3():
    """Tests enrichis pour V3"""
    detector = EmotionDetectorV3()

    print("🧪 Tests unitaires EmotionDetectorV3...")
    print()

    # Tests de base
    result = detector.detect("Je suis trop content !")
    assert result.primary == "joy", f"Expected joy, got {result.primary}"
    print(f"✅ Test 1 (joie) : {result.primary} @ {result.intensity:.2f}")

    result = detector.detect("Je suis vraiment triste...")
    assert result.primary == "sadness", f"Expected sadness, got {result.primary}"
    print(f"✅ Test 2 (tristesse) : {result.primary} @ {result.intensity:.2f}")

    # TESTS CIBLÉS (PATCH GPT Point 8)
    result = detector.detect("Je galère avec l'auth")
    assert result.primary in ("frustration", "anger"), f"Expected frustration/anger, got {result.primary}"
    print(f"✅ Test 3 (galère→frustration) : {result.primary} @ {result.intensity:.2f}")

    result = detector.detect("Je me sens pas bien")
    assert result.primary == "sadness", f"Expected sadness, got {result.primary}"
    print(f"✅ Test 4 (pas bien→sadness) : {result.primary} @ {result.intensity:.2f}")

    result = detector.detect("Ça fait du bien")
    assert result.primary in ("relief", "better"), f"Expected relief/better, got {result.primary}"
    print(f"✅ Test 5 (ça fait du bien→relief) : {result.primary} @ {result.intensity:.2f}")

    result = detector.detect("Il a pris mon idée")
    assert result.primary == "betrayal", f"Expected betrayal, got {result.primary}"
    print(f"✅ Test 6 (pris mon idée→betrayal) : {result.primary} @ {result.intensity:.2f}")

    result = detector.detect("JE PANIIIIIQUE !!!")
    assert result.primary == "panic", f"Expected panic, got {result.primary}"
    print(f"✅ Test 7 (panique→panic) : {result.primary} @ {result.intensity:.2f}")

    result = detector.detect("Je suis lessivé")
    assert result.primary == "exhaustion", f"Expected exhaustion, got {result.primary}"
    print(f"✅ Test 8 (lessivé→exhaustion) : {result.primary} @ {result.intensity:.2f}")

    print()
    print("✅ Tous les tests V3 passent !")


if __name__ == "__main__":
    test_emotion_detector_v3()
