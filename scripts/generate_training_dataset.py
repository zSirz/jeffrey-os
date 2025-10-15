"""
Génération automatique d'un dataset d'entraînement YAML pour Jeffrey OS.
Utilise des phrases réalistes FR/EN avec labels core-8 emotions.
"""

from datetime import datetime
from pathlib import Path

import yaml

# Dataset de qualité FR/EN par émotion (15 exemples/émotion = 120 total)
TRAINING_EXAMPLES = {
    "joy": {
        "fr": [
            "Je suis tellement content aujourd'hui !",
            "C'est fantastique, j'adore cette nouvelle !",
            "Quelle joie de te revoir !",
            "Je suis aux anges, c'est parfait !",
            "Super, exactement ce que j'attendais !",
            "Je suis heureux comme jamais !",
            "C'est merveilleux, je suis ravi !",
            "Génial, ça me rend vraiment joyeux !",
        ],
        "en": [
            "I'm so happy today!",
            "This is fantastic, I love this news!",
            "What a joy to see you again!",
            "I'm over the moon, it's perfect!",
            "Great, exactly what I was hoping for!",
            "I'm happier than ever!",
            "This is wonderful, I'm delighted!",
        ],
    },
    "sadness": {
        "fr": [
            "Je me sens vraiment triste aujourd'hui.",
            "C'est déprimant, je n'ai plus d'espoir.",
            "Je suis déçu et abattu.",
            "Ça me rend mélancolique.",
            "Je suis triste de cette nouvelle.",
            "C'est difficile, je me sens mal.",
            "Je suis découragé par la situation.",
            "Ça me fait de la peine.",
        ],
        "en": [
            "I feel really sad today.",
            "This is depressing, I have no hope left.",
            "I'm disappointed and dejected.",
            "This makes me melancholic.",
            "I'm sad about this news.",
            "It's hard, I feel bad.",
            "I'm discouraged by the situation.",
        ],
    },
    "anger": {
        "fr": [
            "Ça m'énerve vraiment !",
            "Je suis en colère contre toi !",
            "C'est inacceptable, je suis furieux !",
            "Ça me met hors de moi !",
            "Je suis vraiment agacé par ça.",
            "C'est révoltant !",
            "Je ne supporte plus cette situation !",
            "Ça me met en rage !",
        ],
        "en": [
            "This really annoys me!",
            "I'm angry at you!",
            "This is unacceptable, I'm furious!",
            "This drives me crazy!",
            "I'm really irritated by this.",
            "This is revolting!",
            "I can't stand this situation anymore!",
        ],
    },
    "fear": {
        "fr": [
            "J'ai peur de ce qui va se passer.",
            "Ça m'inquiète énormément.",
            "Je suis terrifié par cette idée.",
            "C'est angoissant, je ne sais pas quoi faire.",
            "J'ai des craintes sur l'avenir.",
            "Ça me fait vraiment peur.",
            "Je suis anxieux à propos de ça.",
            "C'est effrayant.",
        ],
        "en": [
            "I'm afraid of what's going to happen.",
            "This worries me a lot.",
            "I'm terrified by this idea.",
            "This is distressing, I don't know what to do.",
            "I have fears about the future.",
            "This really scares me.",
            "I'm anxious about this.",
        ],
    },
    "surprise": {
        "fr": [
            "Wow, je ne m'y attendais vraiment pas !",
            "C'est incroyable, quelle surprise !",
            "Je suis stupéfait par cette nouvelle !",
            "Oh là là, je n'en reviens pas !",
            "C'est inattendu !",
            "Je suis étonné par ce résultat.",
            "Quelle surprise incroyable !",
            "Je ne pensais pas que ça arriverait !",
        ],
        "en": [
            "Wow, I really didn't expect that!",
            "This is incredible, what a surprise!",
            "I'm amazed by this news!",
            "Oh my, I can't believe it!",
            "This is unexpected!",
            "I'm astonished by this result.",
            "What an incredible surprise!",
        ],
    },
    "disgust": {
        "fr": [
            "Beurk, c'est dégoûtant !",
            "C'est répugnant, je ne supporte pas ça.",
            "Ça me dégoûte profondément.",
            "C'est immonde !",
            "Je trouve ça vraiment répulsif.",
            "C'est écœurant.",
            "Ça me donne la nausée.",
            "C'est vraiment dégueulasse.",
        ],
        "en": [
            "Yuck, this is disgusting!",
            "This is repulsive, I can't stand it.",
            "This disgusts me deeply.",
            "This is vile!",
            "I find this really repulsive.",
            "This is nauseating.",
            "This makes me sick.",
        ],
    },
    "neutral": {
        "fr": [
            "OK, j'ai bien noté.",
            "D'accord, je comprends.",
            "Merci pour l'information.",
            "Je vois.",
            "C'est noté.",
            "Bien reçu.",
            "Entendu.",
            "Je prends note.",
        ],
        "en": [
            "OK, noted.",
            "Alright, I understand.",
            "Thanks for the information.",
            "I see.",
            "Got it.",
            "Received.",
            "Understood.",
        ],
    },
    "frustration": {
        "fr": [
            "C'est vraiment frustrant à la longue.",
            "Je suis exaspéré par cette situation.",
            "Ça devient pénible.",
            "Je suis agacé par ces répétitions.",
            "C'est fatiguant de toujours devoir recommencer.",
            "Ça me frustre énormément.",
            "C'est usant.",
            "Je commence à perdre patience.",
        ],
        "en": [
            "This is really frustrating in the long run.",
            "I'm exasperated by this situation.",
            "This is becoming tedious.",
            "I'm annoyed by these repetitions.",
            "It's tiring to always have to start over.",
            "This frustrates me a lot.",
            "This is exhausting.",
        ],
    },
}

# Cas durs : négations, ironie, ambiguïté
HARD_CASES = [
    {"text": "Je ne suis pas heureux.", "label": "sadness", "lang": "fr", "note": "négation"},
    {"text": "Je n'ai pas peur du tout.", "label": "neutral", "lang": "fr", "note": "négation forte"},
    {"text": "I'm not angry at all.", "label": "neutral", "lang": "en", "note": "negation"},
    {"text": "Super, encore un problème... génial.", "label": "frustration", "lang": "fr", "note": "ironie"},
    {"text": "Great, another issue... fantastic.", "label": "frustration", "lang": "en", "note": "irony"},
    {
        "text": "Je suis content mais un peu inquiet aussi.",
        "label": "joy",
        "lang": "fr",
        "note": "émotion mixte - joy dominant",
    },
    {"text": "I'm happy but also a bit worried.", "label": "joy", "lang": "en", "note": "mixed emotion - joy dominant"},
]


def generate_yaml_files(output_dir: str = "data/conversations"):
    """Génère des fichiers YAML pour chaque scénario d'émotion."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    scenario_id = 1
    all_files = []

    # Générer fichiers par émotion
    for emotion, examples in TRAINING_EXAMPLES.items():
        for lang, texts in examples.items():
            for i, text in enumerate(texts, 1):
                scenario = {
                    "scenario_id": f"train_{emotion}_{lang}_{i}",
                    "emotion": emotion,
                    "language": lang,
                    "text": text,
                    "created_at": datetime.now().isoformat(),
                    "source": "auto_generated_v1",
                    "quality": "high",
                }

                filename = f"{output_dir}/scenario_{scenario_id:03d}_{emotion}_{lang}.yaml"
                with open(filename, 'w', encoding='utf-8') as f:
                    yaml.dump(scenario, f, allow_unicode=True, default_flow_style=False)

                all_files.append(filename)
                scenario_id += 1

    # Ajouter cas durs
    for i, case in enumerate(HARD_CASES, 1):
        scenario = {
            "scenario_id": f"hard_case_{i}",
            "emotion": case["label"],
            "language": case["lang"],
            "text": case["text"],
            "note": case["note"],
            "created_at": datetime.now().isoformat(),
            "source": "hard_cases_v1",
            "quality": "challenging",
        }

        filename = f"{output_dir}/scenario_{scenario_id:03d}_hard_{case['label']}.yaml"
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(scenario, f, allow_unicode=True, default_flow_style=False)

        all_files.append(filename)
        scenario_id += 1

    return all_files


def print_dataset_stats(files: list[str]):
    """Affiche les statistiques du dataset généré."""
    emotion_counts = {}
    lang_counts = {"fr": 0, "en": 0}

    for filepath in files:
        with open(filepath, encoding='utf-8') as f:
            data = yaml.safe_load(f)
            emotion = data.get("emotion")
            lang = data.get("language")

            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            if lang in lang_counts:
                lang_counts[lang] += 1

    print("\n📊 STATISTIQUES DU DATASET GÉNÉRÉ")
    print("=" * 50)
    print(f"Total fichiers : {len(files)}")
    print("\nRépartition par émotion :")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  - {emotion:12s} : {count:3d} exemples")
    print("\nRépartition par langue :")
    for lang, count in lang_counts.items():
        print(f"  - {lang:2s} : {count:3d} exemples")
    print("=" * 50)


if __name__ == "__main__":
    print("🚀 Génération du dataset d'entraînement Jeffrey OS...")
    print("📝 Création de YAML avec émotions core-8, FR/EN, cas durs inclus\n")

    files = generate_yaml_files()
    print_dataset_stats(files)

    print("\n✅ Dataset généré avec succès !")
    print("📂 Fichiers disponibles dans : data/conversations/")
    print("🎯 Prêt pour l'entraînement avec train_prototypes_optimized.py")
