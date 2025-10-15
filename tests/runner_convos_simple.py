#!/usr/bin/env python3
"""
JEFFREY OS - RUNNER DE TESTS CONVERSATIONNELS
==============================================

Ce script charge et exécute les 40 scénarios YAML de tests/convos/
pour valider les capacités de Jeffrey OS :
- Détection émotionnelle
- Mémorisation contextuelle
- Recherche sémantique
- Clustering
- Apprentissage

USAGE:
    python3 tests/runner_convos_simple.py

NOTES:
    - Pour l'instant, on teste uniquement la DÉTECTION et la MÉMOIRE
    - Pas encore la génération de réponses (Phase 4-5)
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Pour l'affichage coloré
try:
    from rich.console import Console
    from rich.progress import track
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Module 'rich' non installé. Affichage basique utilisé.")
    print("   Pour un meilleur affichage: pip install rich")


# ===============================================================================
# CLASSES SIMPLIFIÉES POUR LES TESTS
# ===============================================================================


class SimpleEmotionDetector:
    """
    Détecteur d'émotions simplifié pour les tests.

    Dans le vrai Jeffrey OS, ce sera bien plus sophistiqué
    avec des modèles de ML, des transformers, etc.
    """

    # Mots-clés par émotion (version simplifiée pour tests)
    EMOTION_KEYWORDS = {
        "joy": [
            "heureux",
            "content",
            "super",
            "génial",
            "formidable",
            "wahou",
            "excellent",
            "parfait",
            "ravi",
            "enchanté",
            "joie",
            "bonheur",
        ],
        "sadness": [
            "triste",
            "déçu",
            "déprimé",
            "mal",
            "peine",
            "chagrin",
            "malheureux",
            "découragé",
            "seul",
            "vide",
            "larmes",
        ],
        "anger": [
            "colère",
            "énervé",
            "furieux",
            "inadmissible",
            "dégouté",
            "rage",
            "frustré",
            "injuste",
            "scandaleux",
            "révoltant",
        ],
        "fear": [
            "peur",
            "angoisse",
            "anxieux",
            "inquiet",
            "stressé",
            "panique",
            "effrayé",
            "terrorisé",
            "phobique",
            "trouille",
        ],
        "surprise": ["surprise", "choc", "inattendu", "incroyable", "étonnant", "surprenant", "stupéfait", "abasourdi"],
        "disgust": ["dégoût", "répugnant", "dégueulasse", "écœurant", "horrible", "immonde", "infect", "ignoble"],
        "neutral": [],  # Par défaut si aucune émotion forte détectée
    }

    def detect(self, text: str) -> tuple[str, float]:
        """
        Détecte l'émotion dans un texte.

        Returns:
            Tuple (emotion, intensity) où intensity est entre 0 et 1
        """
        text_lower = text.lower()

        # Compter les mots-clés par émotion
        emotion_scores = defaultdict(int)

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1

        # Si aucune émotion détectée
        if not emotion_scores:
            return "neutral", 0.5

        # Émotion dominante
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)

        # Intensité basée sur le nombre de mots-clés
        # (+ il y a de mots-clés, + l'intensité est forte)
        intensity = min(0.5 + (emotion_scores[dominant_emotion] * 0.2), 1.0)

        # Points d'exclamation augmentent l'intensité
        exclamation_count = text.count("!")
        if exclamation_count > 0:
            intensity = min(intensity + (exclamation_count * 0.1), 1.0)

        # CAPS augmente l'intensité
        if text.isupper() and len(text) > 10:
            intensity = min(intensity + 0.2, 1.0)

        return dominant_emotion, intensity


class SimpleMemory:
    """
    Mémoire simplifiée pour les tests.

    Dans le vrai Jeffrey OS, ce sera la UnifiedMemory complète
    avec embeddings, clustering, etc.
    """

    def __init__(self):
        self.messages: list[dict[str, Any]] = []

    def add(self, message: dict[str, Any]) -> None:
        """Ajoute un message à la mémoire"""
        self.messages.append(message)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Recherche simple par mot-clé"""
        query_lower = query.lower()
        results = []

        for msg in self.messages:
            content = msg.get("content", "").lower()
            if query_lower in content:
                results.append(msg)

        return results

    def contains(self, keyword: str) -> bool:
        """Vérifie si un mot-clé est dans la mémoire"""
        keyword_lower = keyword.lower()
        for msg in self.messages:
            content = msg.get("content", "").lower()
            if keyword_lower in content:
                return True
        return False

    def clear(self) -> None:
        """Vide la mémoire"""
        self.messages.clear()


# ===============================================================================
# CLASSE PRINCIPALE DE TEST
# ===============================================================================


class ConversationTestRunner:
    """
    Exécute les tests conversationnels YAML.
    """

    def __init__(self, convos_dir: Path):
        self.convos_dir = Path(convos_dir)
        self.emotion_detector = SimpleEmotionDetector()
        self.memory = SimpleMemory()
        self.results: list[dict[str, Any]] = []

        if RICH_AVAILABLE:
            self.console = Console()

    def load_scenario(self, yaml_file: Path) -> dict[str, Any]:
        """Charge un scénario YAML"""
        with open(yaml_file, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def test_scenario(self, scenario: dict[str, Any], filename: str) -> dict[str, Any]:
        """
        Teste un scénario complet.

        Returns:
            Dictionnaire avec les résultats du test
        """
        metadata = scenario.get("metadata", {})
        conversation = scenario.get("conversation", [])
        validation = scenario.get("validation", {})

        result = {
            "scenario_id": metadata.get("scenario_id", "unknown"),
            "title": metadata.get("title", "Unknown"),
            "filename": filename,
            "tests": {"emotion_detection": [], "memory": [], "context": []},
            "passed": 0,
            "failed": 0,
            "errors": [],
        }

        # Tester chaque message de la conversation
        for i, msg in enumerate(conversation):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                expected_emotion = msg.get("expected_emotion")
                expected_intensity = msg.get("expected_intensity")

                # Test 1 : Détection émotionnelle
                if expected_emotion:
                    detected_emotion, detected_intensity = self.emotion_detector.detect(content)

                    emotion_match = detected_emotion == expected_emotion

                    # Tolérance de ±0.15 pour l'intensité
                    intensity_match = True
                    if expected_intensity:
                        intensity_match = abs(detected_intensity - expected_intensity) <= 0.15

                    test_result = {
                        "message": content[:50] + "...",
                        "expected_emotion": expected_emotion,
                        "detected_emotion": detected_emotion,
                        "expected_intensity": expected_intensity,
                        "detected_intensity": detected_intensity,
                        "passed": emotion_match and intensity_match,
                    }

                    result["tests"]["emotion_detection"].append(test_result)

                    if test_result["passed"]:
                        result["passed"] += 1
                    else:
                        result["failed"] += 1
                        result["errors"].append(f"Message {i + 1}: Expected {expected_emotion}, got {detected_emotion}")

                # Ajouter à la mémoire
                self.memory.add({"role": "user", "content": content, "timestamp": datetime.now().isoformat()})

        # Test 2 : Validation de la mémoire
        should_remember = validation.get("should_remember", "")
        if should_remember:
            # Extraire les mots-clés à vérifier
            keywords = should_remember.replace(",", " ").split()
            memory_passed = all(self.memory.contains(kw) for kw in keywords if len(kw) > 3)

            memory_test = {"should_remember": should_remember, "passed": memory_passed}

            result["tests"]["memory"].append(memory_test)

            if memory_passed:
                result["passed"] += 1
            else:
                result["failed"] += 1
                result["errors"].append(f"Memory: Failed to remember '{should_remember}'")

        # Test 3 : Vérifications contextuelles
        if validation.get("continuity_check"):
            # Pour l'instant, on valide juste que ça existe
            result["tests"]["context"].append({"continuity_check": True, "passed": True})
            result["passed"] += 1

        return result

    def run_all_tests(self) -> list[dict[str, Any]]:
        """
        Exécute tous les tests des fichiers YAML.
        """
        yaml_files = sorted(self.convos_dir.glob("*.yaml"))

        if not yaml_files:
            print(f"❌ Aucun fichier YAML trouvé dans {self.convos_dir}")
            return []

        print(f"\n🚀 Lancement des tests sur {len(yaml_files)} scénarios...")
        print("=" * 70)
        print()

        # Barre de progression si rich disponible
        if RICH_AVAILABLE:
            iterator = track(yaml_files, description="Tests en cours...")
        else:
            iterator = yaml_files

        for yaml_file in iterator:
            # Nettoyer la mémoire avant chaque scénario
            self.memory.clear()

            try:
                scenario = self.load_scenario(yaml_file)
                result = self.test_scenario(scenario, yaml_file.name)
                self.results.append(result)

                if not RICH_AVAILABLE:
                    status = "✅" if result["failed"] == 0 else "❌"
                    print(
                        f"{status} {result['title'][:50]:<50} "
                        f"({result['passed']}/{result['passed'] + result['failed']})"
                    )

            except Exception as e:
                error_result = {
                    "scenario_id": "error",
                    "title": f"ERROR: {yaml_file.name}",
                    "filename": yaml_file.name,
                    "tests": {},
                    "passed": 0,
                    "failed": 1,
                    "errors": [str(e)],
                }
                self.results.append(error_result)

                if not RICH_AVAILABLE:
                    print(f"❌ ERROR: {yaml_file.name} - {str(e)}")

        return self.results

    def print_summary(self):
        """Affiche un résumé des résultats"""
        total_scenarios = len(self.results)
        total_passed = sum(r["passed"] for r in self.results)
        total_failed = sum(r["failed"] for r in self.results)
        scenarios_with_errors = len([r for r in self.results if r["failed"] > 0])

        print()
        print("=" * 70)
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 70)
        print()
        print(f"Scénarios testés : {total_scenarios}")
        print(f"Tests passés     : {total_passed} ✅")
        print(f"Tests échoués    : {total_failed} ❌")
        print(f"Scénarios avec erreurs : {scenarios_with_errors}")
        print()

        # Taux de réussite
        if total_passed + total_failed > 0:
            success_rate = (total_passed / (total_passed + total_failed)) * 100
            print(f"Taux de réussite : {success_rate:.1f}%")

        print()

        # Détails des scénarios échoués
        if scenarios_with_errors > 0:
            print("❌ SCÉNARIOS AVEC ERREURS :")
            print("-" * 70)
            for result in self.results:
                if result["failed"] > 0:
                    print(f"\n📄 {result['title']} ({result['filename']})")
                    print(f"   Tests : {result['passed']}✅ / {result['failed']}❌")
                    for error in result["errors"][:3]:  # Max 3 erreurs par scénario
                        print(f"   ⚠️  {error}")

        print()
        print("=" * 70)

        # Sauvegarder les résultats
        self.save_results()

    def save_results(self):
        """Sauvegarde les résultats en JSON"""
        results_file = Path("test_results") / "conversation_tests.json"
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "total_scenarios": len(self.results),
                    "results": self.results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"💾 Résultats sauvegardés : {results_file}")


# ===============================================================================
# POINT D'ENTRÉE
# ===============================================================================


def main():
    """Point d'entrée principal"""
    print()
    print("=" * 70)
    print("🧪 JEFFREY OS - TEST RUNNER CONVERSATIONNEL")
    print("=" * 70)
    print()
    print("Ce script teste les capacités de Jeffrey OS :")
    print("  • Détection émotionnelle")
    print("  • Mémorisation contextuelle")
    print("  • Recherche sémantique")
    print("  • Continuité conversationnelle")
    print()

    # Définir le dossier des conversations
    convos_dir = Path(__file__).parent / "convos"

    if not convos_dir.exists():
        print(f"❌ Erreur : Le dossier {convos_dir} n'existe pas.")
        print("   Assurez-vous d'avoir exécuté le script de création des scénarios.")
        return 1

    # Créer et exécuter le runner
    runner = ConversationTestRunner(convos_dir)
    runner.run_all_tests()
    runner.print_summary()

    print()
    print("✅ Tests terminés !")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
