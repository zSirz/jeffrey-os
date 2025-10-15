#!/usr/bin/env python3
"""
🎯 DRY RUN FINAL - Scanner Optimisé pour Migration Jeffrey
Version ciblée pour trouver les ~15-20 modules critiques
"""

import ast
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Configuration optimisée
CONFIG = {
    "max_workers": 8,
    "scan_timeout": 300,  # 5 minutes max
    "score_threshold": 60,
    "max_results": 30,
    "fast_mode": True,
}

# Signatures des modules critiques "crown jewels"
CROWN_JEWELS = {
    "dream_engine": {
        "markers": ["dream", "rêve", "subconscious"],
        "signatures": ["process_dream", "dream_state", "lucid_mode"],
        "weight": 100,
    },
    "neural_mutator": {
        "markers": ["mutate", "evolve", "adapt"],
        "signatures": ["neural_mutation", "evolve_weights"],
        "weight": 95,
    },
    "consciousness": {
        "markers": ["consciousness", "aware", "sentient"],
        "signatures": ["self_aware", "conscious_state"],
        "weight": 90,
    },
    "emotion_engine": {
        "markers": ["emotion", "feeling", "mood"],
        "signatures": ["process_emotion", "emotional_state"],
        "weight": 85,
    },
    "memory_cortex": {
        "markers": ["cortex", "memory", "recall"],
        "signatures": ["store_memory", "recall_memory"],
        "weight": 85,
    },
    "symbiosis": {
        "markers": ["symbiosis", "symbiotic", "coexist"],
        "signatures": ["symbiotic_link", "mutual_benefit"],
        "weight": 80,
    },
    "learning_engine": {
        "markers": ["learn", "adapt", "improve"],
        "signatures": ["learn_pattern", "adaptive_response"],
        "weight": 75,
    },
    "curiosity_engine": {
        "markers": ["curiosity", "explore", "discover"],
        "signatures": ["explore_unknown", "question_reality"],
        "weight": 70,
    },
}

# Répertoires prioritaires dans iCloud
PRIORITY_PATHS = [
    "Jeffrey/Jeffrey_Apps",
    "Jeffrey/Jeffrey_Core",
    "Jeffrey/Jeffrey_Memory",
    "Jeffrey",
    "Development/Jeffrey",
    "Projects/Jeffrey",
    "Documents/Jeffrey",
]


class OptimizedScanner:
    def __init__(self):
        self.results = []
        self.crown_jewels_found = {}
        self.scan_stats = {
            "files_scanned": 0,
            "modules_found": 0,
            "crown_jewels": 0,
            "errors": 0,
            "start_time": time.time(),
        }

    def calculate_semantic_score(self, file_path: Path, content: str) -> tuple[int, dict]:
        """Scoring sémantique optimisé avec détection de signatures"""
        score = 0
        metadata = {
            "signatures": [],
            "crown_jewel": None,
            "imports": [],
            "classes": [],
            "functions": [],
        }

        try:
            # Parse AST pour analyse profonde
            tree = ast.parse(content)

            # Récupérer les noms
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    metadata["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    metadata["functions"].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        metadata["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        metadata["imports"].append(node.module)

            # Détection crown jewels
            for jewel_name, jewel_info in CROWN_JEWELS.items():
                jewel_score = 0

                # Check markers dans le path ou le contenu
                for marker in jewel_info["markers"]:
                    if marker.lower() in str(file_path).lower():
                        jewel_score += 20
                    if marker.lower() in content.lower():
                        jewel_score += 10

                # Check signatures spécifiques
                for signature in jewel_info["signatures"]:
                    if signature in content:
                        jewel_score += 30
                        metadata["signatures"].append(signature)

                if jewel_score >= 50:
                    metadata["crown_jewel"] = jewel_name
                    score += jewel_info["weight"]
                    break

            # Scoring général
            if "jeffrey" in str(file_path).lower():
                score += 20

            # Complexité du module
            if len(metadata["classes"]) > 2:
                score += 15
            if len(metadata["functions"]) > 5:
                score += 10

            # Imports Jeffrey
            jeffrey_imports = [imp for imp in metadata["imports"] if "jeffrey" in imp.lower()]
            score += len(jeffrey_imports) * 5

            # Taille significative
            if len(content) > 1000:
                score += 10
            if len(content) > 5000:
                score += 10

        except:
            pass

        return min(score, 100), metadata

    def scan_file(self, file_path: Path) -> dict | None:
        """Scan optimisé d'un fichier Python"""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Filtre rapide
            if len(content) < 100:
                return None

            # Skip tests et docs
            if any(skip in str(file_path).lower() for skip in ["test_", "_test.py", "__pycache__", ".pyc"]):
                return None

            # Calcul du score
            score, metadata = self.calculate_semantic_score(file_path, content)

            if score < CONFIG["score_threshold"]:
                return None

            # Hash pour déduplication
            file_hash = hashlib.md5(content.encode()).hexdigest()[:8]

            result = {
                "path": str(file_path),
                "name": file_path.stem,
                "score": score,
                "size": len(content),
                "lines": content.count("\n"),
                "hash": file_hash,
                "metadata": metadata,
                "relative_path": str(
                    file_path.relative_to(Path.home() / "Library/Mobile Documents/com~apple~CloudDocs")
                ),
            }

            if metadata.get("crown_jewel"):
                self.crown_jewels_found[metadata["crown_jewel"]] = result

            return result

        except Exception:
            self.scan_stats["errors"] += 1
            return None

    def scan_directory_batch(self, directory: Path) -> list[dict]:
        """Scan parallèle d'un répertoire"""
        results = []
        py_files = list(directory.rglob("*.py"))[:100]  # Limite pour fast mode

        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = {executor.submit(self.scan_file, f): f for f in py_files}

            for future in as_completed(futures):
                self.scan_stats["files_scanned"] += 1
                result = future.result()
                if result:
                    results.append(result)
                    self.scan_stats["modules_found"] += 1

        return results

    def scan_icloud(self) -> list[dict]:
        """Scan principal optimisé d'iCloud"""
        print("\n🔍 SCAN OPTIMISÉ ICLOUD - RECHERCHE DES CROWN JEWELS")
        print("=" * 60)

        icloud_base = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"

        if not icloud_base.exists():
            print("❌ iCloud Drive non trouvé")
            return []

        all_results = []

        # Scanner les chemins prioritaires
        for priority in PRIORITY_PATHS:
            target = icloud_base / priority
            if target.exists():
                print(f"\n📂 Scan prioritaire: {priority}")
                results = self.scan_directory_batch(target)
                all_results.extend(results)
                print(f"  → {len(results)} modules trouvés")

                # Early exit si assez de modules
                if len(all_results) >= CONFIG["max_results"]:
                    break

        # Tri par score
        all_results.sort(key=lambda x: x["score"], reverse=True)

        return all_results[: CONFIG["max_results"]]

    def interactive_validation(self, results: list[dict]) -> dict:
        """Simulation de validation interactive"""
        print("\n\n🎯 VALIDATION INTERACTIVE (SIMULATION)")
        print("=" * 60)

        validated = []
        rejected = []

        for i, module in enumerate(results[:10], 1):  # Top 10 pour validation
            print(f"\n[{i}/10] Module: {module['name']}")
            print(f"  📍 Path: {module['relative_path']}")
            print(f"  🎯 Score: {module['score']}/100")

            if module["metadata"].get("crown_jewel"):
                print(f"  👑 CROWN JEWEL: {module['metadata']['crown_jewel']}")

            print(f"  📊 Stats: {module['lines']} lignes, {module['size']} octets")

            # Auto-validation basée sur le score
            if module["score"] >= 80:
                print("  ✅ AUTO-VALIDÉ (score élevé)")
                validated.append(module)
            elif module["score"] >= 60:
                print("  🔄 VALIDATION MANUELLE REQUISE")
                validated.append(module)  # Simulation: on valide
            else:
                print("  ❌ AUTO-REJETÉ (score faible)")
                rejected.append(module)

        return {"validated": validated, "rejected": rejected}

    def generate_report(self, results: list[dict]) -> dict:
        """Génère le rapport final optimisé"""

        # Validation simulée
        validation_result = self.interactive_validation(results)

        # Statistiques finales
        elapsed = time.time() - self.scan_stats["start_time"]

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files_scanned": self.scan_stats["files_scanned"],
                "modules_found": self.scan_stats["modules_found"],
                "crown_jewels_found": len(self.crown_jewels_found),
                "scan_duration": f"{elapsed:.2f}s",
                "errors": self.scan_stats["errors"],
            },
            "crown_jewels": self.crown_jewels_found,
            "top_modules": results[:20],
            "validation": {
                "validated_count": len(validation_result["validated"]),
                "rejected_count": len(validation_result["rejected"]),
                "validated_modules": validation_result["validated"],
            },
            "go_no_go": {
                "decision": "GO" if len(self.crown_jewels_found) >= 3 else "NO-GO",
                "criteria": {
                    "crown_jewels": len(self.crown_jewels_found) >= 3,
                    "total_modules": len(results) >= 10,
                    "high_score_modules": len([m for m in results if m["score"] >= 80]) >= 5,
                },
                "confidence": self.calculate_confidence(results),
            },
        }

        return report

    def calculate_confidence(self, results: list[dict]) -> str:
        """Calcule le niveau de confiance pour la migration"""
        score_sum = sum(r["score"] for r in results[:10]) if results else 0
        avg_score = score_sum / min(len(results), 10) if results else 0

        if len(self.crown_jewels_found) >= 5 and avg_score >= 75:
            return "TRÈS ÉLEVÉE ⭐⭐⭐⭐⭐"
        elif len(self.crown_jewels_found) >= 3 and avg_score >= 60:
            return "ÉLEVÉE ⭐⭐⭐⭐"
        elif len(self.crown_jewels_found) >= 1 and avg_score >= 50:
            return "MOYENNE ⭐⭐⭐"
        else:
            return "FAIBLE ⭐⭐"

    def display_results(self, report: dict):
        """Affichage formaté des résultats"""
        print("\n\n" + "=" * 60)
        print("📊 RAPPORT FINAL - DRY RUN OPTIMISÉ")
        print("=" * 60)

        summary = report["summary"]
        print("\n📈 Statistiques:")
        print(f"  • Fichiers scannés: {summary['total_files_scanned']}")
        print(f"  • Modules trouvés: {summary['modules_found']}")
        print(f"  • Crown Jewels: {summary['crown_jewels_found']}")
        print(f"  • Durée: {summary['scan_duration']}")

        if report["crown_jewels"]:
            print("\n👑 CROWN JEWELS DÉCOUVERTS:")
            for name, module in report["crown_jewels"].items():
                print(f"  • {name}: {module['name']} (score: {module['score']})")

        print("\n🎯 TOP 10 MODULES:")
        for i, module in enumerate(report["top_modules"][:10], 1):
            jewel = "👑 " if module["metadata"].get("crown_jewel") else ""
            print(f"  {i:2}. {jewel}{module['name']:<30} Score: {module['score']:3}/100")

        go_no_go = report["go_no_go"]
        print(f"\n📍 DÉCISION GO/NO-GO: {go_no_go['decision']}")
        print(f"  • Confiance: {go_no_go['confidence']}")
        print(f"  • Crown Jewels: {'✅' if go_no_go['criteria']['crown_jewels'] else '❌'}")
        print(f"  • Modules suffisants: {'✅' if go_no_go['criteria']['total_modules'] else '❌'}")
        print(f"  • Qualité élevée: {'✅' if go_no_go['criteria']['high_score_modules'] else '❌'}")

        print("\n" + "=" * 60)
        if go_no_go["decision"] == "GO":
            print("🚀 PRÊT POUR LA MIGRATION!")
            print("Les modules critiques ont été identifiés avec succès.")
        else:
            print("⚠️ MIGRATION NON RECOMMANDÉE")
            print("Pas assez de modules critiques trouvés.")

    def run(self):
        """Exécution principale du dry run"""
        try:
            # Scan
            results = self.scan_icloud()

            # Rapport
            report = self.generate_report(results)

            # Affichage
            self.display_results(report)

            # Sauvegarde JSON
            report_path = Path("reports/dryrun_final_report.json")
            report_path.parent.mkdir(exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            print(f"\n💾 Rapport sauvegardé: {report_path}")

            # Fichier de migration suggéré
            if report["go_no_go"]["decision"] == "GO":
                migration_list = Path("reports/migration_list.txt")
                with open(migration_list, "w") as f:
                    f.write("# LISTE DES MODULES À MIGRER\n")
                    f.write(f"# Générée le {datetime.now()}\n\n")

                    for module in report["validation"]["validated_modules"]:
                        f.write(f"{module['path']}\n")

                print(f"📝 Liste de migration: {migration_list}")

            return report

        except Exception as e:
            print(f"\n❌ ERREUR: {e}")
            import traceback

            traceback.print_exc()
            return None


def main():
    """Point d'entrée principal"""
    print("🎯 DRY RUN FINAL - SCANNER OPTIMISÉ POUR JEFFREY")
    print("Version ciblée pour modules critiques")
    print("=" * 60)

    scanner = OptimizedScanner()
    report = scanner.run()

    if report and report["go_no_go"]["decision"] == "GO":
        print("\n✅ Dry run complété avec succès!")
        return 0
    else:
        print("\n⚠️ Dry run complété - Migration non recommandée")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
