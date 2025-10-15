#!/usr/bin/env python3
"""
Framework de tests conversationnels pour Jeffrey OS.

Corrections appliquées (selon feedback GPT) :
- enable_vector=None pour auto-détection gracieuse
- assert_reply_includes avec query explicite
- Gestion robuste des erreurs et edge cases
- Reproductibilité (seed 42, PYTHONHASHSEED=0)
"""

import glob
import yaml
import json
import csv
import time
import sys
import random
import os
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import Jeffrey
sys.path.insert(0, 'src')
from jeffrey.memory.unified_memory import UnifiedMemory


class ConversationalTestRunner:
    """Runner de tests conversationnels pour Jeffrey OS."""

    def __init__(self, scenarios_dir: str = "tests/convos"):
        # Reproductibilité
        random.seed(42)
        os.environ['PYTHONHASHSEED'] = '0'

        self.scenarios_dir = Path(scenarios_dir)

        # CORRECTION GPT : enable_vector=None (auto-détection)
        self.memory = UnifiedMemory(enable_vector=None)

        self.results = []
        self.latencies = []
        self.scenario_timeout = 30

        self.start_time = datetime.now()
        self.git_commit = self._get_git_commit()

    def _get_git_commit(self) -> str:
        """Récupère le commit Git court."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.stdout.strip()
        except:
            return "unknown"

    def run_all(self) -> Dict[str, Any]:
        """Lance tous les scénarios et génère un rapport."""
        printf '=%.0s' {1..60}; echo
        print("🧪 JEFFREY OS - TESTS CONVERSATIONNELS")
        printf '=%.0s' {1..60}; echo
        print(f"Git commit: {self.git_commit}")
        print(f"Timestamp: {self.start_time.isoformat()}")

        scenarios = sorted(self.scenarios_dir.glob("*.yaml"))

        if not scenarios:
            print(f"\n⚠️  Aucun scénario dans {self.scenarios_dir}")
            print("    → Exécute le PROMPT 2 pour créer les 40 scénarios YAML")
            return self._empty_report()

        print(f"\n📂 {len(scenarios)} scénarios détectés\n")

        for i, path in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] {path.stem}...")
            result = self.run_scenario(path)
            self.results.append(result)

            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status} ({result['assertions_passed']}/{result['assertions_total']})")

            if result["errors"]:
                for err in result["errors"][:3]:  # Max 3 erreurs affichées
                    print(f"    ⚠️  {err}")

        report = self._generate_report()
        self._save_report(report)
        self._print_summary(report)

        return report

    def run_scenario(self, path: Path) -> Dict[str, Any]:
        """Exécute un scénario YAML."""
        try:
            data = yaml.safe_load(path.read_text())
        except Exception as e:
            return {
                "path": str(path),
                "name": path.stem,
                "error": f"YAML load failed: {e}",
                "passed": False,
                "assertions_total": 0,
                "assertions_passed": 0,
                "memories_added": 0,
                "errors": [f"YAML parse error: {e}"]
            }

        meta = data.get("meta", {})
        user_id = meta.get("user_id", "default")
        name = meta.get("name", path.stem)

        session = data.get("session", [])

        assertions_total = 0
        assertions_passed = 0
        errors = []
        memories_added = 0

        try:
            for step_idx, step in enumerate(session):
                # Action: Ajouter mémoire
                if "user" in step:
                    start = time.time()
                    try:
                        self.memory.add_memory({
                            "user_id": user_id,
                            "content": step["user"],
                            "type": "conversation"
                        })
                        latency = (time.time() - start) * 1000
                        self.latencies.append(latency)
                        memories_added += 1
                    except Exception as e:
                        errors.append(f"Step {step_idx}: add_memory failed: {e}")

                # Assertion: expect_memory_contains
                if "expect_memory_contains" in step:
                    assertions_total += 1
                    expected = step["expect_memory_contains"]

                    try:
                        recent = self.memory.store.list_by_user(user_id)[-5:]
                        if self._check_memory_contains(recent, expected):
                            assertions_passed += 1
                        else:
                            errors.append(f"Step {step_idx}: expect_memory_contains failed")
                    except Exception as e:
                        errors.append(f"Step {step_idx}: expect_memory_contains error: {e}")

                # CORRECTION GPT : assert_reply_includes avec query explicite
                if "assert_reply_includes" in step:
                    assertions_total += 1
                    expected_any = step["assert_reply_includes"].get("any", [])
                    query = step["assert_reply_includes"].get("query", " ".join(expected_any))

                    try:
                        results = self.memory.search_memories(
                            user_id,
                            query=query,
                            limit=5
                        )

                        if results:
                            content = " ".join(r["memory"]["content"].lower() for r in results)
                            if any(term.lower() in content for term in expected_any):
                                assertions_passed += 1
                            else:
                                errors.append(f"Step {step_idx}: assert_reply_includes failed (query='{query}')")
                        else:
                            errors.append(f"Step {step_idx}: No results for assert_reply_includes")
                    except Exception as e:
                        errors.append(f"Step {step_idx}: assert_reply_includes error: {e}")

                # Assertion: assert_topk_semantic
                if "assert_topk_semantic" in step:
                    assertions_total += 1
                    q = step["assert_topk_semantic"]["query"]
                    k = step["assert_topk_semantic"].get("k", 5)
                    must_include_any = step["assert_topk_semantic"].get("must_include_any", [])

                    try:
                        results = self.memory.search_memories(
                            user_id,
                            query=q,
                            semantic_search=True,
                            limit=k
                        )

                        if results:
                            content = " ".join(r["memory"]["content"].lower() for r in results)
                            if any(term.lower() in content for term in must_include_any):
                                assertions_passed += 1
                            else:
                                errors.append(f"Step {step_idx}: assert_topk_semantic failed")
                        else:
                            errors.append(f"Step {step_idx}: No results for assert_topk_semantic")
                    except Exception as e:
                        errors.append(f"Step {step_idx}: assert_topk_semantic error: {e}")

                # Assertion: assert_clusters
                if "assert_clusters" in step:
                    assertions_total += 1
                    expected = step["assert_clusters"]

                    try:
                        user_memories = self.memory.store.list_by_user(user_id)

                        # Re-clustering si seuil atteint
                        if len(user_memories) >= 50:
                            self.memory._recluster_user(user_id)
                            time.sleep(0.5)  # Attendre thread async

                        clusters = self.memory.get_clusters(user_id)
                        min_clusters = expected.get("min_count", 0)

                        if len(clusters) >= min_clusters:
                            assertions_passed += 1
                        else:
                            # Si pas assez de mémoires, warning pas erreur
                            if len(user_memories) < 50:
                                errors.append(f"Step {step_idx}: assert_clusters skipped (N={len(user_memories)} < 50)")
                            else:
                                errors.append(f"Step {step_idx}: assert_clusters failed ({len(clusters)} < {min_clusters})")
                    except Exception as e:
                        errors.append(f"Step {step_idx}: assert_clusters error: {e}")

                # Assertion: assert_feedback_effect
                if "assert_feedback_effect" in step:
                    assertions_total += 1
                    query = step["assert_feedback_effect"]["query"]

                    try:
                        before_results = self.memory.search_memories(
                            user_id,
                            query=query,
                            explain=True,
                            limit=5
                        )

                        if before_results:
                            shown = [r["memory"]["id"] for r in before_results]
                            clicked_idx = step["assert_feedback_effect"].get("clicked_rank", 3) - 1

                            if clicked_idx < len(shown):
                                self.memory.feedback(user_id, shown, [shown[clicked_idx]])

                            after_results = self.memory.search_memories(
                                user_id,
                                query=query,
                                explain=True,
                                limit=5
                            )

                            if after_results:
                                before_weights = before_results[0]["explanation"]["weights_used"]
                                after_weights = after_results[0]["explanation"]["weights_used"]

                                changed = any(
                                    abs(before_weights[k] - after_weights[k]) > 0.001
                                    for k in before_weights
                                )

                                if changed:
                                    assertions_passed += 1
                                else:
                                    errors.append(f"Step {step_idx}: Weights didn't change after feedback")
                            else:
                                errors.append(f"Step {step_idx}: No results after feedback")
                        else:
                            errors.append(f"Step {step_idx}: No results before feedback")
                    except Exception as e:
                        errors.append(f"Step {step_idx}: assert_feedback_effect error: {e}")

        except Exception as e:
            errors.append(f"Scenario execution error: {e}")

        return {
            "path": str(path),
            "name": name,
            "user_id": user_id,
            "assertions_total": assertions_total,
            "assertions_passed": assertions_passed,
            "passed": assertions_passed == assertions_total and assertions_total > 0,
            "memories_added": memories_added,
            "errors": errors
        }

    def _check_memory_contains(self, memories: List[Dict], expected: Dict) -> bool:
        """Vérifie qu'une mémoire récente contient les champs attendus."""
        for mem in memories:
            matches = True
            for key, value in expected.items():
                if key == "tags":
                    if not isinstance(value, list):
                        value = [value]
                    mem_tags = mem.get("tags", []) or []
                    # CORRECTION : Si value est vide, on skip cette vérification
                    if value and not any(tag in mem_tags for tag in value):
                        matches = False
                        break
                else:
                    if mem.get(key) != value:
                        matches = False
                        break

            if matches:
                return True

        return False

    def _generate_report(self) -> Dict:
        """Génère le rapport final."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])

        total_assertions = sum(r["assertions_total"] for r in self.results)
        passed_assertions = sum(r["assertions_passed"] for r in self.results)

        avg_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        p95_lat = sorted(self.latencies)[int(len(self.latencies) * 0.95)] if self.latencies else 0

        return {
            "timestamp": self.start_time.isoformat(),
            "git_commit": self.git_commit,
            "summary": {
                "total_scenarios": total,
                "passed_scenarios": passed,
                "failed_scenarios": total - passed,
                "success_rate": passed / total if total > 0 else 0,
                "total_assertions": total_assertions,
                "passed_assertions": passed_assertions,
                "assertions_success_rate": passed_assertions / total_assertions if total_assertions > 0 else 0
            },
            "performance": {
                "avg_latency_ms": round(avg_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "total_operations": len(self.latencies)
            },
            "scenarios": self.results
        }

    def _save_report(self, report: Dict):
        """Sauvegarde JSON + CSV."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = Path(f"test_results/conversational_tests_{ts}.json")
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        csv_path = Path(f"test_results/conversational_tests_{ts}.csv")
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "name", "passed", "assertions_total",
                "assertions_passed", "memories_added", "errors"
            ])
            writer.writeheader()
            for r in report["scenarios"]:
                writer.writerow({
                    "name": r["name"],
                    "passed": r["passed"],
                    "assertions_total": r["assertions_total"],
                    "assertions_passed": r["assertions_passed"],
                    "memories_added": r["memories_added"],
                    "errors": "; ".join(r["errors"]) if r["errors"] else ""
                })

        print(f"\n📄 Rapport sauvegardé:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")

    def _print_summary(self, report: Dict):
        """Affiche le résumé."""
        summary = report["summary"]
        perf = report["performance"]

        print("\n")
        printf '=%.0s' {1..60}; echo
        print("📊 RÉSULTATS FINAUX")
        printf '=%.0s' {1..60}; echo

        print(f"\n🎯 Scénarios:")
        print(f"  Total: {summary['total_scenarios']}")
        print(f"  ✅ Passed: {summary['passed_scenarios']}")
        print(f"  ❌ Failed: {summary['failed_scenarios']}")
        print(f"  📈 Success rate: {summary['success_rate']:.1%}")

        print(f"\n📝 Assertions:")
        print(f"  Total: {summary['total_assertions']}")
        print(f"  ✅ Passed: {summary['passed_assertions']}")
        print(f"  📈 Success rate: {summary['assertions_success_rate']:.1%}")

        print(f"\n⚡ Performance:")
        print(f"  Avg latency: {perf['avg_latency_ms']:.2f}ms")
        print(f"  P95 latency: {perf['p95_latency_ms']:.2f}ms")
        print(f"  Total operations: {perf['total_operations']}")

        if summary['passed_scenarios'] == summary['total_scenarios']:
            print("\n🎉 TOUS LES TESTS PASSENT !")
        else:
            print("\n⚠️  Certains tests ont échoué. Consultez le rapport.")

    def _empty_report(self) -> Dict:
        """Rapport vide si aucun scénario."""
        return {
            "timestamp": self.start_time.isoformat(),
            "git_commit": self.git_commit,
            "summary": {
                "total_scenarios": 0,
                "passed_scenarios": 0,
                "failed_scenarios": 0,
                "success_rate": 0,
                "total_assertions": 0,
                "passed_assertions": 0,
                "assertions_success_rate": 0
            },
            "performance": {
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "total_operations": 0
            },
            "scenarios": []
        }


if __name__ == "__main__":
    runner = ConversationalTestRunner()
    report = runner.run_all()

    sys.exit(0 if report["summary"]["passed_scenarios"] == report["summary"]["total_scenarios"] else 1)
