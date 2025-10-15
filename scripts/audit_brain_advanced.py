#!/usr/bin/env python3
"""
Audit avancé du cerveau de Jeffrey OS
Détecte modules, stubs, dépendances, cycles, et génère un rapport complet
"""
import ast
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

class BrainAuditor:
    """Auditeur intelligent du cerveau de Jeffrey"""

    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.src = self.root / "src"

        # Patterns étendus pour stubs (avec whitelist return {} # ok)
        self.stub_patterns = [
            r"STUB|stub|MOCK|mock|PLACEHOLDER|placeholder",
            r"TODO(?!\(ok\))|FIXME|HACK",
            r"NotImplementedError|pass\s*#",
            r"raise\s+NotImplementedError",
            r"^if\s+False:",
            r"^\s*return\s+{}\s*(?!#\s*ok)",      # tolère `# ok`
            r"^\s*return\s+None\s*#(?!\s*ok)"      # idem
        ]
        self.stub_regex = re.compile("|".join(self.stub_patterns), re.MULTILINE)

        # Patterns pour NeuralBus (étendu avec on/emit)
        self.subscribe_regex = re.compile(r'\.subscribe\(\s*["\']([^"\']+)')
        self.publish_regex = re.compile(r'\.publish\(\s*topic\s*=\s*["\']([^"\']+)')
        self.on_regex = re.compile(r'bus\.on\(\s*["\']([^"\']+)')
        self.emit_regex = re.compile(r'bus\.emit\(\s*["\']([^"\']+)')

        # Modules attendus avec aliases
        self.expected_modules = {
            # Architecture Cellulaire Bio-Inspirée
            "CellManager|CellFactory|CellularManager": "Gestionnaire de cellules",
            "NeuralBus|EventBus|MessageBus": "Communication événementielle",
            "TissueOrganizer|TissueManager": "Groupes de cellules",

            # Conscience & Cognition
            "ConsciousnessV3|JeffreyLivingConsciousness|Consciousness": "Système de conscience",
            "MetaCognition|MetaCognitive|MetaThinking": "Méta-cognition",
            "TheoryOfMind|ToM|MindTheory": "Théorie de l'esprit",
            "SelfReflection|SelfAnalysis|Introspection": "Auto-réflexion",
            "CognitiveCore|CognitiveOrchestrator|BrainCore": "Noyau cognitif",

            # Mémoire Multi-Couches (CRITIQUE)
            "WorkingMemory|ShortTermMemory|STM": "Mémoire de travail",
            "EmotionalMemory|AffectiveMemory": "Mémoire émotionnelle",
            "EpisodicMemory|EventMemory": "Mémoire épisodique",
            "SensoryMemory|PerceptualMemory": "Mémoire sensorielle",
            "CreativeMemoryWeb|CreativeWeb|AssociativeMemory": "Mémoire créative",
            "NarrativeMemory|StoryMemory": "Mémoire narrative",
            "PhilosophicalMemory|DeepMemory|WisdomMemory": "Mémoire philosophique",
            "MultiLayerMemory|MemoryManager|UnifiedMemory": "Gestionnaire de mémoire unifié",

            # Système Émotionnel Avancé
            "LivingSoulEngine|SoulEngine|EmotionalCore": "Moteur d'âme vivante",
            "ContextualEmpathy|EmpathyEngine|EmpathicResponse": "Empathie contextuelle",
            "EmotionalBonds|BondManager|AttachmentSystem": "Système d'attachement",
            "MicroExpressions|SubtleEmotions": "Micro-expressions",
            "EmotionalJournal|EmotionLog|FeelingsTracker": "Journal émotionnel",

            # Innovations Uniques (PÉPITES BREVETABLES)
            "CircadianRhythm|JeffreyBiorhythms|BioRhythm|BioClock": "Rythme circadien",
            "DreamEngine|DreamProcessor|DreamConsolidation": "Moteur de rêves",
            "SubtleLearning|ImplicitLearning|UnconsciousLearning": "Apprentissage subtil",
            "ProactiveCuriosity|CuriosityEngine|AnticipativeCuriosity": "Curiosité proactive",
            "EvolutiveAttachment|AttachmentEvolution|BondEvolution": "Attachement évolutif",
            "SecureImaginationEngine|ImaginationEngine|CreativeImagination": "Imagination sécurisée",
            "QuantumInspiredDecisionMaker|QuantumDecision|ProbabilisticChoice": "Décision quantique",
            "EcosystemSimulator|MultiAgentEcosystem|IAEcosystem": "Simulateur d'écosystème",
            "NeuralPlasticityEngine|SynapticPlasticity|DynamicConnections": "Plasticité neurale",

            # Sécurité & Éthique
            "GuardianSymphony|GuardianOrchestrator|SecurityOrchestrator": "Orchestrateur de sécurité",
            "EthicalGuardian|EthicsEngine|MoralCompass": "Gardien éthique",
            "PapaControl|ParentalControl|SafetyControl": "Contrôle parental",
            "DigitalImmuneSystem|ImmuneSystem|ThreatDetection": "Système immunitaire digital"
        }

        # Topics NeuralBus attendus
        self.expected_topics = {
            "emotion.ml.detected.v1",
            "memory.store.request.v1",
            "memory.stored.v1",
            "memory.recall.request.v1",
            "memory.recalled.v1",
            "cognition.thought.v1",
            "dream.trigger.v1",
            "dream.generated.v1",
            "state.circadian.updated.v1",
            "ethics.guard.violated.v1",
            "security.threat.detected.v1",
            "curiosity.question.v1",
            "attachment.bond.updated.v1"
        }

        # Résultats
        self.found_modules = {}
        self.missing_modules = []
        self.stubs = []
        self.empty_functions = []
        self.topics_found = {"subscriptions": set(), "publications": set()}
        self.dependency_graph = None
        self.cycles = []

        # Initialiser NetworkX si disponible
        try:
            import networkx as nx
            self.dependency_graph = nx.DiGraph()
            self.networkx_available = True
        except ImportError:
            print("ℹ️ NetworkX non disponible - analyse de dépendances désactivée")
            self.networkx_available = False

    def scan_repository(self):
        """Scan complet avec AST et regex"""
        exclude_dirs = {".venv", "venv", "__pycache__", "dist", "build", "node_modules", "tests"}

        for py_file in self.src.rglob("*.py"):
            # Skip excluded directories
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue

            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")

                # 1. Détecter stubs via regex
                for match in self.stub_regex.finditer(content):
                    self.stubs.append({
                        "file": str(py_file.relative_to(self.root)),
                        "line": content[:match.start()].count('\n') + 1,
                        "type": match.group(0)
                    })

                # 2. Détecter topics NeuralBus
                for match in self.subscribe_regex.finditer(content):
                    self.topics_found["subscriptions"].add(match.group(1))
                for match in self.publish_regex.finditer(content):
                    self.topics_found["publications"].add(match.group(1))
                for match in self.on_regex.finditer(content):
                    self.topics_found["subscriptions"].add(match.group(1))
                for match in self.emit_regex.finditer(content):
                    self.topics_found["publications"].add(match.group(1))

                # 3. Analyse AST
                tree = ast.parse(content, filename=str(py_file))
                self._analyze_ast(tree, py_file)

            except Exception as e:
                print(f"⚠️ Erreur parsing {py_file}: {e}")

    def _analyze_ast(self, tree: ast.AST, file_path: Path):
        """Analyse AST pour classes, fonctions vides, imports"""
        relative_path = str(file_path.relative_to(self.root))

        for node in ast.walk(tree):
            # Détecter classes
            if isinstance(node, ast.ClassDef):
                self.found_modules[node.name] = {
                    "file": relative_path,
                    "line": node.lineno,
                    "has_init": any(n.name == "__init__" for n in node.body if isinstance(n, ast.FunctionDef))
                }

            # Détecter fonctions vides
            if isinstance(node, ast.FunctionDef):
                if len(node.body) == 1:
                    stmt = node.body[0]
                    if isinstance(stmt, (ast.Pass, ast.Raise)) or \
                       (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value == ...):
                        self.empty_functions.append({
                            "file": relative_path,
                            "function": node.name,
                            "line": node.lineno
                        })

            # Construire graphe de dépendances (si NetworkX disponible)
            if self.networkx_available and isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("jeffrey"):
                        target = node.module.split(".")[-1]
                        source = file_path.stem
                        self.dependency_graph.add_edge(source, target)

    def analyze_dependencies(self):
        """Analyser le graphe de dépendances"""
        if not self.networkx_available:
            return

        try:
            import networkx as nx
            # Détecter cycles
            self.cycles = list(nx.simple_cycles(self.dependency_graph))

            # Exporter graphe pour visualisation (best-effort)
            try:
                if self.dependency_graph.nodes():
                    import pydot  # noqa: F401
                    nx.drawing.nx_pydot.write_dot(
                        self.dependency_graph,
                        self.root / "dependencies.dot"
                    )
            except Exception as e:
                print(f"ℹ️ Export DOT ignoré (non-bloquant): {e}")
        except Exception as e:
            print(f"⚠️ Erreur analyse dépendances: {e}")

    def check_missing_modules(self):
        """Identifier modules manquants"""
        found_names = set(self.found_modules.keys())

        for expected in self.expected_modules:
            # Vérifier tous les aliases possibles
            aliases = expected.split("|")
            if not any(alias in found_names for alias in aliases):
                self.missing_modules.append(expected)

    def check_missing_topics(self):
        """Identifier topics NeuralBus manquants"""
        all_topics = self.topics_found["subscriptions"] | self.topics_found["publications"]
        return self.expected_topics - all_topics

    def generate_report(self):
        """Générer rapport Markdown détaillé"""
        missing_topics = self.check_missing_topics()

        report = f"""# 📊 AUDIT COMPLET DU CERVEAU DE JEFFREY OS

## 📈 STATISTIQUES GLOBALES
- **Modules trouvés**: {len(self.found_modules)}/{len(self.expected_modules)}
- **Stubs détectés**: {len(self.stubs)}
- **Fonctions vides**: {len(self.empty_functions)}
- **Topics NeuralBus**: {len(self.topics_found['subscriptions'])} subs, {len(self.topics_found['publications'])} pubs
- **Cycles de dépendances**: {len(self.cycles)}

## ✅ MODULES TROUVÉS ET FONCTIONNELS
"""
        for name, info in sorted(self.found_modules.items()):
            status = "✅" if name not in [s["file"].split("/")[-1].replace(".py", "") for s in self.stubs] else "⚠️"
            report += f"- {status} **{name}** - `{info['file']}` (ligne {info['line']})\n"

        if self.stubs:
            report += "\n## ⚠️ STUBS ET MOCKS DÉTECTÉS\n"
            for stub in self.stubs[:20]:  # Limiter à 20 pour lisibilité
                report += f"- `{stub['file']}:{stub['line']}` - Type: {stub['type']}\n"
            if len(self.stubs) > 20:
                report += f"... et {len(self.stubs) - 20} autres\n"

        if self.missing_modules:
            report += "\n## ❌ MODULES CRITIQUES MANQUANTS\n"
            for module in sorted(self.missing_modules):
                report += f"- **{module.split('|')[0]}** (aliases: {module})\n"

        if missing_topics:
            report += "\n## 🔌 TOPICS NEURALBUS MANQUANTS\n"
            for topic in sorted(missing_topics):
                report += f"- `{topic}`\n"

        if self.cycles:
            report += "\n## 🔄 CYCLES DE DÉPENDANCES DÉTECTÉS\n"
            for cycle in self.cycles[:5]:
                report += f"- Cycle: {' → '.join(cycle)} → {cycle[0]}\n"

        # Plan d'action
        report += "\n## 🎯 PLAN D'ACTION RECOMMANDÉ\n"

        missing_ratio = len(self.missing_modules) / len(self.expected_modules)
        if missing_ratio > 0.5:
            report += """
### Mode Skeletal Recommandé
Plus de 50% des modules sont manquants. Recommandation:
1. Activer le **SKELETAL_MODE** pour démarrer avec stubs intelligents
2. Implémenter une boucle cognitive minimale (Emotion → Memory → Consciousness)
3. Ajouter progressivement les modules manquants
"""

        report += """
### Prochaines Étapes
1. **Récupération**: Chercher les modules manquants dans le cloud (patterns ciblés)
2. **Implémentation**: Créer les modules critiques manquants avec stubs intelligents
3. **Connexion**: Brancher au NeuralBus en suivant l'ordre de dépendance
4. **Validation**: Tests end-to-end avec mode dégradé si nécessaire
"""

        # Sauvegarder le rapport
        report_path = self.root / "audit_report.md"
        report_path.write_text(report)

        # Sauvegarder aussi en JSON pour traitement automatisé
        json_report = {
            "stats": {
                "modules_found": len(self.found_modules),
                "modules_expected": len(self.expected_modules),
                "stubs": len(self.stubs),
                "empty_functions": len(self.empty_functions),
                "cycles": len(self.cycles),
                "missing_ratio": missing_ratio
            },
            "found_modules": self.found_modules,
            "missing_modules": self.missing_modules,
            "stubs": self.stubs[:50],  # Limiter pour la taille
            "cycles": self.cycles,
            "missing_topics": list(missing_topics)
        }

        json_path = self.root / "audit_report.json"
        json_path.write_text(json.dumps(json_report, indent=2))

        return report

    def should_block_pipeline(self) -> bool:
        """Détermine si l'audit doit bloquer le pipeline"""
        critical_missing = ["ConsciousnessV3", "MultiLayerMemory", "LivingSoulEngine"]

        for critical in critical_missing:
            if any(critical in module for module in self.missing_modules):
                return True

        # Bloquer si trop de stubs
        if len(self.stubs) > 100:
            return True

        # Bloquer si cycles critiques
        if len(self.cycles) > 5:
            return True

        return False

def main():
    """Point d'entrée principal"""
    print("🧠 AUDIT DU CERVEAU DE JEFFREY OS - DÉMARRAGE")
    print("=" * 60)

    auditor = BrainAuditor()

    # Phases d'audit
    print("📂 Phase 1: Scan du repository...")
    auditor.scan_repository()

    print("🔍 Phase 2: Analyse des dépendances...")
    auditor.analyze_dependencies()

    print("❓ Phase 3: Vérification des modules manquants...")
    auditor.check_missing_modules()

    print("📝 Phase 4: Génération du rapport...")
    report = auditor.generate_report()

    # Affichage résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE L'AUDIT")
    print("=" * 60)
    print(f"✅ Modules trouvés: {len(auditor.found_modules)}/{len(auditor.expected_modules)}")
    print(f"⚠️ Stubs détectés: {len(auditor.stubs)}")
    print(f"❌ Modules manquants: {len(auditor.missing_modules)}")
    print(f"🔄 Cycles de dépendances: {len(auditor.cycles)}")

    # Calcul du ratio manquant
    missing_ratio = len(auditor.missing_modules) / len(auditor.expected_modules)
    if missing_ratio < 0.3:
        mode = "COMPLETE"
    elif missing_ratio < 0.7:
        mode = "SKELETAL"
    else:
        mode = "RECOVERY"

    print(f"📊 Mode recommandé : {mode}")

    # Décision de blocage
    if auditor.should_block_pipeline():
        print("\n❌ AUDIT ÉCHOUÉ - Pipeline bloqué")
        print("Raisons: Modules critiques manquants ou trop de stubs")
        print("Action: Activer SKELETAL_MODE ou récupérer modules manquants")
        sys.exit(1)
    else:
        print("\n✅ AUDIT RÉUSSI - Pipeline peut continuer")
        print(f"Rapport complet: audit_report.md")
        print(f"Données JSON: audit_report.json")

        # Prochaine commande recommandée
        if mode == "COMPLETE":
            print("\n🎯 Prochaine étape: python scripts/connect_brain_modules.py")
        elif mode == "SKELETAL":
            print("\n🦴 Prochaine étape: python scripts/activate_skeletal_mode.py")
        else:
            print("\n🔍 Prochaine étape: python scripts/recover_cloud_modules.py")

        sys.exit(0)

if __name__ == "__main__":
    main()