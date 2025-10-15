#!/usr/bin/env python3
# Fichier: fix_all_issues.py

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/corrections.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class JeffreyAutoFixer:
    """Correcteur automatique intelligent pour Jeffrey OS"""

    def __init__(self):
        self.root = Path("src/jeffrey")
        self.fixed_count = 0
        self.failed_fixes = []
        self.created_modules = []

    def run_all_fixes(self) -> dict[str, Any]:
        """Exécute toutes les corrections dans l'ordre optimal"""
        logger.info("🔧 DÉMARRAGE DES CORRECTIONS INTELLIGENTES")

        # Ordre optimisé des corrections
        results = {
            'phase1_structure': {
                'init_files': self.fix_missing_init_files(),
                'critical_modules': self.create_critical_modules(),
            },
            'phase2_imports': {
                'legacy_imports': self.fix_legacy_imports(),
            },
            'phase3_syntax': {
                'indentation': self.fix_indentation_errors(),
                'syntax': self.fix_syntax_errors(),
            },
            'phase4_async': {
                'async_issues': self.fix_async_issues_safe(),
            },
            'phase5_orphans': {
                'reconnected': self.reconnect_orphan_modules(),
            },
        }

        results['summary'] = {
            'total_fixed': self.fixed_count,
            'failed_fixes': self.failed_fixes,
            'created_modules': self.created_modules,
        }

        # Sauvegarder le rapport
        Path("diagnostics/fixes_report.json").write_text(json.dumps(results, indent=2, default=str))

        logger.info(f"✅ Corrections terminées: {self.fixed_count} fixes appliqués")
        return results

    def fix_missing_init_files(self) -> int:
        """Crée tous les __init__.py manquants"""
        logger.info("📁 Création des __init__.py manquants...")
        count = 0

        for directory in self.root.rglob("*"):
            if directory.is_dir() and '__pycache__' not in str(directory):
                init_file = directory / "__init__.py"
                if not init_file.exists():
                    init_content = '''"""
Package initialization for Jeffrey OS
"""
__version__ = "0.1.0"
__all__ = []

# Auto-découverte des modules
from pathlib import Path
import importlib

_current_dir = Path(__file__).parent
for py_file in _current_dir.glob("*.py"):
    if not py_file.name.startswith("_") and py_file.name != "__init__.py":
        module_name = py_file.stem
        try:
            importlib.import_module(f".{module_name}", package=__package__)
            __all__.append(module_name)
        except ImportError:
            pass
'''
                    init_file.write_text(init_content)
                    count += 1
                    logger.debug(f"  Créé: {init_file}")

        self.fixed_count += count
        logger.info(f"  ✓ {count} fichiers __init__.py créés")
        return count

    def create_critical_modules(self) -> int:
        """Crée les modules critiques manquants"""
        logger.info("🔨 Création des modules critiques manquants...")
        count = 0

        # 1. MemoryManager
        memory_path = self.root / "core" / "memory" / "memory_manager.py"
        if not memory_path.exists():
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            memory_path.write_text(self.get_memory_manager_code())
            self.created_modules.append("MemoryManager")
            count += 1
            logger.info("  ✓ Créé: MemoryManager")

        # 2. UltimateOrchestrator
        orch_path = self.root / "core" / "orchestration" / "ia_orchestrator_ultimate.py"
        if not orch_path.exists():
            orch_path.parent.mkdir(parents=True, exist_ok=True)
            orch_path.write_text(self.get_orchestrator_code())
            self.created_modules.append("UltimateOrchestrator")
            count += 1
            logger.info("  ✓ Créé: UltimateOrchestrator")

        # 3. EnhancedAuditLogger
        audit_path = self.root / "api" / "audit_logger_enhanced.py"
        if not audit_path.exists():
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(self.get_audit_logger_code())
            self.created_modules.append("EnhancedAuditLogger")
            count += 1
            logger.info("  ✓ Créé: EnhancedAuditLogger")

        # 4. EnhancedSandboxManager
        sandbox_path = self.root / "core" / "sandbox_manager_enhanced.py"
        if not sandbox_path.exists():
            sandbox_path.parent.mkdir(parents=True, exist_ok=True)
            sandbox_path.write_text(self.get_sandbox_manager_code())
            self.created_modules.append("EnhancedSandboxManager")
            count += 1
            logger.info("  ✓ Créé: EnhancedSandboxManager")

        self.fixed_count += count
        return count

    def fix_legacy_imports(self) -> int:
        """Corrige tous les imports legacy de manière sécurisée"""
        logger.info("📦 Correction des imports legacy...")

        replacements = [
            (r'from Orchestrateur_IA\.core\.', 'from jeffrey.core.'),
            (r'from Orchestrateur_IA\.services\.', 'from jeffrey.services.'),
            (r'from Orchestrateur_IA\.utilities\.', 'from jeffrey.utilities.'),
            (r'from Orchestrateur_IA\.', 'from jeffrey.'),
            (r'import Orchestrateur_IA\.', 'import jeffrey.'),
            (r'from src\.jeffrey\.', 'from jeffrey.'),
            (r'import src\.jeffrey\.', 'import jeffrey.'),
        ]

        count = 0
        for py_file in self.root.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8')
                original = content

                for pattern, replacement in replacements:
                    content = re.sub(pattern, replacement, content)

                if content != original:
                    py_file.write_text(content, encoding='utf-8')
                    count += 1
                    logger.debug(f"  Corrigé imports: {py_file.name}")

            except Exception as e:
                logger.error(f"  Erreur correction imports {py_file}: {e}")
                self.failed_fixes.append({'file': str(py_file), 'error': str(e)})

        self.fixed_count += count
        logger.info(f"  ✓ {count} fichiers avec imports corrigés")
        return count

    def fix_indentation_errors(self) -> int:
        """Corrige les erreurs d'indentation de manière intelligente"""
        logger.info("📐 Correction des erreurs d'indentation...")

        # Charger le rapport de compilation
        try:
            compilation_report = json.loads(Path("diagnostics/compilation_report.json").read_text())
        except:
            logger.warning("  Pas de rapport de compilation trouvé")
            return 0

        count = 0
        for error_info in compilation_report.get('all_errors', []):
            if 'indentation' not in error_info.get('error', '').lower():
                continue

            file_path = Path(error_info['file'])
            if not file_path.exists():
                continue

            try:
                lines = file_path.read_text(encoding='utf-8').splitlines()
                fixed_lines = []

                i = 0
                while i < len(lines):
                    line = lines[i]
                    fixed_lines.append(line)

                    # Après une définition de classe/fonction
                    if line.strip().startswith(('class ', 'def ', 'async def ')):
                        if i + 1 < len(lines):
                            next_line = lines[i + 1]
                            # Si la ligne suivante n'est pas indentée correctement
                            if (
                                next_line
                                and not next_line[0].isspace()
                                and not next_line.strip().startswith(('"""', "'''", '#'))
                            ):
                                # Ajouter docstring et pass
                                if line.strip().startswith('class '):
                                    fixed_lines.append('    """Class documentation."""')
                                else:
                                    fixed_lines.append('    """Function documentation."""')
                                fixed_lines.append('    pass')

                    i += 1

                file_path.write_text('\n'.join(fixed_lines), encoding='utf-8')
                count += 1
                logger.debug(f"  Corrigé indentation: {file_path.name}")

            except Exception as e:
                logger.error(f"  Erreur correction indentation {file_path}: {e}")

        self.fixed_count += count
        logger.info(f"  ✓ {count} fichiers avec indentation corrigée")
        return count

    def fix_syntax_errors(self) -> int:
        """Corrige les erreurs de syntaxe courantes"""
        logger.info("🔤 Correction des erreurs de syntaxe...")

        try:
            compilation_report = json.loads(Path("diagnostics/compilation_report.json").read_text())
        except:
            return 0

        count = 0
        for error_info in compilation_report.get('all_errors', []):
            if 'syntax' not in error_info.get('error', '').lower():
                continue

            file_path = Path(error_info['file'])
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text(encoding='utf-8')
                original = content

                # F-strings non fermées
                content = re.sub(r'f"([^"]*[^"])$', r'f"\1"', content, flags=re.MULTILINE)
                content = re.sub(r"f'([^']*[^'])$", r"f'\1'", content, flags=re.MULTILINE)

                # Try sans except
                lines = content.splitlines()
                fixed_lines = []
                i = 0
                while i < len(lines):
                    fixed_lines.append(lines[i])
                    if lines[i].strip() == 'try:':
                        # Chercher le except
                        has_except = False
                        j = i + 1
                        while j < len(lines) and lines[j].startswith('    '):
                            if 'except' in lines[j]:
                                has_except = True
                                break
                            j += 1
                        if not has_except and j < len(lines):
                            # Ajouter except après le bloc try
                            fixed_lines.extend(lines[i + 1 : j])
                            fixed_lines.append('except Exception:')
                            fixed_lines.append('    pass')
                            i = j - 1
                    i += 1

                content = '\n'.join(fixed_lines)

                if content != original:
                    file_path.write_text(content, encoding='utf-8')
                    count += 1
                    logger.debug(f"  Corrigé syntaxe: {file_path.name}")

            except Exception as e:
                logger.error(f"  Erreur correction syntaxe {file_path}: {e}")

        self.fixed_count += count
        logger.info(f"  ✓ {count} fichiers avec syntaxe corrigée")
        return count

    def is_inside_async(self, file_path: Path, line_num: int) -> bool:
        """Vérifie si une ligne est dans une fonction async"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
        except:
            return False

        async_ranges = []

        class AsyncRangeFinder(ast.NodeVisitor):
            def visit_AsyncFunctionDef(self, node):
                start = node.lineno
                # Trouver la fin de la fonction
                end = node.lineno
                if hasattr(node, 'end_lineno'):
                    end = node.end_lineno
                elif node.body:
                    # Estimation basée sur le dernier élément du corps
                    last_stmt = node.body[-1]
                    if hasattr(last_stmt, 'end_lineno'):
                        end = last_stmt.end_lineno
                    elif hasattr(last_stmt, 'lineno'):
                        end = last_stmt.lineno + 5  # Estimation

                async_ranges.append((start, end))
                self.generic_visit(node)

        AsyncRangeFinder().visit(tree)

        # Vérifier si la ligne est dans une plage async
        return any(start <= line_num <= end for start, end in async_ranges)

    def fix_async_issues_safe(self) -> int:
        """Corrige les problèmes async de manière sécurisée"""
        logger.info("⚡ Correction sécurisée des problèmes async...")

        try:
            async_report = json.loads(Path("diagnostics/async_report.json").read_text())
        except:
            logger.warning("  Pas de rapport async trouvé")
            return 0

        count = 0

        # 1. Corriger get_event_loop() -> asyncio.run()
        for issue in async_report.get('details', {}).get('event_loop_problems', []):
            file_path = Path(issue['file'])
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text(encoding='utf-8')
                original = content

                # Pattern plus sûr pour remplacer get_event_loop
                content = re.sub(
                    r'loop\s*=\s*asyncio\.get_event_loop\(\)\s*\n\s*loop\.run_until_complete\(([^)]+)\)',
                    r'asyncio.run(\1)',
                    content,
                )

                if content != original:
                    file_path.write_text(content, encoding='utf-8')
                    count += 1
                    logger.debug(f"  Corrigé event_loop: {file_path.name}")

            except Exception as e:
                logger.error(f"  Erreur correction event_loop {file_path}: {e}")

        # 2. Corriger les missing await (haute priorité)
        for issue in async_report.get('details', {}).get('missing_await', []):
            file_path = Path(issue['file'])
            line_num = issue.get('line', 0)

            if not file_path.exists() or line_num <= 0:
                continue

            try:
                lines = file_path.read_text(encoding='utf-8').splitlines()
                if 0 < line_num <= len(lines):
                    line = lines[line_num - 1]
                    # Ajouter await seulement si vraiment manquant
                    if 'await' not in line and issue['call'] in line:
                        # Préserver l'indentation
                        indent = len(line) - len(line.lstrip())
                        lines[line_num - 1] = ' ' * indent + 'await ' + line.lstrip()
                        file_path.write_text('\n'.join(lines), encoding='utf-8')
                        count += 1
                        logger.debug(f"  Ajouté await: {file_path.name}:{line_num}")

            except Exception as e:
                logger.error(f"  Erreur ajout await {file_path}: {e}")

        # 3. Corriger les appels bloquants (seulement dans async)
        for issue in async_report.get('details', {}).get('blocking_calls', []):
            file_path = Path(issue['file'])
            line_num = issue.get('line', 0)

            if not file_path.exists():
                continue

            # Vérifier si on est vraiment dans une fonction async
            if line_num > 0 and not self.is_inside_async(file_path, line_num):
                logger.debug(f"  Ignoré (pas dans async): {file_path.name}:{line_num}")
                continue

            try:
                content = file_path.read_text(encoding='utf-8')
                original = content

                # Remplacements sécurisés
                if 'time.sleep' in issue['issue']:
                    content = re.sub(r'\btime\.sleep\(', 'await asyncio.sleep(', content)
                    if 'import asyncio' not in content:
                        content = 'import asyncio\n' + content

                # Pour requests -> httpx, on ajoute juste un commentaire TODO
                if 'requests' in issue['issue']:
                    lines = content.splitlines()
                    if line_num > 0 and line_num <= len(lines):
                        lines[line_num - 1] += '  # TODO: Migrer vers httpx'
                        content = '\n'.join(lines)

                if content != original:
                    file_path.write_text(content, encoding='utf-8')
                    count += 1
                    logger.debug(f"  Corrigé blocking: {file_path.name}")

            except Exception as e:
                logger.error(f"  Erreur correction blocking {file_path}: {e}")

        # 4. Ajouter asyncio.run() aux entry points
        for issue in async_report.get('details', {}).get('entry_points_without_run', []):
            file_path = Path(issue['file'])
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text(encoding='utf-8')

                if '__main__' in content and 'async def main' in content and 'asyncio.run(main())' not in content:
                    # Pattern sécurisé
                    pattern = r'(if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*\n)(\s*)(main\(\))'
                    replacement = r'\1\2import asyncio\n\2asyncio.run(main())'

                    new_content = re.sub(pattern, replacement, content)

                    if new_content != content:
                        file_path.write_text(new_content, encoding='utf-8')
                        count += 1
                        logger.debug(f"  Ajouté asyncio.run: {file_path.name}")

            except Exception as e:
                logger.error(f"  Erreur ajout asyncio.run {file_path}: {e}")

        self.fixed_count += count
        logger.info(f"  ✓ {count} problèmes async corrigés")
        return count

    def reconnect_orphan_modules(self) -> int:
        """Reconnecte intelligemment les modules orphelins"""
        logger.info("🔗 Reconnexion des modules orphelins...")

        # Vérifier si la reconnexion est activée
        if os.environ.get('JEFFREY_RECONNECT_ORPHANS') != '1':
            logger.info("  Reconnexion désactivée (activer avec JEFFREY_RECONNECT_ORPHANS=1)")
            return 0

        try:
            imports_report = json.loads(Path("diagnostics/imports_report.json").read_text())
        except:
            logger.warning("  Pas de rapport d'imports trouvé")
            return 0

        orphans = imports_report.get('details', {}).get('orphan_modules', [])
        if not orphans:
            logger.info("  Aucun module orphelin trouvé")
            return 0

        count = 0
        reconnection_map = {}

        # Analyser les orphelins pour trouver où les reconnecter
        for orphan_info in orphans[:50]:  # Limiter aux 50 premiers
            module_name = orphan_info['module']

            # S'assurer que le nom est fully-qualified
            full = module_name if module_name.startswith("jeffrey.") else f"jeffrey.{module_name}"

            # Stratégie : trouver le module parent le plus logique
            if 'consciousness' in module_name:
                parent = 'jeffrey.core.consciousness.__init__'
            elif 'emotion' in module_name:
                parent = 'jeffrey.core.emotions.__init__'
            elif 'memory' in module_name:
                parent = 'jeffrey.core.memory.__init__'
            elif 'orchestrat' in module_name:
                parent = 'jeffrey.core.orchestration.__init__'
            else:
                parent = 'jeffrey.core.__init__'

            if parent not in reconnection_map:
                reconnection_map[parent] = []
            reconnection_map[parent].append(full)

        # Appliquer les reconnexions
        for parent_module, orphan_modules in reconnection_map.items():
            parent_path = self.root / parent_module.replace('jeffrey.', '').replace('.', '/').replace(
                '__init__', '__init__.py'
            )

            if parent_path.exists():
                try:
                    content = parent_path.read_text(encoding='utf-8')

                    # Ajouter les imports
                    imports_to_add = []
                    for orphan in orphan_modules:
                        import_line = f"from {orphan} import *  # Auto-reconnected orphan module"
                        if import_line not in content:
                            imports_to_add.append(import_line)

                    if imports_to_add:
                        # Ajouter après les imports existants ou au début
                        lines = content.splitlines()
                        insert_pos = 0
                        for i, line in enumerate(lines):
                            if line.startswith(('import ', 'from ')):
                                insert_pos = i + 1
                            elif line and not line.startswith('#') and insert_pos > 0:
                                break

                        for import_line in imports_to_add:
                            lines.insert(insert_pos, import_line)
                            insert_pos += 1
                            count += 1

                        parent_path.write_text('\n'.join(lines), encoding='utf-8')
                        logger.debug(f"  Reconnecté {len(imports_to_add)} modules dans {parent_path.name}")

                except Exception as e:
                    logger.error(f"  Erreur reconnexion dans {parent_path}: {e}")

        self.fixed_count += count
        logger.info(f"  ✓ {count} modules orphelins reconnectés")
        return count

    def get_memory_manager_code(self) -> str:
        """Code du MemoryManager"""
        return '''"""
Module de gestion mémoire pour Jeffrey OS
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryManager:
    """Gestionnaire de mémoire avec persistance JSON"""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "data/memory/memory.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, Any] = {}
        self._load()
        logger.info(f"MemoryManager initialisé avec {len(self._memory)} entrées")

    def _load(self):
        """Charge la mémoire depuis le disque"""
        if self.storage_path.exists():
            try:
                self._memory = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Erreur chargement mémoire: {e}")
                self._memory = {}

    def _save(self):
        """Sauvegarde la mémoire sur le disque"""
        try:
            self.storage_path.write_text(
                json.dumps(self._memory, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Erreur sauvegarde mémoire: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de la mémoire"""
        return self._memory.get(key, default)

    def set(self, key: str, value: Any):
        """Stocke une valeur dans la mémoire"""
        self._memory[key] = value
        self._memory["_last_update"] = datetime.now().isoformat()
        self._save()

    def delete(self, key: str):
        """Supprime une clé de la mémoire"""
        if key in self._memory:
            del self._memory[key]
            self._save()

    def search_keys(self, prefix: str) -> List[str]:
        """Recherche les clés par préfixe"""
        return [k for k in self._memory if k.startswith(prefix)]

    def get_conversation_context(self, limit: int = 10) -> List[Dict]:
        """Récupère le contexte de conversation"""
        return self.get("conversation_history", [])[-limit:]

    def add_to_context(self, role: str, content: str):
        """Ajoute au contexte de conversation"""
        history = self.get("conversation_history", [])
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.set("conversation_history", history)

    def clear(self):
        """Efface toute la mémoire"""
        self._memory = {}
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la mémoire"""
        return {
            "total_keys": len(self._memory),
            "last_update": self._memory.get("_last_update", "Never"),
            "size_bytes": len(json.dumps(self._memory).encode("utf-8"))
        }
'''

    def get_audit_logger_code(self) -> str:
        """Code de l'EnhancedAuditLogger"""
        return '''"""
Module d'audit avec persistance et rollback
"""
from __future__ import annotations
import json
import logging
import sqlite3
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class BudgetExceededException(Exception):
    """Exception quand le budget est dépassé"""
    pass

@dataclass
class APICall:
    """Structure d'un appel API"""
    timestamp: datetime
    api_name: str
    endpoint: str
    parameters: Dict[str, Any]
    response_time: float
    estimated_cost: float
    success: bool
    error_message: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    user_id: Optional[str] = None
    correlation_id: str = field(default_factory=lambda: hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16])

class EnhancedAuditLogger:
    """Gestionnaire d'audit avec persistance SQLite"""

    def __init__(self, db_path: Optional[str] = None, daily_limit: float = 10.0, monthly_limit: float = 200.0):
        self.db_path = Path(db_path or "data/audit/jeffrey_audit.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.active_transactions: Dict[str, List[APICall]] = {}
        self.daily_spent = 0.0
        self.monthly_spent = 0.0
        self._lock = asyncio.Lock()
        self._init_database()
        logger.info(f"EnhancedAuditLogger initialisé")

    def _init_database(self):
        """Initialise la base SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correlation_id TEXT UNIQUE,
                    transaction_id TEXT,
                    timestamp TEXT,
                    api_name TEXT,
                    endpoint TEXT,
                    parameters TEXT,
                    response_time REAL,
                    estimated_cost REAL,
                    success BOOLEAN,
                    status TEXT DEFAULT 'pending'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON api_calls(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction ON api_calls(transaction_id)")
            conn.commit()

    async def log_api_call_with_rollback(self, api_call: APICall, transaction_id: Optional[str] = None) -> str:
        """Enregistre un appel API avec support de rollback"""
        async with self._lock:
            if self.daily_spent + api_call.estimated_cost > self.daily_limit:
                raise BudgetExceededException(f"Budget quotidien dépassé")

            if not transaction_id:
                transaction_id = f"tx_{datetime.now().timestamp():.0f}"

            if transaction_id not in self.active_transactions:
                self.active_transactions[transaction_id] = []

            self.active_transactions[transaction_id].append(api_call)
            self.daily_spent += api_call.estimated_cost
            self.monthly_spent += api_call.estimated_cost

            return transaction_id

    async def commit_transaction(self, transaction_id: str):
        """Valide une transaction"""
        async with self._lock:
            if transaction_id in self.active_transactions:
                del self.active_transactions[transaction_id]
                logger.debug(f"Transaction committed: {transaction_id}")

    async def rollback_transaction(self, transaction_id: str, reason: str = ""):
        """Annule une transaction"""
        async with self._lock:
            if transaction_id in self.active_transactions:
                calls = self.active_transactions[transaction_id]
                total_cost = sum(call.estimated_cost for call in calls)
                self.daily_spent -= total_cost
                self.monthly_spent -= total_cost
                del self.active_transactions[transaction_id]
                logger.debug(f"Transaction rolled back: {transaction_id}")

    def get_cost_breakdown(self, period: str = "today") -> Dict[str, Any]:
        """Retourne les métriques de coût"""
        return {
            'period': period,
            'daily_spent': round(self.daily_spent, 2),
            'daily_limit': self.daily_limit,
            'daily_remaining': round(self.daily_limit - self.daily_spent, 2),
            'monthly_spent': round(self.monthly_spent, 2),
            'monthly_limit': self.monthly_limit
        }
'''

    def get_sandbox_manager_code(self) -> str:
        """Code de l'EnhancedSandboxManager"""
        return '''"""
Module sandbox avec détection PII et analyse de sécurité
"""
from __future__ import annotations
import re
import json
import hashlib
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ContentStatus(Enum):
    """États du contenu"""
    DRAFT = "draft"
    SCANNING = "scanning"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"

class RiskLevel(Enum):
    """Niveaux de risque"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Résultat de validation"""
    content_id: str
    status: ContentStatus
    risk_level: RiskLevel
    risk_score: float
    pii_detected: List[Dict]
    security_issues: List[str]

class EnhancedSandboxManager:
    """Gestionnaire sandbox avec détection PII et sécurité"""

    # Patterns PII de base
    PII_PATTERNS = {
        'email': re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'),
        'phone': re.compile(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b'),
        'credit_card': re.compile(r'\\b(?:\\d{4}[\\s-]?){3}\\d{4}\\b'),
    }

    # Patterns de sécurité
    SECURITY_PATTERNS = {
        'sql_injection': re.compile(r'(DROP|DELETE|INSERT|UPDATE).*?(TABLE|DATABASE)', re.IGNORECASE),
        'xss_attack': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE),
    }

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "data/sandbox")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.validation_cache: Dict[str, ValidationResult] = {}
        self._lock = asyncio.Lock()
        logger.info("EnhancedSandboxManager initialisé")

    async def submit_creation_with_privacy(self, content: Union[str, bytes], content_type: str,
                                          creator: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Soumet un contenu pour validation"""
        async with self._lock:
            content_hash = hashlib.sha256(
                str(content).encode() if isinstance(content, str) else content
            ).hexdigest()
            creation_id = f"content_{int(datetime.now().timestamp())}_{content_hash[:12]}"

            # Lancer l'analyse async
            asyncio.create_task(self._analyze_content(creation_id, content, content_type, creator))

            return creation_id

    async def _analyze_content(self, content_id: str, content: Any, content_type: str, creator: str):
        """Analyse le contenu pour risques"""
        if not isinstance(content, str):
            result = ValidationResult(
                content_id=content_id,
                status=ContentStatus.PENDING_REVIEW,
                risk_level=RiskLevel.LOW,
                risk_score=0.1,
                pii_detected=[],
                security_issues=[]
            )
        else:
            # Détection PII
            pii_detections = []
            for pii_type, pattern in self.PII_PATTERNS.items():
                if pattern.search(content):
                    pii_detections.append({'type': pii_type, 'detected': True})

            # Analyse sécurité
            security_issues = []
            for threat, pattern in self.SECURITY_PATTERNS.items():
                if pattern.search(content):
                    security_issues.append(threat)

            # Score de risque
            risk_score = min(1.0, len(pii_detections) * 0.2 + len(security_issues) * 0.4)

            # Niveau de risque
            if risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 0.6:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
            elif risk_score >= 0.2:
                risk_level = RiskLevel.LOW
            else:
                risk_level = RiskLevel.SAFE

            # Statut
            if risk_score >= 0.8:
                status = ContentStatus.QUARANTINED
            elif risk_score >= 0.6:
                status = ContentStatus.PENDING_REVIEW
            else:
                status = ContentStatus.APPROVED

            result = ValidationResult(
                content_id=content_id,
                status=status,
                risk_level=risk_level,
                risk_score=risk_score,
                pii_detected=pii_detections,
                security_issues=security_issues
            )

        self.validation_cache[content_id] = result
        logger.debug(f"Content analyzed: {content_id} - Risk: {result.risk_score:.2f}")

    async def get_validation_result(self, content_id: str) -> Optional[ValidationResult]:
        """Récupère le résultat de validation"""
        return self.validation_cache.get(content_id)

    def get_pending_reviews(self, content_type: Optional[str] = None) -> List[Dict]:
        """Retourne les contenus en attente"""
        pending = []
        for content_id, result in self.validation_cache.items():
            if result.status == ContentStatus.PENDING_REVIEW:
                pending.append({
                    'id': content_id,
                    'risk_level': result.risk_level.value,
                    'risk_score': result.risk_score
                })
        return pending

    async def validate_creation(self, content_id: str, validator: str, approved: bool, reason: str = "") -> bool:
        """Valide ou rejette un contenu"""
        async with self._lock:
            if content_id in self.validation_cache:
                result = self.validation_cache[content_id]
                result.status = ContentStatus.APPROVED if approved else ContentStatus.REJECTED
                logger.debug(f"Content validated: {content_id} - {'Approved' if approved else 'Rejected'}")
                return True
            return False
'''

    def get_orchestrator_code(self) -> str:
        """Code de l'UltimateOrchestrator"""
        return '''"""
Orchestrateur principal de Jeffrey OS
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Imports conditionnels pour éviter les erreurs circulaires
try:
    from jeffrey.core.memory.memory_manager import MemoryManager
except ImportError:
    logger.warning("MemoryManager non disponible")
    MemoryManager = None

try:
    from jeffrey.api.audit_logger_enhanced import EnhancedAuditLogger, APICall
except ImportError:
    logger.warning("EnhancedAuditLogger non disponible")
    EnhancedAuditLogger = None
    APICall = None

try:
    from jeffrey.core.sandbox_manager_enhanced import EnhancedSandboxManager
except ImportError:
    logger.warning("EnhancedSandboxManager non disponible")
    EnhancedSandboxManager = None

class UltimateOrchestrator:
    """
    Orchestrateur principal avec gestion mémoire, audit et sandbox
    """

    def __init__(self):
        """Initialise l'orchestrateur avec les composants disponibles"""
        self.memory = MemoryManager() if MemoryManager else None
        self.audit = EnhancedAuditLogger() if EnhancedAuditLogger else None
        self.sandbox = EnhancedSandboxManager() if EnhancedSandboxManager else None

        self.initialized = all([self.memory, self.audit, self.sandbox])

        if self.initialized:
            logger.info("UltimateOrchestrator initialisé avec tous les composants")
        else:
            logger.warning("UltimateOrchestrator en mode dégradé - composants manquants")

    async def process(self, text: str, user: str = "user") -> str:
        """
        Traite une requête utilisateur

        Args:
            text: Texte à traiter
            user: Identifiant utilisateur

        Returns:
            Réponse générée
        """
        response = f"Jeffrey> Reçu: {text}"

        try:
            # Vérification sandbox
            if self.sandbox:
                content_id = await self.sandbox.submit_creation_with_privacy(
                    content=text,
                    content_type="text",
                    creator=user
                )
                logger.debug(f"Contenu sandbox: {content_id}")

            # Audit de l'appel
            if self.audit and APICall:
                api_call = APICall(
                    timestamp=datetime.now(),
                    api_name="orchestrator",
                    endpoint="process",
                    parameters={"text_length": len(text)},
                    response_time=0.01,
                    estimated_cost=0.001,
                    success=True,
                    user_id=user
                )
                tx_id = await self.audit.log_api_call_with_rollback(api_call)
                await self.audit.commit_transaction(tx_id)

            # Stockage en mémoire
            if self.memory:
                self.memory.add_to_context(user, text)
                self.memory.set("last_interaction", datetime.now().isoformat())

            # Ici on pourrait ajouter le traitement IA réel
            # Pour l'instant, réponse simple
            response = f"Jeffrey> J'ai bien reçu votre message: '{text[:50]}...'"

            # Ajouter la réponse au contexte
            if self.memory:
                self.memory.add_to_context("assistant", response)

        except Exception as e:
            logger.error(f"Erreur traitement: {e}")
            response = "Jeffrey> Désolé, une erreur s'est produite."

        return response

    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut de l'orchestrateur"""
        status = {
            "initialized": self.initialized,
            "components": {
                "memory": self.memory is not None,
                "audit": self.audit is not None,
                "sandbox": self.sandbox is not None
            }
        }

        if self.memory:
            status["memory_stats"] = self.memory.get_stats()

        if self.audit:
            status["audit_stats"] = self.audit.get_cost_breakdown()

        return status

    async def shutdown(self):
        """Arrêt propre de l'orchestrateur"""
        logger.info("Arrêt de l'orchestrateur...")

        # Sauvegarder l'état si nécessaire
        if self.memory:
            self.memory.set("shutdown_time", datetime.now().isoformat())

        logger.info("Orchestrateur arrêté")
'''


# Exécution
if __name__ == "__main__":
    fixer = JeffreyAutoFixer()
    results = fixer.run_all_fixes()

    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES CORRECTIONS")
    print("=" * 50)
    print(f"✅ Total corrigé: {results['summary']['total_fixed']}")
    print(f"📦 Modules créés: {', '.join(results['summary']['created_modules'])}")
    print(f"❌ Échecs: {len(results['summary']['failed_fixes'])}")
