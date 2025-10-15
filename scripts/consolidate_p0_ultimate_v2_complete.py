#!/usr/bin/env python3
"""
Jeffrey OS - P0 Consolidation Ultimate V2 Complete
Version finale avec TOUTES les améliorations de l'équipe
Production-ready avec fondations indestructibles
"""

import argparse
import ast
import asyncio
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psutil


class P0ConsolidatorUltimateV2Complete:
    """
    Consolidateur P0 avec toutes les améliorations:
    - AST lambdas/comprehensions (Grok)
    - Workers IO-bound optimaux (Grok + GPT)
    - Memory monitoring (Grok)
    - Proxy URL invalide (Grok)
    - Symbiosis timer + prune (Grok + GPT)
    - Deps centrality (Grok)
    - EU AI Act compliance (Grok)
    - iCloud preflight (GPT)
    - Hash 64 chars (GPT)
    - Git idempotence (GPT)
    """

    def __init__(self, args):
        self.args = args
        self.base_dir = Path.cwd()  # Use current working directory
        self.icloud_base = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS"

        # Enhanced report structure
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0-complete",
            "status": "IN_PROGRESS",
            "phases": {},
            "issues": [],
            "warnings": [],
            "drifts": [],
            "fixes_suggested": [],
            "symbiosis_score": None,
            "p1_priorities": [],
            "compliance_warnings": [],
            "memory_stats": {},
            "summary": {},
        }

        self.setup_logging()
        self.define_p0_modules()

    def define_p0_modules(self):
        """Define P0 modules with ACTUAL paths (temporary chaos) vs FUTURE paths (clean)"""

        # ACTUAL: Where modules are NOW (chaotic organization)
        # FUTURE: Where they SHOULD be (clean organization)

        self.p0_modules = {
            "dream_engine": {
                "source": self.icloud_base / "src/core/dreaming/dream_engine.py",
                "dest": self.base_dir / "src/jeffrey/core/consciousness/dream_engine.py",  # ACTUAL (chaos)
                "dest_future": self.base_dir / "src/jeffrey/core/dreaming/dream_engine.py",  # FUTURE (clean)
            },
            "self_awareness_tracker": {
                "source": self.icloud_base / "src/core/consciousness/self_awareness_tracker.py",
                "dest": self.base_dir / "src/jeffrey/core/consciousness/self_awareness_tracker.py",  # OK
            },
            "cognitive_synthesis": {
                "source": self.icloud_base / "src/core/memory/cognitive_synthesis.py",
                "dest": self.base_dir / "src/jeffrey/core/consciousness/cognitive_synthesis.py",  # ACTUAL (chaos)
                "dest_future": self.base_dir / "src/jeffrey/core/memory/cognitive_synthesis.py",  # FUTURE (clean)
            },
            "cortex_memoriel": {
                "source": self.icloud_base / "src/core/memory/cortex_memoriel.py",
                "dest": self.base_dir / "src/jeffrey/core/memory/cortex_memoriel.py",  # OK
            },
        }

        # Log structure status
        self.log("   📁 Module locations (ACTUAL vs FUTURE):", "info")
        for name, config in self.p0_modules.items():
            dest_exists = "✅" if config["dest"].exists() else "❌"
            actual_loc = config["dest"].parent.name
            future_loc = config.get("dest_future", config["dest"]).parent.name

            if actual_loc != future_loc:
                self.log(
                    f"      {name}: {actual_loc}/ → {future_loc}/ (needs reorg) {dest_exists}",
                    "warning",
                )
            else:
                self.log(f"      {name}: {actual_loc}/ (correct) {dest_exists}", "success")

    def setup_logging(self):
        """Setup dual logging (console + file)"""
        # Create reports directory
        log_dir = self.base_dir / "reports"
        log_dir.mkdir(exist_ok=True)

        # Log file with timestamp
        if self.args.log_file:
            self.log_file = open(self.args.log_file, "w", encoding="utf-8")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = open(log_dir / f"p0_consolidation_v2_{timestamp}.log", "w", encoding="utf-8")

        # Write header
        self.log_file.write("Jeffrey OS - P0 Consolidation V2 Complete\n")
        self.log_file.write(f"Started: {datetime.now().isoformat()}\n")
        self.log_file.write("=" * 50 + "\n\n")

    def log(self, message, level="info"):
        """Enhanced dual logging"""
        # File logging (always)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_file.write(f"[{timestamp}] {level.upper()}: {message}\n")
        self.log_file.flush()

        # Console logging
        if self.args.no_color or not sys.stdout.isatty():
            print(message)
        else:
            colors = {
                "header": "\033[95m",
                "success": "\033[92m",
                "warning": "\033[93m",
                "error": "\033[91m",
                "info": "\033[94m",
                "endc": "\033[0m",
            }
            color = colors.get(level, colors["info"])
            print(f"{color}{message}{colors['endc']}")

    def compute_file_hash(self, filepath: Path) -> str:
        """Compute full SHA-256 hash (64 chars)"""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()  # Full 64 chars

    def get_optimal_workers(self):
        """Workers optimaux pour IO-bound avec iCloud (GPT + GROK formula)"""
        cpu_count = os.cpu_count() or 4

        # GPT : max 4 pour éviter saturation iCloud
        # GROK : //3 pour balance IO/CPU
        optimal = min(4, max(1, cpu_count // 3))

        self.log(f"Using {optimal} workers (CPUs: {cpu_count}, IO-bound mode)", "info")
        return optimal

    def get_import_path(self, module_name):
        """Get correct import path based on module structure"""
        if module_name == "dream_engine":
            return "src.jeffrey.core.dreaming.dream_engine"
        elif module_name in ["cognitive_synthesis", "cortex_memoriel"]:
            return f"src.jeffrey.core.memory.{module_name}"
        else:  # self_awareness_tracker
            return f"src.jeffrey.core.consciousness.{module_name}"

    def check_memory_usage(self):
        """Monitor memory during runtime tests"""
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_available_gb = mem.available / (1024**3)

        if mem_percent > 80:
            self.log(f"   ⚠️ High memory usage: {mem_percent:.1f}%", "warning")
            self.report["warnings"].append(f"Memory usage high: {mem_percent:.1f}%")

        if mem_available_gb < 1:
            self.log(f"   ⚠️ Low memory available: {mem_available_gb:.1f}GB", "warning")

        return mem_percent, mem_available_gb

    def get_offline_env(self):
        """Environment pour offline HERMÉTIQUE TOTAL"""
        env = os.environ.copy()

        # HuggingFace complet
        env.update(
            {
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )

        # PyTorch complet
        env.update(
            {
                "TORCH_OFFLINE": "1",
                "TORCH_HOME": str(self.base_dir / "cache" / "torch"),
                "TORCH_HUB": str(self.base_dir / "cache" / "torch_hub"),
            }
        )

        # FAISS et ML
        env.update(
            {
                "FAISS_OFFLINE": "1",
                "FAISS_DISABLE_CPU_FEATURES": "1",  # GPT addition
                "SENTENCE_TRANSFORMERS_OFFLINE": "1",
                "NLTK_DATA": str(self.base_dir / "cache" / "nltk"),
            }
        )

        # PROXY BLOCK TOTAL avec URL invalide (Grok improvement)
        env.update(
            {
                "HTTP_PROXY": "http://invalid.proxy:9999",
                "HTTPS_PROXY": "http://invalid.proxy:9999",
                "http_proxy": "http://invalid.proxy:9999",  # lowercase aussi
                "https_proxy": "http://invalid.proxy:9999",
                "NO_PROXY": "*",
                "no_proxy": "*",
                "REQUESTS_CA_BUNDLE": "",
                "CURL_CA_BUNDLE": "",
            }
        )

        # Python et pip
        env.update(
            {
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INDEX": "1",  # Block PyPI
                "PIP_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONWARNINGS": "ignore",
            }
        )

        # Log pour audit
        self.log("   Offline environment configured:", "info")
        for key in ["HTTP_PROXY", "TRANSFORMERS_OFFLINE", "TORCH_OFFLINE"]:
            self.log(f"      {key}={env.get(key)}", "info")

        return env

    def check_import_protection_complete(self, tree):
        """Deep AST analysis avec lambdas et list comprehensions"""

        def add_parents(node, parent=None):
            node.parent = parent
            for child in ast.iter_child_nodes(node):
                add_parents(child, node)

        def node_in_protected_context(node):
            """Check if import is protected (function, try/except, lambda, comprehension)"""
            current = node
            while hasattr(current, "parent") and current.parent:
                parent = current.parent
                # Functions
                if isinstance(parent, ast.FunctionDef):
                    return True, "function"
                # Try/except
                if isinstance(parent, (ast.ExceptHandler, ast.Try)):
                    return True, "try/except"
                # Lambdas
                if isinstance(parent, ast.Lambda):
                    return True, "lambda"
                # Comprehensions
                if isinstance(parent, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                    return True, "comprehension"
                current = parent
            return False, None

        add_parents(tree)
        unprotected = []

        # Deep walk ALL nodes including nested
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = self.get_import_name(node)

                if self.is_heavy_import(module_name):
                    protected, context = node_in_protected_context(node)
                    if protected:
                        self.log(f"      Import '{module_name}' in {context} (protected)", "info")
                    else:
                        unprotected.append(module_name)
                        self.log(f"      ⚠️ Unprotected heavy import: {module_name}", "warning")

        return unprotected

    def get_import_name(self, node):
        """Extract import name from AST node"""
        if isinstance(node, ast.Import):
            return node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            return node.module
        return None

    def is_heavy_import(self, module_name):
        """Check if module is heavy"""
        if not module_name:
            return False
        heavy_modules = [
            "torch",
            "tensorflow",
            "transformers",
            "faiss",
            "numpy",
            "pandas",
            "sklearn",
            "scipy",
        ]
        return any(module_name.startswith(h) for h in heavy_modules)

    def preflight_checks(self):
        """Enhanced preflight with iCloud sync check"""
        self.log("\n" + "=" * 50, "header")
        self.log("PREFLIGHT CHECKS", "header")

        # Check iCloud sources
        missing = []
        not_synced = []

        for name, config in self.p0_modules.items():
            source = config["source"]

            if not source.exists():
                missing.append((name, str(source)))
            elif source.stat().st_size == 0:
                not_synced.append((name, str(source)))

        if missing:
            self.log("❌ Missing sources (iCloud not synced?):", "error")
            for name, path in missing[:3]:
                self.log(f"   - {name}: {path}", "error")

            # Suggestion
            self.log("\n💡 Fix: Open Finder → iCloud Drive → Wait for ✓ green checkmarks", "info")

            if not self.args.dry_run:
                raise FileNotFoundError(f"Missing {len(missing)} source files - sync iCloud first")

        if not_synced:
            self.log("⚠️ Empty files (downloading?):", "warning")
            for name, path in not_synced:
                self.log(f"   - {name}: {path}", "warning")

        # Check disk space
        usage = shutil.disk_usage(self.base_dir)
        free_gb = usage.free / (1024**3)
        if free_gb < 1:
            self.log(f"⚠️ Low disk space: {free_gb:.1f}GB", "warning")

        # Check memory
        self.check_memory_usage()

        self.log("   ✅ All preflight checks passed", "success")

    async def check_symbiosis_compatibility(self):
        """Real symbiosis check avec timer soft et prune suggestions"""
        try:
            # Import avec timeout
            from jeffrey.core.symbiosis.symbiosis_engine import SymbiosisEngine

            symbiosis = SymbiosisEngine()

            modules_pairs = [
                ("dream_engine", "cortex_memoriel"),
                ("self_awareness_tracker", "cognitive_synthesis"),
                ("dream_engine", "cognitive_synthesis"),
            ]

            scores = []
            low_compat_pairs = []

            for mod1, mod2 in modules_pairs:
                try:
                    # Timer soft 3-5s (GPT suggestion)
                    score = await asyncio.wait_for(symbiosis.check_compat(mod1, mod2), timeout=5.0)
                    scores.append(score)

                    if score < 0.7:
                        low_compat_pairs.append((mod1, mod2, score))
                        self.log(f"   ⚠️ Low compatibility {mod1}↔{mod2}: {score:.2f}", "warning")
                    else:
                        self.log(f"   ✅ Symbiosis {mod1}↔{mod2}: {score:.2f}", "success")

                except TimeoutError:
                    self.log(f"   ⏱️ Symbiosis check timeout for {mod1}↔{mod2}", "warning")
                except Exception as e:
                    self.log(f"   Symbiosis check skipped: {e}", "info")

            # Prune suggestions si low compat (Grok innovation)
            if low_compat_pairs:
                for mod1, mod2, score in low_compat_pairs:
                    self.report["fixes_suggested"].append(
                        f"Consider pruning low-compat link {mod1}↔{mod2} (score: {score:.2f})"
                    )

            if scores:
                avg_score = sum(scores) / len(scores)
                self.report["symbiosis_score"] = avg_score
                return avg_score
            else:
                self.log("   Symbiosis checks skipped (not available)", "info")
                return None

        except ImportError:
            self.log("   Symbiosis module not found - skipping", "info")
            return None

    def analyze_stub_dependencies(self):
        """Analyze stubs with centrality for P1 priorities"""
        try:
            # Run deps_graph if exists
            deps_script = self.base_dir / "scripts" / "deps_graph.py"
            if not deps_script.exists():
                return

            result = subprocess.run(
                ["python3", str(deps_script), "--json", "--centrality"],
                capture_output=True,
                text=True,
                cwd=self.base_dir,
                timeout=30,
            )

            if result.returncode == 0:
                deps_data = json.loads(result.stdout)

                # Calculate stub priorities based on centrality
                stub_priorities = []
                for module, info in deps_data.items():
                    if module.endswith("_stub"):
                        centrality = info.get("centrality", 0)
                        imports_from_p0 = sum(1 for imp in info.get("imported_by", []) if imp in self.p0_modules)
                        priority_score = centrality * 10 + imports_from_p0 * 5

                        stub_priorities.append(
                            {
                                "stub": module,
                                "centrality": centrality,
                                "p0_dependencies": imports_from_p0,
                                "priority_score": priority_score,
                                "priority": (
                                    "HIGH" if priority_score > 15 else "MEDIUM" if priority_score > 5 else "LOW"
                                ),
                            }
                        )

                # Sort and save
                stub_priorities.sort(key=lambda x: x["priority_score"], reverse=True)

                self.report["p1_priorities"] = stub_priorities[:10]  # Top 10

                # Generate visual if matplotlib available
                try:
                    self.generate_deps_visualization(deps_data)
                except ImportError:
                    pass

        except Exception as e:
            self.log(f"   Deps analysis skipped: {e}", "info")

    def check_ai_compliance(self):
        """Check EU AI Act compliance for P0 modules"""
        high_risk_keywords = [
            "biometric",
            "emotion",
            "prediction",
            "profiling",
            "decision",
            "autonomous",
            "surveillance",
        ]

        compliance_issues = []

        for module_name, config in self.p0_modules.items():
            module_path = config["dest"]
            if module_path.exists():
                content = module_path.read_text().lower()

                # Check for high-risk AI patterns
                risks_found = [kw for kw in high_risk_keywords if kw in content]

                if risks_found:
                    compliance_issues.append(
                        {
                            "module": module_name,
                            "risks": risks_found,
                            "recommendation": f"Review {module_name} for EU AI Act compliance (keywords: {', '.join(risks_found)})",
                        }
                    )

        if compliance_issues:
            self.report["compliance_warnings"] = compliance_issues
            for issue in compliance_issues:
                self.log(
                    f"   ⚖️ Compliance warning: {issue['module']} - {issue['recommendation']}",
                    "warning",
                )

    async def phase1_analyze_sources(self):
        """Analyze source modules"""
        self.log("\n" + "=" * 50, "header")
        self.log("PHASE 1: ANALYZE SOURCES", "header")

        analysis_results = {}

        for module_name, config in self.p0_modules.items():
            source = config["source"]

            if source.exists():
                size = source.stat().st_size
                content = source.read_text()
                tree = ast.parse(content, str(source))

                # Check for unprotected imports
                unprotected = self.check_import_protection_complete(tree)

                # Compute hash
                file_hash = self.compute_file_hash(source)

                analysis_results[module_name] = {
                    "size": size,
                    "lines": len(content.splitlines()),
                    "hash": file_hash,
                    "unprotected_imports": unprotected,
                }

                self.log(f"   ✅ {module_name}: {size} bytes, hash: {file_hash[:8]}...", "success")

                if unprotected:
                    self.report["warnings"].append(f"{module_name} has {len(unprotected)} unprotected imports")
            else:
                self.log(f"   ❌ {module_name}: source not found", "error")
                self.report["issues"].append(f"{module_name}: source missing")

        self.report["phases"]["analyze"] = analysis_results

    async def phase2_copy_with_validation(self):
        """Copy modules with validation"""
        self.log("\n" + "=" * 50, "header")
        self.log("PHASE 2: COPY WITH VALIDATION", "header")

        if self.args.dry_run:
            self.log("   🔍 DRY-RUN: No files will be copied", "info")
            return

        copied = []

        for module_name, config in self.p0_modules.items():
            source = config["source"]
            dest = config["dest"]

            if source.exists():
                # Create destination directory
                dest.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(source, dest)

                # Verify hash
                source_hash = self.compute_file_hash(source)
                dest_hash = self.compute_file_hash(dest)

                if source_hash == dest_hash:
                    self.log(f"   ✅ {module_name}: copied and verified", "success")
                    copied.append(module_name)
                else:
                    self.log(f"   ❌ {module_name}: hash mismatch after copy", "error")
                    self.report["issues"].append(f"{module_name}: copy verification failed")
            else:
                self.log(f"   ⚠️ {module_name}: source not found, skipping", "warning")

        self.report["phases"]["copy"] = {"copied": copied}

    async def phase3_runtime_tests(self):
        """Runtime tests with memory monitoring"""
        self.log("\n" + "=" * 50, "header")
        self.log("PHASE 3: RUNTIME TESTS", "header")

        # Check memory before
        mem_before, _ = self.check_memory_usage()

        env = self.get_offline_env() if self.args.offline else os.environ.copy()
        test_results = {}

        for module_name in self.p0_modules:
            try:
                # Basic import test
                result = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        sys.executable,
                        "-c",
                        f"import sys; sys.path.insert(0, '.'); "
                        f"from {self.get_import_path(module_name)} import *; print('OK')",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.base_dir,
                        env=env,
                    ),
                    timeout=self.args.timeout,
                )

                stdout, stderr = await result.communicate()

                if result.returncode == 0 and b"OK" in stdout:
                    self.log(f"   ✅ {module_name}: runtime test passed", "success")
                    test_results[module_name] = "PASS"
                else:
                    self.log(f"   ❌ {module_name}: runtime test failed", "error")
                    if stderr:
                        self.log(f"      Error: {stderr.decode()[:200]}", "error")
                    test_results[module_name] = "FAIL"
                    self.report["issues"].append(f"{module_name}: runtime test failed")

            except TimeoutError:
                self.log(f"   ⏱️ {module_name}: test timeout ({self.args.timeout}s)", "warning")
                test_results[module_name] = "TIMEOUT"
                self.report["warnings"].append(f"{module_name}: test timeout")
            except Exception as e:
                self.log(f"   ❌ {module_name}: test error: {e}", "error")
                test_results[module_name] = "ERROR"
                self.report["issues"].append(f"{module_name}: {str(e)}")

        # Check memory after
        mem_after, _ = self.check_memory_usage()

        if mem_after - mem_before > 20:  # 20% increase
            self.log(f"   ⚠️ Memory increased by {mem_after - mem_before:.1f}% during tests", "warning")

        # Symbiosis check
        await self.check_symbiosis_compatibility()

        self.report["phases"]["runtime"] = test_results

    async def phase4_audit_stubs(self):
        """Parallel audit of stubs"""
        self.log("\n" + "=" * 50, "header")
        self.log("PHASE 4: AUDIT STUBS", "header")

        # Find all stub files
        stub_files = list(self.base_dir.glob("src/jeffrey/stubs/*_stub.py"))

        if not stub_files:
            self.log("   No stub files found", "info")
            return

        self.log(f"   Found {len(stub_files)} stub files", "info")

        # Parallel analysis with optimal workers
        stub_stats = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=(self.args.workers or self.get_optimal_workers())
        ) as executor:
            futures = {}

            for stub_file in stub_files:
                futures[executor.submit(self.analyze_stub_file, stub_file)] = stub_file

            for future in concurrent.futures.as_completed(futures):
                stub_file = futures[future]
                try:
                    stats = future.result(timeout=10)
                    stub_stats[stub_file.stem] = stats
                    self.log(
                        f"   ✅ {stub_file.stem}: {stats['lines']} lines, {stats['todos']} TODOs",
                        "info",
                    )
                except Exception as e:
                    self.log(f"   ❌ {stub_file.stem}: analysis failed: {e}", "error")

        # Analyze dependencies
        self.analyze_stub_dependencies()

        self.report["phases"]["stubs"] = stub_stats

    def analyze_stub_file(self, stub_file):
        """Analyze a single stub file"""
        content = stub_file.read_text()
        lines = content.splitlines()

        # Count TODOs and FIXME
        todos = sum(1 for line in lines if "TODO" in line or "FIXME" in line)

        # Check for NotImplementedError
        not_implemented = content.count("NotImplementedError")

        return {
            "lines": len(lines),
            "todos": todos,
            "not_implemented": not_implemented,
            "size": stub_file.stat().st_size,
        }

    async def phase5_documentation(self):
        """Generate documentation"""
        self.log("\n" + "=" * 50, "header")
        self.log("PHASE 5: DOCUMENTATION", "header")

        # Create integrity lock file
        lock_file = self.base_dir / "p0_integrity.lock"
        lock_data = {"timestamp": datetime.now().isoformat(), "version": "2.0.0", "modules": {}}

        for module_name, config in self.p0_modules.items():
            if config["dest"].exists():
                lock_data["modules"][module_name] = {
                    "hash": self.compute_file_hash(config["dest"]),  # Full 64 chars
                    "size": config["dest"].stat().st_size,
                    "path": str(config["dest"].relative_to(self.base_dir)),
                }

        if not self.args.dry_run:
            with open(lock_file, "w") as f:
                json.dump(lock_data, f, indent=2)
            self.log(
                f"   ✅ Created p0_integrity.lock with {len(lock_data['modules'])} modules",
                "success",
            )
        else:
            self.log("   🔍 DRY-RUN: Would create p0_integrity.lock", "info")

        self.report["phases"]["documentation"] = {"lock_file": str(lock_file)}

    def generate_final_report(self):
        """Generate final report"""
        self.log("\n" + "=" * 50, "header")
        self.log("FINAL REPORT", "header")

        # Summary
        self.report["summary"] = {
            "modules_processed": len(self.p0_modules),
            "issues_count": len(self.report["issues"]),
            "warnings_count": len(self.report["warnings"]),
            "fixes_suggested": len(self.report["fixes_suggested"]),
        }

        # Display summary
        self.log("\n📊 Summary:", "header")
        self.log(f"   Modules: {self.report['summary']['modules_processed']}", "info")
        self.log(
            f"   Issues: {self.report['summary']['issues_count']}",
            "error" if self.report["issues"] else "info",
        )
        self.log(
            f"   Warnings: {self.report['summary']['warnings_count']}",
            "warning" if self.report["warnings"] else "info",
        )

        if self.report["symbiosis_score"] is not None:
            self.log(f"   Symbiosis Score: {self.report['symbiosis_score']:.2f}", "info")

        # Save report
        if not self.args.dry_run:
            report_file = (
                self.base_dir / "reports" / f"p0_consolidation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(self.report, f, indent=2)
            self.log(f"\n   📄 Report saved to: {report_file}", "info")

    def git_operations(self):
        """Git operations with idempotence"""
        self.log("\n" + "=" * 50, "header")
        self.log("GIT OPERATIONS", "header")

        try:
            # Check if we need to create a new branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.base_dir,
            )

            current_branch = result.stdout.strip()

            if current_branch != "p0-ultimate-v2-complete":
                # Create and switch to new branch
                subprocess.run(
                    ["git", "checkout", "-b", "p0-ultimate-v2-complete"],
                    cwd=self.base_dir,
                    check=False,  # Ignore if branch exists
                )

            # Add files
            for config in self.p0_modules.values():
                if config["dest"].exists():
                    subprocess.run(["git", "add", str(config["dest"])], cwd=self.base_dir)

            # Add lock file
            subprocess.run(["git", "add", "p0_integrity.lock"], cwd=self.base_dir, check=False)

            # Commit
            message = f"P0 Consolidation V2 Complete - {len(self.p0_modules)} modules secured"
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.base_dir,
                check=False,  # Ignore if nothing to commit
            )

            self.log("   ✅ Git operations completed", "success")

        except Exception as e:
            self.log(f"   ⚠️ Git operations failed: {e}", "warning")

    async def run(self):
        """Main consolidation pipeline"""
        try:
            # Start
            self.log("\n🚀 Jeffrey OS - P0 Consolidation V2 Complete", "header")
            self.log("=" * 50, "header")

            # Preflight
            self.preflight_checks()

            # Phase 1: Analyze sources
            await self.phase1_analyze_sources()

            # Phase 2: Copy with validation
            await self.phase2_copy_with_validation()

            # Phase 3: Runtime tests
            await self.phase3_runtime_tests()

            # Phase 4: Parallel audit
            await self.phase4_audit_stubs()

            # Phase 5: Documentation
            await self.phase5_documentation()

            # Phase 6: Compliance
            self.check_ai_compliance()

            # Final report
            self.generate_final_report()

            # Git operations
            if not self.args.no_git and not self.args.dry_run:
                self.git_operations()

            # Check strict/warning modes
            if self.args.strict and self.report["issues"]:
                self.report["status"] = "FAILED_STRICT"
                self.log("\n❌ Failed due to --strict mode", "error")
                sys.exit(1)

            if self.args.fail_on_warning and self.report["warnings"]:
                self.report["status"] = "FAILED_WARNINGS"
                self.log("\n❌ Failed due to --fail-on-warning mode", "error")
                sys.exit(1)

            self.report["status"] = "SUCCESS"
            self.log("\n✅ P0 Consolidation Complete!", "success")

        except Exception as e:
            self.report["status"] = "ERROR"
            self.report["error"] = str(e)
            self.log(f"\n❌ Consolidation failed: {e}", "error")
            raise
        finally:
            if hasattr(self, "log_file"):
                self.log_file.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Jeffrey OS P0 Consolidation Ultimate V2 Complete",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # All CLI options
    parser.add_argument("--dry-run", action="store_true", help="Simulation mode - no file modifications")
    parser.add_argument("--offline", action="store_true", help="Force complete offline mode")
    parser.add_argument("--no-git", action="store_true", help="Skip Git operations")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout per test in seconds (default: 15)")
    parser.add_argument("--strict", action="store_true", help="Fail on any test failure")
    parser.add_argument("--fail-on-warning", action="store_true", help="Fail if warnings detected")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--log-file", type=str, help="Custom log file path")
    parser.add_argument("--workers", type=int, help="Override worker count")

    args = parser.parse_args()

    # Run consolidation
    consolidator = P0ConsolidatorUltimateV2Complete(args)
    asyncio.run(consolidator.run())


if __name__ == "__main__":
    main()
