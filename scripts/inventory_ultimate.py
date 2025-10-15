#!/usr/bin/env python3
"""
Ultimate Module Inventory for Jeffrey OS v3.1 HARDENED
- All team corrections integrated
- fnmatch for exclusions
- Dynamic activity scoring
- Ethical/GPU bonuses
- Network blocking in sandbox
"""

import ast
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration
BUDGET_SECONDS = 60
MIN_LINES_DEFAULT = 60
TIMEOUT_PER_FILE_MS = 200
CACHE_FILE = "artifacts/module_cache.json"
OUTPUT_FILE = "artifacts/inventory_ultimate.json"

# Whitelist pour petits modules critiques (Gemini suggestion)
WHITELIST_SMALL_MODULES = {
    "response_generator",
    "ollama_interface",
    "bridge_*",
    "adapter_*",
    "config",
    "constants",
    "simple_*",  # Fallbacks simples
}

# Exclusions strictes (GPT requirement)
EXCLUSION_PATTERNS = [
    "*/tests/*",
    "*/test/*",
    "*test*.py",
    "*_test.py",
    "*/__pycache__/*",
    "*/.venv/*",
    "*/venv/*",
    "*/archive/*",
    "*/deprecated/*",
    "*/old/*",
    "*/backup/*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
]

# Patterns avancés pour scoring (Grok enhancement)
ADVANCED_PATTERNS = {
    "ethics": ["ethics", "bias", "alignment", "moral", "safe", "responsible"],
    "gpu": ["torch", "cuda", "gpu", "tensor", "accelerate"],
    "ai_ml": ["model", "train", "predict", "neural", "embedding", "transformer"],
    "realtime": ["async", "await", "asyncio", "concurrent", "thread"],
    "memory": ["memory", "cache", "storage", "recall", "episodic", "semantic"],
    "consciousness": ["consciousness", "aware", "meta", "introspect", "reflect"],
    "emotion": ["emotion", "feel", "mood", "sentiment", "empathy"],
    "decision": ["decide", "choice", "verdict", "judge", "evaluate"],
}


class UltimateInventory:
    def __init__(self):
        self.start_time = time.time()
        self.cache = self._load_cache()
        self.modules = []
        self.stats = {
            "files_scanned": 0,
            "modules_found": 0,
            "tests_excluded": 0,
            "cached_hits": 0,
            "timeouts": 0,
            "errors": [],
        }

    def _load_cache(self) -> dict:
        """Load SHA256 cache for speed"""
        if Path(CACHE_FILE).exists():
            try:
                with open(CACHE_FILE) as f:
                    return json.load(f).get("modules", {})
            except:
                pass
        return {}

    def _save_cache(self):
        """Save cache for next run"""
        cache_data = {
            "version": "3.0",
            "timestamp": datetime.now().isoformat(),
            "modules": self.cache,
        }
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)

    def _should_exclude(self, path: Path) -> bool:
        """Check if file should be excluded using fnmatch"""
        path_str = os.path.normpath(str(path))

        # Patterns normalisés pour fnmatch
        NORMALIZED_PATTERNS = [
            "*/tests/*",
            "*/test/*",
            "*test_*.py",
            "*_test.py",
            "*/__pycache__/*",
            "*/.venv/*",
            "*/venv/*",
            "*/archive/*",
            "*/deprecated/*",
            "*/old/*",
            "*/backup/*",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.so",
        ]

        for pattern in NORMALIZED_PATTERNS:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    def _should_whitelist(self, name: str) -> bool:
        """Check if module is whitelisted despite size using fnmatch"""
        return any(fnmatch.fnmatch(name, pat) for pat in WHITELIST_SMALL_MODULES)

    def _extract_jeffrey_meta_ast(self, content: str) -> dict | None:
        """Extract __jeffrey_meta__ using AST only - NO EXEC (GPT requirement)"""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if hasattr(target, "id") and target.id == "__jeffrey_meta__":
                            try:
                                # Safe literal evaluation
                                meta = ast.literal_eval(node.value)
                                if isinstance(meta, dict):
                                    return meta
                            except:
                                pass
            return None
        except:
            return None

    def _calculate_health_score(self, path: Path, content: str, meta: dict) -> tuple[int, dict]:
        """Calculate Health Score 0-100 with all bonuses"""

        scores = {
            "size_complexity": 0,  # 35%
            "tests": 0,  # 5% (réduit de 30%)
            "activity": 0,  # 20%
            "integration": 0,  # 20%
            "bonuses": 0,  # 20% (augmenté)
        }

        lines = len(content.splitlines())

        # 1. Size & Complexity (35 points max)
        if lines >= 500:
            scores["size_complexity"] = 35
        elif lines >= 200:
            scores["size_complexity"] = 30
        elif lines >= 100:
            scores["size_complexity"] = 25
        elif lines >= 60:
            scores["size_complexity"] = 15
        else:
            scores["size_complexity"] = 10

        # Bonus for classes and functions
        try:
            tree = ast.parse(content)
            classes = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))
            functions = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))

            if classes >= 3:
                scores["size_complexity"] = min(35, scores["size_complexity"] + 5)
            if functions >= 10:
                scores["size_complexity"] = min(35, scores["size_complexity"] + 5)
        except:
            pass

        # 2. Tests (5 points max - RÉDUIT)
        test_file = path.parent / f"test_{path.stem}.py"
        if test_file.exists() and test_file.stat().st_size > 1000:
            scores["tests"] = 5
        elif test_file.exists():
            scores["tests"] = 3
        elif "test" in str(path.parent):
            scores["tests"] = 2

        # 3. Activity (20 points max - DYNAMIQUE même depuis cache)
        try:
            mtime = path.stat().st_mtime
            days_old = (time.time() - mtime) / 86400

            if days_old < 7:
                scores["activity"] = 20
            elif days_old < 30:
                scores["activity"] = 15
            elif days_old < 90:
                scores["activity"] = 10
            elif days_old < 365:
                scores["activity"] = 5
            else:
                scores["activity"] = 0
        except:
            scores["activity"] = 5

        # 4. Integration via AST (20 points max - PLUS FIABLE)
        try:
            tree = ast.parse(content)
            jeffrey_imports = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and "jeffrey" in node.module:
                        jeffrey_imports += 1
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if "jeffrey" in alias.name:
                            jeffrey_imports += 1

            if jeffrey_imports >= 5:
                scores["integration"] = 20
            elif jeffrey_imports >= 3:
                scores["integration"] = 15
            elif jeffrey_imports >= 1:
                scores["integration"] = 10
            else:
                scores["integration"] = 5
        except:
            scores["integration"] = 5

        # 5. Bonuses ÉTENDUS (20 points max)
        content_lower = content.lower()

        # Ethics/Alignment bonus (CRITIQUE pour AGI safe)
        ETHICS_PATTERNS = [
            "ethics",
            "bias",
            "alignment",
            "moral",
            "safe",
            "responsible",
            "fairness",
            "transparency",
        ]
        if any(word in content_lower for word in ETHICS_PATTERNS):
            scores["bonuses"] += 7  # Augmenté

        # GPU/Performance bonus
        GPU_PATTERNS = ["torch", "cuda", "gpu", "tensor", "accelerate", "tpu"]
        if any(word in content_lower for word in GPU_PATTERNS):
            scores["bonuses"] += 5

        # AI/ML bonus
        if any(word in content_lower for word in ["model", "train", "predict", "neural", "transformer"]):
            scores["bonuses"] += 3

        # Real-time bonus
        if "async" in content_lower or "await" in content_lower:
            scores["bonuses"] += 3

        # Consciousness/Meta bonus
        if any(word in content_lower for word in ["consciousness", "aware", "meta", "introspect"]):
            scores["bonuses"] += 5

        # Meta stability bonus
        if meta:
            if meta.get("stability") == "stable":
                scores["bonuses"] += 5
            elif meta.get("stability") == "beta":
                scores["bonuses"] += 2

            if meta.get("critical"):
                scores["bonuses"] += 5

        # Cap bonuses at 20
        scores["bonuses"] = min(20, scores["bonuses"])

        # Calculate total
        total = sum(scores.values())
        total = min(100, total)  # Cap at 100

        # Grade avec seuils ajustés
        if total >= 80:
            grade = "A"
        elif total >= 65:
            grade = "B"
        elif total >= 50:
            grade = "C"
        elif total >= 35:
            grade = "D"
        else:
            grade = "F"

        return total, {
            "score": total,
            "grade": grade,
            "breakdown": scores,
            "capabilities": self._detect_capabilities(content),
        }

    def _detect_capabilities(self, content: str) -> list[str]:
        """Detect module capabilities"""
        capabilities = []
        content_lower = content.lower()

        for capability, keywords in ADVANCED_PATTERNS.items():
            if any(keyword in content_lower for keyword in keywords):
                capabilities.append(capability)

        return capabilities

    def _measure_runtime_sandboxed(self, path: Path) -> dict:
        """Measure runtime in HARDENED isolated subprocess"""

        # Runner script durci avec toutes les protections
        runner_script = """
import importlib.util
import sys
import time
import json
import resource
import os

# BLOCK NETWORK - Monkeypatch socket
import socket
class _BlockedSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Network blocked in health sandbox")
socket.socket = _BlockedSocket

# Try to block requests if available
try:
    import requests
    requests.get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Network blocked"))
    requests.post = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Network blocked"))
except ImportError:
    pass

# Block subprocess spawning
import subprocess as _sp
import os as _os
class _NoSpawn:
    def __init__(self, *a, **k):
        raise RuntimeError("Subprocess blocked in health sandbox")

_sp.Popen = _NoSpawn  # type: ignore
_sp.call = _NoSpawn   # type: ignore
_sp.run = _NoSpawn    # type: ignore
_os.system = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("os.system blocked"))

# Insert project root if provided
proj_root = os.environ.get("PYTHONPATH")
if proj_root and proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Set resource limits with macOS fallback
try:
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    # RLIMIT_AS problematic on macOS, use RLIMIT_DATA instead
    if sys.platform == 'darwin':
        resource.setrlimit(resource.RLIMIT_DATA, (256*1024*1024, 256*1024*1024))
    else:
        resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
except Exception:
    pass  # Continue without limits rather than fail

path = sys.argv[1]
samples = []
budget_ms = int(os.environ.get("JEFFREY_RUNTIME_BUDGET_MS", "200"))
t0 = time.perf_counter()

try:
    spec = importlib.util.spec_from_file_location("test_module", path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find health_check
        health_check = None

        if hasattr(module, 'health_check'):
            health_check = module.health_check
        else:
            for attr_name in dir(module):
                if not attr_name.startswith('_'):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, 'health_check'):
                        try:
                            instance = attr()
                            health_check = instance.health_check
                            break
                        except:
                            pass

        if health_check:
            # Measure with budget limit
            for i in range(10):
                if (time.perf_counter() - t0) * 1000 > budget_ms:
                    break

                start = time.perf_counter()
                health_check()
                elapsed_ms = (time.perf_counter() - start) * 1000
                samples.append(elapsed_ms)

            if len(samples) >= 3:  # Need at least 3 samples
                samples.sort()
                result = {
                    'measured': True,
                    'p50_latency_ms': samples[len(samples)//2],
                    'p95_latency_ms': samples[int(len(samples)*0.95)] if len(samples) > 1 else samples[-1],
                    'samples': len(samples)
                }
            else:
                result = {'measured': False, 'reason': 'insufficient_samples'}
        else:
            result = {'measured': False, 'reason': 'no_health_check'}
    else:
        result = {'measured': False, 'reason': 'import_failed'}

except Exception as e:
    result = {'measured': False, 'reason': str(e)[:100]}

print(json.dumps(result))
"""

        try:
            # Calculate PYTHONPATH for project root
            from pathlib import Path

            proj_root = None
            try:
                p = Path(path)
                # Go up until "src", then take its parent as project root
                for parent in p.parents:
                    if parent.name == "src":
                        proj_root = str(parent.parent)
                        break
            except Exception:
                pass

            # Create safe environment
            env = dict(os.environ)
            env.update(
                {
                    "JEFFREY_SAFE_MODE": "1",
                    "JEFFREY_NO_NETWORK": "1",
                    "JEFFREY_RUNTIME_BUDGET_MS": str(TIMEOUT_PER_FILE_MS),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONIOENCODING": "utf-8",
                }
            )
            if proj_root:
                env["PYTHONPATH"] = proj_root

            # Support relaxed mode for easier measurements
            relaxed = os.environ.get("JEFFREY_MEASURE_RELAXED") == "1"

            cmd = [
                sys.executable,
                "-I",  # Isolated mode
                "-E",  # Ignore environment
            ]

            # Mode relaxé : pas de -S (permet site-packages)
            if not relaxed:
                cmd.append("-S")  # No site packages

            cmd += ["-c", runner_script, str(path)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1.5,
                env=env,  # Increased timeout
            )

            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout.strip())
            else:
                stderr_msg = result.stderr[:100] if result.stderr else "unknown"
                return {"measured": False, "reason": f"subprocess_failed: {stderr_msg}"}

        except subprocess.TimeoutExpired:
            return {"measured": False, "reason": "timeout"}
        except Exception as e:
            return {"measured": False, "reason": str(e)[:50]}

    def _process_file(self, path: Path) -> dict | None:
        """Process a single Python file with all checks"""

        # Check budget
        if time.time() - self.start_time > BUDGET_SECONDS:
            self.stats["timeouts"] += 1
            return None

        # Check exclusions
        if self._should_exclude(path):
            if "test" in str(path).lower():
                self.stats["tests_excluded"] += 1
            return None

        try:
            # Get file hash
            with open(path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Check cache
            cache_key = str(path)
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if cached.get("sha256") == file_hash:
                    self.stats["cached_hits"] += 1
                    return cached.get("module_data")

            # Read content with robust encoding
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = len(content.splitlines())
            name = path.stem

            # Check size with whitelist
            if lines < MIN_LINES_DEFAULT and not self._should_whitelist(name):
                return None

            # Extract metadata
            meta = self._extract_jeffrey_meta_ast(content)

            # Calculate health score
            score, health_data = self._calculate_health_score(path, content, meta)

            # Measure runtime
            runtime_data = self._measure_runtime_sandboxed(path)

            # Build module data
            module_data = {
                "path": str(path),
                "name": name,
                "lines": lines,
                "size_bytes": path.stat().st_size,
                "health_score": score,
                "grade": health_data["grade"],
                "health_breakdown": health_data["breakdown"],
                "capabilities": health_data["capabilities"],
                "has_jeffrey_meta": meta is not None,
                "jeffrey_meta": meta,
                "metrics": runtime_data,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }

            # Update cache
            self.cache[cache_key] = {"sha256": file_hash, "module_data": module_data}

            return module_data

        except Exception as e:
            self.stats["errors"].append(f"{path}: {str(e)[:50]}")
            return None

    def run_inventory(self, root_path: str = "src/jeffrey") -> dict:
        """Run the complete inventory"""

        print("🧠 ULTIMATE MODULE INVENTORY v3.0")
        print("=" * 60)
        print(f"⏱️  Budget: {BUDGET_SECONDS}s | Cache: {Path(CACHE_FILE).exists()}")
        print(f"📁 Scanning: {root_path}")
        print("-" * 60)

        # Find all Python files
        root = Path(root_path)
        py_files = list(root.rglob("*.py"))

        print(f"📊 Found {len(py_files)} Python files")

        # Process files with progress
        processed = 0
        for i, path in enumerate(py_files):
            # Check budget
            if time.time() - self.start_time > BUDGET_SECONDS:
                print(f"\n⚠️  Budget exceeded at file {i}/{len(py_files)}")
                break

            # Process file
            module_data = self._process_file(path)
            if module_data:
                self.modules.append(module_data)
                self.stats["modules_found"] += 1

            processed += 1
            self.stats["files_scanned"] += 1

            # Progress indicator
            if processed % 50 == 0:
                elapsed = time.time() - self.start_time
                print(f"  Processed {processed}/{len(py_files)} files ({elapsed:.1f}s)...")

        # Save cache
        self._save_cache()

        # Sort modules by health score
        self.modules.sort(key=lambda x: x["health_score"], reverse=True)

        # Generate brain mapping
        brain_regions = self._map_brain_regions()

        # Select Bundle 1 candidates
        bundle1 = self._select_bundle1()

        # Calculate statistics
        execution_time = time.time() - self.start_time

        # Build final report (avec TOUS les modules, pas seulement top 100)
        report = {
            "version": "3.0",
            "timestamp": datetime.now().isoformat(),
            "execution_time_s": round(execution_time, 2),
            "stats": self.stats,
            "summary": {
                "total_modules": len(self.modules),
                "grade_a": sum(1 for m in self.modules if m["grade"] == "A"),
                "grade_b": sum(1 for m in self.modules if m["grade"] == "B"),
                "grade_c": sum(1 for m in self.modules if m["grade"] == "C"),
                "grade_d": sum(1 for m in self.modules if m["grade"] == "D"),
                "grade_f": sum(1 for m in self.modules if m["grade"] == "F"),
                "measured_runtime": sum(1 for m in self.modules if m["metrics"].get("measured")),
            },
            "brain_regions": brain_regions,
            "bundle1_recommendations": bundle1,
            "modules": self.modules,  # TOUS les modules sauvegardés
        }

        # Save report
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        self._print_summary(report)

        return report

    def _map_brain_regions(self) -> dict:
        """Map modules to brain regions"""

        regions = {
            "tronc_cerebral": {
                "emoji": "⚡",
                "description": "Infrastructure & Core Systems",
                "keywords": ["bus", "kernel", "core", "pipeline", "infrastructure"],
                "modules": [],
            },
            "cortex_occipital": {
                "emoji": "👁️",
                "description": "Perception & Input Processing",
                "keywords": ["input", "parse", "vision", "perception", "sensor"],
                "modules": [],
            },
            "cortex_temporal": {
                "emoji": "🧩",
                "description": "Memory & Knowledge",
                "keywords": ["memory", "storage", "recall", "episodic", "semantic"],
                "modules": [],
            },
            "systeme_limbique": {
                "emoji": "💭",
                "description": "Emotions & Feelings",
                "keywords": ["emotion", "feel", "mood", "sentiment", "empathy"],
                "modules": [],
            },
            "broca_wernicke": {
                "emoji": "🗣️",
                "description": "Language & Communication",
                "keywords": ["language", "response", "generate", "speak", "llm"],
                "modules": [],
            },
            "cortex_frontal": {
                "emoji": "🎭",
                "description": "Executive & Decision Making",
                "keywords": ["executive", "decision", "orchestrat", "control", "verdict"],
                "modules": [],
            },
            "hippocampe": {
                "emoji": "🔄",
                "description": "Learning & Adaptation",
                "keywords": ["learn", "adapt", "train", "curiosity", "discover"],
                "modules": [],
            },
            "corps_calleux": {
                "emoji": "🌟",
                "description": "Integration & Bridges",
                "keywords": ["bridge", "integrate", "connect", "glue", "adapter"],
                "modules": [],
            },
        }

        # Map modules to regions
        for module in self.modules:
            name_lower = module["name"].lower()
            capabilities = module.get("capabilities", [])

            # Find best matching region
            best_region = None
            best_confidence = 0

            for region_name, region_data in regions.items():
                confidence = 0

                # Check keywords in name
                for keyword in region_data["keywords"]:
                    if keyword in name_lower:
                        confidence += 30

                # Check capabilities
                for capability in capabilities:
                    if capability in ["memory"] and region_name == "cortex_temporal":
                        confidence += 40
                    elif capability in ["emotion"] and region_name == "systeme_limbique":
                        confidence += 40
                    elif capability in ["consciousness"] and region_name == "cortex_frontal":
                        confidence += 40
                    elif capability in ["decision"] and region_name == "cortex_frontal":
                        confidence += 30
                    elif capability in ["ai_ml"] and region_name == "hippocampe":
                        confidence += 30

                # Check jeffrey_meta
                if module.get("jeffrey_meta"):
                    meta_regions = module["jeffrey_meta"].get("brain_regions", [])
                    for meta_region in meta_regions:
                        if meta_region.lower() in region_name.lower():
                            confidence += 50

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_region = region_name

            # Add module to best region
            if best_region and best_confidence >= 30:
                regions[best_region]["modules"].append(
                    {
                        "name": module["name"],
                        "score": module["health_score"],
                        "confidence": min(100, best_confidence),
                    }
                )

        # Sort modules in each region by score
        for region in regions.values():
            region["modules"].sort(key=lambda x: x["score"], reverse=True)
            region["count"] = len(region["modules"])

        return regions

    def _select_bundle1(self) -> dict:
        """Select Bundle 1 modules with strict criteria"""

        # Temporary exclusion for modules with heavy dependencies
        EXCLUDED_FROM_BUNDLE1 = {
            "autonomous_language_system",  # timeout issues
            "jeffrey_chat_integration",  # may have other issues
            "agi_orchestrator",  # too many missing dependencies
        }

        bundle1_modules = []
        regions_covered = set()
        total_p95 = 0
        measured_count = 0

        # Get brain regions first
        brain_regions = self._map_brain_regions()

        # First pass: Select at least 1 module from each region to ensure coverage
        for region_name, region_data in brain_regions.items():
            if region_data["modules"] and len(bundle1_modules) < 10:
                # Find first stable module in this region
                for module_ref in region_data["modules"][:3]:  # Check top 3
                    # Skip excluded modules
                    if module_ref["name"] in EXCLUDED_FROM_BUNDLE1:
                        continue
                    module = next((m for m in self.modules if m["name"] == module_ref["name"]), None)
                    if not module:
                        continue

                    # Check stability (beta allowed, experimental not)
                    meta = module.get("jeffrey_meta") or {}
                    stability = (meta.get("stability") or "beta").lower()
                    if stability in ["experimental", "deprecated"]:
                        continue

                    # Add this module to ensure region coverage
                    bundle1_modules.append(module)
                    regions_covered.add(region_name)

                    # Count measured metrics
                    if module.get("metrics", {}).get("measured"):
                        measured_count += 1
                        total_p95 += module["metrics"].get("p95_latency_ms", 0)

                    break  # One module per region in first pass

        # Second pass: Add best remaining modules up to 10 total
        if len(bundle1_modules) < 10:
            # Sort all modules by score
            remaining = [m for m in self.modules if m not in bundle1_modules]
            remaining.sort(key=lambda x: x["health_score"], reverse=True)

            for module in remaining[:10]:
                if len(bundle1_modules) >= 10:
                    break

                # Skip excluded modules
                if module["name"] in EXCLUDED_FROM_BUNDLE1:
                    continue

                # Check stability
                meta = module.get("jeffrey_meta") or {}
                stability = (meta.get("stability") or "beta").lower()
                if stability not in ["experimental", "deprecated"]:
                    bundle1_modules.append(module)

                    # Update metrics
                    if module.get("metrics", {}).get("measured"):
                        measured_count += 1
                        total_p95 += module["metrics"].get("p95_latency_ms", 0)

        # Determine status with STRICT criteria (6/8 regions required)
        if len(regions_covered) >= 6 and measured_count >= 3 and total_p95 <= 250:
            status = "ready"
        elif len(regions_covered) >= 6 and measured_count >= 3:
            status = "needs_performance_tuning"  # P95 too high
        elif len(regions_covered) >= 6:
            status = "needs_measurement"  # Not enough measured
        else:
            status = "needs_more_coverage"  # Not enough regions

        return {
            "modules": bundle1_modules[:10],
            "regions_covered": f"{len(regions_covered)}/8",
            "measured_modules": measured_count,
            "total_p95_budget_ms": round(total_p95, 2),
            "status": status,
        }

    def _print_summary(self, report: dict):
        """Print beautiful summary"""

        print("\n" + "=" * 60)
        print("📊 INVENTORY COMPLETE!")
        print("=" * 60)

        print(f"\n⏱️  Execution: {report['execution_time_s']}s")
        print(f"📁 Files scanned: {report['stats']['files_scanned']}")
        print(f"✅ Modules found: {report['summary']['total_modules']}")
        print(f"🚫 Tests excluded: {report['stats']['tests_excluded']}")
        print(f"💾 Cache hits: {report['stats']['cached_hits']}")

        print("\n📈 Module Grades:")
        print(f"  A (85-100): {report['summary']['grade_a']} modules")
        print(f"  B (70-84):  {report['summary']['grade_b']} modules")
        print(f"  C (55-69):  {report['summary']['grade_c']} modules")
        print(f"  D (40-54):  {report['summary']['grade_d']} modules")
        print(f"  F (0-39):   {report['summary']['grade_f']} modules")

        print("\n🧠 Brain Regions Coverage:")
        for name, data in report["brain_regions"].items():
            print(f"  {data['emoji']} {name}: {data['count']} modules")

        print("\n🚀 Bundle 1 Recommendations:")
        bundle1 = report["bundle1_recommendations"]
        print(f"  Modules: {len(bundle1['modules'])}")
        print(f"  Regions: {bundle1['regions_covered']}")
        print(f"  P95 Budget: {bundle1['total_p95_budget_ms']}ms")
        print(f"  Status: {bundle1['status']}")

        print(f"\n✅ Report saved to: {OUTPUT_FILE}")
        print(f"💾 Cache saved to: {CACHE_FILE}")


if __name__ == "__main__":
    inventory = UltimateInventory()
    inventory.run_inventory()
