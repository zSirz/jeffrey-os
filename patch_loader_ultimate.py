#!/usr/bin/env python3
"""
Patch ultime du loader pour gérer toutes les signatures de modules
"""

import os
import re
import textwrap

CANDIDATES = [
    "src/jeffrey/core/consciousness_loop.py",
    "src/jeffrey/consciousness/consciousness_loop.py",
]

path = next((p for p in CANDIDATES if os.path.exists(p)), None)
if not path:
    print("❌ consciousness_loop.py introuvable")
    exit(1)

src = open(path, encoding="utf-8", errors="ignore").read()

# Backup ultime
bak = path + ".backup_ultimate"
if not os.path.exists(bak):
    open(bak, "w").write(src)

# Assurer les imports nécessaires
for imp in ("import os", "import importlib", "import inspect", "import asyncio"):
    if imp not in src:
        src = imp + "\n" + src

# Méthode de chargement ULTIME - gère toutes les signatures possibles
loader_method = '''
    async def _load_module_dynamic(self, module_info: dict):
        """Charge et INSTANCIE un module réel. Refuse les stubs. Gère toutes les signatures."""
        import importlib, inspect, os, sys

        # Amélioration GPT: assurer le sys.path
        root = os.getcwd()
        src_dir = os.path.join(root, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        module_name = module_info.get("name", "")
        module_path = module_info.get("path", "")
        region = module_info.get("brain_region", "")

        if not module_path or not os.path.exists(module_path):
            print(f"  ⚠️ Path not found: {module_path}")
            return

        # Convertir path en import
        import_path = module_path.replace("/", ".").replace(".py", "")
        for prefix in ("src.", "."):
            if import_path.startswith(prefix):
                import_path = import_path[len(prefix):]

        # Importer le module
        mod = None
        for candidate in [
            import_path,
            f"jeffrey.{import_path.split('jeffrey.')[-1]}" if "jeffrey" in import_path else None,
            f"src.{import_path}" if not import_path.startswith("jeffrey") else None
        ]:
            if not candidate:
                continue
            try:
                mod = importlib.import_module(candidate)
                break
            except Exception as e:
                print(f"    Import attempt failed for {candidate}: {e}")
                pass

        if not mod:
            print(f"  ❌ Import failed: {import_path}")
            return

        # TROUVER UNE VRAIE INSTANCE - Stratégies multiples
        instance = None

        # Stratégie 1: Variables exportées directement
        for attr in [
            module_name, f"{module_name}_instance", "engine", "module",
            "emotion_engine", "conscience_engine", "pipeline", "parser",
            "generator", "orchestrator", "memory", "system", "manager",
            "explainer", "provider_manager", "agi_orchestrator"
        ]:
            if hasattr(mod, attr):
                candidate = getattr(mod, attr)
                if inspect.isclass(candidate):
                    instance = self._try_instantiate_class(candidate, region)
                elif callable(candidate):
                    try:
                        instance = candidate()
                    except:
                        instance = candidate
                else:
                    instance = candidate
                if instance:
                    break

        # Stratégie 2: Classe CamelCase
        if not instance and module_name:
            camel_name = "".join(w.capitalize() for w in module_name.split("_"))
            if hasattr(mod, camel_name):
                cls = getattr(mod, camel_name)
                if inspect.isclass(cls):
                    instance = self._try_instantiate_class(cls, region)

        # Stratégie 3: Fonction initialize()
        if not instance and hasattr(mod, "initialize"):
            init_func = getattr(mod, "initialize")
            try:
                if inspect.iscoroutinefunction(init_func):
                    instance = await init_func()
                else:
                    instance = init_func()
            except Exception as e:
                print(f"    Initialize function failed: {e}")

        # Stratégie 4: Chercher automatiquement des classes appropriées
        if not instance:
            target_keywords = {
                "executive": ["Executive", "Planner", "Decision", "Command", "Explainer"],
                "motor": ["Motor", "Generator", "Response", "Output", "Provider", "Manager"],
                "language": ["Language", "NLP", "Linguistic", "Text", "Provider", "Manager"],
                "integration": ["Integration", "Orchestrator", "Bridge", "Pipeline", "Synthesis", "AGI"],
                "emotion": ["Emotion", "Feeling", "Mood", "Core"],
                "conscience": ["Conscience", "Synthesis", "Cognitive", "Awareness"],
                "perception": ["Perception", "Input", "Parser", "Sensor"],
                "memory": ["Memory", "Storage", "Cache", "Working"]
            }

            keywords = target_keywords.get(region, ["Engine", "Module", "System"])

            for name in dir(mod):
                obj = getattr(mod, name)
                if inspect.isclass(obj) and any(kw in name for kw in keywords) and "Stub" not in name:
                    instance = self._try_instantiate_class(obj, region)
                    if instance:
                        break

        if not instance:
            print(f"  ❌ No instance found in {import_path}")
            return

        # REFUSER LES STUBS
        if "Stub" in instance.__class__.__name__:
            print(f"  ❌ Stub rejected for {region}")
            return

        # Initialiser si nécessaire
        if hasattr(instance, "initialize"):
            init = instance.initialize
            try:
                if inspect.iscoroutinefunction(init):
                    await init()
                else:
                    init()
            except Exception as e:
                print(f"    Initialization warning: {e}")

        # VÉRIFIER qu'il y a une méthode process/analyze/parse/generate
        required_methods = ["process", "analyze", "parse", "generate", "execute", "synthesize", "orchestrate", "run"]
        has_method = any(hasattr(instance, m) for m in required_methods)
        if not has_method:
            print(f"  ⚠️ No standard method found for {region}, but keeping instance: {instance.__class__.__name__}")

        # STOCKER L'INSTANCE RÉELLE
        self.regions[region] = instance
        print(f"  ✅ Loaded REAL module for {region}: {instance.__class__.__name__}")

    def _try_instantiate_class(self, cls, region):
        """Essaie d'instancier une classe avec différentes signatures"""
        try:
            # Signature 1: Pas d'arguments
            return cls()
        except TypeError:
            pass

        # Signature 2: bus=None
        try:
            return cls(bus=None)
        except TypeError:
            pass

        # Signature 3: config={}
        try:
            return cls(config={})
        except TypeError:
            pass

        # Signature 4: bus + config
        try:
            return cls(bus=None, config={})
        except TypeError:
            pass

        # Signature 5: Avec NeuralBus
        try:
            from jeffrey.neuralbus.core import NeuralBus
            bus = NeuralBus(namespace="consciousness")
            return cls(bus)
        except Exception:
            pass

        # Signature 6: Avec des arguments par défaut common
        try:
            return cls(logger=None)
        except TypeError:
            pass

        # Signature 7: Provider-style avec model
        try:
            return cls(model="gpt-4")
        except TypeError:
            pass

        # Signature 8: Orchestrator-style
        try:
            return cls(modules=[])
        except TypeError:
            pass

        print(f"    Failed to instantiate {cls.__name__} for {region}")
        return None
'''

# Remplacer la méthode existante
if "_load_module_dynamic" in src:
    # Pattern pour trouver la méthode complète
    pattern = r"async def _load_module_dynamic\(self.*?\):.*?(?=\n    async def|\n    def|\nclass|\Z)"
    new_src = re.sub(pattern, loader_method.strip(), src, flags=re.DOTALL)

    # Fallback si rien remplacé
    if new_src == src:
        cls = re.search(r"class\s+ConsciousnessLoop\b.*?:", src)
        if cls:
            insert_at = cls.end()
            indented = "\n" + textwrap.indent(loader_method.strip(), "    ") + "\n"
            new_src = src[:insert_at] + indented + src[insert_at:]
    src = new_src
else:
    # Ajouter après la classe
    class_match = re.search(r"class\s+ConsciousnessLoop.*?:", src)
    if class_match:
        insert_pos = class_match.end()
        indented = "\n" + textwrap.indent(loader_method.strip(), "    ") + "\n"
        src = src[:insert_pos] + indented + src[insert_pos:]

open(path, "w").write(src)
print("✅ Loader ULTIMATE injecté (gère toutes les signatures)")
