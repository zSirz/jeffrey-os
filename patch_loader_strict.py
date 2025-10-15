#!/usr/bin/env python3
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

# Backup
bak = path + ".backup_strict"
if not os.path.exists(bak):
    open(bak, "w").write(src)

# Assurer les imports nécessaires
for imp in ("import os", "import importlib", "import inspect", "import asyncio"):
    if imp not in src:
        src = imp + "\n" + src

# Méthode de chargement STRICTE qui crée de VRAIES instances - avec améliorations GPT
loader_method = '''
    async def _load_module_dynamic(self, module_info: dict):
        """Charge et INSTANCIE un module réel. Refuse les stubs."""
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
        for candidate in [import_path, f"jeffrey.{import_path.split('jeffrey.')[-1]}" if "jeffrey" in import_path else None]:
            if not candidate:
                continue
            try:
                mod = importlib.import_module(candidate)
                break
            except:
                pass

        if not mod:
            print(f"  ❌ Import failed: {import_path}")
            return

        # TROUVER UNE VRAIE INSTANCE
        instance = None

        # 1. Variables exportées
        for attr in [module_name, f"{module_name}_instance", "engine", "module", "emotion_engine", "conscience_engine"]:
            if hasattr(mod, attr):
                candidate = getattr(mod, attr)
                if inspect.isclass(candidate):
                    instance = candidate()
                else:
                    instance = candidate
                break

        # 2. Classe CamelCase
        if not instance and module_name:
            camel_name = "".join(w.capitalize() for w in module_name.split("_"))
            if hasattr(mod, camel_name):
                cls = getattr(mod, camel_name)
                if inspect.isclass(cls):
                    instance = cls()

        # 3. Fonction initialize()
        if not instance and hasattr(mod, "initialize"):
            init_func = getattr(mod, "initialize")
            if inspect.iscoroutinefunction(init_func):
                instance = await init_func()
            else:
                instance = init_func()

        # 4. Chercher une classe Engine/Module
        if not instance:
            for name in dir(mod):
                obj = getattr(mod, name)
                if inspect.isclass(obj) and ("Engine" in name or "Module" in name) and "Stub" not in name:
                    try:
                        instance = obj()
                        break
                    except:
                        continue

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
            if inspect.iscoroutinefunction(init):
                await init()
            else:
                init()

        # VÉRIFIER qu'il y a une méthode process
        has_process = any(hasattr(instance, m) for m in ["process", "analyze", "analyze_emotion"])
        if not has_process:
            print(f"  ❌ No process method for {region}")
            return

        # STOCKER L'INSTANCE RÉELLE
        self.regions[region] = instance
        print(f"  ✅ Loaded REAL module for {region}: {instance.__class__.__name__}")
'''

# Injecter ou remplacer la méthode
if "_load_module_dynamic" in src:
    # Remplacer l'existante
    pattern = r"async def _load_module_dynamic\(self.*?\):.*?(?=\n    async def|\n    def|\nclass|\Z)"
    new_src = re.sub(pattern, loader_method.strip(), src, flags=re.DOTALL)

    # Amélioration GPT: fallback si rien remplacé
    if new_src == src:
        cls = re.search(r"class\s+ConsciousnessLoop\b.*?:", src)
        assert cls, "ConsciousnessLoop introuvable"
        insert_at = cls.end()
        # indenter la méthode à 4 espaces
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
print("✅ Loader STRICT injecté")
