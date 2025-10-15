# 📦 Jeffrey OS - Requirements Documentation

## Fichiers de dépendances

### 1. `requirements.txt` - Production
Contient toutes les dépendances nécessaires pour faire tourner Jeffrey OS en production.

**Installation:**
```bash
pip install -r requirements.txt
```

### 2. `requirements-minimal.txt` - Installation minimale
Version allégée avec seulement les dépendances critiques.

**Installation:**
```bash
pip install -r requirements-minimal.txt
```

### 3. `requirements-dev.txt` - Développement
Inclut toutes les dépendances de production + outils de développement (linting, testing, etc.)

**Installation:**
```bash
pip install -r requirements-dev.txt
```

### 4. `requirements_full.txt` - État actuel complet
Généré avec `pip freeze`, contient l'état exact de l'environnement avec toutes les versions.

**Génération:**
```bash
pip freeze > requirements_full.txt
```

## Scripts d'installation

### `install_jeffrey.sh`
Script automatique qui:
- Vérifie Python 3.11+
- Crée un environnement virtuel
- Installe toutes les dépendances
- Configure l'environnement
- Vérifie l'installation

**Utilisation:**
```bash
chmod +x install_jeffrey.sh
./install_jeffrey.sh
```

### `check_deps.py`
Vérifie que toutes les dépendances sont installées correctement.

**Utilisation:**
```bash
python check_deps.py
```

### `test_installation.py`
Test rapide pour vérifier que Jeffrey OS peut démarrer.

**Utilisation:**
```bash
python test_installation.py
```

## Dépendances critiques

| Package | Version | Description | Obligatoire |
|---------|---------|-------------|-------------|
| Python | 3.11+ | Langage de base | ✅ |
| httpx | 0.27+ | Client HTTP async | ✅ |
| networkx | 3.4+ | Graphes et réseaux | ✅ |
| kivy | 2.3+ | Interface graphique | ✅ |
| msgpack | 1.0+ | Sérialisation binaire | ✅ |
| pydantic | 2.9+ | Validation de données | ✅ |
| numpy | 2.2+ | Calcul numérique | ✅ |
| redis | 5.0+ | Cache mémoire | ⚠️ Recommandé |
| openai | 1.50+ | Interface LLM/Ollama | ⚠️ Recommandé |
| uvloop | 0.21+ | Performance async | ⚠️ Linux/Mac |

## Services externes

### Ollama (recommandé)
Pour les capacités de conversation avec LLM.

**Installation:**
```bash
# Mac
brew install ollama
ollama pull mistral:7b-instruct

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral:7b-instruct
```

### Redis (optionnel)
Pour le cache distribué et la persistance.

**Installation:**
```bash
# Mac
brew install redis
brew services start redis

# Linux
sudo apt install redis-server
sudo systemctl start redis
```

## Environnements testés

- ✅ **macOS 14+ (Apple Silicon)** - Pleinement supporté
- ✅ **Ubuntu 22.04+** - Pleinement supporté
- ⚠️ **Windows 11** - Supporté sans uvloop
- ✅ **Python 3.11.6** - Version recommandée
- ✅ **Python 3.12+** - Compatible

## Résolution de problèmes

### Erreur: Module not found
```bash
# Réinstaller avec force
pip install --force-reinstall -r requirements.txt
```

### Erreur: Version conflicts
```bash
# Créer un nouvel environnement propre
rm -rf .venv_prod
python3 -m venv .venv_prod
source .venv_prod/bin/activate
pip install -r requirements.txt
```

### Kivy ne démarre pas
```bash
# Mac - Installer les dépendances système
brew install sdl2 sdl2_image sdl2_ttf sdl2_mixer

# Linux
sudo apt-get install python3-kivy
```

## Commande rapide pour tout installer

```bash
# One-liner pour installation complète
curl -sSL https://raw.githubusercontent.com/votre-repo/jeffrey-os/main/install_jeffrey.sh | bash
```

## Support

En cas de problème:
1. Exécuter `python test_installation.py` pour diagnostic
2. Vérifier les logs dans `logs/`
3. Consulter INSTALLATION.md pour guide détaillé

---

*Dernière mise à jour: 2025-10-01*
