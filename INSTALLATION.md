# 🚀 Jeffrey OS - Guide d'Installation

## Prérequis

- **Python 3.11+** (obligatoire)
- **Ollama** (recommandé pour les capacités LLM)
- **Redis** (optionnel, pour cache distribué)
- **macOS/Linux** (recommandé pour uvloop)

## Installation Rapide

### 1. Cloner le repository

```bash
git clone https://github.com/votre-repo/jeffrey-os.git
cd jeffrey-os
```

### 2. Installation automatique

```bash
# Rendre le script exécutable
chmod +x install_jeffrey.sh

# Lancer l'installation
./install_jeffrey.sh
```

Le script va:
- ✅ Vérifier la version Python
- ✅ Créer un environnement virtuel
- ✅ Installer toutes les dépendances
- ✅ Créer les dossiers nécessaires
- ✅ Générer un fichier .env de configuration

### 3. Installation manuelle (alternative)

```bash
# Créer l'environnement virtuel
python3 -m venv .venv_prod

# Activer l'environnement
source .venv_prod/bin/activate  # Linux/Mac
# ou
.venv_prod\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Pour les développeurs
pip install -r requirements-dev.txt
```

## Configuration

### 1. Ollama (pour LLM)

```bash
# Installer Ollama (Mac)
brew install ollama

# Démarrer Ollama
ollama serve

# Télécharger le modèle Mistral
ollama pull mistral:7b-instruct
```

### 2. Redis (optionnel)

```bash
# Mac
brew install redis
brew services start redis

# Linux
sudo apt install redis-server
sudo systemctl start redis
```

### 3. Variables d'environnement

Éditer le fichier `.env`:

```env
# Configuration Jeffrey
JEFFREY_ENV=production
JEFFREY_LOG_LEVEL=INFO

# Ollama
OLLAMA_HOST=http://localhost:9010
OLLAMA_MODEL=mistral:7b-instruct

# Redis (optionnel)
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Vérification de l'installation

### 1. Vérifier les dépendances

```bash
python check_deps.py
```

### 2. Test basique

```bash
python tests/test_bridge_v3_basic.py
```

### 3. Test conversation

```bash
python tests/test_bridge_v3_conversation.py
```

### 4. Interface graphique

```bash
python test_kivy_integration.py
```

## Structure des dépendances

### Core (obligatoires)
- `httpx` - Client HTTP async
- `networkx` - Opérations sur graphes
- `kivy` - Interface graphique
- `msgpack` - Sérialisation binaire
- `pydantic` - Validation de données

### Neural Architecture
- `redis` - Cache et mémoire partagée
- `nats-py` - Message bus neural

### Machine Learning
- `numpy` - Calcul numérique
- `scikit-learn` - ML classique
- `langdetect` - Détection de langue

### Performance
- `uvloop` - Optimisation asyncio (Linux/Mac)
- `prometheus-client` - Métriques

## Troubleshooting

### Python version incorrecte
```bash
# Installer Python 3.11 avec pyenv
pyenv install 3.11.6
pyenv local 3.11.6
```

### Import errors
```bash
# Réinstaller toutes les dépendances
pip install --force-reinstall -r requirements.txt
```

### Ollama ne répond pas
```bash
# Vérifier qu'Ollama est lancé
curl http://localhost:9010/api/tags

# Redémarrer Ollama
killall ollama
ollama serve
```

### Interface Kivy ne se lance pas
```bash
# Installer les dépendances système (Mac)
brew install sdl2 sdl2_image sdl2_ttf sdl2_mixer

# Linux
sudo apt-get install python3-kivy
```

## Support

Pour toute question ou problème:
1. Vérifier les logs dans `logs/`
2. Consulter la documentation dans `docs/`
3. Ouvrir une issue sur GitHub

## Licence

MIT License - Voir LICENSE.md
