# 🧠 Jeffrey OS - Guide LLM (Ollama/vLLM)

## 🚀 Quick Start pour Mac

```bash
# 1. Installer Ollama et télécharger les modèles
./scripts/install_ollama.sh

# 2. Démarrer le serveur LLM adaptatif
./scripts/start_llm_server.sh

# 3. Tester que tout fonctionne
python tests/test_llm_adaptive.py

# 4. Lancer Jeffrey OS
python main.py
```

## 📦 Architecture Dual-Mode

Jeffrey OS détecte automatiquement votre environnement :

| Environnement | Backend | Modèle | Performance |
|---------------|---------|--------|-------------|
| **macOS** | Ollama | Mistral 7B | ⚡ Optimisé Apple Silicon |
| **Linux+GPU** | vLLM | Apertus 8B | 🎮 Max performance GPU |
| **Linux CPU** | Ollama | Mistral 7B | 💻 Mode CPU efficace |

## 🔧 Configuration

Le fichier `.env.p2` contient :
```env
LLM_MODEL_OLLAMA=mistral:7b-instruct  # Pour Mac/CPU
LLM_MODEL_VLLM=swiss-ai/Apertus-8B    # Pour GPU
```

## 🧪 Tests

```bash
# Test adaptatif (détecte l'environnement)
python tests/test_llm_adaptive.py

# Tests complets avec pytest
pytest tests/test_llm_adaptive.py -v

# Test de charge
python tests/test_apertus_load.py
```

## 📊 Monitoring

```bash
# Dashboard temps réel
python scripts/monitor_apertus.py

# Métriques Prometheus
curl http://localhost:9010/metrics
```

## 🌉 API OpenAI Compatible

Les deux backends exposent la même API :
- Endpoint : `http://localhost:9010/v1`
- Modèles : `/v1/models`
- Chat : `/v1/chat/completions`
- Métriques : `/metrics`

## 🐙 Modèles Ollama disponibles

```bash
# Recommandé (installé par défaut)
ollama pull mistral:7b-instruct

# Alternatives
ollama pull llama3.2:3b      # Plus léger
ollama pull phi3:medium       # Ultra-rapide
ollama pull codellama:13b     # Pour le code
```

## ☁️ Options Cloud pour Apertus

Si vous voulez utiliser le vrai modèle Apertus :

### RunPod ($0.39/h)
```bash
# Sur RunPod
vllm serve swiss-ai/Apertus-8B-Instruct-2509 --port 8000

# Sur votre Mac
LLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1
```

### Google Colab (Gratuit)
```python
!pip install vllm
!vllm serve swiss-ai/Apertus-8B-Instruct-2509 --share
```

## 🔍 Troubleshooting

### Ollama ne démarre pas sur Mac
```bash
# Installer via Homebrew
brew install ollama

# Ou réinstaller
curl -fsSL https://ollama.com/install.sh | sh
```

### Port 9010 occupé
```bash
# Changer le port dans .env.p2
LLM_PORT=9011

# Relancer
./scripts/start_llm_server.sh
```

### Modèle trop lent
```bash
# Utiliser un modèle plus léger
ollama pull phi3:mini
export LLM_MODEL_OLLAMA=phi3:mini
```

## 📈 Performance Tips

1. **Sur Mac M1/M2** : Ollama utilise automatiquement Metal
2. **RAM minimum** : 8GB pour Mistral 7B
3. **Cache** : Les modèles sont cachés dans `~/.ollama/models`
4. **Concurrence** : Le proxy supporte plusieurs requêtes simultanées

## 🎯 Prochaines étapes

1. ✅ Installation complète
2. ✅ Tests passent
3. 🚀 Jeffrey OS avec LLM local
4. 📊 Monitor les performances
5. 🔧 Ajuster les paramètres selon vos besoins
