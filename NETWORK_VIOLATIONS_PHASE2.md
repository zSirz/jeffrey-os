# 🌐 Audit Architecture - Phase 2 : Violations du Pattern Bridge
**Date:** 2025-09-23
**Périmètre:** src/jeffrey/ (excluant src/jeffrey/bridge/)

## 📊 Résumé des Violations

### 🔴 Fichiers utilisant des bibliothèques réseau (hors bridge/)

#### 1. **requests** (2 fichiers)
- `src/jeffrey/core/emotions/profiles/emotional_profile_manager.py`
  - **Ligne 140:** `requests.get()` - API OpenWeatherMap
- `src/jeffrey/services/providers/provider_manager.py`
  - **Ligne 546:** `requests.post()` - Appels API génériques

#### 2. **aiohttp** (4 fichiers uniques + backups)
- `src/jeffrey/services/voice/engine/elevenlabs_v3_engine.py`
  - Import aiohttp (ligne 20)
  - Multiple usages : ClientSession, get/post requests
  - API ElevenLabs pour la synthèse vocale

- `src/jeffrey/services/voice/engine/elevenlabs_client.py`
  - Import aiohttp (ligne 13)
  - ClientSession, ClientTimeout, ClientError
  - Client API ElevenLabs réutilisable

- `src/jeffrey/interfaces/api/websocket/websocket_handler.py`
  - Import from aiohttp (ligne 21): `web, WSMsgType`
  - **Note:** Utilisation légitime pour le serveur WebSocket, pas pour des appels HTTP externes

#### 3. **httpx** (0 fichier)
- Aucune utilisation détectée

#### 4. **urllib** (0 fichier)
- Aucune utilisation détectée

## 📋 Plan d'Action

### 🎯 Adaptateurs Bridge à créer

1. **`src/jeffrey/bridge/adapters/weather_adapter.py`**
   - Pour : `emotional_profile_manager.py`
   - API : OpenWeatherMap
   - Méthodes : `get_weather(location: str)`

2. **`src/jeffrey/bridge/adapters/provider_api_adapter.py`**
   - Pour : `provider_manager.py`
   - API : Endpoints génériques de providers
   - Méthodes : `call_provider_api(endpoint, headers, data)`

3. **`src/jeffrey/bridge/adapters/elevenlabs_adapter.py`**
   - Pour : `elevenlabs_v3_engine.py` et `elevenlabs_client.py`
   - API : ElevenLabs TTS
   - Méthodes :
     - `get_voices()`
     - `text_to_speech(text, voice_id, settings)`
     - `get_user_info()`
     - `check_api_availability()`

### ✅ Exceptions (pas de changement nécessaire)

- **`websocket_handler.py`** : Utilise aiohttp.web pour servir des WebSockets, pas pour des appels HTTP sortants. C'est un usage légitime du framework web.

## 📁 Structure proposée

```
src/jeffrey/bridge/
├── __init__.py
├── adapters/
│   ├── __init__.py
│   ├── weather_adapter.py
│   ├── provider_api_adapter.py
│   └── elevenlabs_adapter.py
└── base/
    ├── __init__.py
    └── http_adapter.py  # Classe de base pour tous les adaptateurs
```

## 🔄 Prochaines étapes

1. Créer la structure de répertoires bridge/
2. Implémenter la classe de base HTTPAdapter
3. Créer chaque adaptateur spécifique
4. Modifier les fichiers originaux pour utiliser les adaptateurs
5. Supprimer les imports directs de requests/aiohttp
