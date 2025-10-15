# 🔐 Jeffrey OS - Forteresse de Sécurité V2

## ✅ Installation Réussie !

Jeffrey OS est maintenant prêt à démarrer avec tous les composants de sécurité fonctionnels.

## 🚀 Démarrage Rapide

```bash
# 1. Installation des dépendances
pip install -r requirements.txt

# 2. Configuration initiale
make setup

# 3. Tests de fumée
make smoke

# 4. Démarrage de l'API
make start

# 5. Vérifier le statut
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## 📊 État Actuel

### ✅ Composants Fonctionnels
- **Control Plane API** - FastAPI avec monitoring Prometheus
- **Cache Guardian** - Rate limiting intelligent (mode basique en DEV)
- **Anti-Replay System** - Protection contre les rejeux (fallback mémoire sans Redis)
- **mTLS Bridge** - Authentification mutuelle TLS
- **FFI Bridge** - Mode stub pour intégration Rust future
- **Guardians Hub** - Point d'entrée pour les gardiens
- **EdDSA Signer** - Signatures cryptographiques Ed25519
- **PII Redactor** - Protection des données sensibles

### ⚠️ Warnings Normaux en DEV
- Anti-Replay en mode mémoire (Redis non disponible)
- FFI Bridge en mode stub (lib Rust non compilée)
- Guardians Symphony/Ethical non chargés (modules optionnels)

## 🔧 Configuration

### Variables d'Environnement
```bash
# .env
SECURITY_MODE=dev      # dev ou prod
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

### Structure des Fichiers
```
jeffrey_os/
├── src/jeffrey/core/
│   ├── control/         # API Control Plane
│   ├── guardians/       # Gardiens éthiques
│   ├── security/        # Modules de sécurité
│   │   ├── signer/      # Algorithmes de signature
│   │   └── ...
│   └── neuralbus/       # Event bus et FFI
├── keys/                # Certificats et clés
├── scripts/             # Scripts d'installation
└── tests/               # Tests unitaires
```

## 🛠️ Commandes Makefile

| Commande | Description |
|----------|-------------|
| `make help` | Affiche l'aide |
| `make setup` | Installation complète |
| `make start` | Démarre l'API |
| `make stop` | Arrête l'API |
| `make smoke` | Tests de fumée (20s) |
| `make test` | Tests complets |
| `make audit` | Audit de sécurité |
| `make clean` | Nettoyage |

## 📈 Endpoints API

### Health & Monitoring
- `GET /` - Page d'accueil avec liens
- `GET /health` - Health check basique
- `GET /ready` - Readiness check détaillé
- `GET /status` - Statut complet du système
- `GET /metrics` - Métriques Prometheus

### Test de Sécurité
- `POST /test/security` - Test de la chaîne de sécurité

## 🔍 Tests de Validation

```bash
# Test rapide de tous les composants
python scripts/smoke_boot.py

# Résultat attendu:
# ✅ Réussis: 15+
# ⚠️ Warnings: 3-4 (normal en DEV)
# ❌ Échecs: 0
```

## 🚨 Prochaines Étapes

### Phase 1 - Consolidation (En cours)
- [x] Structure de base
- [x] Composants de sécurité
- [x] API de monitoring
- [ ] Tests unitaires complets
- [ ] Documentation API

### Phase 2 - Production Ready
- [ ] Redis pour Anti-Replay distribué
- [ ] ML pour Cache Guardian
- [ ] Compilation modules Rust
- [ ] Certificats de production
- [ ] Guardian Symphony & Ethical

### Phase 3 - Post-Quantum
- [ ] liboqs-python pour signature PQ
- [ ] Dilithium/Falcon
- [ ] Migration des signatures

## 📝 Notes Importantes

1. **Mode DEV vs PROD**: Le système est configuré en mode DEV par défaut. Les fallbacks sont activés pour permettre le développement sans dépendances externes.

2. **Redis Optionnel**: En DEV, l'Anti-Replay utilise la mémoire. Pour activer Redis:
   ```bash
   docker run -d -p 6379:6379 redis:alpine
   ```

3. **Guardians Manquants**: Les modules Guardian Symphony et Ethical Guardian ne sont pas inclus dans cette version de base. Ils peuvent être ajoutés depuis les archives iCloud si disponibles.

4. **Sécurité**: Les certificats générés sont pour le développement uniquement. Ne jamais utiliser en production !

## 🐛 Troubleshooting

### Port 8000 occupé
```bash
lsof -i :8000
kill -9 [PID]
```

### Redis non disponible
C'est normal en mode DEV. Pour l'activer:
```bash
brew install redis
redis-server
```

### Import errors
```bash
pip install -r requirements.txt
python -m pip install --upgrade pip
```

## 📚 Documentation

- [Architecture Technique](docs/ARCHITECTURE.md) - À créer
- [Guide de Sécurité](docs/SECURITY.md) - À créer
- [API Reference](docs/API.md) - À créer

## 📄 Licence

Jeffrey OS - Propriétaire
© 2024 - Tous droits réservés

---

✅ **Jeffrey OS V2 est maintenant opérationnel !**

L'API est accessible sur http://localhost:8000 après `make start`.
