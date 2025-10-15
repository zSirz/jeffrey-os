# 🐤 JEFFREY OS - Canary Deployment Guide

## Objectif
Déployer v2.4.2 en production de manière progressive et sécurisée.

## Prérequis
- ✅ v2.4.2 validée (F1≥0.543, smoke≥62.5%)
- ✅ Tie-break anger/frustration actif (1.7% règle application rate)
- ✅ PII protection dans logs active
- ✅ Monitoring JSON fonctionnel avec 482 predictions loggées

## Plan de Déploiement

### Phase 1: Canary 10% (Jour 1)
```
Ancien système: 90% trafic
Nouveau v2.4.2: 10% trafic
```

**Métriques à surveiller (24h) :**
- Route distribution: linear_head ≥90%, regex ≤1%
- Accuracy proxy (si feedback users disponibles)
- Latence P50/P95 (baseline: P90 = 0.726s confidence)
- Taux low_confidence ≤25% (baseline: 24.7%)
- Rule applications: tiebreak_anger_keyword ~1-2%
- proto_dimension_mismatch = 0

**Critères de succès :**
- Aucune augmentation des erreurs
- Latence stable (±10% vs baseline)
- Aucun feedback négatif utilisateur majeur
- Tie-break rule applications dans plage attendue (1-5%)

**Rollback automatique si :**
- Regex fallback >2% pendant >15 min
- Erreurs critiques augmentation >50%
- Latence P95 >200ms (ou votre seuil)
- Dimension mismatch errors >0

### Phase 2: Canary 20% (Jour 2)
Si Phase 1 OK → monter à 20%

**Durée observation :** 24h
**Mêmes métriques que Phase 1**

### Phase 3: Progressive 50% → 100% (Jours 3-4)
- Jour 3 matin : 50%
- Jour 3 soir : 75%
- Jour 4 : 100% si tout OK

## Commandes

### Démarrer canary 10%
```bash
# Si load balancer (Nginx, etc.)
# Configurer routing : 90% → old, 10% → v2.4.2

# Exemple Nginx:
upstream jeffrey_backend {
    server old-version.com weight=9;
    server new-v2.4.2.com weight=1;
}
```

### Monitoring en temps réel
```bash
# Analyser les logs toutes les 15 min
watch -n 900 'PYTHONPATH=src python scripts/analyze_monitoring_logs.py'

# Ou dashboard (à venir)
```

### Rollback instantané
```bash
# Si problème détecté
# Remettre 100% sur ancienne version
# Via load balancer ou feature flag
```

## Alertes à configurer

**Critiques (rollback immédiat) :**
- Regex fallback >2% pendant >15 min
- proto_dimension_mismatch >0
- Spike erreurs 5xx
- Rule application rate >10% (signe de problème de classification)

**Warnings (investigation) :**
- Low_confidence >25%
- Latence P95 augmentation >20%
- Tie-break applications <0.5% (peut indiquer un problème)

## Métriques de Baseline v2.4.2

**Performance actuelle (482 predictions analysées) :**
- **Accuracy smoke test:** 62.5% (target ≥60% ✅)
- **Route distribution:** linear_head 100.0% ✅
- **Regex fallback:** 0.0% ✅
- **Low confidence rate:** 24.7% (légèrement élevé mais acceptable)
- **Mean confidence:** 0.514
- **Tie-break applications:** 1.7% (8/482 predictions)
- **PII protection:** Active ✅

**Émotions les plus fréquentes :**
1. joy: 21.8%
2. fear: 19.9%
3. frustration: 14.5%
4. anger: 13.9%

## Après 100% Déployé

1. Observer 7 jours
2. Collecter feedback users
3. Analyser métriques hebdomadaires
4. Préparer v2.5 (annotations + fine-tuning)

## Tests de Validation Pre-Deployment

**Avant chaque étape de canary, exécuter :**

```bash
# 1. Test tie-break (doit passer ≥80%)
PYTHONPATH=src python scripts/test_anger_tiebreak.py

# 2. Test PII masking (doit passer 100%)
PYTHONPATH=src python scripts/test_pii_masking.py

# 3. Smoke test global (doit être ≥60%)
export PYTHONPATH=src && python scripts/smoke_test_fr_en.py

# 4. Analyse monitoring (doit être "MOSTLY READY" ou mieux)
PYTHONPATH=src python scripts/analyze_monitoring_logs.py
```

**Tous les tests doivent passer avant d'augmenter le % de canary.**

## Configuration Tie-Break

**Variable d'environnement :**
```bash
# Ajuster le seuil de tie-break si nécessaire (default: 0.05)
export JEFFREY_TIEBREAK_DELTA=0.05  # Plus bas = plus strict
```

**Monitoring tie-break :**
- Applications normales : 1-3% du trafic
- Si >5% : peut indiquer des scores trop serrés (modèle incertain)
- Si <0.5% : peut indiquer que le seuil est trop strict

## Exemple Dashboard KPIs

**Production Health :**
- ✅ Linear head usage: >90%
- ✅ Regex fallback: <1%
- ⚠️ Low confidence: <15% (actuel: 24.7%)
- ✅ PII protection: Active
- ✅ Tie-break rate: 1-5%

**Performance Evolution :**
- Accuracy: baseline 62.5%
- Mean confidence: baseline 0.514
- P90 confidence: baseline 0.726

---

**Generated:** 2025-10-13
**Version:** v2.4.2-prod-final
**Status:** Canary-Ready ✅
