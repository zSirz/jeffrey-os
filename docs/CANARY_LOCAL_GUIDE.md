# Guide de Test Canary Local - Jeffrey OS v2.4.2

## Vue d'ensemble

Ce guide explique comment lancer un test canary local avec génération de trafic réaliste pour valider Jeffrey OS avant déploiement production.

## Architecture du test

```
Terminal 1 (watch)       Terminal 2 (checks)      Terminal 3 (traffic)
     |                         |                         |
     |                         |                         |
  watch_canary.sh          monitor_canary_slo.py    generate_canary_traffic.py
     |                         |                         |
     v                         v                         v
  Checks auto              Checks manuels           Génère trafic
  toutes 5 min             à H+1, H+4, etc.         continu pondéré
     |                         |                         |
     +-------------------------+-------------------------+
                               |
                               v
                    logs/predictions/*.jsonl
```

## Phase 1 : Smoke Run (10 min)

### 1.1. Lancement

**Terminal 1 : Watch automatique**
```bash
cd ~/Desktop/jeffrey_OS
source .venv_jeffrey/bin/activate
./scripts/watch_canary.sh
# Laisser tourner, ne pas fermer
```

**Terminal 3 : Génération de trafic (smoke)**
```bash
cd ~/Desktop/jeffrey_OS
source .venv_jeffrey/bin/activate
chmod +x scripts/launch_canary_local.sh
./scripts/launch_canary_local.sh
# Lance 200 events (~10 min)
```

### 1.2. Vérification après 10-15 min

**Terminal 2 : Checks manuels**
```bash
cd ~/Desktop/jeffrey_OS
source .venv_jeffrey/bin/activate
export PYTHONPATH=src

# Synthèse SLO
python scripts/monitor_canary_slo.py

# Analyse détaillée
python scripts/analyze_monitoring_logs.py | head -50
```

### 1.3. Critères Go/No-Go

✅ **GO pour 24h** si tous ces critères sont OK pendant 15-30 min :
- Linear head ≥ 90%
- Regex fallback ≤ 1%
- P95 latency ≤ 200ms
- Tie-break ≤ 5% (normal avec borderline)
- Low confidence ≤ 25% (normal avec borderline)

⚠️ **NO-GO / Ajuster** si :
- Regex > 1% → Réduire proportion "borderline" dans le générateur
- P95 > 200ms → Baisser TRAFFIC_RATE_PER_MIN ou BURST_SIZE
- Low confidence > 30% → Réduire proportion "borderline"

## Phase 2 : Run 24h (après smoke OK)

### 2.1. Ajuster la configuration

```bash
# Terminal 3 (si smoke run terminé)
cd ~/Desktop/jeffrey_OS
source .venv_jeffrey/bin/activate

# Configuration 24h
export PYTHONPATH=src
export JEFFREY_TIEBREAK_ENABLED=true
export JEFFREY_TIEBREAK_DELTA=0.05
export JEFFREY_TIEBREAK_EXTENDED_DELTA=0.15

# Traffic parameters
export TRAFFIC_RATE_PER_MIN=20
export BURST_EVERY=120
export BURST_SIZE=5
export JITTER_MS=300
export PII_MAX_PER_MIN=2
unset MAX_EVENTS  # Infinite pour 24h

# Lancer
python scripts/generate_canary_traffic.py
```

### 2.2. Points de contrôle (H+1, H+4, H+8, H+24)

À chaque point de contrôle, dans Terminal 2 :

```bash
# Synthèse SLO
PYTHONPATH=src python scripts/monitor_canary_slo.py

# Analyse complète
python scripts/analyze_monitoring_logs.py | head -50

# Histogramme latences
tail -1000 logs/predictions/predictions_$(date +%Y-%m-%d).jsonl \
  | jq '.latency_ms' | sort -n | tail -50

# Taux tie-break et low confidence
tail -1000 logs/predictions/predictions_$(date +%Y-%m-%d).jsonl \
  | jq -r '[.prediction.low_confidence, .prediction.rule_applied] | @tsv' \
  | awk '{lc+=$1=="true"; tb+=$2!=""} END{printf("low_conf=%.1f%%  tie_break=%.1f%%\n",100*lc/NR,100*tb/NR)}'
```

### 2.3. Journal de bord (à remplir manuellement)

```
📅 Date : __/__/2025
⏰ H0 (__:__) : Smoke run OK, lancement 24h
   Baseline: Linear __% | Regex __% | P95 __ms | Tie-break __% | Low-conf __%

✅ H+1  (__:__) : Linear __% | Regex __% | P95 __ms | Notes : _______________
✅ H+4  (__:__) : Linear __% | Regex __% | P95 __ms | Notes : _______________
✅ H+8  (__:__) : Linear __% | Regex __% | P95 __ms | Notes : _______________
✅ H+24 (__:__) : Linear __% | Regex __% | P95 __ms | Notes : _______________

🚨 Incidents : _________________________________________________________________
```

## Commandes utiles

### Vérifier PII redaction

```bash
grep "text_preview" logs/predictions/predictions_$(date +%Y-%m-%d).jsonl | head -10
# Doit afficher [EMAIL], [URL], [PHONE], [IPV6] - JAMAIS d'emails/numéros bruts
```

### Distribution des émotions

```bash
tail -1000 logs/predictions/predictions_$(date +%Y-%m-%d).jsonl \
  | jq -r '.prediction.primary' \
  | sort | uniq -c | sort -rn
```

### Trafic par heure

```bash
tail -1000 logs/predictions/predictions_$(date +%Y-%m-%d).jsonl \
  | jq -r '.timestamp' \
  | cut -d'T' -f2 | cut -d':' -f1 \
  | sort | uniq -c
```

### Vérifier les bursts

```bash
tail -500 logs/predictions/predictions_$(date +%Y-%m-%d).jsonl \
  | jq -r '.timestamp' \
  | awk -F'[T:]' '{print $2":"$3}' \
  | uniq -c \
  | awk '$1 > 5 {print "🔥 Burst detected:", $0}'
```

## Ajustements possibles

### Augmenter le volume

```bash
export TRAFFIC_RATE_PER_MIN=40  # 40/min au lieu de 20
export BURST_SIZE=10            # Bursts plus gros
```

### Stresser le tie-break

Modifier `scripts/generate_canary_traffic.py` :
```python
WEIGHTS = {
    "normal": 0.60,      # Réduire normal
    "borderline": 0.30,  # Augmenter borderline
    "pii": 0.05,
    "edge": 0.05
}
```

### Stresser PII

```bash
export PII_MAX_PER_MIN=5  # Plus de PII par minute
```

### Mode haute charge (stress test)

```bash
export TRAFFIC_RATE_PER_MIN=60
export BURST_EVERY=60
export BURST_SIZE=15
# Surveiller P95 latency de près
```

## Critères de rollback immédiat

🚨 **ROLLBACK si dépassé > 15 minutes** :
- Linear head < 90%
- Regex fallback > 2%
- P95 latency > 250ms
- Dimension mismatch > 0

**Procédure de rollback :**
```bash
# 1. Stopper le traffic (Ctrl+C dans Terminal 3)
# 2. Noter l'incident
echo "ROLLBACK à $(date) - Raison: [LINEAR/REGEX/LATENCY]" >> logs/rollback_history.log
# 3. Revenir à la version précédente
git checkout [BRANCHE_PRECEDENTE]
# 4. Redémarrer Jeffrey OS avec l'ancienne version
```

## Métriques attendues (normales)

Avec le trafic pondéré (70% normal, 20% borderline, 5% PII, 5% edge) :

```
✅ Linear head:      98-100%
✅ Regex fallback:   0-1%
⚠️ Low confidence:   15-25% (normal avec borderline)
⚠️ Tie-break rate:   3-8% (normal avec anger/frustration mix)
✅ P95 latency:      100-150ms
✅ P99 latency:      180-220ms
```

## Après 24h stable

1. **Analyser les logs complets**
```bash
python scripts/analyze_monitoring_logs.py > canary_24h_report.txt
```

2. **Exporter les métriques**
```bash
tail -10000 logs/predictions/predictions_$(date +%Y-%m-%d).jsonl \
  | jq -r '[.timestamp, .route, .prediction.primary, .prediction.confidence, .latency_ms] | @csv' \
  > canary_24h_metrics.csv
```

3. **Tag la version**
```bash
git tag -a v2.4.2-canary-ok -m "Local canary 24h passed - ready for production"
git push origin --tags
```

4. **Préparer le déploiement réel**
- Setup load balancer / feature flag
- Configurer alertes Slack/Email
- Préparer dashboard Grafana
- Documenter la procédure de rollback production

## FAQ

**Q : Le terminal 3 s'arrête après 200 events ?**
R : C'est normal pour le smoke run. Pour 24h, lance avec `unset MAX_EVENTS`.

**Q : Low confidence à 25%, c'est normal ?**
R : Oui, avec 20% de borderline cases. Si > 30%, réduire la proportion borderline.

**Q : Aucun PII dans les logs ?**
R : Vérifie que PII_MAX_PER_MIN n'est pas à 0. Les PII sont limités à 5% du trafic.

**Q : P95 latency augmente progressivement ?**
R : Possible memory leak ou cache overflow. Surveiller RAM et relancer si nécessaire.

**Q : Comment savoir si tie-break fonctionne ?**
R : Cherche `rule_applied` dans les logs :
```bash
grep "rule_applied" logs/predictions/predictions_$(date +%Y-%m-%d).jsonl | head -10
```

## Support

En cas de problème, vérifier :
1. Logs d'erreurs : `tail -100 logs/errors.log`
2. Processus actifs : `ps aux | grep jeffrey`
3. Utilisation RAM/CPU : `top` ou `htop`
4. Espace disque : `df -h`
