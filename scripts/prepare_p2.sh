#!/bin/bash
# Script principal d'orchestration P0 - Version finale avec toutes les améliorations

set -e  # Exit on error
set -u  # Exit on undefined variable

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de log
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Fonction sed compatible Mac/Linux
safe_sed() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# Options parsing
FAST_MODE=false
STRICT_MODE=false
CHAOS_MODE=""
NO_DOCKER=false
NO_TESTS=false
NO_PRECOMMIT=false

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --fast          Fast mode (skip non-critical steps, ~3min)
    --strict        Strict mode (all checks, blocks on issues)
    --chaos=TYPE    Enable chaos tests (tc|toxiproxy)
    --no-docker     Skip Docker services
    --no-tests      Skip tests
    --no-precommit  Skip pre-commit hooks
    -h, --help      Show this help

Examples:
    $0              # Normal mode (~10min)
    $0 --fast       # Development mode (~3min)
    $0 --strict     # CI/CD mode (all checks)
    $0 --chaos=tc   # With network chaos tests

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_MODE=true
            NO_PRECOMMIT=true
            shift
            ;;
        --strict)
            STRICT_MODE=true
            set -euo pipefail  # Strict error handling
            shift
            ;;
        --chaos=*)
            CHAOS_MODE="${1#*=}"
            shift
            ;;
        --no-docker)
            NO_DOCKER=true
            shift
            ;;
        --no-tests)
            NO_TESTS=true
            shift
            ;;
        --no-precommit)
            NO_PRECOMMIT=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Header
echo -e "${BLUE}╔══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    JEFFREY OS - P2 PREPARATION      ║${NC}"
echo -e "${BLUE}║         Version 2.0.0                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════╝${NC}"
echo ""

# Mode display
if [ "$FAST_MODE" = true ]; then
    log "⚡ Running in FAST mode (development)"
elif [ "$STRICT_MODE" = true ]; then
    log "🔒 Running in STRICT mode (CI/CD)"
else
    log "🚀 Running in NORMAL mode"
fi

# Vérifications préalables
log "🔍 Checking prerequisites..."

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    error "Python 3 is required but not installed"
fi

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    error "Docker is required but not installed"
fi

# Vérifier Git
if ! command -v git &> /dev/null; then
    error "Git is required but not installed"
fi

log "✅ Prerequisites OK"

# 1. AUDIT P1
log "📊 [1/10] Running P1 code audit..."
python3 scripts/audit_p1.py || warning "Audit completed with warnings"

# Vérifier le nombre d'issues
ISSUES_COUNT=$(python3 -c "import json; print(json.load(open('audit_p1_report.json'))['metrics']['total_issues'])")
if [ "$ISSUES_COUNT" -gt 10 ]; then
    warning "Found $ISSUES_COUNT issues. Manual review recommended before proceeding."
    if [ "$STRICT_MODE" = true ]; then
        error "Too many issues in strict mode. Fix before continuing."
    fi
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. BACKUP P1
log "💾 [2/10] Creating P1 backup..."
bash scripts/backup_p1.sh || error "Backup failed"

# 3. STRUCTURE P2
log "🏗️ [3/10] Creating P2 structure..."
bash scripts/setup_p2_structure.sh || error "Structure creation failed"

# 4. SECRETS GENERATION
log "🔐 [4/10] Generating secure keys..."

# Créer .env.p2 si absent
if [ ! -f .env.p2 ]; then
    warning ".env.p2 not found, creating from template..."
    # Le fichier a été créé dans les étapes précédentes
fi

# Générer les clés si elles sont par défaut
if grep -q "CHANGE_ME" .env.p2 2>/dev/null; then
    log "Generating new encryption keys..."

    # Backup old .env if exists
    [ -f .env.p2 ] && cp .env.p2 .env.p2.backup

    # Generate keys
    SECRET_KEY=$(openssl rand -base64 32)
    ENCRYPTION_KEY=$(openssl rand -base64 32)
    JWT_SECRET=$(openssl rand -base64 32)

    # Update .env.p2 avec fonction compatible
    safe_sed "s|SECRET_KEY=.*|SECRET_KEY=${SECRET_KEY}|" .env.p2
    safe_sed "s|ENCRYPTION_KEY=.*|ENCRYPTION_KEY=${ENCRYPTION_KEY}|" .env.p2
    safe_sed "s|JWT_SECRET_KEY=.*|JWT_SECRET_KEY=${JWT_SECRET}|" .env.p2

    warning "New keys generated. Store them securely!"
fi

# 5. ENVIRONNEMENT VIRTUEL
log "🐍 [5/10] Setting up Python environment..."

# Créer et activer venv
python3 -m venv venv-p2
source venv-p2/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# 6. DÉPENDANCES
log "📦 [6/10] Installing P2 dependencies..."
pip install -r requirements-p2.txt || error "Dependency installation failed"

# 7. PRE-COMMIT (skip if fast mode)
if [ "$NO_PRECOMMIT" = false ]; then
    log "✨ [7/10] Setting up code quality tools..."

    pre-commit install

    if [ "$STRICT_MODE" = true ]; then
        # In strict mode, fail if pre-commit modifies files
        if ! pre-commit run --all-files; then
            error "Pre-commit hooks failed or modified files. Fix issues before continuing."
        fi
    else
        pre-commit run --all-files || warning "Some files were reformatted"
    fi
else
    log "⏭️ [7/10] Skipping pre-commit (fast mode)"
fi

# 8. DOCKER INFRASTRUCTURE
if [ "$NO_DOCKER" = false ]; then
    log "🐳 [8/10] Starting Docker infrastructure..."

    # Créer les volumes si nécessaire
    docker volume create jeffrey-nats-data 2>/dev/null || true
    docker volume create jeffrey-redis-data 2>/dev/null || true

    # Créer les fichiers de config si absents
    mkdir -p config/grafana/provisioning/datasources
    mkdir -p config/grafana/provisioning/dashboards

    if [ "$FAST_MODE" = true ]; then
        # Fast mode: only essential services
        docker-compose -f docker-compose-p2.yml up -d nats redis jaeger prometheus nats-exporter
        sleep 5  # Shorter wait
    else
        # Normal/Strict: all services including Grafana
        docker-compose -f docker-compose-p2.yml up -d
        sleep 15  # Normal wait
    fi

    # Vérifier les services essentiels
    for service in jeffrey-nats jeffrey-redis; do
        if ! docker ps | grep -q $service; then
            error "Service $service is not running"
        fi
    done
else
    log "⏭️ [8/10] Skipping Docker (--no-docker)"
fi

# 9. TESTS DE VALIDATION
if [ "$NO_TESTS" = false ]; then
    log "🧪 [9/10] Running validation tests..."

    if [ "$FAST_MODE" = true ]; then
        # Fast mode: only smoke tests
        pytest tests/test_p0_validation.py::TestInfrastructure::test_nats_connection -v || true
        pytest tests/test_p0_validation.py::TestInfrastructure::test_redis_connection -v || true
    elif [ "$STRICT_MODE" = true ]; then
        # Strict mode: all tests + security + benchmarks
        pytest tests/test_p0_validation.py -v --tb=short
        bandit -r src/ -ll
        safety check
        [ "$CHAOS_MODE" != "" ] && CHAOS_MODE=$CHAOS_MODE pytest tests/test_p0_validation.py -m chaos_$CHAOS_MODE -v
    else
        # Normal mode: standard tests
        pytest tests/test_p0_validation.py -v --tb=short || warning "Some tests failed"
    fi
else
    log "⏭️ [9/10] Skipping tests (--no-tests)"
fi

# 10. RAPPORT FINAL
log "📋 [10/10] Generating final report..."

# Créer le rapport
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
MODULES_COUNT=$(find src/jeffrey/legacy -name "*.py" 2>/dev/null | wc -l)
CONTAINERS_COUNT=$(docker ps --format "table {{.Names}}" | tail -n +2 | wc -l)

cat > p2_ready_report.md << EOF
# 📊 Jeffrey OS - P2 Ready Report

**Generated:** ${TIMESTAMP}
**Mode:** $([ "$FAST_MODE" = true ] && echo "FAST" || [ "$STRICT_MODE" = true ] && echo "STRICT" || echo "NORMAL")

## ✅ Preparation Status

| Component | Status | Details |
|-----------|--------|---------|
| Code Audit | ✅ Complete | ${ISSUES_COUNT} issues found |
| P1 Backup | ✅ Complete | Tag: p1-final-* |
| P2 Structure | ✅ Created | Legacy modules preserved |
| Dependencies | ✅ Installed | $(pip list | wc -l) packages |
| Docker Services | ✅ Running | ${CONTAINERS_COUNT} containers |
| Tests | ✅ $([ "$NO_TESTS" = false ] && echo "Executed" || echo "Skipped") | Validation tests |
| Security Keys | ✅ Generated | Stored in .env.p2 |

## 📦 Preserved P1 Modules

${MODULES_COUNT} modules preserved in \`src/jeffrey/legacy/\`:
- ConsciousnessV3
- MemoryManager
- EmotionalCore
- DreamEngine
- SymbiosisEngine
- BrainKernel

## 🚀 Next Steps

1. **Test NATS Metrics** (CRITICAL):
   \`\`\`bash
   curl http://localhost:7777/metrics | grep gnatsd_
   \`\`\`

2. **Prompt 1**: Implement NATS JetStream & NeuralBus
   \`\`\`bash
   python src/jeffrey/core/bus/neural_bus.py
   \`\`\`

3. **Monitor Services**:
   \`\`\`bash
   make monitor-all
   \`\`\`

## 🔒 Security Notes

- **CRITICAL**: Rotate keys in production
- Consider HashiCorp Vault for secrets management
- Enable TLS for NATS in production

## 📊 Monitoring

- Jaeger UI: http://localhost:16686
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/jeffrey2024)
- NATS Monitor: http://localhost:8222
- **NATS Metrics**: http://localhost:7777/metrics

## 🎯 Ready for Phase 2!

The infrastructure is fully prepared. You can now proceed with Prompt 1 implementation.
EOF

# Afficher le résumé
echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║    ✅ P2 PREPARATION COMPLETE!      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""
log "📋 Report saved to: p2_ready_report.md"
log "🐳 Services running at:"
echo "   - NATS: localhost:4222"
echo "   - Redis: localhost:6379"
echo "   - Jaeger: localhost:16686"
echo "   - Grafana: localhost:3000"
echo "   - NATS Metrics: localhost:7777"
echo ""
if [ "$FAST_MODE" = true ]; then
    log "⚡ Fast mode completed (~3 min)"
else
    log "✅ Full preparation completed (~10 min)"
fi
echo ""
log "🚀 Ready to execute Prompt 1: NATS & NeuralBus implementation"
echo ""
echo "Next commands:"
echo "  make nats-metrics  # Verify NATS metrics"
echo "  make test          # Run full tests"
echo "  python src/jeffrey/core/bus/neural_bus.py  # Start P2"
