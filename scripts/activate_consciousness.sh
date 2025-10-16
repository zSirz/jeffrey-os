#!/bin/bash
set -e

echo "🚀 JEFFREY OS CONSCIOUSNESS ACTIVATION"
echo "====================================="

PHASE=$1

if [ -z "$PHASE" ]; then
    echo "Usage: $0 [j1|j2|j3]"
    echo ""
    echo "  j1 - Jour 1: Dry-run (WRITE=false)"
    echo "  j2 - Jour 2: Write minimal (MAX_BONDS=5)"
    echo "  j3 - Jour 3: Scale up (MAX_BONDS=20)"
    exit 1
fi

case $PHASE in
    j1)
        echo "📊 JOUR 1: Dry-run activation"
        # GPT correction #6: sed -i compatible macOS
        sed -i.bak 's/ENABLE_CONSCIOUSNESS=false/ENABLE_CONSCIOUSNESS=true/' .env
        echo "✅ ENABLE_CONSCIOUSNESS=true"
        echo "⏳ Observer metrics 24h avant J2"
        ;;
    j2)
        echo "✍️ JOUR 2: Write minimal"
        sed -i.bak 's/ENABLE_CONSCIOUSNESS_WRITE=false/ENABLE_CONSCIOUSNESS_WRITE=true/' .env
        sed -i.bak 's/CONSCIOUSNESS_MAX_BONDS_UPDATES=5/CONSCIOUSNESS_MAX_BONDS_UPDATES=10/' .env
        echo "✅ ENABLE_CONSCIOUSNESS_WRITE=true"
        echo "✅ MAX_BONDS_UPDATES=10"
        echo "⏳ Observer metrics 24h avant J3"
        ;;
    j3)
        echo "🚀 JOUR 3: Scale up"
        sed -i.bak 's/CONSCIOUSNESS_MAX_BONDS_UPDATES=10/CONSCIOUSNESS_MAX_BONDS_UPDATES=20/' .env
        sed -i.bak 's/CONSCIOUSNESS_MAX_NEW_MEMORIES=1/CONSCIOUSNESS_MAX_NEW_MEMORIES=3/' .env
        echo "✅ MAX_BONDS_UPDATES=20"
        echo "✅ MAX_NEW_MEMORIES=3"
        echo "🎯 Full activation complete!"
        ;;
    *)
        echo "❌ Invalid phase: $PHASE"
        exit 1
        ;;
esac

echo ""
echo "📋 Next steps:"
echo "  1. Restart API: docker-compose restart jeffrey-api"
echo "  2. Check Grafana dashboard"
echo "  3. Monitor for 24h before next phase"