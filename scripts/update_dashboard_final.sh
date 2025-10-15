#!/bin/bash
# Script de mise à jour FINAL pour le Dashboard Jeffrey OS
set -euo pipefail

echo "🚀 Mise à jour du Dashboard Jeffrey OS..."

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Backup du dashboard existant si présent
if [ -f "dashboard_pro_ultime.html" ]; then
    echo -e "${YELLOW}📦 Backup du dashboard existant...${NC}"
    cp dashboard_pro_ultime.html "dashboard_backup_$(date +%Y%m%d_%H%M%S).html"
fi

# 2. Créer le dossier static si nécessaire
if [ ! -d "static" ]; then
    echo -e "${GREEN}📁 Création du dossier static...${NC}"
    mkdir -p static
fi

# 3. Télécharger les libs si absentes
echo "📚 Vérification des librairies locales..."
if [ ! -f "static/chart.min.js" ]; then
    echo "  ⬇️  Téléchargement Chart.js..."
    curl -sL https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js -o static/chart.min.js
fi
if [ ! -f "static/three.min.js" ]; then
    echo "  ⬇️  Téléchargement Three.js..."
    curl -sL https://cdn.jsdelivr.net/npm/three@0.157.0/build/three.min.js -o static/three.min.js
fi
if [ ! -f "static/particles.min.js" ]; then
    echo "  ⬇️  Téléchargement Particles.js..."
    curl -sL https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js -o static/particles.min.js
fi

# 4. Arrêter tout serveur existant sur port 9000
echo -e "${YELLOW}⏹️  Arrêt du serveur existant...${NC}"
lsof -ti:9000 | xargs kill -9 2>/dev/null || true
sleep 1

# 5. Installer les dépendances Python
echo "🐍 Installation des dépendances Python..."
if [ -d ".venv_prod" ]; then
    source .venv_prod/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "venv-p2" ]; then
    source venv-p2/bin/activate
fi

python -c "import aiohttp" 2>/dev/null || {
    echo "  ⬇️  Installation aiohttp..."
    pip install aiohttp --quiet
}
python -c "import psutil" 2>/dev/null || {
    echo "  ⬇️  Installation psutil..."
    pip install psutil --quiet
}

# 6. Vérifier les fichiers requis
echo "✔️  Vérification des fichiers..."
if [ ! -f "src/jeffrey/core/loops/metrics_server.py" ]; then
    echo -e "${RED}❌ Erreur : metrics_server.py manquant${NC}"
    exit 1
fi
if [ ! -f "dashboard_pro_ultime.html" ]; then
    echo -e "${RED}❌ Erreur : dashboard_pro_ultime.html manquant${NC}"
    exit 1
fi

# 7. Lancer le serveur de métriques
echo -e "${GREEN}📊 Lancement du serveur de métriques...${NC}"
python src/jeffrey/core/loops/metrics_server.py \
    --port 9000 \
    --dashboard dashboard_pro_ultime.html &
SERVER_PID=$!

# Attendre que le serveur soit prêt
sleep 3

# 8. Vérifier que le serveur est lancé
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Le serveur n'a pas pu démarrer${NC}"
    exit 1
fi

# 9. Ouvrir le navigateur
echo -e "${GREEN}🌐 Ouverture du dashboard...${NC}"
if command -v open &> /dev/null; then
    open http://localhost:9000/
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:9000/
else
    echo -e "${YELLOW}👉 Ouvrez manuellement : http://localhost:9000/${NC}"
fi

# 10. Afficher les infos
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║   ✅ DASHBOARD JEFFREY OS LANCÉ !             ║"
echo "╠════════════════════════════════════════════════╣"
echo "║   📊 Dashboard : http://localhost:9000/       ║"
echo "║   📈 Métriques : http://localhost:9000/metrics ║"
echo "║   ❤️  Santé    : http://localhost:9000/health  ║"
echo "║   📋 Statut    : http://localhost:9000/status ║"
echo "╠════════════════════════════════════════════════╣"
echo "║   PID Serveur  : $SERVER_PID                  ║"
echo "║   Pour arrêter : kill $SERVER_PID             ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Garder le script actif et nettoyer à la sortie
trap "kill $SERVER_PID 2>/dev/null; echo -e '${YELLOW}⏹️  Dashboard arrêté${NC}'" EXIT

echo "Appuyez sur Ctrl+C pour arrêter le dashboard..."
wait $SERVER_PID
