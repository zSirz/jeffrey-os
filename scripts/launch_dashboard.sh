#!/bin/bash
# Lance le dashboard avec le serveur de métriques
set -euo pipefail  # GPT improvement: strict mode

echo "🚀 Lancement du Dashboard Jeffrey OS..."

# Vérifier que le dashboard HTML existe
if [ ! -f "dashboard_pro.html" ]; then
    echo "❌ Erreur : dashboard_pro.html n'existe pas"
    echo "Créez d'abord le fichier dashboard avec le contenu fourni"
    exit 1
fi

# Tuer tout ancien serveur sur port 9000
lsof -ti:9000 | xargs kill -9 2>/dev/null || true

# Activer venv si disponible
if [ -d .venv_prod ]; then
    source .venv_prod/bin/activate
elif [ -d venv ]; then
    source venv/bin/activate
fi

# Installer aiohttp si nécessaire
python -c "import aiohttp" 2>/dev/null || pip install aiohttp psutil --quiet

# Lancer le serveur
echo "📊 Démarrage du serveur de métriques sur port 9000..."
python src/jeffrey/core/loops/metrics_server.py --port 9000 --dashboard dashboard_pro.html &
SERVER_PID=$!

# Attendre que le serveur soit prêt
sleep 3

# Vérifier que le serveur est lancé
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Le serveur n'a pas pu démarrer"
    exit 1
fi

# Ouvrir le navigateur
echo "🌐 Ouverture du dashboard dans le navigateur..."
if command -v open &> /dev/null; then
    open http://localhost:9000/
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:9000/
else
    echo "Ouvrez manuellement : http://localhost:9000/"
fi

echo ""
echo "✅ Dashboard lancé !"
echo "📊 URL : http://localhost:9000/"
echo ""
echo "Pour arrêter le serveur : kill $SERVER_PID"
echo "Ou Ctrl+C dans ce terminal"

# Garder le script actif
trap "kill $SERVER_PID 2>/dev/null; echo '⏹️ Dashboard stopped'" EXIT
wait $SERVER_PID
