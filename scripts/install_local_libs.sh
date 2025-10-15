#!/bin/bash
# Télécharge les libs pour fallback local

set -euo pipefail

echo "📦 Installation des librairies locales pour fallback..."

# Créer le dossier static
mkdir -p static

# Télécharger Chart.js
echo "Downloading Chart.js..."
curl -L https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js -o static/chart.min.js

# Télécharger Three.js
echo "Downloading Three.js..."
curl -L https://cdn.jsdelivr.net/npm/three@0.157.0/build/three.min.js -o static/three.min.js

# Télécharger Particles.js
echo "Downloading Particles.js..."
curl -L https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js -o static/particles.min.js

echo "✅ Librairies téléchargées dans /static"
