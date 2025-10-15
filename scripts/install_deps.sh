#!/bin/bash
# Script d'installation propre des dépendances P2

set -e

echo "🧹 Nettoyage de l'environnement..."

# Si on est déjà dans un venv, on en sort
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate 2>/dev/null || true
fi

# Supprimer l'ancien venv s'il existe
if [ -d "venv-p2" ]; then
    echo "Suppression de l'ancien environnement virtuel..."
    rm -rf venv-p2
fi

# Créer un nouveau venv propre
echo "🐍 Création du nouvel environnement virtuel..."
python3 -m venv venv-p2

# Activer le venv
source venv-p2/bin/activate

# Upgrade pip/setuptools/wheel
echo "📦 Mise à jour des outils de base..."
pip install --upgrade pip setuptools wheel

# Installer les dépendances
echo "📦 Installation des dépendances P2..."
pip install -r requirements-p2.txt

# Vérifier qu'il n'y a pas de conflits
echo "✅ Vérification des dépendances..."
pip check

# Afficher un résumé
echo ""
echo "✅ Installation terminée avec succès!"
echo "📦 Packages installés : $(pip list | wc -l)"
echo ""
echo "Pour activer l'environnement :"
echo "  source venv-p2/bin/activate"
echo ""
echo "Prochaine étape :"
echo "  ./scripts/prepare_p2.sh --fast"
