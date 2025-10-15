#!/bin/bash
# Audit de sécurité pour le code Rust

set -e

echo "🦀 Audit de sécurité Rust..."
echo "================================"

# Vérifier si cargo est installé
if ! command -v cargo &> /dev/null; then
    echo "⚠️  Cargo non installé - Rust audit skipped"
    echo "   Pour installer: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 0
fi

# Vérifier si cargo-audit est installé
if ! cargo audit --version &> /dev/null; then
    echo "📦 Installation de cargo-audit..."
    cargo install cargo-audit
fi

# Chercher les projets Rust
RUST_PROJECTS=$(find . -name "Cargo.toml" 2>/dev/null | head -5)

if [ -z "$RUST_PROJECTS" ]; then
    echo "📝 Aucun projet Rust trouvé (Cargo.toml)"
    echo "   Les modules Rust sont en src/jeffrey/quarantine/"

    # Vérifier les fichiers .rs en quarantine
    if [ -d "src/jeffrey/quarantine" ]; then
        RS_FILES=$(find src/jeffrey/quarantine -name "*.rs" 2>/dev/null | wc -l)
        echo "   Trouvé $RS_FILES fichiers .rs en quarantine"
    fi
    exit 0
fi

# Auditer chaque projet Rust
for cargo_file in $RUST_PROJECTS; do
    PROJECT_DIR=$(dirname "$cargo_file")
    echo ""
    echo "🔍 Audit de: $PROJECT_DIR"
    echo "----------------------------"

    cd "$PROJECT_DIR"

    # Audit des dépendances
    if cargo audit 2>/dev/null; then
        echo "✅ Pas de vulnérabilités connues"
    else
        echo "⚠️  Vulnérabilités détectées - vérifier cargo audit"
    fi

    # Retour au répertoire racine
    cd - > /dev/null
done

echo ""
echo "✅ Audit Rust terminé"
