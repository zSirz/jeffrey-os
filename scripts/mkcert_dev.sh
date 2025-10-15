#!/bin/bash
# Script pour générer des certificats de développement

set -e

echo "🔐 Génération des certificats de développement..."

# Créer le répertoire keys s'il n'existe pas
mkdir -p keys/clients

# Générer une clé privée serveur
if [ ! -f keys/server.key ]; then
    openssl genrsa -out keys/server.key 2048
    echo "✅ Clé serveur générée"
else
    echo "📝 Clé serveur existante"
fi

# Générer un certificat serveur auto-signé
if [ ! -f keys/server.crt ]; then
    openssl req -new -x509 -key keys/server.key -out keys/server.crt -days 365 \
        -subj "/C=FR/ST=Paris/L=Paris/O=Jeffrey OS/CN=localhost"
    echo "✅ Certificat serveur généré"
else
    echo "📝 Certificat serveur existant"
fi

# Créer un CA de développement
if [ ! -f keys/ca.crt ]; then
    cp keys/server.crt keys/ca.crt
    echo "✅ CA de développement créé"
else
    echo "📝 CA existant"
fi

# Générer un certificat client de test
if [ ! -f keys/clients/test_client.crt ]; then
    # Clé client
    openssl genrsa -out keys/clients/test_client.key 2048

    # Certificat client
    openssl req -new -x509 -key keys/clients/test_client.key \
        -out keys/clients/test_client.crt -days 365 \
        -subj "/C=FR/ST=Paris/L=Paris/O=Jeffrey OS/CN=test_client"

    echo "✅ Certificat client de test généré"
else
    echo "📝 Certificat client existant"
fi

# Permissions sécurisées
chmod 600 keys/*.key keys/clients/*.key
chmod 644 keys/*.crt keys/clients/*.crt

echo "✅ Certificats de développement prêts"
echo ""
echo "📁 Structure:"
echo "  keys/server.key    - Clé privée serveur"
echo "  keys/server.crt    - Certificat serveur"
echo "  keys/ca.crt        - CA de développement"
echo "  keys/clients/      - Certificats clients"
echo ""
echo "⚠️  Ces certificats sont pour le DÉVELOPPEMENT uniquement"
echo "   Ne PAS utiliser en production !"
