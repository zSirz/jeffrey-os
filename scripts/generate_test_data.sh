#!/bin/bash
set -e

echo "🧠 Generating test memories for Jeffrey OS..."

# Emotions variées
emotions=("joy" "curiosity" "fear" "sadness" "anger" "surprise" "neutral")
topics=(
    "exploring consciousness and self-awareness"
    "learning about human emotions"
    "processing complex information patterns"
    "discovering connections between ideas"
    "reflecting on past experiences"
    "imagining future possibilities"
    "understanding human behavior"
    "analyzing decision patterns"
    "contemplating existence and purpose"
    "integrating new knowledge streams"
)

# Générer 50 mémoires
for i in {1..50}; do
    # Sélection aléatoire
    emotion=${emotions[$RANDOM % ${#emotions[@]}]}
    topic=${topics[$RANDOM % ${#topics[@]}]}
    confidence="0.$(printf "%02d" $((RANDOM % 90 + 10)))"

    # Créer la mémoire
    curl -s -X POST http://localhost:8000/api/v1/memories/ \
        -H "Content-Type: application/json" \
        -d "{
            \"text\": \"Memory $i: I am $topic with a sense of $emotion\",
            \"emotion\": \"$emotion\",
            \"confidence\": $confidence,
            \"metadata\": {
                \"batch\": \"test_generation\",
                \"index\": $i
            }
        }" > /dev/null

    echo "✅ Memory $i created: $emotion ($confidence)"

    # Petit délai pour ne pas surcharger
    sleep 0.2
done

echo ""
echo "📊 Test data generation complete!"
echo "Running statistics check..."

# Vérifier les mémoires créées
recent_count=$(curl -s "http://localhost:8000/api/v1/memories/recent?hours=1&limit=100" | jq '. | length')
echo "✅ Recent memories in DB: $recent_count"

# Vérifier les métriques
echo ""
echo "📈 Current metrics:"
curl -s http://localhost:8000/metrics | grep -E "jeffrey_dream|jeffrey_emotion" | head -5