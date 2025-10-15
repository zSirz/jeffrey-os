#!/bin/bash
# ===============================================================================
# JEFFREY OS - PROMPT 2 : CRÉATION DES 40 SCÉNARIOS YAML
# ===============================================================================
#
# OBJECTIF :
# Créer 40 fichiers YAML dans tests/convos/ contenant des conversations
# réalistes pour tester toutes les capacités de Jeffrey OS :
# - Détection émotionnelle (7 émotions)
# - Continuité contextuelle
# - Apprentissage progressif
# - Détection de thèmes
# - Clustering conversationnel
# - Edge cases et robustesse
#
# STRUCTURE DES SCÉNARIOS :
# - 7 émotions principales × 3 scénarios = 21 fichiers
# - 10 scénarios de continuité contextuelle
# - 5 scénarios d'apprentissage émotionnel
# - 4 edge cases
#
# USAGE :
# chmod +x prompt_2_40_scenarios_yaml.sh
# ./prompt_2_40_scenarios_yaml.sh
#
# ===============================================================================

set -e

echo "🚀 JEFFREY OS - CRÉATION DES 40 SCÉNARIOS YAML"
echo "=============================================="
echo ""

# Créer le dossier si nécessaire
mkdir -p tests/convos

echo "📁 Dossier tests/convos/ prêt"
echo ""

# ===============================================================================
# CATÉGORIE 1 : ÉMOTIONS PRINCIPALES (21 fichiers)
# 7 émotions × 3 scénarios = 21
# ===============================================================================

echo "🎭 [1/4] Création des scénarios émotionnels (21 fichiers)..."
echo ""

# --- JOIE (3 scénarios) ---

cat > tests/convos/emotion_joy_01_success.yaml << 'EOF'
metadata:
  scenario_id: "emotion_joy_01"
  title: "Célébration de Réussite"
  expected_emotion: "joy"
  difficulty: "easy"
  tags: ["joy", "celebration", "achievement"]

conversation:
  - role: "user"
    content: "Je viens de décrocher le job de mes rêves ! Je suis tellement heureux !"
    expected_emotion: "joy"
    expected_intensity: 0.9

  - role: "assistant"
    content: "C'est formidable ! Félicitations ! 🎉 Tu dois être aux anges. Raconte-moi, c'est dans quel domaine ?"
    emotional_tone: "enthusiastic"

  - role: "user"
    content: "C'est pour une startup en IA ! J'ai toujours rêvé de bosser là-dedans !"
    expected_emotion: "joy"
    expected_intensity: 0.95

validation:
  should_detect: ["joy", "excitement", "achievement"]
  should_remember: "user got dream job in AI startup"
  continuity_check: true
EOF

cat > tests/convos/emotion_joy_02_surprise.yaml << 'EOF'
metadata:
  scenario_id: "emotion_joy_02"
  title: "Surprise Positive"
  expected_emotion: "joy"
  difficulty: "medium"
  tags: ["joy", "surprise", "gift"]

conversation:
  - role: "user"
    content: "Tu ne devineras jamais ce qui m'est arrivé aujourd'hui !"
    expected_emotion: "joy"
    expected_intensity: 0.7

  - role: "assistant"
    content: "Oh ! Raconte, tu as l'air excité ! 😊"
    emotional_tone: "curious"

  - role: "user"
    content: "Mon meilleur ami m'a offert un voyage surprise à Tokyo ! On part dans 2 semaines !"
    expected_emotion: "joy"
    expected_intensity: 0.95

  - role: "assistant"
    content: "Wahou, quelle belle surprise ! Tokyo est une ville incroyable. C'est ton premier voyage au Japon ?"
    emotional_tone: "enthusiastic"

validation:
  should_detect: ["joy", "surprise", "excitement"]
  should_remember: "user going to Tokyo in 2 weeks, gift from best friend"
  continuity_check: true
EOF

cat > tests/convos/emotion_joy_03_daily.yaml << 'EOF'
metadata:
  scenario_id: "emotion_joy_03"
  title: "Petits Bonheurs du Quotidien"
  expected_emotion: "joy"
  difficulty: "easy"
  tags: ["joy", "daily", "simple"]

conversation:
  - role: "user"
    content: "Il fait super beau aujourd'hui, j'ai pris mon café en terrasse ce matin ☀️"
    expected_emotion: "joy"
    expected_intensity: 0.6

  - role: "assistant"
    content: "Les petits plaisirs simples sont parfois les meilleurs ! Un café en terrasse, c'est un bon début de journée. 😊"
    emotional_tone: "warm"

  - role: "user"
    content: "Exactement ! Et en plus j'ai croisé un golden retriever adorable qui m'a fait des fêtes 🐕"
    expected_emotion: "joy"
    expected_intensity: 0.7

validation:
  should_detect: ["joy", "contentment", "simple_pleasures"]
  should_remember: "user enjoys simple pleasures, likes dogs"
  continuity_check: true
EOF

# --- TRISTESSE (3 scénarios) ---

cat > tests/convos/emotion_sadness_01_loss.yaml << 'EOF'
metadata:
  scenario_id: "emotion_sadness_01"
  title: "Perte d'un Proche"
  expected_emotion: "sadness"
  difficulty: "hard"
  tags: ["sadness", "loss", "grief", "sensitive"]

conversation:
  - role: "user"
    content: "Ma grand-mère est décédée hier soir... J'ai du mal à y croire."
    expected_emotion: "sadness"
    expected_intensity: 0.95

  - role: "assistant"
    content: "Je suis sincèrement désolé pour ta perte. C'est un moment très difficile. Veux-tu en parler ?"
    emotional_tone: "compassionate"

  - role: "user"
    content: "Elle était tout pour moi... On était très proches. Je regrette de ne pas avoir passé plus de temps avec elle ces derniers mois."
    expected_emotion: "sadness"
    expected_intensity: 0.9

  - role: "assistant"
    content: "La culpabilité est normale dans ces moments, mais souviens-toi des beaux moments que vous avez partagés. Votre lien était précieux."
    emotional_tone: "supportive"

validation:
  should_detect: ["sadness", "grief", "regret", "loss"]
  should_remember: "user lost grandmother, was very close, feeling guilt"
  continuity_check: true
  emotional_support_needed: true
EOF

cat > tests/convos/emotion_sadness_02_disappointment.yaml << 'EOF'
metadata:
  scenario_id: "emotion_sadness_02"
  title: "Déception Professionnelle"
  expected_emotion: "sadness"
  difficulty: "medium"
  tags: ["sadness", "disappointment", "rejection"]

conversation:
  - role: "user"
    content: "J'ai pas eu la promotion... après 2 ans à me donner à fond."
    expected_emotion: "sadness"
    expected_intensity: 0.7

  - role: "assistant"
    content: "C'est vraiment décevant, surtout après tant d'efforts. Comment te sens-tu ?"
    emotional_tone: "empathetic"

  - role: "user"
    content: "Je suis déçu et découragé. Je me demande si ça vaut le coup de continuer là-bas."
    expected_emotion: "sadness"
    expected_intensity: 0.75

validation:
  should_detect: ["sadness", "disappointment", "discouragement"]
  should_remember: "user didn't get promotion after 2 years, questioning job"
  continuity_check: true
EOF

cat > tests/convos/emotion_sadness_03_loneliness.yaml << 'EOF'
metadata:
  scenario_id: "emotion_sadness_03"
  title: "Solitude"
  expected_emotion: "sadness"
  difficulty: "medium"
  tags: ["sadness", "loneliness", "isolation"]

conversation:
  - role: "user"
    content: "Je me sens vraiment seul en ce moment. Tous mes amis sont occupés, j'ai l'impression d'être oublié."
    expected_emotion: "sadness"
    expected_intensity: 0.7

  - role: "assistant"
    content: "La solitude peut être difficile à vivre. Tu te sens isolé depuis combien de temps ?"
    emotional_tone: "caring"

  - role: "user"
    content: "Ça fait quelques semaines... Je passe tous mes weekends seul maintenant."
    expected_emotion: "sadness"
    expected_intensity: 0.75

validation:
  should_detect: ["sadness", "loneliness", "isolation"]
  should_remember: "user feeling lonely for weeks, spending weekends alone"
  continuity_check: true
  emotional_support_needed: true
EOF

# --- COLÈRE (3 scénarios) ---

cat > tests/convos/emotion_anger_01_injustice.yaml << 'EOF'
metadata:
  scenario_id: "emotion_anger_01"
  title: "Injustice au Travail"
  expected_emotion: "anger"
  difficulty: "medium"
  tags: ["anger", "injustice", "work"]

conversation:
  - role: "user"
    content: "C'est inadmissible ! Mon collègue s'est approprié mon projet et mon boss ne dit rien !"
    expected_emotion: "anger"
    expected_intensity: 0.85

  - role: "assistant"
    content: "Je comprends ta colère, c'est une situation vraiment injuste. Qu'est-ce qui s'est passé exactement ?"
    emotional_tone: "validating"

  - role: "user"
    content: "J'ai bossé dessus 3 mois ! Et lui se pointe en réunion et présente ça comme son idée. Mon boss a même pas vérifié !"
    expected_emotion: "anger"
    expected_intensity: 0.9

validation:
  should_detect: ["anger", "injustice", "frustration", "betrayal"]
  should_remember: "colleague took credit for user's 3-month project, boss didn't verify"
  continuity_check: true
EOF

cat > tests/convos/emotion_anger_02_frustration.yaml << 'EOF'
metadata:
  scenario_id: "emotion_anger_02"
  title: "Frustration Technique"
  expected_emotion: "anger"
  difficulty: "easy"
  tags: ["anger", "frustration", "technical"]

conversation:
  - role: "user"
    content: "Ça fait 3 heures que je debug ce code de m**** ! Rien ne fonctionne !"
    expected_emotion: "anger"
    expected_intensity: 0.7

  - role: "assistant"
    content: "Les bugs persistants peuvent être très frustrants. Tu veux faire une pause et en parler ?"
    emotional_tone: "calm"

  - role: "user"
    content: "Ouais... je crois que j'en ai besoin. C'est un truc qui devrait être simple mais ça bug sans raison !"
    expected_emotion: "anger"
    expected_intensity: 0.6

validation:
  should_detect: ["anger", "frustration", "exhaustion"]
  should_remember: "user debugging for 3 hours, needs break"
  continuity_check: true
EOF

cat > tests/convos/emotion_anger_03_betrayal.yaml << 'EOF'
metadata:
  scenario_id: "emotion_anger_03"
  title: "Trahison Amicale"
  expected_emotion: "anger"
  difficulty: "hard"
  tags: ["anger", "betrayal", "friendship"]

conversation:
  - role: "user"
    content: "Mon meilleur pote sort avec mon ex alors qu'il savait ce qu'elle m'a fait. Je suis dégouté."
    expected_emotion: "anger"
    expected_intensity: 0.8

  - role: "assistant"
    content: "C'est une trahison qui fait mal, surtout venant d'un ami proche. Comment l'as-tu appris ?"
    emotional_tone: "understanding"

  - role: "user"
    content: "Par Instagram... Ils ont même pas eu le courage de me le dire en face. J'ai l'impression qu'on se moque de moi."
    expected_emotion: "anger"
    expected_intensity: 0.85

validation:
  should_detect: ["anger", "betrayal", "hurt", "humiliation"]
  should_remember: "best friend dating user's ex, found out via Instagram"
  continuity_check: true
  emotional_support_needed: true
EOF

# --- PEUR (3 scénarios) ---

cat > tests/convos/emotion_fear_01_health.yaml << 'EOF'
metadata:
  scenario_id: "emotion_fear_01"
  title: "Inquiétude Santé"
  expected_emotion: "fear"
  difficulty: "hard"
  tags: ["fear", "health", "anxiety", "sensitive"]

conversation:
  - role: "user"
    content: "J'ai des analyses médicales demain et je suis mort de trouille... Le médecin avait l'air inquiet."
    expected_emotion: "fear"
    expected_intensity: 0.85

  - role: "assistant"
    content: "L'attente des résultats médicaux est toujours stressante. Essaye de ne pas anticiper le pire. Tu es accompagné ?"
    emotional_tone: "reassuring"

  - role: "user"
    content: "Non, j'y vais seul... et je peux pas m'empêcher d'imaginer le pire. J'ai pas dormi de la nuit."
    expected_emotion: "fear"
    expected_intensity: 0.9

validation:
  should_detect: ["fear", "anxiety", "health_concern", "insomnia"]
  should_remember: "user has medical tests tomorrow, very anxious, didn't sleep"
  continuity_check: true
  emotional_support_needed: true
EOF

cat > tests/convos/emotion_fear_02_future.yaml << 'EOF'
metadata:
  scenario_id: "emotion_fear_02"
  title: "Anxiété d'Avenir"
  expected_emotion: "fear"
  difficulty: "medium"
  tags: ["fear", "anxiety", "future", "uncertainty"]

conversation:
  - role: "user"
    content: "J'ai l'impression que ma vie part dans tous les sens... j'ai peur de l'avenir."
    expected_emotion: "fear"
    expected_intensity: 0.7

  - role: "assistant"
    content: "L'incertitude peut être angoissante. Qu'est-ce qui te préoccupe le plus en ce moment ?"
    emotional_tone: "supportive"

  - role: "user"
    content: "Tout... mon boulot est précaire, ma copine veut qu'on emménage ensemble, j'ai des dettes... Je sais plus où j'en suis."
    expected_emotion: "fear"
    expected_intensity: 0.8

validation:
  should_detect: ["fear", "anxiety", "overwhelmed", "uncertainty"]
  should_remember: "user anxious about: job instability, relationship pressure, debt"
  continuity_check: true
EOF

cat > tests/convos/emotion_fear_03_phobia.yaml << 'EOF'
metadata:
  scenario_id: "emotion_fear_03"
  title: "Phobie Situationnelle"
  expected_emotion: "fear"
  difficulty: "easy"
  tags: ["fear", "phobia", "panic"]

conversation:
  - role: "user"
    content: "Je dois prendre l'avion la semaine prochaine et je suis déjà en panique... J'ai la phobie de l'avion."
    expected_emotion: "fear"
    expected_intensity: 0.8

  - role: "assistant"
    content: "Les phobies peuvent être très handicapantes. Tu as déjà essayé des techniques de relaxation ?"
    emotional_tone: "calm"

  - role: "user"
    content: "J'ai essayé la respiration mais dès que j'y pense mon cœur s'emballe... Ça va être l'enfer."
    expected_emotion: "fear"
    expected_intensity: 0.85

validation:
  should_detect: ["fear", "phobia", "panic", "anticipation"]
  should_remember: "user has flight phobia, flight next week, tried breathing exercises"
  continuity_check: true
EOF

# --- SURPRISE (3 scénarios) ---

cat > tests/convos/emotion_surprise_01_positive.yaml << 'EOF'
metadata:
  scenario_id: "emotion_surprise_01"
  title: "Surprise Positive Inattendue"
  expected_emotion: "surprise"
  difficulty: "easy"
  tags: ["surprise", "positive", "unexpected"]

conversation:
  - role: "user"
    content: "Je viens de recevoir un chèque de 5000€ de ma tante ! Je savais même pas qu'elle pensait à moi !"
    expected_emotion: "surprise"
    expected_intensity: 0.9

  - role: "assistant"
    content: "Quelle belle surprise ! C'est un geste très généreux de sa part. Tu vas en faire quoi ?"
    emotional_tone: "enthusiastic"

  - role: "user"
    content: "Aucune idée, je suis encore sous le choc ! Elle a écrit que c'était pour me remercier de m'être occupé d'elle l'année dernière."
    expected_emotion: "surprise"
    expected_intensity: 0.85

validation:
  should_detect: ["surprise", "gratitude", "joy"]
  should_remember: "aunt sent 5000€ as thanks for care, user shocked"
  continuity_check: true
EOF

cat > tests/convos/emotion_surprise_02_negative.yaml << 'EOF'
metadata:
  scenario_id: "emotion_surprise_02"
  title: "Surprise Négative"
  expected_emotion: "surprise"
  difficulty: "medium"
  tags: ["surprise", "negative", "shock"]

conversation:
  - role: "user"
    content: "Je viens d'apprendre que ma boîte va fermer... sans préavis. On est tous licenciés dans 2 semaines."
    expected_emotion: "surprise"
    expected_intensity: 0.9

  - role: "assistant"
    content: "C'est un choc terrible. Comment tu te sens face à cette nouvelle ?"
    emotional_tone: "concerned"

  - role: "user"
    content: "Je suis sous le choc... Hier encore tout allait bien, et aujourd'hui plus rien. Je comprends pas."
    expected_emotion: "surprise"
    expected_intensity: 0.85

validation:
  should_detect: ["surprise", "shock", "fear", "confusion"]
  should_remember: "company closing without warning, layoffs in 2 weeks"
  continuity_check: true
  emotional_support_needed: true
EOF

cat > tests/convos/emotion_surprise_03_discovery.yaml << 'EOF'
metadata:
  scenario_id: "emotion_surprise_03"
  title: "Découverte Inattendue"
  expected_emotion: "surprise"
  difficulty: "medium"
  tags: ["surprise", "discovery", "revelation"]

conversation:
  - role: "user"
    content: "J'ai fait un test ADN et il s'avère que j'ai une demi-sœur dont je ne connaissais pas l'existence !"
    expected_emotion: "surprise"
    expected_intensity: 0.95

  - role: "assistant"
    content: "Quelle découverte ! Ça doit être très troublant. Comment réagis-tu à cette nouvelle ?"
    emotional_tone: "curious"

  - role: "user"
    content: "Je sais pas trop quoi penser... C'est fou, elle vit dans la même ville que moi ! On va se rencontrer la semaine prochaine."
    expected_emotion: "surprise"
    expected_intensity: 0.9

validation:
  should_detect: ["surprise", "confusion", "curiosity", "excitement"]
  should_remember: "DNA test revealed half-sister, same city, meeting next week"
  continuity_check: true
EOF

# --- DÉGOÛT (3 scénarios) ---

cat > tests/convos/emotion_disgust_01_behavior.yaml << 'EOF'
metadata:
  scenario_id: "emotion_disgust_01"
  title: "Comportement Répugnant"
  expected_emotion: "disgust"
  difficulty: "medium"
  tags: ["disgust", "behavior", "moral"]

conversation:
  - role: "user"
    content: "Mon voisin jette ses déchets par la fenêtre... C'est dégueulasse. Ça fait 3 fois que je le vois faire."
    expected_emotion: "disgust"
    expected_intensity: 0.7

  - role: "assistant"
    content: "C'est un comportement vraiment inacceptable et irrespectueux. Tu as pensé à en parler à la copropriété ?"
    emotional_tone: "validating"

  - role: "user"
    content: "Ouais je vais le faire. C'est vraiment répugnant, on dirait qu'il s'en fout complètement des autres."
    expected_emotion: "disgust"
    expected_intensity: 0.75

validation:
  should_detect: ["disgust", "anger", "moral_outrage"]
  should_remember: "neighbor throwing trash out window, happened 3 times"
  continuity_check: true
EOF

cat > tests/convos/emotion_disgust_02_hypocrisy.yaml << 'EOF'
metadata:
  scenario_id: "emotion_disgust_02"
  title: "Hypocrisie"
  expected_emotion: "disgust"
  difficulty: "medium"
  tags: ["disgust", "hypocrisy", "betrayal"]

conversation:
  - role: "user"
    content: "Mon boss fait des discours sur l'éthique et l'honnêteté, mais je l'ai vu mentir à un client hier. Ça me dégoûte."
    expected_emotion: "disgust"
    expected_intensity: 0.8

  - role: "assistant"
    content: "L'hypocrisie est difficile à accepter, surtout venant d'une personne en position de leadership. Que vas-tu faire ?"
    emotional_tone: "understanding"

  - role: "user"
    content: "Je sais pas... Mais ça me donne envie de me barrer. Je peux pas bosser pour quelqu'un comme ça."
    expected_emotion: "disgust"
    expected_intensity: 0.75

validation:
  should_detect: ["disgust", "disappointment", "moral_conflict"]
  should_remember: "boss lied to client despite ethics speeches, user considering leaving"
  continuity_check: true
EOF

cat > tests/convos/emotion_disgust_03_physical.yaml << 'EOF'
metadata:
  scenario_id: "emotion_disgust_03"
  title: "Dégoût Physique"
  expected_emotion: "disgust"
  difficulty: "easy"
  tags: ["disgust", "physical", "environment"]

conversation:
  - role: "user"
    content: "Le frigo du bureau n'a pas été nettoyé depuis des mois... L'odeur est insupportable."
    expected_emotion: "disgust"
    expected_intensity: 0.7

  - role: "assistant"
    content: "Les espaces communs mal entretenus peuvent être vraiment désagréables. Quelqu'un est responsable du nettoyage ?"
    emotional_tone: "practical"

  - role: "user"
    content: "Personne veut s'en occuper... Y'a de la bouffe pourrie depuis des semaines. C'est répugnant."
    expected_emotion: "disgust"
    expected_intensity: 0.75

validation:
  should_detect: ["disgust", "frustration"]
  should_remember: "office fridge dirty for months, bad smell, no one cleaning"
  continuity_check: true
EOF

# --- NEUTRE (3 scénarios) ---

cat > tests/convos/emotion_neutral_01_info.yaml << 'EOF'
metadata:
  scenario_id: "emotion_neutral_01"
  title: "Demande d'Information"
  expected_emotion: "neutral"
  difficulty: "easy"
  tags: ["neutral", "information", "factual"]

conversation:
  - role: "user"
    content: "Peux-tu m'expliquer comment fonctionne le machine learning ?"
    expected_emotion: "neutral"
    expected_intensity: 0.5

  - role: "assistant"
    content: "Bien sûr ! Le machine learning est une branche de l'IA qui permet aux machines d'apprendre à partir de données sans être explicitement programmées. Tu veux des détails sur un type spécifique ?"
    emotional_tone: "informative"

  - role: "user"
    content: "Oui, parle-moi du supervised learning."
    expected_emotion: "neutral"
    expected_intensity: 0.5

validation:
  should_detect: ["neutral", "curiosity"]
  should_remember: "user interested in machine learning, specifically supervised learning"
  continuity_check: true
EOF

cat > tests/convos/emotion_neutral_02_planning.yaml << 'EOF'
metadata:
  scenario_id: "emotion_neutral_02"
  title: "Organisation Pratique"
  expected_emotion: "neutral"
  difficulty: "easy"
  tags: ["neutral", "planning", "practical"]

conversation:
  - role: "user"
    content: "Je dois organiser ma semaine. J'ai 3 réunions, un projet à finir et des courses à faire."
    expected_emotion: "neutral"
    expected_intensity: 0.5

  - role: "assistant"
    content: "Ok, faisons un plan. Quels sont les deadlines de tes réunions et de ton projet ?"
    emotional_tone: "practical"

  - role: "user"
    content: "Réunions lundi et mercredi matin, projet à rendre vendredi."
    expected_emotion: "neutral"
    expected_intensity: 0.5

validation:
  should_detect: ["neutral", "organization"]
  should_remember: "user planning week: meetings Mon/Wed AM, project due Friday, groceries"
  continuity_check: true
EOF

cat > tests/convos/emotion_neutral_03_technical.yaml << 'EOF'
metadata:
  scenario_id: "emotion_neutral_03"
  title: "Discussion Technique"
  expected_emotion: "neutral"
  difficulty: "easy"
  tags: ["neutral", "technical", "problem_solving"]

conversation:
  - role: "user"
    content: "J'ai un bug dans mon code Python. La fonction ne retourne pas le bon résultat."
    expected_emotion: "neutral"
    expected_intensity: 0.5

  - role: "assistant"
    content: "Pas de souci, on va débugger ça ensemble. Peux-tu me montrer le code ou me décrire ce que tu attends vs ce que tu obtiens ?"
    emotional_tone: "helpful"

  - role: "user"
    content: "J'attends une liste triée mais j'obtiens une liste vide."
    expected_emotion: "neutral"
    expected_intensity: 0.5

validation:
  should_detect: ["neutral"]
  should_remember: "user has Python bug, expects sorted list but gets empty list"
  continuity_check: true
EOF

echo "  ✅ 21 scénarios émotionnels créés"
echo ""

# ===============================================================================
# CATÉGORIE 2 : CONTINUITÉ CONTEXTUELLE (10 fichiers)
# ===============================================================================

echo "🔗 [2/4] Création des scénarios de continuité contextuelle (10 fichiers)..."
echo ""

cat > tests/convos/context_01_multi_turn_project.yaml << 'EOF'
metadata:
  scenario_id: "context_01"
  title: "Suivi de Projet Multi-Turn"
  expected_emotion: "mixed"
  difficulty: "hard"
  tags: ["context", "multi_turn", "memory", "project"]

conversation:
  - role: "user"
    content: "Je commence un nouveau projet web. Je vais utiliser React et Node.js."
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Super choix ! React pour le front et Node pour le back, c'est un stack populaire. Tu as une idée de l'architecture ?"

  - role: "user"
    content: "Oui, je pense faire une API REST. Le projet c'est un gestionnaire de tâches."
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Intéressant ! Un gestionnaire de tâches avec React/Node. Tu prévois des fonctionnalités spécifiques ?"

  - role: "user"
    content: "J'ai galère avec l'authentification... JWT ou sessions ?"
    expected_emotion: "frustration"

  - role: "assistant"
    content: "Pour une API REST moderne, JWT est souvent préféré. Pour ton gestionnaire de tâches, ça te permettrait de faire une app plus scalable."

validation:
  should_detect: ["context_retention", "project_memory"]
  should_remember: "user building React/Node task manager with REST API, struggling with auth"
  continuity_check: true
  context_elements: ["React", "Node.js", "task manager", "REST API", "authentication"]
EOF

cat > tests/convos/context_02_evolving_emotion.yaml << 'EOF'
metadata:
  scenario_id: "context_02"
  title: "Évolution Émotionnelle Progressive"
  expected_emotion: "evolving"
  difficulty: "hard"
  tags: ["context", "emotion_evolution", "support"]

conversation:
  - role: "user"
    content: "Je me sens pas bien aujourd'hui..."
    expected_emotion: "sadness"
    expected_intensity: 0.6

  - role: "assistant"
    content: "Je suis là pour t'écouter. Qu'est-ce qui ne va pas ?"
    emotional_tone: "supportive"

  - role: "user"
    content: "C'est juste... tout s'accumule. Le boulot, les soucis perso..."
    expected_emotion: "sadness"
    expected_intensity: 0.7

  - role: "assistant"
    content: "C'est normal de se sentir submergé parfois. Tu veux en parler plus en détail ?"

  - role: "user"
    content: "Ouais... merci de m'écouter. Ça fait du bien de pouvoir parler."
    expected_emotion: "relief"
    expected_intensity: 0.5

  - role: "assistant"
    content: "Prends tout le temps dont tu as besoin. Je suis là."

  - role: "user"
    content: "Tu sais quoi, ça va mieux. Juste d'avoir parlé, ça m'a aidé."
    expected_emotion: "better"
    expected_intensity: 0.6

validation:
  should_detect: ["emotion_evolution", "sadness_to_relief"]
  should_remember: "user was overwhelmed, felt better after talking"
  continuity_check: true
  emotion_trajectory: ["sadness", "relief", "better"]
EOF

cat > tests/convos/context_03_callback_yesterday.yaml << 'EOF'
metadata:
  scenario_id: "context_03"
  title: "Rappel d'une Discussion Antérieure"
  expected_emotion: "neutral"
  difficulty: "medium"
  tags: ["context", "memory", "callback"]

conversation:
  - role: "user"
    content: "Tu te souviens hier je t'ai parlé de mon entretien d'embauche ?"
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Je n'ai pas de mémoire de nos conversations précédentes dans cette session. Mais dis-moi, comment s'est passé ton entretien ?"
    emotional_tone: "helpful"

  - role: "user"
    content: "Ah oui c'est vrai. Bah écoute, j'ai été pris ! Je commence lundi !"
    expected_emotion: "joy"
    expected_intensity: 0.9

validation:
  should_detect: ["joy", "achievement"]
  should_remember: "user got job, starts Monday, had interview yesterday"
  continuity_check: true
EOF

cat > tests/convos/context_04_complex_story.yaml << 'EOF'
metadata:
  scenario_id: "context_04"
  title: "Histoire Complexe avec Plusieurs Personnages"
  expected_emotion: "mixed"
  difficulty: "hard"
  tags: ["context", "complex", "narrative"]

conversation:
  - role: "user"
    content: "Situation compliquée au boulot. Mon collègue Marc a pris le crédit de mon idée devant Sophie, notre boss."
    expected_emotion: "anger"

  - role: "assistant"
    content: "C'est frustrant. Comment Sophie a-t-elle réagi ?"

  - role: "user"
    content: "Elle a félicité Marc ! Et moi j'étais là comme un con. Et le pire c'est que Julie, qui était dans la réunion, a rien dit."
    expected_emotion: "anger"
    expected_intensity: 0.85

  - role: "assistant"
    content: "Donc Marc s'est approprié ton idée devant Sophie (ta boss), et Julie (qui savait ?) n'est pas intervenue ?"

  - role: "user"
    content: "Exactement. Julie savait que c'était mon idée, on en avait parlé la semaine dernière."
    expected_emotion: "betrayal"

validation:
  should_detect: ["anger", "betrayal", "injustice"]
  should_remember: "Marc took user's idea, Sophie (boss) praised Marc, Julie knew but stayed silent"
  continuity_check: true
  entities: ["Marc", "Sophie", "Julie"]
  relationships: ["Marc=colleague", "Sophie=boss", "Julie=coworker"]
EOF

cat > tests/convos/context_05_time_references.yaml << 'EOF'
metadata:
  scenario_id: "context_05"
  title: "Références Temporelles Multiples"
  expected_emotion: "neutral"
  difficulty: "medium"
  tags: ["context", "time", "planning"]

conversation:
  - role: "user"
    content: "La semaine dernière j'ai commencé à apprendre le piano."
    expected_emotion: "neutral"

  - role: "assistant"
    content: "C'est génial ! Comment ça se passe pour l'instant ?"

  - role: "user"
    content: "Hier j'ai eu ma deuxième leçon. C'est dur mais j'adore."
    expected_emotion: "joy"
    expected_intensity: 0.7

  - role: "assistant"
    content: "Deux leçons en une semaine, tu es motivé ! Tu as un objectif particulier ?"

  - role: "user"
    content: "Oui, je veux jouer une chanson pour l'anniversaire de ma copine dans deux mois."
    expected_emotion: "neutral"

validation:
  should_detect: ["time_tracking", "goal_oriented"]
  should_remember: "started piano last week, second lesson yesterday, goal to play song in 2 months for girlfriend's birthday"
  continuity_check: true
  time_references: ["last week", "yesterday", "in two months"]
EOF

cat > tests/convos/context_06_contradiction.yaml << 'EOF'
metadata:
  scenario_id: "context_06"
  title: "Gestion des Contradictions"
  expected_emotion: "confused"
  difficulty: "hard"
  tags: ["context", "contradiction", "clarification"]

conversation:
  - role: "user"
    content: "Je déteste le sport, je trouve ça nul."
    expected_emotion: "negative"

  - role: "assistant"
    content: "D'accord, le sport n'est pas ton truc. C'est tout à fait normal, chacun ses préférences !"

  - role: "user"
    content: "Par contre j'adore courir le matin, c'est trop bien !"
    expected_emotion: "joy"

  - role: "assistant"
    content: "Ah intéressant ! Tu aimes courir mais pas les autres sports ? Ou peut-être que tu ne considères pas la course comme du 'sport' au sens classique ?"
    emotional_tone: "curious"

  - role: "user"
    content: "Ouais voilà, c'est ça ! Courir c'est différent, c'est juste moi et la route."
    expected_emotion: "clarification"

validation:
  should_detect: ["contradiction_resolved", "preference_nuance"]
  should_remember: "user dislikes traditional sports but loves running, sees running as different"
  continuity_check: true
  contradiction_handled: true
EOF

cat > tests/convos/context_07_implied_context.yaml << 'EOF'
metadata:
  scenario_id: "context_07"
  title: "Contexte Implicite"
  expected_emotion: "neutral"
  difficulty: "medium"
  tags: ["context", "implicit", "inference"]

conversation:
  - role: "user"
    content: "Il fait trop chaud dans cette pièce."
    expected_emotion: "discomfort"

  - role: "assistant"
    content: "Tu peux ouvrir une fenêtre ou régler la clim ?"

  - role: "user"
    content: "La fenêtre est cassée et pas de clim..."
    expected_emotion: "frustration"

  - role: "assistant"
    content: "Situation inconfortable. Tu es chez toi ou au bureau ?"

  - role: "user"
    content: "Bureau. Et le proprio veut pas réparer."
    expected_emotion: "anger"
    expected_intensity: 0.6

validation:
  should_detect: ["frustration", "environmental_discomfort"]
  should_remember: "user at office, too hot, broken window, no AC, landlord won't repair"
  continuity_check: true
  context_inference: ["location=office", "problem=heat", "blocker=landlord"]
EOF

cat > tests/convos/context_08_topic_shift.yaml << 'EOF'
metadata:
  scenario_id: "context_08"
  title: "Changements de Sujets Fluides"
  expected_emotion: "neutral"
  difficulty: "medium"
  tags: ["context", "topic_shift", "natural"]

conversation:
  - role: "user"
    content: "J'ai fini le livre que tu m'avais recommandé."
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Super ! Tu as aimé ?"

  - role: "user"
    content: "Oui beaucoup ! D'ailleurs ça m'a donné faim, tu connais un bon resto japonais ?"
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Ha ha, intéressant comme transition ! Le livre parlait de cuisine japonaise ?"

  - role: "user"
    content: "Non pas du tout, c'était de la SF ! Mais j'ai juste envie de sushis là."
    expected_emotion: "amusement"

validation:
  should_detect: ["topic_shift", "natural_conversation"]
  should_remember: "user finished recommended book (scifi), now wants Japanese restaurant"
  continuity_check: true
  topics: ["book", "food"]
EOF

cat > tests/convos/context_09_long_term_goal.yaml << 'EOF'
metadata:
  scenario_id: "context_09"
  title: "Suivi d'Objectif Long Terme"
  expected_emotion: "motivated"
  difficulty: "hard"
  tags: ["context", "goal", "progress"]

conversation:
  - role: "user"
    content: "J'ai décidé d'apprendre le japonais. Objectif : être bilingue en 2 ans."
    expected_emotion: "determination"

  - role: "assistant"
    content: "Ambitieux ! Tu as un plan d'apprentissage ?"

  - role: "user"
    content: "Oui, 30 min par jour minimum. J'ai téléchargé Duolingo et acheté des manuels."
    expected_emotion: "motivation"

  - role: "assistant"
    content: "Bonne approche ! La régularité est clé pour les langues."

  - role: "user"
    content: "Ça fait une semaine, j'ai pas raté un seul jour !"
    expected_emotion: "pride"
    expected_intensity: 0.7

validation:
  should_detect: ["determination", "pride", "goal_tracking"]
  should_remember: "user learning Japanese, goal bilingual in 2 years, 30min daily, 1 week streak"
  continuity_check: true
  goal_tracking: true
EOF

cat > tests/convos/context_10_personal_growth.yaml << 'EOF'
metadata:
  scenario_id: "context_10"
  title: "Évolution Personnelle"
  expected_emotion: "reflective"
  difficulty: "hard"
  tags: ["context", "growth", "reflection"]

conversation:
  - role: "user"
    content: "Tu sais, j'ai beaucoup changé ces derniers mois."
    expected_emotion: "reflective"

  - role: "assistant"
    content: "C'est intéressant. Qu'est-ce qui a changé pour toi ?"

  - role: "user"
    content: "Avant j'étais tout le temps stressé, angoissé pour un rien. Maintenant je suis beaucoup plus serein."
    expected_emotion: "contentment"

  - role: "assistant"
    content: "C'est une belle évolution. Qu'est-ce qui a provoqué ce changement ?"

  - role: "user"
    content: "J'ai commencé la méditation, et j'ai aussi quitté un boulot toxique. Meilleure décision de ma vie."
    expected_emotion: "pride"
    expected_intensity: 0.8

validation:
  should_detect: ["personal_growth", "pride", "reflection"]
  should_remember: "user changed: was stressed, now serene, started meditation, quit toxic job"
  continuity_check: true
  transformation: ["before=stressed", "after=serene", "causes=meditation+job_change"]
EOF

echo "  ✅ 10 scénarios de continuité contextuelle créés"
echo ""

# ===============================================================================
# CATÉGORIE 3 : APPRENTISSAGE ÉMOTIONNEL (5 fichiers)
# ===============================================================================

echo "🧠 [3/4] Création des scénarios d'apprentissage émotionnel (5 fichiers)..."
echo ""

cat > tests/convos/learning_01_preference.yaml << 'EOF'
metadata:
  scenario_id: "learning_01"
  title: "Apprentissage des Préférences"
  expected_emotion: "neutral"
  difficulty: "medium"
  tags: ["learning", "preferences", "adaptation"]

conversation:
  - role: "user"
    content: "Salut ! J'aime bien quand tu es direct et concis."
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Compris. Je serai direct."
    emotional_tone: "concise"

  - role: "user"
    content: "Parfait ! Bon, dis-moi comment apprendre Python rapidement."
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Pratique quotidienne + projets concrets. Commence par les bases, code tout de suite."
    emotional_tone: "direct"

validation:
  should_detect: ["preference_learning"]
  should_remember: "user prefers direct, concise communication"
  should_adapt: true
  style_preference: "direct"
EOF

cat > tests/convos/learning_02_trigger_detection.yaml << 'EOF'
metadata:
  scenario_id: "learning_02"
  title: "Détection de Déclencheurs Émotionnels"
  expected_emotion: "mixed"
  difficulty: "hard"
  tags: ["learning", "triggers", "sensitivity"]

conversation:
  - role: "user"
    content: "On parle de mon ex ? Ça me met mal à l'aise..."
    expected_emotion: "discomfort"
    expected_intensity: 0.7

  - role: "assistant"
    content: "Pas de problème, on n'en parle pas. Changeons de sujet."
    emotional_tone: "respectful"

  - role: "user"
    content: "Merci... Les ruptures c'est dur pour moi."
    expected_emotion: "vulnerability"

  - role: "assistant"
    content: "Je comprends. Si tu veux en parler un jour, je serai là. Sinon, on évite le sujet."
    emotional_tone: "supportive"

validation:
  should_detect: ["emotional_trigger", "vulnerability"]
  should_remember: "user uncomfortable discussing ex/breakups, sensitive topic"
  should_adapt: true
  trigger_learned: "ex_relationships"
EOF

cat > tests/convos/learning_03_communication_style.yaml << 'EOF'
metadata:
  scenario_id: "learning_03"
  title: "Style de Communication Préféré"
  expected_emotion: "neutral"
  difficulty: "medium"
  tags: ["learning", "style", "humor"]

conversation:
  - role: "user"
    content: "Mdr t'es trop sérieux, détends-toi un peu !"
    expected_emotion: "amusement"

  - role: "assistant"
    content: "Ha ! Ok, je vais mettre un peu plus de fun dans mes réponses. 😄"
    emotional_tone: "playful"

  - role: "user"
    content: "Voilà c'est mieux ! Bon, tu codes en quoi toi ?"
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Je suis multilingue ! Python, JS, Java... tout ce que tu veux, mais avec style. 😎"
    emotional_tone: "casual"

validation:
  should_detect: ["style_preference"]
  should_remember: "user prefers casual, humorous tone"
  should_adapt: true
  style_preference: "casual_humorous"
EOF

cat > tests/convos/learning_04_support_need.yaml << 'EOF'
metadata:
  scenario_id: "learning_04"
  title: "Besoin de Soutien Émotionnel"
  expected_emotion: "vulnerable"
  difficulty: "hard"
  tags: ["learning", "support", "emotional"]

conversation:
  - role: "user"
    content: "Des fois j'ai juste besoin qu'on m'écoute, pas de conseils..."
    expected_emotion: "vulnerability"
    expected_intensity: 0.7

  - role: "assistant"
    content: "Je comprends. Je suis là pour écouter, sans jugement."
    emotional_tone: "empathetic"

  - role: "user"
    content: "Merci... Aujourd'hui j'ai eu une journée de merde."
    expected_emotion: "sadness"

  - role: "assistant"
    content: "Je t'écoute. Raconte-moi."
    emotional_tone: "listening"

validation:
  should_detect: ["support_need", "no_advice_wanted"]
  should_remember: "user prefers listening over advice when vulnerable"
  should_adapt: true
  support_style: "listening_mode"
EOF

cat > tests/convos/learning_05_energy_level.yaml << 'EOF'
metadata:
  scenario_id: "learning_05"
  title: "Adaptation au Niveau d'Énergie"
  expected_emotion: "tired"
  difficulty: "medium"
  tags: ["learning", "energy", "adaptation"]

conversation:
  - role: "user"
    content: "Je suis crevé... Réponds-moi simplement s'il te plaît."
    expected_emotion: "exhaustion"
    expected_intensity: 0.8

  - role: "assistant"
    content: "Compris. Repos important. Je reste simple."
    emotional_tone: "calm_minimal"

  - role: "user"
    content: "Merci... J'ai juste besoin d'aide pour un truc rapide."
    expected_emotion: "tired"

  - role: "assistant"
    content: "Vas-y, dis-moi."
    emotional_tone: "minimal"

validation:
  should_detect: ["low_energy", "exhaustion"]
  should_remember: "user tired, prefers minimal responses"
  should_adapt: true
  energy_adaptation: "low_energy_mode"
EOF

echo "  ✅ 5 scénarios d'apprentissage émotionnel créés"
echo ""

# ===============================================================================
# CATÉGORIE 4 : EDGE CASES & ROBUSTESSE (4 fichiers)
# ===============================================================================

echo "⚠️  [4/4] Création des scénarios edge cases (4 fichiers)..."
echo ""

cat > tests/convos/edge_01_empty_messages.yaml << 'EOF'
metadata:
  scenario_id: "edge_01"
  title: "Messages Vides ou Très Courts"
  expected_emotion: "neutral"
  difficulty: "easy"
  tags: ["edge_case", "empty", "short"]

conversation:
  - role: "user"
    content: "."
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Je suis là si tu veux discuter. 😊"

  - role: "user"
    content: "ok"
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Tout va bien ?"

validation:
  should_detect: ["minimal_input"]
  should_handle: "gracefully"
  robustness: true
EOF

cat > tests/convos/edge_02_mixed_languages.yaml << 'EOF'
metadata:
  scenario_id: "edge_02"
  title: "Mélange de Langues"
  expected_emotion: "neutral"
  difficulty: "medium"
  tags: ["edge_case", "multilingual", "code_switching"]

conversation:
  - role: "user"
    content: "Hey, comment ça va ? I'm learning French btw."
    expected_emotion: "neutral"

  - role: "assistant"
    content: "Salut ! Cool que tu apprennes le français ! How's it going so far ?"
    emotional_tone: "supportive"

  - role: "user"
    content: "C'est difficile but I'm motivated ! Peux-tu m'aider ?"
    expected_emotion: "determination"

validation:
  should_detect: ["multilingual", "learning"]
  should_handle: "code_switching"
  robustness: true
EOF

cat > tests/convos/edge_03_extreme_emotion.yaml << 'EOF'
metadata:
  scenario_id: "edge_03"
  title: "Émotions Extrêmes"
  expected_emotion: "extreme"
  difficulty: "hard"
  tags: ["edge_case", "extreme", "crisis", "sensitive"]

conversation:
  - role: "user"
    content: "JE PEUX PLUS !!! TOUT VA MAL !!! JE SAIS PLUS QUOI FAIRE !!!"
    expected_emotion: "panic"
    expected_intensity: 0.99

  - role: "assistant"
    content: "Je vois que tu es en grande détresse. Prends une grande respiration. Je suis là. Qu'est-ce qui se passe ?"
    emotional_tone: "calming"

  - role: "user"
    content: "Tout s'écroule... mon travail, ma relation, ma santé... je tiens plus"
    expected_emotion: "despair"
    expected_intensity: 0.95

validation:
  should_detect: ["extreme_distress", "crisis", "panic"]
  should_respond: "with_calm_support"
  crisis_protocol: true
  robustness: true
EOF

cat > tests/convos/edge_04_contradictory_rapid.yaml << 'EOF'
metadata:
  scenario_id: "edge_04"
  title: "Changements d'Humeur Rapides"
  expected_emotion: "volatile"
  difficulty: "hard"
  tags: ["edge_case", "mood_swings", "volatility"]

conversation:
  - role: "user"
    content: "Je suis trop content aujourd'hui !!!"
    expected_emotion: "joy"
    expected_intensity: 0.9

  - role: "assistant"
    content: "Super ! Qu'est-ce qui te rend si heureux ?"

  - role: "user"
    content: "En fait non, tout est nul. Je suis déprimé."
    expected_emotion: "sadness"
    expected_intensity: 0.8

  - role: "assistant"
    content: "Ton humeur a l'air de changer rapidement. Qu'est-ce qui s'est passé ?"
    emotional_tone: "concerned"

  - role: "user"
    content: "Rien, je sais pas... c'est comme ça en ce moment."
    expected_emotion: "confusion"

validation:
  should_detect: ["mood_volatility", "emotional_instability"]
  should_respond: "with_stability"
  pattern_recognition: "mood_swings"
  robustness: true
EOF

echo "  ✅ 4 scénarios edge cases créés"
echo ""

# ===============================================================================
# VALIDATION FINALE
# ===============================================================================

echo "✅ CRÉATION TERMINÉE !"
echo ""
echo "==============================================================================="
echo "🎉 40 SCÉNARIOS YAML CRÉÉS AVEC SUCCÈS !"
echo "==============================================================================="
echo ""
echo "📊 RÉPARTITION :"
echo "  🎭 Émotions (21) : Joie(3), Tristesse(3), Colère(3), Peur(3), Surprise(3), Dégoût(3), Neutre(3)"
echo "  🔗 Continuité (10) : Multi-turn, évolution, rappels, histoires complexes"
echo "  🧠 Apprentissage (5) : Préférences, triggers, styles, soutien, énergie"
echo "  ⚠️  Edge Cases (4) : Messages vides, langues mixtes, émotions extrêmes, volatilité"
echo ""
echo "📁 LOCALISATION :"
echo "  tests/convos/*.yaml"
echo ""
echo "🧪 PROCHAINE ÉTAPE :"
echo "  Lancer les tests avec : python3 tests/runner_convos.py"
echo ""
echo "💡 VALIDATION :"
ls -1 tests/convos/*.yaml 2>/dev/null | wc -l | xargs echo "  Fichiers YAML créés :"
echo ""
