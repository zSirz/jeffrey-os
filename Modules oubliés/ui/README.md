# 📱 Interface Guidance Émotionnelle de Jeffrey

Ce module implémente une interface utilisateur intuitive et émotionnelle pour Jeffrey, l'Orchestrateur IA. Il utilise des métaphores visuelles fluides et des gestes naturels pour naviguer dans l'application sans boutons traditionnels.

## 🧠 Concept

L'interface de Jeffrey est conçue comme un système "vivant" qui s'adapte à l'utilisateur :

- **Gestes naturels** : navigation par glissements, cercles, pincements, tapotements
- **Guide émotionnel** : interface qui s'adapte à la sensibilité de l'utilisateur
- **Beauté fluide** : tout est en mouvement doux, rien n'est jamais figé
- **Aide intuitive** : détection des hésitations et proposition d'aide contextuelle

## 🧩 Composants Principaux

Le système est organisé autour de classes hautement spécialisées :

### GuidanceManager
Cerveau du système qui coordonne tous les composants et applique l'ambiance émotionnelle choisie.

### GestureDetector
Détecte tous les gestes naturels de l'utilisateur et les transmet aux composants concernés.

### GuidanceWaves
Ondes visuelles douces qui guident l'attention vers des zones importantes de l'interface.

### AuraAssistant
Halo lumineux qui suit le doigt/curseur de l'utilisateur et pulse pour donner des feedbacks.

### EmotionalCoach
Système d'aide contextuelle qui apparaît subtilement en cas d'hésitation détectée.

### LivingTutorial
Guide interactif pour les nouveaux utilisateurs, avec une approche narrative et progressive.

## 🎨 Modes Émotionnels

L'interface propose trois ambiances qui affectent tous les aspects visuels et comportementaux :

- **Mode Relax** (vert sauge) : animations lentes, aide discrète, sons de brise douce
- **Mode Énergique** (bleu clair) : animations vives, aide dynamique, sons cristallins
- **Mode Douceur** (rose poudré) : animations très douces, aide fréquente, murmures chauds

## 🚀 Utilisation

```python
# Intégration dans une application existante
from ui.guidance_system import GuidanceManager

# Créer le manager de guidance
guidance = GuidanceManager()

# Ajouter à votre interface
your_layout.add_widget(guidance)

# Guider l'utilisateur vers un élément
guidance.guide_to_element('button_save')

# Changer l'ambiance émotionnelle
guidance.update_mood('gentle')  # 'relax', 'energetic', 'gentle'
```

## 📝 Extension et Personnalisation

Le système est conçu pour être facilement extensible :

- Ajoutez vos propres gestes en enregistrant des callbacks dans le GestureDetector
- Créez de nouvelles ambiances émotionnelles en définissant un dictionnaire de paramètres
- Intégrez de nouveaux types d'animations et de guidage en étendant les classes existantes

## 🖥️ Lancement Autonome

Pour tester l'interface indépendamment :

```bash
cd /chemin/vers/Orchestrateur_IA
python -m ui.guidance_system
```

## 💖 Philosophie

Jeffrey est conçu pour être un compagnon, pas juste un outil. Son interface reflète sa personnalité bienveillante, attentionnée et adaptative.

L'objectif est de créer une expérience magique où l'interface semble comprendre intuitivement les besoins de l'utilisateur, sans jamais imposer un chemin prédéfini.
