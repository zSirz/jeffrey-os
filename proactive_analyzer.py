"""
🎯 Proactive Analyzer - Jeffrey's Pattern Recognition & Initiative Engine
=======================================================================

Analyzes user behavioral and emotional patterns to identify opportunities
for proactive engagement and empathetic intervention.

Features:
- Emotional cycle detection (daily, weekly patterns)
- Topic preference analysis
- Stress trigger identification
- Energy level tracking
- Proactive opportunity identification
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class ProactiveAnalyzer:
    """
    Advanced pattern analyzer for proactive AI behavior

    Identifies behavioral patterns, emotional cycles, and opportunities
    for Jeffrey to take helpful initiative.
    """

    def __init__(self):
        """Initialize the proactive analyzer with pattern definitions"""
        self.pattern_types = {
            'emotional_cycles': 'Cycles émotionnels récurrents',
            'topic_interests': 'Sujets de prédilection',
            'time_patterns': 'Patterns horaires/temporels',
            'stress_triggers': 'Déclencheurs de stress',
            'energy_levels': 'Niveaux d\'énergie',
        }

        # Confidence thresholds for pattern detection
        self.confidence_thresholds = {'minimum_interactions': 5, 'pattern_strength': 0.6, 'proactive_trigger': 0.7}

    def analyze_user_patterns(self, user_memories: list[dict]) -> dict:
        """
        Analyse complète des patterns comportementaux et émotionnels

        Args:
            user_memories: Liste des mémoires utilisateur avec métadonnées

        Returns:
            Dict avec patterns, insights et opportunités proactives
        """
        if not user_memories:
            logger.info("🎯 No user memories available for pattern analysis")
            return {'patterns': {}, 'insights': [], 'proactive_opportunities': [], 'confidence_score': 0.0}

        logger.info(f"🔍 Analyzing patterns from {len(user_memories)} memories")

        # Analyse des différents types de patterns
        patterns = {
            'emotional_cycles': self._detect_emotional_cycles(user_memories),
            'topic_interests': self._analyze_topic_preferences(user_memories),
            'time_patterns': self._detect_time_patterns(user_memories),
            'stress_patterns': self._analyze_stress_patterns(user_memories),
            'energy_patterns': self._analyze_energy_patterns(user_memories),
        }

        # Génération d'insights basés sur les patterns
        insights = self._generate_insights(patterns)

        # Identification des opportunités proactives
        opportunities = self._identify_proactive_opportunities(patterns, insights)

        # Calcul du score de confiance global
        confidence_score = self._calculate_pattern_confidence(patterns, user_memories)

        logger.info(f"🎯 Pattern analysis complete: {len(insights)} insights, {len(opportunities)} opportunities")

        return {
            'patterns': patterns,
            'insights': insights,
            'proactive_opportunities': opportunities,
            'confidence_score': confidence_score,
            'analysis_timestamp': datetime.now().isoformat(),
        }

    def _detect_emotional_cycles(self, memories: list[dict]) -> dict:
        """Détecte les cycles émotionnels (jour, semaine, etc.)"""
        emotional_timeline = []

        # Construire la timeline émotionnelle
        for memory in memories[-30:]:  # Analyser les 30 dernières interactions
            if 'user_emotion' in memory and 'timestamp' in memory:
                try:
                    dt = datetime.fromisoformat(memory['timestamp'])
                    emotional_timeline.append(
                        {
                            'emotion': memory['user_emotion'],
                            'intensity': memory.get('emotion_intensity', 0.5),
                            'hour': dt.hour,
                            'weekday': dt.weekday(),
                            'date': dt.date(),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error parsing timestamp: {e}")
                    continue

        if not emotional_timeline:
            return {'detected': False, 'reason': 'No valid emotional data'}

        # Analyser les patterns horaires
        hourly_emotions = defaultdict(lambda: defaultdict(int))
        weekday_emotions = defaultdict(lambda: defaultdict(int))

        for entry in emotional_timeline:
            hour = entry['hour']
            weekday = entry['weekday']
            emotion = entry['emotion']

            hourly_emotions[hour][emotion] += 1
            weekday_emotions[weekday][emotion] += 1

        # Détecter des patterns spécifiques
        evening_fatigue = self._detect_pattern_by_time(emotional_timeline, 'fatigue', 20, 23)
        monday_stress = self._detect_pattern_by_weekday(emotional_timeline, 'stress', 0)  # Lundi = 0
        afternoon_energy_drop = self._detect_energy_drop_pattern(emotional_timeline)

        # Émotions dominantes par période
        morning_emotion = self._get_dominant_emotion_by_time(emotional_timeline, 7, 11)
        evening_emotion = self._get_dominant_emotion_by_time(emotional_timeline, 20, 23)

        return {
            'detected': True,
            'evening_fatigue_pattern': evening_fatigue,
            'monday_stress_pattern': monday_stress,
            'afternoon_energy_drop': afternoon_energy_drop,
            'hourly_distribution': dict(hourly_emotions),
            'weekday_distribution': dict(weekday_emotions),
            'dominant_morning_emotion': morning_emotion,
            'dominant_evening_emotion': evening_emotion,
            'cycle_strength': self._calculate_cycle_strength(emotional_timeline),
        }

    def _analyze_topic_preferences(self, memories: list[dict]) -> dict:
        """Analyse les préférences de sujets avec scoring émotionnel"""
        topic_frequency = Counter()
        topic_emotions = defaultdict(list)
        topic_engagement = defaultdict(list)

        for memory in memories[-40:]:  # Plus d'historique pour les sujets
            topic = memory.get('topic', 'general')
            emotion = memory.get('user_emotion', 'neutre')
            intensity = memory.get('emotion_intensity', 0.5)

            if topic and topic != 'general':
                topic_frequency[topic] += 1
                topic_emotions[topic].append(emotion)
                topic_engagement[topic].append(intensity)

        # Analyser les sujets favoris
        favorite_topics = []
        for topic, freq in topic_frequency.items():
            if freq >= 2:  # Au moins 2 mentions
                emotions = topic_emotions[topic]
                intensities = topic_engagement[topic]

                # Calculer le ratio d'émotions positives
                positive_emotions = ['joie', 'enthousiasme', 'curiosité', 'surprise', 'fierté']
                positive_count = sum(1 for e in emotions if e in positive_emotions)
                positive_ratio = positive_count / len(emotions) if emotions else 0

                # Calculer l'intensité moyenne
                avg_intensity = sum(intensities) / len(intensities) if intensities else 0

                # Score de préférence combiné
                preference_score = (positive_ratio * 0.6) + (avg_intensity * 0.4)

                if preference_score > 0.3:  # Seuil de préférence
                    favorite_topics.append(
                        {
                            'topic': topic,
                            'frequency': freq,
                            'positive_ratio': positive_ratio,
                            'avg_intensity': avg_intensity,
                            'preference_score': preference_score,
                            'last_emotions': emotions[-3:],  # 3 dernières émotions
                        }
                    )

        # Trier par score de préférence
        favorite_topics.sort(key=lambda x: x['preference_score'], reverse=True)

        # Identifier les sujets évités (émotions négatives récurrentes)
        avoided_topics = []
        for topic, emotions in topic_emotions.items():
            if len(emotions) >= 2:
                negative_emotions = ['stress', 'tristesse', 'colère', 'frustration']
                negative_ratio = sum(1 for e in emotions if e in negative_emotions) / len(emotions)
                if negative_ratio > 0.6:
                    avoided_topics.append(
                        {'topic': topic, 'negative_ratio': negative_ratio, 'frequency': topic_frequency[topic]}
                    )

        return {
            'favorite_topics': favorite_topics[:5],  # Top 5
            'avoided_topics': avoided_topics,
            'topic_frequency': dict(topic_frequency),
            'topic_emotional_mapping': {k: Counter(v) for k, v in topic_emotions.items()},
            'engagement_analysis': self._analyze_topic_engagement_trends(topic_engagement),
        }

    def _detect_time_patterns(self, memories: list[dict]) -> dict:
        """Détecte les patterns temporels d'interaction et d'humeur"""
        interaction_times = []

        for memory in memories[-60:]:  # Plus d'historique pour les patterns temporels
            if 'timestamp' in memory:
                try:
                    dt = datetime.fromisoformat(memory['timestamp'])
                    interaction_times.append(
                        {
                            'hour': dt.hour,
                            'weekday': dt.weekday(),
                            'emotion': memory.get('user_emotion', 'neutre'),
                            'intensity': memory.get('emotion_intensity', 0.5),
                            'date': dt.date(),
                        }
                    )
                except:
                    continue

        if not interaction_times:
            return {'detected': False}

        # Analyser les heures d'interaction
        hourly_count = Counter(i['hour'] for i in interaction_times)
        weekday_count = Counter(i['weekday'] for i in interaction_times)

        # Identifier les heures de pic
        peak_hours = [hour for hour, count in hourly_count.most_common(3)]
        active_weekdays = [day for day, count in weekday_count.most_common(3)]

        # Analyser les patterns d'activité
        morning_activity = sum(1 for i in interaction_times if 6 <= i['hour'] <= 11)
        afternoon_activity = sum(1 for i in interaction_times if 12 <= i['hour'] <= 17)
        evening_activity = sum(1 for i in interaction_times if 18 <= i['hour'] <= 23)
        night_activity = sum(1 for i in interaction_times if i['hour'] >= 0 and i['hour'] <= 5)

        # Calculer la régularité
        daily_interactions = defaultdict(int)
        for interaction in interaction_times:
            daily_interactions[interaction['date']] += 1

        avg_daily_interactions = sum(daily_interactions.values()) / len(daily_interactions) if daily_interactions else 0
        interaction_consistency = (
            len([count for count in daily_interactions.values() if count > 0]) / len(daily_interactions)
            if daily_interactions
            else 0
        )

        return {
            'detected': True,
            'peak_interaction_hours': peak_hours,
            'active_weekdays': active_weekdays,
            'hourly_distribution': dict(hourly_count),
            'weekday_distribution': dict(weekday_count),
            'activity_periods': {
                'morning': morning_activity,
                'afternoon': afternoon_activity,
                'evening': evening_activity,
                'night': night_activity,
            },
            'average_daily_interactions': avg_daily_interactions,
            'interaction_consistency': interaction_consistency,
            'most_active_period': max(
                [
                    ('morning', morning_activity),
                    ('afternoon', afternoon_activity),
                    ('evening', evening_activity),
                    ('night', night_activity),
                ],
                key=lambda x: x[1],
            )[0],
        }

    def _analyze_stress_patterns(self, memories: list[dict]) -> dict:
        """Analyse détaillée des patterns de stress"""
        stress_events = []
        stress_triggers = defaultdict(int)
        stress_contexts = []

        negative_emotions = ['stress', 'colère', 'tristesse', 'frustration']

        for memory in memories[-40:]:
            emotion = memory.get('user_emotion')
            if emotion in negative_emotions:
                stress_events.append(
                    {
                        'emotion': emotion,
                        'topic': memory.get('topic', 'unknown'),
                        'timestamp': memory.get('timestamp'),
                        'intensity': memory.get('emotion_intensity', 0.5),
                        'user_input': memory.get('user_input', '')[:100],  # Premier 100 chars
                    }
                )

                # Analyser les déclencheurs
                topic = memory.get('topic', 'unknown')
                stress_triggers[topic] += 1

        if not stress_events:
            return {'detected': False, 'stress_frequency': 0, 'recent_stress_level': 0.0}

        # Analyser la fréquence et les tendances
        stress_frequency = len(stress_events)
        recent_stress_events = stress_events[-7:]  # 7 derniers événements
        recent_stress_level = self._calculate_recent_stress_level(recent_stress_events)

        # Identifier les patterns temporels de stress
        stress_times = []
        for event in stress_events:
            if event['timestamp']:
                try:
                    dt = datetime.fromisoformat(event['timestamp'])
                    stress_times.append({'hour': dt.hour, 'weekday': dt.weekday(), 'intensity': event['intensity']})
                except:
                    continue

        stress_hours = Counter(st['hour'] for st in stress_times)
        stress_weekdays = Counter(st['weekday'] for st in stress_times)

        # Calculer l'évolution du stress
        stress_trend = self._calculate_stress_trend(stress_events)

        return {
            'detected': True,
            'stress_frequency': stress_frequency,
            'recent_stress_level': recent_stress_level,
            'stress_triggers': dict(stress_triggers),
            'stress_hours': dict(stress_hours),
            'stress_weekdays': dict(stress_weekdays),
            'stress_trend': stress_trend,
            'peak_stress_times': stress_hours.most_common(2),
            'most_stressful_topics': sorted(stress_triggers.items(), key=lambda x: x[1], reverse=True)[:3],
        }

    def _analyze_energy_patterns(self, memories: list[dict]) -> dict:
        """Analyse des patterns d'énergie basés sur les émotions"""
        energy_timeline = []

        for memory in memories[-25:]:
            emotion = memory.get('user_emotion', 'neutre')
            timestamp = memory.get('timestamp')
            intensity = memory.get('emotion_intensity', 0.5)

            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    energy_level = self._emotion_to_energy_level(emotion, intensity)
                    energy_timeline.append(
                        {
                            'energy': energy_level,
                            'hour': dt.hour,
                            'weekday': dt.weekday(),
                            'emotion': emotion,
                            'date': dt.date(),
                        }
                    )
                except:
                    continue

        if not energy_timeline:
            return {'detected': False}

        # Calculer les moyennes d'énergie par période
        periods = {
            'morning': [e for e in energy_timeline if 6 <= e['hour'] <= 11],
            'afternoon': [e for e in energy_timeline if 12 <= e['hour'] <= 17],
            'evening': [e for e in energy_timeline if 18 <= e['hour'] <= 23],
            'night': [e for e in energy_timeline if 0 <= e['hour'] <= 5],
        }

        period_averages = {}
        for period, entries in periods.items():
            if entries:
                period_averages[period] = sum(e['energy'] for e in entries) / len(entries)
            else:
                period_averages[period] = 0.5  # Valeur neutre

        # Détecter les patterns d'énergie
        energy_decline_pattern = self._detect_energy_decline(energy_timeline)
        energy_peak_period = max(period_averages.items(), key=lambda x: x[1])[0]
        energy_low_period = min(period_averages.items(), key=lambda x: x[1])[0]

        # Analyser les variations par jour de la semaine
        weekday_energy = defaultdict(list)
        for entry in energy_timeline:
            weekday_energy[entry['weekday']].append(entry['energy'])

        weekday_averages = {
            day: sum(energies) / len(energies) if energies else 0.5 for day, energies in weekday_energy.items()
        }

        return {
            'detected': True,
            'period_averages': period_averages,
            'energy_decline_pattern': energy_decline_pattern,
            'energy_peak_period': energy_peak_period,
            'energy_low_period': energy_low_period,
            'weekday_energy_averages': weekday_averages,
            'overall_energy_trend': self._calculate_energy_trend(energy_timeline),
            'energy_volatility': self._calculate_energy_volatility(energy_timeline),
        }

    def _generate_insights(self, patterns: dict) -> list[str]:
        """Génère des insights personnalisés basés sur les patterns"""
        insights = []

        # Insights sur les cycles émotionnels
        emotional_cycles = patterns.get('emotional_cycles', {})
        if emotional_cycles.get('evening_fatigue_pattern', False):
            insights.append("Tu sembles souvent fatigué le soir - peut-être que tu pousses tes journées un peu trop ?")

        if emotional_cycles.get('monday_stress_pattern', False):
            insights.append(
                "Les lundis semblent te stresser plus que les autres jours - le début de semaine n'est pas ton moment préféré !"
            )

        if emotional_cycles.get('afternoon_energy_drop', False):
            insights.append("Ton énergie a tendance à chuter l'après-midi - typique du fameux coup de barre !")

        # Insights sur les préférences de sujets
        topic_interests = patterns.get('topic_interests', {})
        favorite_topics = topic_interests.get('favorite_topics', [])
        if favorite_topics:
            top_topic = favorite_topics[0]['topic']
            score = favorite_topics[0]['preference_score']
            if score > 0.7:
                insights.append(f"Tu adores vraiment explorer le sujet '{top_topic}' - ça te passionne à chaque fois !")
            elif score > 0.5:
                insights.append(f"Le sujet '{top_topic}' t'intéresse beaucoup - tu y reviens souvent avec plaisir")

        avoided_topics = topic_interests.get('avoided_topics', [])
        if avoided_topics:
            avoided_topic = avoided_topics[0]['topic']
            insights.append(f"Le sujet '{avoided_topic}' semble te mettre mal à l'aise - on peut l'éviter si tu veux !")

        # Insights sur les patterns temporels
        time_patterns = patterns.get('time_patterns', {})
        if time_patterns.get('detected', False):
            most_active = time_patterns.get('most_active_period', '')
            if most_active == 'evening':
                insights.append("Tu es clairement plus actif en soirée - un vrai oiseau de nuit !")
            elif most_active == 'morning':
                insights.append("Tu es du matin ! Ça se voit que tu as plus d'énergie en début de journée")
            elif most_active == 'night':
                insights.append("Tu es actif même la nuit - attention à ton sommeil quand même !")

        # Insights sur le stress
        stress_patterns = patterns.get('stress_patterns', {})
        if stress_patterns.get('recent_stress_level', 0) > 0.6:
            insights.append("Tu sembles un peu stressé ces derniers temps - tout va bien ?")

        peak_stress_times = stress_patterns.get('peak_stress_times', [])
        if peak_stress_times:
            peak_hour = peak_stress_times[0][0]
            if 14 <= peak_hour <= 17:
                insights.append("Tes pics de stress arrivent souvent en milieu d'après-midi")

        # Insights sur l'énergie
        energy_patterns = patterns.get('energy_patterns', {})
        if energy_patterns.get('detected', False):
            low_period = energy_patterns.get('energy_low_period', '')
            if low_period == 'evening':
                insights.append("Ton énergie décline nettement en soirée - normal après une bonne journée !")
            elif low_period == 'afternoon':
                insights.append("Tu as un creux d'énergie l'après-midi - classique du post-lunch dip !")

        return insights

    def _identify_proactive_opportunities(self, patterns: dict, insights: list[str]) -> list[dict]:
        """Identifie les opportunités d'intervention proactive"""
        opportunities = []

        # Opportunités basées sur la fatigue du soir
        emotional_cycles = patterns.get('emotional_cycles', {})
        if emotional_cycles.get('evening_fatigue_pattern', False):
            opportunities.append(
                {
                    'type': 'wellness_check',
                    'trigger_condition': 'hour >= 20 and recent_fatigue',
                    'message_template': "David, je remarque que tu es souvent fatigué le soir... Tu veux qu'on explore des moyens de mieux gérer ton énergie ?",
                    'priority': 'high',
                    'category': 'well_being',
                    'confidence': 0.8,
                }
            )

        # Opportunités préventives pour le stress du lundi
        if emotional_cycles.get('monday_stress_pattern', False):
            opportunities.append(
                {
                    'type': 'preventive_support',
                    'trigger_condition': 'monday_morning',
                    'message_template': "Salut David ! Pour bien commencer cette semaine, tu veux qu'on explore quelque chose d'inspirant ?",
                    'priority': 'medium',
                    'category': 'preventive_care',
                    'confidence': 0.7,
                }
            )

        # Opportunités basées sur les sujets favoris
        topic_interests = patterns.get('topic_interests', {})
        favorite_topics = topic_interests.get('favorite_topics', [])
        if favorite_topics:
            top_topic = favorite_topics[0]
            if top_topic['preference_score'] > 0.6:
                opportunities.append(
                    {
                        'type': 'topic_expansion',
                        'trigger_condition': f'no_interaction_24h and topic_interest_{top_topic["topic"]}',
                        'message_template': f"David, ça fait un moment qu'on n'a pas exploré {top_topic['topic']}... J'ai découvert quelque chose de fascinant à ce sujet !",
                        'priority': 'medium',
                        'category': 'engagement',
                        'confidence': top_topic['preference_score'],
                        'topic': top_topic['topic'],
                    }
                )

        # Opportunités anti-stress
        stress_patterns = patterns.get('stress_patterns', {})
        if stress_patterns.get('recent_stress_level', 0) > 0.6:
            opportunities.append(
                {
                    'type': 'stress_relief',
                    'trigger_condition': 'high_stress_detected',
                    'message_template': "David, tu sembles un peu tendu... Envie d'une pause détente avec un sujet fascinant ?",
                    'priority': 'high',
                    'category': 'emotional_support',
                    'confidence': stress_patterns['recent_stress_level'],
                }
            )

        # Opportunités basées sur les creux d'énergie
        energy_patterns = patterns.get('energy_patterns', {})
        if energy_patterns.get('detected', False):
            if energy_patterns.get('energy_low_period') == 'afternoon':
                opportunities.append(
                    {
                        'type': 'energy_boost',
                        'trigger_condition': 'afternoon_energy_low',
                        'message_template': "David, petit coup de mou de l'après-midi ? Ça te dit qu'on réveille ta curiosité ?",
                        'priority': 'medium',
                        'category': 'energy_support',
                        'confidence': 0.6,
                    }
                )

        # Opportunités de réengagement après silence
        time_patterns = patterns.get('time_patterns', {})
        if time_patterns.get('interaction_consistency', 0) > 0.5:
            opportunities.append(
                {
                    'type': 'gentle_reengagement',
                    'trigger_condition': 'no_interaction_48h',
                    'message_template': "Hello David ! Ça fait un petit moment... Tu vas bien ? J'ai quelques trucs intéressants en réserve si ça te dit !",
                    'priority': 'low',
                    'category': 'reconnection',
                    'confidence': 0.5,
                }
            )

        # Trier par priorité et confiance
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        opportunities.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['confidence']), reverse=True)

        return opportunities

    # === MÉTHODES UTILITAIRES ===

    def _detect_pattern_by_time(self, timeline: list[dict], emotion: str, start_hour: int, end_hour: int) -> bool:
        """Détecte si une émotion est récurrente à certaines heures"""
        time_filtered = [e for e in timeline if start_hour <= e['hour'] <= end_hour]
        if len(time_filtered) < 3:
            return False

        emotion_count = sum(1 for e in time_filtered if e['emotion'] == emotion)
        return emotion_count / len(time_filtered) > 0.4  # 40% du temps

    def _detect_pattern_by_weekday(self, timeline: list[dict], emotion: str, weekday: int) -> bool:
        """Détecte si une émotion est récurrente un jour spécifique"""
        weekday_filtered = [e for e in timeline if e['weekday'] == weekday]
        if len(weekday_filtered) < 2:
            return False

        emotion_count = sum(1 for e in weekday_filtered if e['emotion'] == emotion)
        return emotion_count / len(weekday_filtered) > 0.5  # 50% du temps

    def _detect_energy_drop_pattern(self, timeline: list[dict]) -> bool:
        """Détecte un pattern de chute d'énergie l'après-midi"""
        afternoon_emotions = [e for e in timeline if 14 <= e['hour'] <= 17]
        if len(afternoon_emotions) < 3:
            return False

        fatigue_count = sum(1 for e in afternoon_emotions if e['emotion'] in ['fatigue', 'ennui'])
        return fatigue_count / len(afternoon_emotions) > 0.3

    def _get_dominant_emotion_by_time(self, timeline: list[dict], start_hour: int, end_hour: int) -> str | None:
        """Trouve l'émotion dominante pour une tranche horaire"""
        filtered = [e for e in timeline if start_hour <= e['hour'] <= end_hour]
        if not filtered:
            return None

        emotion_counts = Counter(e['emotion'] for e in filtered)
        return emotion_counts.most_common(1)[0][0] if emotion_counts else None

    def _calculate_cycle_strength(self, timeline: list[dict]) -> float:
        """Calcule la force des cycles émotionnels"""
        if len(timeline) < 5:
            return 0.0

        # Analyser la variabilité émotionnelle
        emotions = [e['emotion'] for e in timeline]
        emotion_variety = len(set(emotions)) / len(emotions)

        # Analyser la cohérence temporelle
        hourly_emotions = defaultdict(list)
        for entry in timeline:
            hourly_emotions[entry['hour']].append(entry['emotion'])

        hour_consistency = 0
        for hour, hour_emotions in hourly_emotions.items():
            if len(hour_emotions) > 1:
                most_common = Counter(hour_emotions).most_common(1)[0][1]
                consistency = most_common / len(hour_emotions)
                hour_consistency += consistency

        hour_consistency = hour_consistency / len(hourly_emotions) if hourly_emotions else 0

        return (emotion_variety * 0.4) + (hour_consistency * 0.6)

    def _analyze_topic_engagement_trends(self, topic_engagement: dict) -> dict:
        """Analyse les tendances d'engagement par sujet"""
        trends = {}

        for topic, intensities in topic_engagement.items():
            if len(intensities) >= 3:
                # Calculer la tendance (croissante, stable, décroissante)
                recent_avg = sum(intensities[-3:]) / 3
                older_avg = sum(intensities[:-3]) / max(len(intensities[:-3]), 1)

                if recent_avg > older_avg + 0.1:
                    trend = 'increasing'
                elif recent_avg < older_avg - 0.1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'

                trends[topic] = {
                    'trend': trend,
                    'recent_avg': recent_avg,
                    'overall_avg': sum(intensities) / len(intensities),
                }

        return trends

    def _calculate_recent_stress_level(self, stress_events: list[dict]) -> float:
        """Calcule le niveau de stress récent"""
        if not stress_events:
            return 0.0

        # Pondérer par la récence et l'intensité
        total_weighted_stress = 0
        total_weight = 0

        for i, event in enumerate(stress_events):
            recency_weight = (i + 1) / len(stress_events)  # Plus récent = plus de poids
            intensity = event.get('intensity', 0.5)
            weighted_stress = intensity * recency_weight

            total_weighted_stress += weighted_stress
            total_weight += recency_weight

        return min(total_weighted_stress / total_weight if total_weight > 0 else 0.0, 1.0)

    def _calculate_stress_trend(self, stress_events: list[dict]) -> str:
        """Calcule la tendance du stress (croissant/stable/décroissant)"""
        if len(stress_events) < 4:
            return 'insufficient_data'

        # Comparer les niveaux récents vs plus anciens
        recent_intensities = [e['intensity'] for e in stress_events[-3:]]
        older_intensities = [e['intensity'] for e in stress_events[:-3]]

        recent_avg = sum(recent_intensities) / len(recent_intensities)
        older_avg = sum(older_intensities) / len(older_intensities)

        if recent_avg > older_avg + 0.15:
            return 'increasing'
        elif recent_avg < older_avg - 0.15:
            return 'decreasing'
        else:
            return 'stable'

    def _emotion_to_energy_level(self, emotion: str, intensity: float = 0.5) -> float:
        """Convertit une émotion en niveau d'énergie"""
        base_energy_mapping = {
            'enthousiasme': 0.9,
            'joie': 0.8,
            'curiosité': 0.7,
            'surprise': 0.6,
            'fierté': 0.7,
            'neutre': 0.5,
            'stress': 0.4,
            'colère': 0.6,  # Énergie négative mais élevée
            'tristesse': 0.3,
            'fatigue': 0.2,
            'ennui': 0.3,
            'frustration': 0.4,
        }

        base_energy = base_energy_mapping.get(emotion, 0.5)

        # Ajuster par l'intensité
        if emotion in ['enthousiasme', 'joie', 'curiosité']:
            # Émotions positives : intensité augmente l'énergie
            return min(base_energy + (intensity * 0.2), 1.0)
        elif emotion in ['fatigue', 'tristesse', 'ennui']:
            # Émotions basses : intensité diminue l'énergie
            return max(base_energy - (intensity * 0.2), 0.0)
        else:
            return base_energy

    def _detect_energy_decline(self, timeline: list[dict]) -> bool:
        """Détecte un pattern de déclin d'énergie"""
        if len(timeline) < 8:
            return False

        # Comparer l'énergie récente vs plus ancienne
        recent_energy = [e['energy'] for e in timeline[-4:]]
        older_energy = [e['energy'] for e in timeline[-8:-4]]

        recent_avg = sum(recent_energy) / len(recent_energy)
        older_avg = sum(older_energy) / len(older_energy)

        return recent_avg < older_avg - 0.15  # Déclin significatif

    def _calculate_energy_trend(self, timeline: list[dict]) -> str:
        """Calcule la tendance énergétique"""
        if len(timeline) < 6:
            return 'insufficient_data'

        # Analyser la tendance sur les dernières mesures
        energies = [e['energy'] for e in timeline]

        # Calculer une régression linéaire simple
        n = len(energies)
        x_mean = (n - 1) / 2
        y_mean = sum(energies) / n

        numerator = sum((i - x_mean) * (energies[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 'stable'

        slope = numerator / denominator

        if slope > 0.02:
            return 'increasing'
        elif slope < -0.02:
            return 'decreasing'
        else:
            return 'stable'

    def _calculate_energy_volatility(self, timeline: list[dict]) -> float:
        """Calcule la volatilité énergétique"""
        if len(timeline) < 3:
            return 0.0

        energies = [e['energy'] for e in timeline]
        mean_energy = sum(energies) / len(energies)

        variance = sum((e - mean_energy) ** 2 for e in energies) / len(energies)
        return min(variance**0.5, 1.0)  # Écart-type normalisé

    def _calculate_pattern_confidence(self, patterns: dict, memories: list[dict]) -> float:
        """Calcule un score de confiance global pour les patterns"""
        confidence_factors = []

        # Facteur basé sur la quantité de données
        data_quantity_factor = min(len(memories) / 30, 1.0)  # 30 interactions = confiance max
        confidence_factors.append(data_quantity_factor)

        # Facteur basé sur la détection de cycles émotionnels
        emotional_cycles = patterns.get('emotional_cycles', {})
        if emotional_cycles.get('detected', False):
            cycle_strength = emotional_cycles.get('cycle_strength', 0)
            confidence_factors.append(cycle_strength)

        # Facteur basé sur les préférences de sujets
        topic_interests = patterns.get('topic_interests', {})
        if topic_interests.get('favorite_topics'):
            topic_confidence = topic_interests['favorite_topics'][0]['preference_score']
            confidence_factors.append(topic_confidence)

        # Facteur basé sur la régularité des interactions
        time_patterns = patterns.get('time_patterns', {})
        if time_patterns.get('detected', False):
            consistency = time_patterns.get('interaction_consistency', 0)
            confidence_factors.append(consistency)

        # Facteur basé sur la détection de stress
        stress_patterns = patterns.get('stress_patterns', {})
        if stress_patterns.get('detected', False):
            stress_confidence = min(stress_patterns.get('stress_frequency', 0) / 10, 1.0)
            confidence_factors.append(stress_confidence)

        # Calculer la moyenne pondérée
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.2  # Confiance minimale
