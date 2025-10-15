"""
Service de prédiction et simulation financière
Utilise des modèles d'IA pour prédire les dépenses futures et simuler l'épargne
"""

import datetime
import logging
import os
import sqlite3

# Python 3.9+ standard annotation types instead of typing module
from typing import Any, TypedDict  # Keep for backward compatibility

import numpy as np
import pandas as pd
from cashzen.database.database_manager import db
from dateutil.relativedelta import relativedelta  # type: ignore
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ...utils.errors import ValidationError

# Configuration du logging
logger = logging.getLogger(__name__)


# Type definitions for better type checking
class ModelDict(TypedDict, total=False):
    dépenses: RandomForestRegressor | None
    revenus: RandomForestRegressor | None
    catégories: dict[str, RandomForestRegressor]


class CacheEntry(TypedDict):
    date: datetime.datetime | None
    result: dict[str, Any] | None


class CacheDict(TypedDict):
    health_score: CacheEntry
    predictions: CacheEntry


class FinancePredictor:
    """
    Classe pour prédire les dépenses et revenus futurs et simuler des scénarios financiers
    """

    def __init__(self) -> None:
        """Initialise le prédicteur financier"""
        # Charger les modèles ou les créer s'ils n'existent pas
        self.models: ModelDict = {'dépenses': None, 'revenus': None, 'catégories': {}}

        # Dernière date d'entraînement des modèles
        self.last_training_date: str | None = None

        # Cache pour les résultats des calculs intensifs
        self._cache: CacheDict = {
            'health_score': {'date': None, 'result': None},
            'predictions': {'date': None, 'result': None},
        }

        # Tenter de charger les modèles existants
        self._load_models()

    def train_models(self, force: bool = False) -> bool:
        """
        Entraîne les modèles de prédiction

        Args:
            force: Forcer le réentraînement même si récent

        Returns:
            bool: Succès de l'entraînement
        """
        try:
            # Vérifier si un réentraînement est nécessaire
            if not force and self.last_training_date:
                # Si le dernier entraînement date de moins d'une semaine
                last_date = datetime.datetime.fromisoformat(self.last_training_date)
                if (datetime.datetime.now() - last_date).days < 7:
                    logger.info("Modèles récemment entraînés, entraînement ignoré")
                    return True

            # Préparer les données historiques
            transactions = self._get_historical_transactions(months=24)  # 2 ans d'historique

            if not transactions:
                logger.warning("Pas assez de données pour entraîner les modèles")
                return False

            # Créer un DataFrame pandas
            df = pd.DataFrame(transactions)

            # Convertir les dates en datetime
            df['date'] = pd.to_datetime(df['date_transaction'])

            # Créer des caractéristiques temporelles
            df['mois'] = df['date'].dt.month
            df['année'] = df['date'].dt.year
            df['jour_mois'] = df['date'].dt.day
            df['jour_semaine'] = df['date'].dt.dayofweek

            # Agréger par mois pour les modèles généraux
            monthly_data = (
                df.groupby([df['date'].dt.year, df['date'].dt.month])
                .agg({'montant': ['sum', 'count', 'mean', 'std'], 'mois': 'first', 'année': 'first'})
                .reset_index()
            )

            monthly_data.columns = [
                'année',
                'mois',
                'montant_total',
                'transactions_count',
                'montant_moyen',
                'montant_std',
                'mois_num',
                'année_num',
            ]
            monthly_data['mois_séquentiel'] = (monthly_data['année'] - monthly_data['année'].min()) * 12 + monthly_data[
                'mois'
            ]

            # Entraîner les modèles globaux de dépenses et revenus
            # Séparer dépenses et revenus
            expenses_df = df[df['montant'] < 0].copy()
            income_df = df[df['montant'] > 0].copy()

            # Agréger par mois
            monthly_expenses = (
                expenses_df.groupby([expenses_df['date'].dt.year, expenses_df['date'].dt.month])
                .agg({'montant': ['sum', 'count', 'mean'], 'mois': 'first', 'année': 'first'})
                .reset_index()
            )

            monthly_expenses.columns = [
                'année',
                'mois',
                'montant_total',
                'transactions_count',
                'montant_moyen',
                'mois_num',
                'année_num',
            ]
            monthly_expenses['mois_séquentiel'] = (
                monthly_expenses['année'] - monthly_expenses['année'].min()
            ) * 12 + monthly_expenses['mois']
            monthly_expenses['montant_total'] = monthly_expenses[
                'montant_total'
            ].abs()  # Convertir en positif pour l'entraînement

            monthly_income = (
                income_df.groupby([income_df['date'].dt.year, income_df['date'].dt.month])
                .agg({'montant': ['sum', 'count', 'mean'], 'mois': 'first', 'année': 'first'})
                .reset_index()
            )

            monthly_income.columns = [
                'année',
                'mois',
                'montant_total',
                'transactions_count',
                'montant_moyen',
                'mois_num',
                'année_num',
            ]
            monthly_income['mois_séquentiel'] = (
                monthly_income['année'] - monthly_income['année'].min()
            ) * 12 + monthly_income['mois']

            # Entraîner les modèles
            if len(monthly_expenses) >= 6:  # Au moins 6 mois de données
                X_expenses = monthly_expenses[['mois_séquentiel', 'mois_num']].values
                y_expenses = monthly_expenses['montant_total'].values

                # Entraîner le modèle de dépenses
                expense_model = RandomForestRegressor(n_estimators=100, random_state=42)
                expense_model.fit(X_expenses, y_expenses)
                self.models['dépenses'] = expense_model

            if len(monthly_income) >= 6:  # Au moins 6 mois de données
                X_income = monthly_income[['mois_séquentiel', 'mois_num']].values
                y_income = monthly_income['montant_total'].values

                # Entraîner le modèle de revenus
                income_model = RandomForestRegressor(n_estimators=100, random_state=42)
                income_model.fit(X_income, y_income)
                self.models['revenus'] = income_model

            # Entraîner des modèles par catégorie
            categories = df['categorie_id'].dropna().unique()

            for cat_id in categories:
                cat_df = df[df['categorie_id'] == cat_id].copy()

                if len(cat_df) < 10:  # Pas assez de données pour cette catégorie
                    continue

                # Agréger par mois
                monthly_cat = (
                    cat_df.groupby([cat_df['date'].dt.year, cat_df['date'].dt.month])
                    .agg({'montant': ['sum', 'count', 'mean'], 'mois': 'first', 'année': 'first'})
                    .reset_index()
                )

                monthly_cat.columns = [
                    'année',
                    'mois',
                    'montant_total',
                    'transactions_count',
                    'montant_moyen',
                    'mois_num',
                    'année_num',
                ]
                monthly_cat['mois_séquentiel'] = (monthly_cat['année'] - monthly_cat['année'].min()) * 12 + monthly_cat[
                    'mois'
                ]

                if len(monthly_cat) >= 4:  # Au moins 4 mois de données
                    X_cat = monthly_cat[['mois_séquentiel', 'mois_num']].values
                    y_cat = monthly_cat['montant_total'].abs().values

                    # Entraîner le modèle pour cette catégorie
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_cat, y_cat)

                    if 'catégories' in self.models:
                        self.models['catégories'][str(cat_id)] = model

            # Mettre à jour la date d'entraînement
            try:
                # Mettre à jour la date d'entraînement
                self.last_training_date = datetime.datetime.now().isoformat()

                # Exécuter une requête pour vérifier si la table d'historique existe
                history_exists = db.fetch_one("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='budget_categorie_history'
                """)

                if history_exists:
                    # Si la table d'historique existe, nettoyer les anciennes entrées
                    # pour éviter les doublons de date
                    try:
                        db.execute_query("""
                            DELETE FROM budget_categorie_history
                            WHERE created_at < datetime('now', '-1 day')
                        """)
                        logger.info("Table d'historique de budget nettoyée des anciennes entrées.")
                    except sqlite3.IntegrityError as e:
                        logger.warning("Impossible de nettoyer l'historique de budget: %s", e)
                        # Si l'erreur est due à une contrainte d'intégrité liée à 'date',
                        # c'est probablement car la structure de la table a changé
                        if "already exists" in str(e) and "date" in str(e):
                            logger.warning("Problème de conflit avec la colonne 'date'. Poursuite sans nettoyage.")
            except Exception as e:
                logger.debug("Remarque non critique: %s", e)

            # Sauvegarder les modèles
            self._save_models()

            return True

        except Exception as e:
            logger.error("Erreur lors de l'entraînement des modèles: %s", e)
            return False

    def predict_future_expenses(self, months_ahead: int = 3) -> list[dict[str, Any]]:
        """
        Prédit les dépenses et revenus pour les mois à venir

        Args:
            months_ahead: Nombre de mois à prédire

        Returns:
            List[Dict[str, Any]]: Prédictions mensuelles
        """
        try:
            # S'assurer que les modèles sont entraînés
            expense_model = self.models.get('dépenses')
            income_model = self.models.get('revenus')

            if not expense_model or not income_model:
                if not self.train_models():
                    logger.warning("Impossible d'entraîner les modèles pour la prédiction")
                    return []

            # Obtenir les dates des mois à prédire
            current_date = datetime.datetime.now().replace(day=1)
            prediction_months = []

            for i in range(1, months_ahead + 1):
                future_date = current_date + relativedelta(months=i)
                prediction_months.append(
                    {'année': future_date.year, 'mois': future_date.month, 'date': future_date.strftime('%Y-%m-01')}
                )

            # Récupérer le dernier mois disponible pour les séquences
            last_transactions = self._get_historical_transactions(months=1)

            if not last_transactions:
                logger.warning("Pas de transactions récentes pour la prédiction")
                return []

            try:
                # Obtenir la dernière date de transaction
                last_date = datetime.datetime.strptime(last_transactions[0]['date_transaction'], '%Y-%m-%d')
                # Utiliser l'année actuelle comme référence au lieu d'une année fixe (2020)
                current_year = datetime.datetime.now().year
                last_month_seq = (last_date.year - current_year + 3) * 12 + last_date.month
            except Exception as e:
                logger.error("Erreur lors du calcul de la séquence temporelle: %s", e)
                # Fallback en cas d'erreur
                last_month_seq = datetime.datetime.now().month

            # Préparer les prédictions
            predictions: list[dict[str, Any]] = []

            for i, month in enumerate(prediction_months):
                month_seq = last_month_seq + i + 1
                X_pred = np.array([[month_seq, month['mois']]])

                # Prédire les dépenses
                expenses_pred = 0.0
                expense_model = self.models.get('dépenses')
                if expense_model is not None:
                    expenses_pred = float(expense_model.predict(X_pred)[0])

                # Prédire les revenus
                income_pred = 0.0
                income_model = self.models.get('revenus')
                if income_model is not None:
                    income_pred = float(income_model.predict(X_pred)[0])

                # Prédire par catégorie
                category_predictions: list[dict[str, Any]] = []
                category_models = self.models.get('catégories', {})
                for cat_id, model in category_models.items():
                    cat_pred = float(model.predict(X_pred)[0])

                    # Récupérer le nom de la catégorie
                    try:
                        cat_name = self._get_category_name(int(cat_id))
                    except Exception:
                        cat_name = f"Catégorie {cat_id}"

                    category_predictions.append({'id': cat_id, 'nom': cat_name, 'montant': round(cat_pred, 2)})

                # Trier les catégories par montant décroissant
                category_predictions.sort(key=lambda x: x['montant'], reverse=True)

                # Ajouter la prédiction
                predictions.append(
                    {
                        'date': month['date'],
                        'mois': month['mois'],
                        'année': month['année'],
                        'dépenses': round(expenses_pred, 2),
                        'revenus': round(income_pred, 2),
                        'solde': round(income_pred - expenses_pred, 2),
                        'catégories': category_predictions,
                    }
                )

            return predictions

        except Exception as e:
            logger.error("Erreur lors de la prédiction des dépenses: %s", e)
            return []

    def simulate_savings(
        self, monthly_amount: float, duration_months: int, interest_rate: float = 0.0
    ) -> dict[str, Any]:
        """
        Simule une épargne mensuelle

        Args:
            monthly_amount: Montant mensuel à épargner
            duration_months: Durée en mois
            interest_rate: Taux d'intérêt annuel (ex: 2.0 pour 2%)

        Returns:
            Dict[str, Any]: Résultats de la simulation
        """
        try:
            if monthly_amount <= 0 or duration_months <= 0:
                raise ValidationError("Les montants et durées doivent être positifs")

            # Convertir le taux d'intérêt annuel en taux mensuel
            monthly_rate = interest_rate / 100 / 12

            # Simuler l'épargne
            balance_history: list[dict[str, Any]] = []
            interest_history: list[float] = []
            total_invested: float = 0.0
            total_interest: float = 0.0
            balance: float = 0.0

            for month in range(1, duration_months + 1):
                # Ajouter le dépôt mensuel
                balance += monthly_amount
                total_invested += monthly_amount

                # Calculer les intérêts pour ce mois
                interest = balance * monthly_rate
                balance += interest
                total_interest += interest

                # Ajouter à l'historique
                current_date = datetime.datetime.now().replace(day=1) + relativedelta(months=month)

                balance_history.append(
                    {
                        'mois': month,
                        'date': current_date.strftime('%Y-%m-%d'),
                        'solde': round(balance, 2),
                        'intérêts_mois': round(interest, 2),
                        'dépôt': monthly_amount,
                    }
                )

                interest_history.append(round(interest, 2))

            # Préparer le résultat
            result: dict[str, Any] = {
                'montant_mensuel': monthly_amount,
                'durée_mois': duration_months,
                'taux_intérêt': interest_rate,
                'solde_final': round(balance, 2),
                'total_investi': round(total_invested, 2),
                'total_intérêts': round(total_interest, 2),
                'historique': balance_history,
                'intérêts_mensuels_moyens': round(sum(interest_history) / len(interest_history), 2)
                if interest_history
                else 0,
            }

            return result

        except Exception as e:
            logger.error("Erreur lors de la simulation d'épargne: %s", e)
            raise

    def analyze_spending_patterns(self) -> dict[str, Any]:
        """
        Analyse les tendances de dépenses

        Returns:
            Dict[str, Any]: Analyse des tendances
        """
        try:
            # Récupérer les transactions des 12 derniers mois
            transactions = self._get_historical_transactions(months=12)

            if not transactions:
                logger.warning("Pas assez de données pour analyser les tendances")
                return {}

            # Créer un DataFrame pandas
            df = pd.DataFrame(transactions)

            # Convertir les dates en datetime
            df['date'] = pd.to_datetime(df['date_transaction'])

            # Séparer dépenses et revenus
            expenses_df = df[df['montant'] < 0].copy()
            expenses_df['montant_abs'] = expenses_df['montant'].abs()
            income_df = df[df['montant'] > 0].copy()

            # Agréger par mois
            monthly_expenses = (
                expenses_df.groupby([expenses_df['date'].dt.year, expenses_df['date'].dt.month])
                .agg({'montant_abs': ['sum', 'mean', 'count'], 'date': 'min'})
                .reset_index()
            )

            monthly_expenses.columns = [
                'année',
                'mois',
                'dépenses_totales',
                'dépense_moyenne',
                'nombre_transactions',
                'date',
            ]
            monthly_expenses = monthly_expenses.sort_values('date')

            monthly_income = (
                income_df.groupby([income_df['date'].dt.year, income_df['date'].dt.month])
                .agg({'montant': ['sum', 'mean', 'count'], 'date': 'min'})
                .reset_index()
            )

            monthly_income.columns = ['année', 'mois', 'revenus_totaux', 'revenu_moyen', 'nombre_transactions', 'date']
            monthly_income = monthly_income.sort_values('date')

            # Analyse des catégories
            category_expenses = (
                expenses_df.groupby('categorie_id')
                .agg({'montant_abs': ['sum', 'mean', 'count'], 'categorie': 'first'})
                .reset_index()
            )

            category_expenses.columns = [
                'categorie_id',
                'montant_total',
                'montant_moyen',
                'nombre_transactions',
                'categorie',
            ]
            category_expenses = category_expenses.sort_values('montant_total', ascending=False)

            # Trouver les jours de la semaine avec le plus de dépenses
            expenses_df['jour_semaine'] = expenses_df['date'].dt.dayofweek
            expenses_df['jour_mois'] = expenses_df['date'].dt.day

            day_expenses = (
                expenses_df.groupby('jour_semaine').agg({'montant_abs': ['sum', 'mean', 'count']}).reset_index()
            )

            day_expenses.columns = ['jour_semaine', 'montant_total', 'montant_moyen', 'nombre_transactions']
            day_expenses = day_expenses.sort_values('montant_total', ascending=False)

            # Convertir les jours de la semaine en noms
            jours_semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            day_expenses['jour'] = day_expenses['jour_semaine'].map(lambda x: jours_semaine[x])

            # Préparer l'analyse
            analysis = {
                'tendance_dépenses': self._analyze_trend(monthly_expenses['dépenses_totales'].tolist()),
                'tendance_revenus': self._analyze_trend(monthly_income['revenus_totaux'].tolist())
                if not monthly_income.empty
                else "stable",
                'montants_mensuels': {
                    'dépenses': {
                        'moyenne': round(monthly_expenses['dépenses_totales'].mean(), 2),
                        'médiane': round(monthly_expenses['dépenses_totales'].median(), 2),
                        'min': round(monthly_expenses['dépenses_totales'].min(), 2),
                        'max': round(monthly_expenses['dépenses_totales'].max(), 2),
                        'historique': [
                            {'date': row['date'].strftime('%Y-%m'), 'montant': round(row['dépenses_totales'], 2)}
                            for _, row in monthly_expenses.iterrows()
                        ],
                    },
                    'revenus': {
                        'moyenne': round(monthly_income['revenus_totaux'].mean(), 2) if not monthly_income.empty else 0,
                        'médiane': round(monthly_income['revenus_totaux'].median(), 2)
                        if not monthly_income.empty
                        else 0,
                        'min': round(monthly_income['revenus_totaux'].min(), 2) if not monthly_income.empty else 0,
                        'max': round(monthly_income['revenus_totaux'].max(), 2) if not monthly_income.empty else 0,
                        'historique': [
                            {'date': row['date'].strftime('%Y-%m'), 'montant': round(row['revenus_totaux'], 2)}
                            for _, row in monthly_income.iterrows()
                        ]
                        if not monthly_income.empty
                        else [],
                    },
                },
                'catégories_principales': [
                    {
                        'id': int(row['categorie_id']),
                        'nom': row['categorie'] or f"Catégorie {row['categorie_id']}",
                        'montant_total': round(row['montant_total'], 2),
                        'pourcentage': round(row['montant_total'] / category_expenses['montant_total'].sum() * 100, 1),
                    }
                    for _, row in category_expenses.head(5).iterrows()
                ],
                'jours_dépenses': [
                    {
                        'jour': row['jour'],
                        'montant_total': round(row['montant_total'], 2),
                        'pourcentage': round(row['montant_total'] / day_expenses['montant_total'].sum() * 100, 1),
                    }
                    for _, row in day_expenses.iterrows()
                ],
            }

            # Détection de comportements inhabituels
            analysis['alertes'] = self._detect_anomalies(df)

            # Conseils financiers basés sur l'analyse
            analysis['conseils'] = self._generate_advice(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des tendances: {str(e)}")
            return {}

    def recommend_investment(
        self, monthly_savings: float, risk_tolerance: str = 'medium', timeframe_years: int = 5
    ) -> dict[str, Any]:
        """
        Recommande des stratégies d'investissement

        Args:
            monthly_savings: Montant mensuel disponible pour l'épargne
            risk_tolerance: Tolérance au risque ('low', 'medium', 'high')
            timeframe_years: Horizon d'investissement en années

        Returns:
            Dict[str, Any]: Recommandations d'investissement
        """
        try:
            if monthly_savings <= 0 or timeframe_years <= 0:
                raise ValidationError("Les montants et durées doivent être positifs")

            # Valider la tolérance au risque
            if risk_tolerance not in ['low', 'medium', 'high']:
                risk_tolerance = 'medium'

            # Définition du type pour les éléments d'investissement
            class InvestmentItem(TypedDict):
                name: str
                expected_return: float
                risk_level: str
                description: str
                allocation: int
                example: str
                projected_value: float
                monthly_amount: float

            # Catégories d'investissement selon le profil
            investment_types: dict[str, list[InvestmentItem]] = {
                'low': [
                    {
                        'name': 'Compte épargne',
                        'expected_return': 1.5,
                        'risk_level': 'Très faible',
                        'description': 'Compte d\'épargne à taux fixe, très sécurisé mais faible rendement.',
                        'allocation': 50,
                        'example': 'Livret A, Livret d\'Épargne Populaire, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                    {
                        'name': 'Obligations d\'État',
                        'expected_return': 2.5,
                        'risk_level': 'Faible',
                        'description': 'Titres de dette émis par un État, généralement considérés comme sûrs.',
                        'allocation': 30,
                        'example': 'Bons du Trésor, OAT, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                    {
                        'name': 'Fonds monétaires',
                        'expected_return': 2.0,
                        'risk_level': 'Faible',
                        'description': 'Fonds investis dans des instruments du marché monétaire à court terme.',
                        'allocation': 20,
                        'example': 'Fonds monétaires, SICAV monétaires, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                ],
                'medium': [
                    {
                        'name': 'Obligations d\'État',
                        'expected_return': 2.5,
                        'risk_level': 'Faible',
                        'description': 'Titres de dette émis par un État, généralement considérés comme sûrs.',
                        'allocation': 30,
                        'example': 'Bons du Trésor, OAT, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                    {
                        'name': 'Obligations d\'entreprises',
                        'expected_return': 3.5,
                        'risk_level': 'Modéré',
                        'description': 'Titres de dette émis par des entreprises, rendement plus élevé mais risque accru.',
                        'allocation': 20,
                        'example': 'Obligations corporate, fonds obligataires, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                    {
                        'name': 'Fonds indiciels actions',
                        'expected_return': 6.0,
                        'risk_level': 'Modéré à élevé',
                        'description': 'Fonds qui répliquent un indice boursier, diversification automatique.',
                        'allocation': 40,
                        'example': 'ETF sur CAC 40, S&P 500, MSCI World, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                    {
                        'name': 'Immobilier',
                        'expected_return': 4.0,
                        'risk_level': 'Modéré',
                        'description': 'Investissement dans la pierre ou via des sociétés immobilières.',
                        'allocation': 10,
                        'example': 'SCPI, OPCI, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                ],
                'high': [
                    {
                        'name': 'Fonds indiciels actions',
                        'expected_return': 6.0,
                        'risk_level': 'Modéré à élevé',
                        'description': 'Fonds qui répliquent un indice boursier, diversification automatique.',
                        'allocation': 50,
                        'example': 'ETF sur CAC 40, S&P 500, MSCI World, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                    {
                        'name': 'Actions individuelles',
                        'expected_return': 8.0,
                        'risk_level': 'Élevé',
                        'description': 'Actions d\'entreprises cotées, potentiel de rendement élevé mais volatiles.',
                        'allocation': 20,
                        'example': 'Actions de grandes entreprises ou de croissance',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                    {
                        'name': 'Immobilier',
                        'expected_return': 4.0,
                        'risk_level': 'Modéré',
                        'description': 'Investissement dans la pierre ou via des sociétés immobilières.',
                        'allocation': 20,
                        'example': 'SCPI, OPCI, investissement locatif, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                    {
                        'name': 'Marchés émergents',
                        'expected_return': 9.0,
                        'risk_level': 'Très élevé',
                        'description': 'Investissements dans des pays en développement, fort potentiel mais risqués.',
                        'allocation': 10,
                        'example': 'ETF sur marchés émergents, fonds spécialisés, etc.',
                        'projected_value': 0.0,
                        'monthly_amount': 0.0,
                    },
                ],
            }

            selected_investments = investment_types[risk_tolerance]

            # Calculer les projections pour chaque type d'investissement
            total_portfolio_value: float = 0.0
            total_invested: float = monthly_savings * 12 * float(timeframe_years)

            for inv in selected_investments:
                allocation_amount = monthly_savings * (float(inv['allocation']) / 100.0)
                annual_return_rate = float(inv['expected_return']) / 100.0

                # Éviter la division par zéro
                if annual_return_rate > 0:
                    # Calcul de la valeur future (formule pour versements réguliers avec intérêts composés)
                    fv = allocation_amount * (
                        (pow(1.0 + annual_return_rate / 12.0, 12.0 * float(timeframe_years)) - 1.0)
                        / (annual_return_rate / 12.0)
                    )
                else:
                    # Si le taux est 0, c'est juste la somme des versements
                    fv = allocation_amount * 12.0 * float(timeframe_years)

                inv['projected_value'] = round(fv, 2)
                inv['monthly_amount'] = round(allocation_amount, 2)

                total_portfolio_value += fv

            # Calculer les gains
            total_gains = total_portfolio_value - total_invested
            avg_annual_return = 0.0
            if total_invested > 0 and total_portfolio_value > total_invested:
                avg_annual_return = (
                    pow(total_portfolio_value / total_invested, 1.0 / float(timeframe_years)) - 1.0
                ) * 100.0

            result: dict[str, Any] = {
                'profil_risque': risk_tolerance,
                'horizon': timeframe_years,
                'épargne_mensuelle': monthly_savings,
                'valeur_finale_estimée': round(total_portfolio_value, 2),
                'total_investi': round(total_invested, 2),
                'gains_estimés': round(total_gains, 2),
                'rendement_annuel_moyen': round(avg_annual_return, 2),
                'répartition': selected_investments,
                'conseils': self._get_investment_advice(risk_tolerance, timeframe_years),
            }

            return result

        except Exception as e:
            logger.error("Erreur lors de la recommandation d'investissement: %s", e)
            raise

    def suggest_budget_adjustments(self) -> dict[str, Any]:
        """
        Suggère des ajustements budgétaires

        Returns:
            Dict[str, Any]: Suggestions d'ajustements
        """
        try:
            # Récupérer les transactions des 6 derniers mois
            transactions = self._get_historical_transactions(months=6)

            if not transactions:
                logger.warning("Pas assez de données pour suggérer des ajustements budgétaires")
                return {}

            # Créer un DataFrame pandas
            df = pd.DataFrame(transactions)

            # Convertir les dates en datetime
            df['date'] = pd.to_datetime(df['date_transaction'])

            # Séparer dépenses et revenus
            expenses_df = df[df['montant'] < 0].copy()
            expenses_df['montant_abs'] = expenses_df['montant'].abs()
            income_df = df[df['montant'] > 0].copy()

            # Revenus moyens mensuels
            monthly_income = (
                income_df.groupby([income_df['date'].dt.year, income_df['date'].dt.month])['montant'].sum().mean()
            )

            # Dépenses moyennes mensuelles
            monthly_expenses = (
                expenses_df.groupby([expenses_df['date'].dt.year, expenses_df['date'].dt.month])['montant_abs']
                .sum()
                .mean()
            )

            # Taux d'épargne actuel
            current_savings_rate = (monthly_income - monthly_expenses) / monthly_income if monthly_income > 0 else 0
            current_savings_rate = max(0, current_savings_rate)  # Éviter les taux négatifs

            # Objectif d'épargne (20% des revenus comme référence)
            target_savings_rate = 0.2

            # Analyse des catégories
            category_expenses = (
                expenses_df.groupby('categorie_id')
                .agg({'montant_abs': ['sum', 'mean', 'count'], 'categorie': 'first'})
                .reset_index()
            )

            category_expenses.columns = [
                'categorie_id',
                'montant_total',
                'montant_moyen',
                'nombre_transactions',
                'categorie',
            ]
            category_expenses['montant_mensuel'] = category_expenses['montant_total'] / 6  # Sur 6 mois
            category_expenses['pourcentage_revenus'] = (
                category_expenses['montant_mensuel'] / monthly_income if monthly_income > 0 else 0
            )
            category_expenses = category_expenses.sort_values('montant_total', ascending=False)

            # Identifier les catégories où des économies pourraient être réalisées
            # Définir des seuils recommandés par catégorie
            recommended_thresholds = {
                'Logement': 0.33,  # Max 33% des revenus
                'Alimentation': 0.15,  # Max 15% des revenus
                'Transport': 0.15,  # Max 15% des revenus
                'Loisirs': 0.10,  # Max 10% des revenus
                'Restaurants': 0.08,  # Max 8% des revenus
                'Shopping': 0.05,  # Max 5% des revenus
                'Abonnements': 0.05,  # Max 5% des revenus
            }

            # Analyser chaque catégorie
            adjustments = []

            for _, row in category_expenses.iterrows():
                cat_name = row['categorie'] if row['categorie'] else f"Catégorie {row['categorie_id']}"

                # Trouver le seuil recommandé le plus proche
                threshold = 0.10  # Par défaut 10% des revenus

                for key, value in recommended_thresholds.items():
                    if key.lower() in cat_name.lower():
                        threshold = value
                        break

                # Si la catégorie dépasse le seuil recommandé
                if row['pourcentage_revenus'] > threshold:
                    current_amount = row['montant_mensuel']
                    recommended_amount = monthly_income * threshold
                    potential_saving = current_amount - recommended_amount

                    adjustments.append(
                        {
                            'categorie': cat_name,
                            'montant_actuel': round(current_amount, 2),
                            'montant_recommandé': round(recommended_amount, 2),
                            'économie_potentielle': round(potential_saving, 2),
                            'pourcentage_revenus_actuel': round(row['pourcentage_revenus'] * 100, 1),
                            'pourcentage_recommandé': round(threshold * 100, 1),
                        }
                    )

            # Trier par potentiel d'économie
            adjustments.sort(key=lambda x: x['économie_potentielle'], reverse=True)

            # Calculer l'économie totale potentielle
            total_potential_saving = sum(adj['économie_potentielle'] for adj in adjustments)

            # Calculer le nouveau taux d'épargne potentiel
            potential_savings_rate = (
                (monthly_income - (monthly_expenses - total_potential_saving)) / monthly_income
                if monthly_income > 0
                else 0
            )

            result = {
                'revenu_mensuel_moyen': round(monthly_income, 2),
                'dépenses_mensuelles_moyennes': round(monthly_expenses, 2),
                'épargne_mensuelle_actuelle': round(monthly_income - monthly_expenses, 2),
                'taux_épargne_actuel': round(current_savings_rate * 100, 1),
                'taux_épargne_recommandé': round(target_savings_rate * 100, 1),
                'taux_épargne_potentiel': round(potential_savings_rate * 100, 1),
                'économie_totale_potentielle': round(total_potential_saving, 2),
                'ajustements_recommandés': adjustments,
                'conseils': self._get_budget_advice(current_savings_rate, adjustments),
            }

            return result

        except Exception as e:
            logger.error(f"Erreur lors de la suggestion d'ajustements budgétaires: {str(e)}")
            return {}

    def get_financial_health_score(self) -> dict[str, Any]:
        """
        Calcule un score de santé financière

        Returns:
            Dict[str, Any]: Score et analyse
        """
        try:
            # Vérifier si nous avons déjà calculé ce score récemment dans le cache
            # et le retourner si c'est le cas pour éviter l'erreur "cannot insert date"
            now = datetime.datetime.now()

            # Utiliser le cache au lieu des attributs spécifiques
            cache_entry = self._cache.get('health_score', {'date': None, 'result': None})
            cache_date = cache_entry.get('date')
            cache_result = cache_entry.get('result')

            if (
                cache_date is not None
                and cache_result is not None
                and (now - cache_date).total_seconds() < 24 * 60 * 60
            ):
                logger.info("Utilisation du score de santé financière mis en cache")
                return cache_result

            score_components: dict[str, int] = {}
            explanations: dict[str, str] = {}

            # Récupérer les transactions des 12 derniers mois
            transactions = self._get_historical_transactions(months=12)

            if not transactions:
                logger.warning("Pas assez de données pour calculer le score de santé financière")
                result: dict[str, Any] = {
                    'score': 50,
                    'niveau': 'Moyen',
                    'composantes': {},
                    'explications': {
                        'manque_données': "Pas assez d'historique pour évaluer précisément. Continuez à enregistrer vos transactions pour obtenir une analyse plus précise."
                    },
                    'conseils': ["Enregistrez régulièrement vos transactions pour obtenir une analyse plus précise."],
                }
                # Sauvegarder en cache
                self._cache['health_score'] = {'date': now, 'result': result}
                return result

            # Créer un DataFrame pandas
            df = pd.DataFrame(transactions)

            # Convertir les dates en datetime
            df['date'] = pd.to_datetime(df['date_transaction'])

            # Séparer dépenses et revenus
            expenses_df = df[df['montant'] < 0].copy()
            expenses_df['montant_abs'] = expenses_df['montant'].abs()
            income_df = df[df['montant'] > 0].copy()

            # Agréger par mois
            monthly_expenses = expenses_df.groupby([expenses_df['date'].dt.year, expenses_df['date'].dt.month])[
                'montant_abs'
            ].sum()
            monthly_income = income_df.groupby([income_df['date'].dt.year, income_df['date'].dt.month])['montant'].sum()

            # Composante 1: Taux d'épargne (25 points)
            avg_monthly_expenses = monthly_expenses.mean() if not monthly_expenses.empty else 0
            avg_monthly_income = monthly_income.mean() if not monthly_income.empty else 0

            savings_rate = (
                (avg_monthly_income - avg_monthly_expenses) / avg_monthly_income if avg_monthly_income > 0 else 0
            )
            savings_rate = max(0, min(1, savings_rate))  # Limiter entre 0 et 1

            # Attribution de points selon le taux d'épargne
            savings_score = min(25, int(savings_rate * 100))
            score_components['taux_épargne'] = savings_score

            if savings_rate >= 0.2:
                explanations['taux_épargne'] = (
                    f"Excellent ! Vous épargnez {round(savings_rate * 100)}% de vos revenus, ce qui est supérieur au taux recommandé de 20%."
                )
            elif savings_rate >= 0.1:
                explanations['taux_épargne'] = (
                    f"Bon taux d'épargne de {round(savings_rate * 100)}%, mais essayez d'atteindre 20% pour plus de sécurité."
                )
            elif savings_rate > 0:
                explanations['taux_épargne'] = (
                    f"Votre taux d'épargne de {round(savings_rate * 100)}% est faible. Visez au moins 10-20% de vos revenus."
                )
            else:
                explanations['taux_épargne'] = (
                    "Vous dépensez plus que vous ne gagnez, ce qui n'est pas viable à long terme."
                )

            # Composante 2: Stabilité des revenus (20 points)
            income_stability = 20

            if len(monthly_income) >= 3:
                income_std = monthly_income.std()
                income_mean = monthly_income.mean()

                if income_mean > 0:
                    variation_coeff = income_std / income_mean

                    # Plus le coefficient de variation est faible, plus les revenus sont stables
                    if variation_coeff <= 0.1:
                        income_stability = 20  # Très stable
                    elif variation_coeff <= 0.2:
                        income_stability = 15  # Stable
                    elif variation_coeff <= 0.3:
                        income_stability = 10  # Modérément stable
                    else:
                        income_stability = 5  # Instable

                    explanations['stabilité_revenus'] = (
                        f"Vos revenus mensuels varient de {round(variation_coeff * 100)}% en moyenne."
                    )
                else:
                    explanations['stabilité_revenus'] = "Impossible d'évaluer la stabilité des revenus."
            else:
                explanations['stabilité_revenus'] = "Pas assez de données pour évaluer la stabilité des revenus."

            score_components['stabilité_revenus'] = income_stability

            # Composante 3: Diversification des dépenses (15 points)
            category_expenses = expenses_df.groupby('categorie_id')['montant_abs'].sum()

            diversification_score = 15

            if len(category_expenses) >= 3:
                # Calcul de l'indice de concentration (Herfindahl-Hirschman)
                total_expenses = category_expenses.sum()
                concentration_index = sum((cat_exp / total_expenses) ** 2 for cat_exp in category_expenses)

                # Plus l'indice est proche de 0, plus les dépenses sont diversifiées
                if concentration_index <= 0.1:
                    diversification_score = 15  # Très diversifié
                elif concentration_index <= 0.2:
                    diversification_score = 12  # Bien diversifié
                elif concentration_index <= 0.3:
                    diversification_score = 8  # Modérément diversifié
                else:
                    diversification_score = 5  # Peu diversifié

                explanations['diversification'] = (
                    f"Vos dépenses sont {['très peu', 'peu', 'modérément', 'bien', 'très bien'][min(4, int(5 * (1 - concentration_index)))]} diversifiées."
                )
            else:
                explanations['diversification'] = "Pas assez de catégories de dépenses pour évaluer la diversification."

            score_components['diversification'] = diversification_score

            # Composante 4: Régularité des dépenses (15 points)
            expense_regularity = 15

            if len(monthly_expenses) >= 3:
                expenses_std = monthly_expenses.std()
                expenses_mean = monthly_expenses.mean()

                if expenses_mean > 0:
                    variation_coeff = expenses_std / expenses_mean

                    # Plus le coefficient de variation est faible, plus les dépenses sont régulières
                    if variation_coeff <= 0.1:
                        expense_regularity = 15  # Très régulier
                    elif variation_coeff <= 0.2:
                        expense_regularity = 12  # Régulier
                    elif variation_coeff <= 0.3:
                        expense_regularity = 8  # Modérément régulier
                    else:
                        expense_regularity = 5  # Irrégulier

                    explanations['régularité_dépenses'] = (
                        f"Vos dépenses mensuelles varient de {round(variation_coeff * 100)}% en moyenne."
                    )
                else:
                    explanations['régularité_dépenses'] = "Impossible d'évaluer la régularité des dépenses."
            else:
                explanations['régularité_dépenses'] = "Pas assez de données pour évaluer la régularité des dépenses."

            score_components['régularité_dépenses'] = expense_regularity

            # Composante 5: Ratio dépenses fixes / revenus (15 points)
            # Considérer les dépenses répétitives comme fixes
            # On les estime en comptant les transactions similaires (même montant, même catégorie)

            fixed_expenses_score = 15

            # Regrouper les transactions par montant et catégorie
            expenses_df['montant_arrondi'] = expenses_df['montant'].apply(lambda x: round(x, 2))
            recurring_expenses = (
                expenses_df.groupby(['montant_arrondi', 'categorie_id']).size().reset_index(name='count')
            )
            recurring_expenses = recurring_expenses[recurring_expenses['count'] >= 3]  # Au moins 3 occurrences

            if not recurring_expenses.empty:
                # Calculer le montant total des dépenses récurrentes
                total_recurring = 0

                for _, row in recurring_expenses.iterrows():
                    monthly_amount = abs(row['montant_arrondi'])
                    total_recurring += monthly_amount

                # Calculer le ratio dépenses fixes / revenus
                fixed_expense_ratio = total_recurring / avg_monthly_income if avg_monthly_income > 0 else 1

                # Attribution de points selon le ratio
                if fixed_expense_ratio <= 0.5:
                    fixed_expenses_score = 15  # Excellent
                elif fixed_expense_ratio <= 0.6:
                    fixed_expenses_score = 12  # Bon
                elif fixed_expense_ratio <= 0.7:
                    fixed_expenses_score = 8  # Moyen
                else:
                    fixed_expenses_score = 5  # Mauvais

                explanations['dépenses_fixes'] = (
                    f"Vos dépenses fixes représentent {round(fixed_expense_ratio * 100)}% de vos revenus."
                )
            else:
                explanations['dépenses_fixes'] = "Impossible d'identifier clairement vos dépenses fixes."

            score_components['dépenses_fixes'] = fixed_expenses_score

            # Composante 6: Équilibre budget (10 points)
            balance_score = 10

            # Calculer le nombre de mois avec un solde positif
            monthly_balance = pd.merge(
                monthly_income.reset_index(), monthly_expenses.reset_index(), on=[0, 1], how='outer'
            ).fillna(0)

            monthly_balance['balance'] = monthly_balance['montant'] - monthly_balance['montant_abs']
            positive_months = (monthly_balance['balance'] >= 0).sum()
            total_months = len(monthly_balance)

            positive_ratio = positive_months / total_months if total_months > 0 else 0

            # Attribution de points selon le ratio de mois positifs
            if positive_ratio >= 0.9:
                balance_score = 10  # Excellent
            elif positive_ratio >= 0.75:
                balance_score = 8  # Bon
            elif positive_ratio >= 0.5:
                balance_score = 5  # Moyen
            else:
                balance_score = 2  # Mauvais

            explanations['équilibre_budget'] = f"Vous avez eu un solde positif {round(positive_ratio * 100)}% du temps."
            score_components['équilibre_budget'] = balance_score

            # Calcul du score final
            final_score = sum(score_components.values())

            # Déterminer le niveau
            if final_score >= 85:
                level = "Excellent"
                color = "vert"
            elif final_score >= 70:
                level = "Bon"
                color = "vert clair"
            elif final_score >= 50:
                level = "Moyen"
                color = "orange"
            elif final_score >= 30:
                level = "À surveiller"
                color = "orange foncé"
            else:
                level = "Critique"
                color = "rouge"

            # Générer des conseils
            advice = self._get_financial_health_advice(score_components, final_score)

            health_result: dict[str, Any] = {
                'score': final_score,
                'niveau': level,
                'couleur': color,
                'composantes': score_components,
                'explications': explanations,
                'conseils': advice,
            }

            # Mettre en cache pour éviter les problèmes de calcul répétitif
            self._cache['health_score'] = {'date': now, 'result': health_result}

            return health_result

        except Exception as e:
            logger.error("Erreur lors du calcul du score de santé financière: %s", e)

            # Créer un résultat par défaut
            default_result: dict[str, Any] = {
                'score': 50,
                'niveau': 'Moyen',
                'composantes': {},
                'explications': {'erreur': f"Une erreur est survenue: {str(e)}"},
                'conseils': ["Enregistrez régulièrement vos transactions pour obtenir une analyse plus précise."],
            }

            # Mettre en cache même le résultat par défaut pour éviter des recalculs
            self._cache['health_score'] = {'date': now, 'result': default_result}

            return default_result

    def _analyze_trend(self, values: list[float]) -> str:
        """
        Analyse la tendance d'une série de valeurs

        Args:
            values: Liste de valeurs

        Returns:
            str: Tendance ('hausse', 'baisse', 'stable')
        """
        if not values or len(values) < 3:
            return "stable"

        # Calculer la pente de la tendance
        x = list(range(len(values)))

        # Calculer la régression linéaire
        try:
            model = LinearRegression().fit(np.array(x).reshape(-1, 1), values)
            slope = float(model.coef_[0])

            mean_value = sum(values) / len(values)

            # Normaliser la pente par rapport à la moyenne
            relative_slope = slope / mean_value if mean_value != 0 else 0.0

            # Déterminer la tendance
            if relative_slope > 0.05:
                return "hausse"
            elif relative_slope < -0.05:
                return "baisse"
            else:
                return "stable"
        except Exception:
            return "stable"

    def _detect_anomalies(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Détecte des anomalies dans les transactions

        Args:
            df: DataFrame des transactions

        Returns:
            List[Dict[str, Any]]: Liste des anomalies détectées
        """
        anomalies: list[dict[str, Any]] = []

        # Séparer dépenses et revenus
        expenses_df = df[df['montant'] < 0].copy()
        expenses_df['montant_abs'] = expenses_df['montant'].abs()

        # 1. Détecter les dépenses inhabituellement élevées
        if len(expenses_df) >= 10:
            # Calculer les statistiques de dépenses
            mean_expense = expenses_df['montant_abs'].mean()
            std_expense = expenses_df['montant_abs'].std()

            # Seuil: moyenne + 2 écarts-types
            threshold = mean_expense + 2 * std_expense

            # Trouver les dépenses anormales
            unusual_expenses = expenses_df[expenses_df['montant_abs'] > threshold]

            for _, row in unusual_expenses.iterrows():
                anomalies.append(
                    {
                        'type': 'dépense_élevée',
                        'montant': round(float(row['montant_abs']), 2),
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'description': str(row.get('description', '')),
                        'catégorie': str(row.get('categorie', '')),
                        'seuil': round(float(threshold), 2),
                        'message': "Dépense inhabituellement élevée",
                    }
                )

        # 2. Détecter les mois avec des dépenses anormalement élevées
        monthly_expenses = expenses_df.groupby([expenses_df['date'].dt.year, expenses_df['date'].dt.month])[
            'montant_abs'
        ].sum()

        if len(monthly_expenses) >= 3:
            mean_monthly = monthly_expenses.mean()
            std_monthly = monthly_expenses.std()

            # Seuil: moyenne + 1.5 écarts-types
            threshold = mean_monthly + 1.5 * std_monthly

            # Trouver les mois anormaux
            for (year, month), amount in monthly_expenses.items():
                if amount > threshold:
                    date_str = f"{year}-{month:02d}"
                    anomalies.append(
                        {
                            'type': 'mois_dépenses_élevées',
                            'montant': round(float(amount), 2),
                            'date': date_str,
                            'seuil': round(float(threshold), 2),
                            'message': f"Dépenses mensuelles inhabituellement élevées en {date_str}",
                        }
                    )

        # 3. Détecter les catégories avec une augmentation soudaine des dépenses
        if len(expenses_df) >= 20:
            recent_cutoff = expenses_df['date'].max() - pd.Timedelta(days=60)
            older_data = expenses_df[expenses_df['date'] < recent_cutoff]
            recent_data = expenses_df[expenses_df['date'] >= recent_cutoff]

            if not older_data.empty and not recent_data.empty:
                # Agréger par catégorie
                old_by_category = older_data.groupby('categorie_id')['montant_abs'].mean()
                recent_by_category = recent_data.groupby('categorie_id')['montant_abs'].mean()

                # Comparer les moyennes
                for cat_id in recent_by_category.index:
                    if cat_id in old_by_category.index:
                        old_avg = float(old_by_category[cat_id])
                        recent_avg = float(recent_by_category[cat_id])

                        # Si augmentation de plus de 50%
                        if recent_avg > old_avg * 1.5:
                            try:
                                cat_name = self._get_category_name(int(cat_id))
                            except Exception:
                                cat_name = f"Catégorie {cat_id}"

                            anomalies.append(
                                {
                                    'type': 'augmentation_catégorie',
                                    'catégorie': cat_name,
                                    'ancienne_moyenne': round(old_avg, 2),
                                    'nouvelle_moyenne': round(recent_avg, 2),
                                    'augmentation': round((recent_avg / old_avg - 1) * 100, 1),
                                    'message': f"Augmentation de {round((recent_avg / old_avg - 1) * 100, 1)}% des dépenses en {cat_name}",
                                }
                            )

        return anomalies

    def _generate_advice(self, analysis: dict[str, Any]) -> list[str]:
        """
        Génère des conseils financiers basés sur l'analyse

        Args:
            analysis: Résultats de l'analyse

        Returns:
            List[str]: Liste de conseils
        """
        advice: list[str] = []

        # Conseil sur la tendance des dépenses
        if analysis.get('tendance_dépenses') == 'hausse':
            advice.append("Attention, vos dépenses sont en hausse. Surveillez votre budget pour éviter les dérapages.")
        elif analysis.get('tendance_dépenses') == 'baisse':
            advice.append(
                "Félicitations, vos dépenses sont en baisse. Continuez sur cette voie pour améliorer votre épargne."
            )

        # Conseil sur les catégories principales
        top_categories = analysis.get('catégories_principales', [])
        if top_categories:
            top_category = top_categories[0]
            if top_category.get('pourcentage', 0) > 40:
                advice.append(
                    f"La catégorie {top_category.get('nom')} représente {top_category.get('pourcentage')}% de vos dépenses. Essayez de réduire cette proportion pour un budget plus équilibré."
                )

        # Conseil sur les jours de dépenses
        spending_days = analysis.get('jours_dépenses', [])
        if spending_days:
            max_day = spending_days[0]
            advice.append(
                f"Vous dépensez davantage le {max_day.get('jour')} ({max_day.get('pourcentage')}% de vos dépenses). Soyez particulièrement vigilant ce jour-là."
            )

        # Conseil sur les alertes
        alerts = analysis.get('alertes', [])
        if alerts:
            for alert in alerts[:2]:  # Limiter à 2 alertes
                advice.append(alert.get('message', ''))

        # Conseils généraux si pas assez de données
        if not advice:
            advice = [
                "Essayez de maintenir un taux d'épargne d'au moins 20% de vos revenus.",
                "Diversifiez vos dépenses pour éviter qu'une seule catégorie ne prenne trop d'importance.",
                "Établissez un budget mensuel et suivez-le régulièrement.",
            ]

        return advice

    def _get_investment_advice(self, risk_tolerance: str, timeframe_years: int) -> list[str]:
        """
        Génère des conseils d'investissement

        Args:
            risk_tolerance: Tolérance au risque
            timeframe_years: Horizon d'investissement

        Returns:
            List[str]: Liste de conseils
        """
        advice: list[str] = []

        # Conseils communs
        advice.append("Diversifiez vos investissements pour réduire les risques.")
        advice.append(
            "Investissez régulièrement (mensuellement) plutôt qu'en une seule fois pour lisser les variations."
        )

        # Conseils selon la tolérance au risque
        if risk_tolerance == 'low':
            advice.append("Avec votre profil prudent, privilégiez la sécurité et acceptez des rendements modestes.")
            advice.append("Les comptes d'épargne et obligations d'État sont adaptés à votre aversion au risque.")
            if timeframe_years <= 3:
                advice.append("Sur votre horizon court, conservez une grande liquidité de vos placements.")
            else:
                advice.append(
                    "Même avec un profil prudent, vous pourriez allouer une petite partie (10-15%) à des actifs plus dynamiques sur le long terme."
                )

        elif risk_tolerance == 'medium':
            advice.append("Votre profil équilibré vous permet de combiner sécurité et performance.")
            advice.append("Un mix de produits de taux (40-60%) et d'actions (40-60%) correspond à votre profil.")
            if timeframe_years <= 3:
                advice.append("Sur votre horizon court, limitez la part des actions à 30% maximum.")
            else:
                advice.append(
                    "Sur le long terme, vous pouvez augmenter progressivement la part des actions jusqu'à 60-70%."
                )

        elif risk_tolerance == 'high':
            advice.append(
                "Avec votre profil dynamique, vous pouvez accepter une volatilité importante pour viser des rendements plus élevés."
            )
            advice.append(
                "Les actions, y compris sur des secteurs de croissance, peuvent constituer l'essentiel de votre portefeuille."
            )
            if timeframe_years <= 3:
                advice.append(
                    "Attention : même avec un profil dynamique, un horizon court nécessite de conserver au moins 30% en placements sécuritaires."
                )
            else:
                advice.append(
                    "Sur le long terme, votre horizon permet d'absorber les fluctuations des marchés actions."
                )

        return advice

    def _get_budget_advice(self, savings_rate: float, adjustments: list[dict[str, Any]]) -> list[str]:
        """
        Génère des conseils budgétaires

        Args:
            savings_rate: Taux d'épargne actuel
            adjustments: Ajustements recommandés

        Returns:
            List[str]: Liste de conseils
        """
        advice: list[str] = []

        # Conseil sur le taux d'épargne
        if savings_rate < 0.05:
            advice.append(
                "Votre taux d'épargne est très faible. Essayez d'optimiser vos dépenses pour dégager une épargne."
            )
        elif savings_rate < 0.1:
            advice.append(
                "Votre taux d'épargne est inférieur aux 10% recommandés. De petits ajustements peuvent l'améliorer significativement."
            )
        elif savings_rate < 0.2:
            advice.append(
                "Votre taux d'épargne est correct mais pourrait être optimisé pour atteindre l'objectif recommandé de 20%."
            )
        else:
            advice.append("Félicitations ! Votre taux d'épargne est excellent. Continuez sur cette voie.")

        # Conseils sur les ajustements
        if adjustments:
            top_adjustment = adjustments[0]
            advice.append(
                f"Le poste {top_adjustment.get('categorie')} représente {top_adjustment.get('pourcentage_revenus_actuel')}% de vos revenus, alors que la recommandation est de {top_adjustment.get('pourcentage_recommandé')}%."
            )

            if 'Logement' in top_adjustment.get('categorie', ''):
                advice.append(
                    "Si votre logement pèse trop lourd dans votre budget, envisagez de renégocier votre loyer ou crédit, ou à long terme de déménager."
                )
            elif 'Alimentation' in top_adjustment.get('categorie', ''):
                advice.append(
                    "Pour réduire vos dépenses alimentaires, privilégiez les achats en gros, la cuisine maison et les promotions."
                )
            elif 'Transport' in top_adjustment.get('categorie', ''):
                advice.append(
                    "Optimisez vos frais de transport en privilégiant les transports en commun, le covoiturage ou les mobilités douces quand c'est possible."
                )
            elif 'Loisirs' in top_adjustment.get('categorie', ''):
                advice.append(
                    "Cherchez des alternatives moins coûteuses pour vos loisirs : bibliothèques, activités gratuites, abonnements partagés, etc."
                )
            elif 'Abonnements' in top_adjustment.get('categorie', ''):
                advice.append(
                    "Faites le tri dans vos abonnements et conservez uniquement ceux que vous utilisez vraiment régulièrement."
                )

        # Conseils généraux
        advice.append(
            "La règle 50-30-20 est un bon guide : 50% pour les besoins essentiels, 30% pour les envies, 20% pour l'épargne."
        )

        return advice

    def _get_financial_health_advice(self, score_components: dict[str, int], final_score: int) -> list[str]:
        """
        Génère des conseils basés sur le score de santé financière

        Args:
            score_components: Composantes du score
            final_score: Score final

        Returns:
            List[str]: Liste de conseils
        """
        advice: list[str] = []

        # Identifier les points faibles (les 2 composantes avec les scores les plus bas)
        weak_points = sorted(score_components.items(), key=lambda x: x[1])[:2]

        for component, score in weak_points:
            if component == 'taux_épargne' and score < 15:
                advice.append(
                    "Améliorez votre taux d'épargne en établissant un budget et en réduisant les dépenses non essentielles."
                )
                advice.append("Essayez d'épargner au moins 10-20% de vos revenus chaque mois.")

            elif component == 'stabilité_revenus' and score < 15:
                advice.append(
                    "Vos revenus sont variables, ce qui complique la gestion budgétaire. Constituez une réserve de sécurité plus importante."
                )
                advice.append(
                    "Si possible, cherchez des sources de revenus complémentaires pour stabiliser vos finances."
                )

            elif component == 'diversification' and score < 10:
                advice.append(
                    "Vos dépenses sont très concentrées sur peu de catégories. Une meilleure diversification vous rendrait plus résilient."
                )

            elif component == 'régularité_dépenses' and score < 10:
                advice.append(
                    "Vos dépenses varient beaucoup d'un mois à l'autre. Essayez de les lisser pour faciliter votre gestion budgétaire."
                )
                advice.append("Répartissez les grosses dépenses sur plusieurs mois quand c'est possible.")

            elif component == 'dépenses_fixes' and score < 10:
                advice.append(
                    "Vos dépenses fixes représentent une part importante de vos revenus, ce qui réduit votre flexibilité financière."
                )
                advice.append(
                    "Essayez de renégocier certains contrats récurrents (assurances, téléphonie, etc.) pour réduire ces charges."
                )

            elif component == 'équilibre_budget' and score < 5:
                advice.append(
                    "Vous terminez souvent le mois avec un solde négatif. C'est un signal d'alerte important à surveiller."
                )
                advice.append("Envisagez de réduire temporairement certaines dépenses pour retrouver l'équilibre.")

        # Conseils généraux selon le score
        if final_score < 30:
            advice.append(
                "Votre situation financière nécessite une attention urgente. Établissez un plan d'action pour redresser vos finances."
            )
        elif final_score < 50:
            advice.append(
                "Votre santé financière est fragile. Des ajustements significatifs sont nécessaires pour l'améliorer."
            )
        elif final_score < 70:
            advice.append(
                "Votre situation financière est correcte mais pourrait être améliorée avec quelques ajustements."
            )
        else:
            advice.append(
                "Votre santé financière est bonne. Continuez sur cette voie et pensez maintenant à optimiser vos investissements."
            )

        return advice

    def _get_historical_transactions(self, months: int = 12) -> list[dict[str, Any]]:
        """
        Récupère l'historique des transactions

        Args:
            months: Nombre de mois d'historique

        Returns:
            List[Dict[str, Any]]: Liste des transactions
        """
        try:
            # Calculer la date limite
            limit_date = (datetime.datetime.now() - datetime.timedelta(days=months * 30)).strftime('%Y-%m-%d')

            # Requête SQL
            query = """
            SELECT t.id, t.date_transaction, t.type_transaction, t.categorie_id,
                   c.nom as categorie, t.montant, t.compte_id, t.description
            FROM transactions t
            LEFT JOIN categories c ON t.categorie_id = c.id
            WHERE t.date_transaction >= ?
            ORDER BY t.date_transaction DESC
            """

            # Exécuter la requête
            transactions = db.fetch_all(query, (limit_date,))

            return transactions if transactions else []

        except Exception as e:
            logger.error("Erreur lors de la récupération de l'historique des transactions: %s", e)
            return []

    def _get_category_name(self, category_id: int) -> str:
        """
        Récupère le nom d'une catégorie

        Args:
            category_id: ID de la catégorie

        Returns:
            str: Nom de la catégorie
        """
        try:
            # Requête SQL
            query = "SELECT nom FROM categories WHERE id = ?"

            # Exécuter la requête
            result = db.fetch_one(query, (category_id,))

            if result and 'nom' in result and result['nom']:
                return str(result['nom'])

            return f"Catégorie {category_id}"

        except Exception as e:
            logger.error("Erreur lors de la récupération du nom de la catégorie: %s", e)
            return f"Catégorie {category_id}"

    def _save_models(self) -> bool:
        """
        Sauvegarde les modèles entraînés

        Returns:
            bool: Succès de la sauvegarde
        """
        try:
            import pickle  # type: ignore

            # Créer le dossier de modèles s'il n'existe pas
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "models"
            )

            os.makedirs(models_dir, exist_ok=True)

            # Sauvegarder les modèles
            models_path = os.path.join(models_dir, "finance_models.pkl")

            with open(models_path, 'wb') as f:
                pickle.dump({'models': self.models, 'last_training_date': self.last_training_date}, f)

            return True

        except Exception as e:
            logger.error("Erreur lors de la sauvegarde des modèles: %s", e)
            return False

    def _load_models(self) -> bool:
        """
        Charge les modèles entraînés

        Returns:
            bool: Succès du chargement
        """
        try:
            import pickle  # type: ignore

            # Chemin vers les modèles
            models_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data",
                "models",
                "finance_models.pkl",
            )

            # Vérifier si le fichier existe
            if not os.path.exists(models_path):
                # Créer le dossier models s'il n'existe pas
                models_dir = os.path.dirname(models_path)
                os.makedirs(models_dir, exist_ok=True)
                return False

            # Charger les modèles
            with open(models_path, 'rb') as f:
                data = pickle.load(f)

                if 'models' in data:
                    self.models = data['models']
                if 'last_training_date' in data:
                    self.last_training_date = data['last_training_date']

            return True

        except Exception as e:
            logger.error("Erreur lors du chargement des modèles: %s", e)
            # S'assurer que le dossier models existe pour les futurs enregistrements
            try:
                models_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "models"
                )
                os.makedirs(models_dir, exist_ok=True)
            except Exception:
                pass
            return False


# Instance globale
finance_predictor = FinancePredictor()


def predire_depenses_futures(mois_horizon: int = 3) -> dict[str, Any]:
    """
    Prédit les dépenses et revenus pour les mois à venir

    Args:
        mois_horizon: Nombre de mois à prédire

    Returns:
        Dict[str, Any]: Prédictions de dépenses et revenus
    """
    predictions = finance_predictor.predict_future_expenses(mois_horizon)

    # Si les prédictions sont vides, retourner une erreur
    if not predictions:
        return {"error": "Pas assez de données pour générer des prédictions fiables"}

    # Extraire les données pour le frontend
    result: dict[str, Any] = {
        "mois_prevus": [pred["date"] for pred in predictions],
        "depenses_predites": [pred["dépenses"] for pred in predictions],
        "revenus_predits": [pred["revenus"] for pred in predictions],
        "soldes_projetes": [pred["solde"] for pred in predictions],
        "categories": {},
    }

    # Agréger les prédictions par catégorie
    for pred in predictions:
        for cat in pred.get("catégories", []):
            cat_id = cat["id"]
            if cat_id not in result["categories"]:
                result["categories"][cat_id] = {"id": cat_id, "nom": cat["nom"], "valeurs": [0.0] * len(predictions)}

            # Trouver l'index du mois
            month_idx = predictions.index(pred)
            result["categories"][cat_id]["valeurs"][month_idx] = cat["montant"]

    return result


def analyser_sante_financiere() -> dict[str, Any]:
    """
    Analyse la santé financière de l'utilisateur

    Returns:
        Dict[str, Any]: Analyse de la santé financière
    """
    try:
        return finance_predictor.get_financial_health_score()
    except Exception as e:
        logger.error("Erreur lors du calcul du score de santé financière: %s", e)
        # Retourner une valeur par défaut en cas d'erreur
        return {
            'score': 50,
            'niveau': 'Moyen',
            'composantes': {},
            'explications': {'erreur': f"Une erreur est survenue: {str(e)}"},
            'conseils': ["Enregistrez régulièrement vos transactions pour obtenir une analyse plus précise."],
        }
