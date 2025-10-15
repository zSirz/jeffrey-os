"""
Service d'analyse et d'insights financiers pour CashZen
Fournit des analyses avancées, des tendances et des recommandations personnalisées
"""

import calendar
import datetime
import logging
import statistics
from typing import Any

from services.ai.finance_predictor import predire_depenses_futures
from services.finance_api import get_financial_health
from services.gestion_budget import get_budgets_par_categorie
from services.gestion_comptes import get_all_comptes
from services.gestion_transactions import get_transactions_periode
from services.goal_service import goal_service

# Configuration du logging
logger = logging.getLogger(__name__)


class InsightsService:
    """
    Service pour générer des insights financiers personnalisés
    """

    def __init__(self):
        """Initialisation du service d'insights"""
        pass

    def generate_monthly_report(self, month: int | None = None, year: int | None = None) -> dict[str, Any]:
        """
        Génère un rapport financier mensuel

        Args:
            month: Mois (1-12, None=mois actuel)
            year: Année (None=année actuelle)

        Returns:
            Dict[str, Any]: Rapport mensuel
        """
        try:
            # Utiliser le mois et l'année courants si non spécifiés
            today = datetime.date.today()
            month = month or today.month
            year = year or today.year

            # Début et fin du mois
            first_day = datetime.date(year, month, 1)
            last_day = datetime.date(year, month, calendar.monthrange(year, month)[1])

            # Obtenir les transactions du mois
            transactions = get_transactions_periode(first_day.isoformat(), last_day.isoformat())

            # Séparer les dépenses et les revenus
            expenses = [t for t in transactions if t.get("type") == "debit"]
            incomes = [t for t in transactions if t.get("type") == "credit"]

            # Calculer les totaux
            total_expenses = sum(t.get("montant", 0) for t in expenses)
            total_incomes = sum(t.get("montant", 0) for t in incomes)
            net_cash_flow = total_incomes - total_expenses

            # Comparer avec le mois précédent
            prev_month = month - 1 if month > 1 else 12
            prev_year = year if month > 1 else year - 1

            prev_first_day = datetime.date(prev_year, prev_month, 1)
            prev_last_day = datetime.date(prev_year, prev_month, calendar.monthrange(prev_year, prev_month)[1])

            prev_transactions = get_transactions_periode(prev_first_day.isoformat(), prev_last_day.isoformat())
            prev_expenses = [t for t in prev_transactions if t.get("type") == "debit"]
            prev_incomes = [t for t in prev_transactions if t.get("type") == "credit"]

            prev_total_expenses = sum(t.get("montant", 0) for t in prev_expenses)
            prev_total_incomes = sum(t.get("montant", 0) for t in prev_incomes)
            prev_net_cash_flow = prev_total_incomes - prev_total_expenses

            # Calculer les variations
            expense_change_pct = (
                ((total_expenses - prev_total_expenses) / prev_total_expenses * 100) if prev_total_expenses > 0 else 0
            )
            income_change_pct = (
                ((total_incomes - prev_total_incomes) / prev_total_incomes * 100) if prev_total_incomes > 0 else 0
            )

            # Analyser les dépenses par catégorie
            expense_by_category = {}
            for expense in expenses:
                category = expense.get("categorie", "Non catégorisé")
                if category not in expense_by_category:
                    expense_by_category[category] = 0
                expense_by_category[category] += expense.get("montant", 0)

            # Trier les catégories par montant
            sorted_categories = sorted(expense_by_category.items(), key=lambda x: x[1], reverse=True)
            top_categories = [{"categorie": cat, "montant": amt} for cat, amt in sorted_categories[:5]]

            # Calculer le taux d'épargne
            savings_rate = ((total_incomes - total_expenses) / total_incomes * 100) if total_incomes > 0 else 0

            # Obtenir les objectifs d'épargne et leur progression
            savings_goals = goal_service.get_goals(active_only=True, goal_type="épargne")

            # Obtenir les budgets du mois et leur état
            budgets = get_budgets_par_categorie(month, year)

            # Extraire les dépassements de budget
            budget_overruns = []
            for cat, budget in budgets.items():
                if budget.get("depense_actuelle", 0) > budget.get("montant", 0) and budget.get("montant", 0) > 0:
                    overrun_pct = (
                        (budget.get("depense_actuelle", 0) - budget.get("montant", 0)) / budget.get("montant", 0) * 100
                    )
                    budget_overruns.append(
                        {
                            "categorie": cat,
                            "budget": budget.get("montant", 0),
                            "depense": budget.get("depense_actuelle", 0),
                            "depassement": budget.get("depense_actuelle", 0) - budget.get("montant", 0),
                            "depassement_pct": overrun_pct,
                        }
                    )

            # Trier par pourcentage de dépassement
            budget_overruns.sort(key=lambda x: x["depassement_pct"], reverse=True)

            # Générer des insights basés sur les données
            insights = self._generate_monthly_insights(
                total_expenses,
                total_incomes,
                net_cash_flow,
                expense_change_pct,
                income_change_pct,
                expense_by_category,
                budgets,
                budget_overruns,
            )

            # Compiler le rapport
            return {
                "month": month,
                "year": year,
                "period": {"start": first_day.isoformat(), "end": last_day.isoformat()},
                "summary": {
                    "total_expenses": total_expenses,
                    "total_incomes": total_incomes,
                    "net_cash_flow": net_cash_flow,
                    "savings_rate": savings_rate,
                    "expense_change_pct": expense_change_pct,
                    "income_change_pct": income_change_pct,
                },
                "expenses": {"by_category": expense_by_category, "top_categories": top_categories},
                "budget": {
                    "total_budget": sum(b.get("montant", 0) for b in budgets.values()),
                    "total_spent": sum(b.get("depense_actuelle", 0) for b in budgets.values()),
                    "overruns": budget_overruns,
                },
                "savings": {
                    "goals_count": len(savings_goals),
                    "total_contributed": sum(g.get("montant_actuel", 0) for g in savings_goals),
                    "goals": savings_goals,
                },
                "insights": insights,
                "transaction_count": len(transactions),
            }

        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport mensuel: {e}")
            return {"error": str(e), "month": month, "year": year}

    def generate_spending_anomalies(self, lookback_months: int = 6) -> list[dict[str, Any]]:
        """
        Détecte les anomalies de dépenses

        Args:
            lookback_months: Nombre de mois d'historique à analyser

        Returns:
            List[Dict[str, Any]]: Liste des anomalies détectées
        """
        try:
            # Date de début pour l'analyse (lookback_months mois en arrière)
            today = datetime.date.today()
            start_date = today.replace(day=1) - datetime.timedelta(days=lookback_months * 30)

            # Récupérer les transactions
            transactions = get_transactions_periode(start_date.isoformat(), today.isoformat())

            # Ne garder que les dépenses
            expenses = [t for t in transactions if t.get("type") == "debit"]

            # Organiser les dépenses par mois et par catégorie
            expenses_by_month_category = {}

            for expense in expenses:
                try:
                    date = datetime.date.fromisoformat(expense.get("date"))
                    month_key = f"{date.year}-{date.month:02d}"
                    category = expense.get("categorie", "Non catégorisé")

                    if month_key not in expenses_by_month_category:
                        expenses_by_month_category[month_key] = {}

                    if category not in expenses_by_month_category[month_key]:
                        expenses_by_month_category[month_key][category] = 0

                    expenses_by_month_category[month_key][category] += expense.get("montant", 0)

                except ValueError:
                    # Date invalide, ignorer
                    continue

            # Calculer les moyennes et écarts-types par catégorie
            stats_by_category = {}

            for month_key, categories in expenses_by_month_category.items():
                for category, amount in categories.items():
                    if category not in stats_by_category:
                        stats_by_category[category] = []

                    stats_by_category[category].append(amount)

            # Calculer les statistiques descriptives
            category_stats = {}
            for category, amounts in stats_by_category.items():
                if len(amounts) >= 3:  # Au moins 3 points de données pour des statistiques fiables
                    category_stats[category] = {
                        "mean": statistics.mean(amounts),
                        "median": statistics.median(amounts),
                        "stdev": statistics.stdev(amounts) if len(amounts) > 1 else 0,
                        "min": min(amounts),
                        "max": max(amounts),
                    }

            # Obtenir les dépenses du mois en cours
            current_month = f"{today.year}-{today.month:02d}"
            current_expenses = expenses_by_month_category.get(current_month, {})

            # Détecter les anomalies
            anomalies = []

            for category, amount in current_expenses.items():
                if category in category_stats:
                    stats = category_stats[category]
                    mean = stats["mean"]
                    stdev = stats["stdev"]

                    # Considérer comme anomalie si la dépense dépasse la moyenne de plus de 2 écarts-types,
                    # ou si elle dépasse le maximum historique de 50%
                    if stdev > 0 and (amount > mean + 2 * stdev) or (amount > stats["max"] * 1.5):
                        anomalies.append(
                            {
                                "category": category,
                                "current_amount": amount,
                                "average_amount": mean,
                                "deviation": (amount - mean) / mean * 100 if mean > 0 else 0,
                                "std_deviations": (amount - mean) / stdev if stdev > 0 else 0,
                                "max_historical": stats["max"],
                                "severity": "high" if amount > mean + 3 * stdev else "medium",
                            }
                        )

            # Trier par déviation par rapport à la moyenne
            anomalies.sort(key=lambda x: x["deviation"], reverse=True)

            return anomalies

        except Exception as e:
            logger.error(f"Erreur lors de la détection des anomalies: {e}")
            return []

    def analyze_spending_patterns(self, time_range: str = "year") -> dict[str, Any]:
        """
        Analyse les tendances et patterns de dépenses

        Args:
            time_range: Période d'analyse ("month", "quarter", "year")

        Returns:
            Dict[str, Any]: Analyse des patterns de dépenses
        """
        try:
            # Déterminer la date de début selon la période
            today = datetime.date.today()

            if time_range == "month":
                start_date = today.replace(day=1)
            elif time_range == "quarter":
                month = today.month
                quarter_start_month = ((month - 1) // 3) * 3 + 1
                start_date = today.replace(month=quarter_start_month, day=1)
            else:  # year
                start_date = today.replace(month=1, day=1)

            # Récupérer les transactions
            transactions = get_transactions_periode(start_date.isoformat(), today.isoformat())

            # Ne garder que les dépenses
            expenses = [t for t in transactions if t.get("type") == "debit"]

            # Analyse par catégorie
            by_category = {}
            for expense in expenses:
                category = expense.get("categorie", "Non catégorisé")

                if category not in by_category:
                    by_category[category] = {"total": 0, "count": 0, "max": 0, "min": float('inf'), "transactions": []}

                amount = expense.get("montant", 0)
                by_category[category]["total"] += amount
                by_category[category]["count"] += 1
                by_category[category]["max"] = max(by_category[category]["max"], amount)
                by_category[category]["min"] = min(by_category[category]["min"], amount)
                by_category[category]["transactions"].append(expense)

            # Calcul des moyennes et ajout des pourcentages
            total_spend = sum(cat_data["total"] for cat_data in by_category.values())

            for category, data in by_category.items():
                data["average"] = data["total"] / data["count"] if data["count"] > 0 else 0
                data["percentage"] = (data["total"] / total_spend * 100) if total_spend > 0 else 0

            # Analyse par jour de la semaine
            by_day_of_week = {i: 0 for i in range(7)}  # 0 = lundi, 6 = dimanche

            for expense in expenses:
                try:
                    date = datetime.date.fromisoformat(expense.get("date"))
                    day_of_week = date.weekday()
                    by_day_of_week[day_of_week] += expense.get("montant", 0)
                except ValueError:
                    # Date invalide, ignorer
                    continue

            # Convertir en jours nommés
            days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
            by_day_named = {days[i]: amount for i, amount in by_day_of_week.items()}

            # Analyse par semaine du mois
            by_week_of_month = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

            for expense in expenses:
                try:
                    date = datetime.date.fromisoformat(expense.get("date"))
                    week_of_month = (date.day - 1) // 7 + 1
                    by_week_of_month[min(week_of_month, 5)] += expense.get("montant", 0)
                except ValueError:
                    # Date invalide, ignorer
                    continue

            # Identifier les commerçants les plus fréquents
            merchants = {}
            for expense in expenses:
                merchant = expense.get("tiers", "").strip()
                if merchant and merchant != "":
                    if merchant not in merchants:
                        merchants[merchant] = {"total": 0, "count": 0, "average": 0}

                    merchants[merchant]["total"] += expense.get("montant", 0)
                    merchants[merchant]["count"] += 1

            # Calculer les moyennes
            for merchant, data in merchants.items():
                data["average"] = data["total"] / data["count"]

            # Trier par fréquence
            top_merchants_by_frequency = sorted(
                [(merchant, data) for merchant, data in merchants.items()], key=lambda x: x[1]["count"], reverse=True
            )[:10]

            # Trier par montant total
            top_merchants_by_amount = sorted(
                [(merchant, data) for merchant, data in merchants.items()], key=lambda x: x[1]["total"], reverse=True
            )[:10]

            # Trouver les patterns de dépenses récurrentes
            recurring_candidates = []

            for merchant, data in merchants.items():
                # Considérer comme récurrent si fréquence > 2 et écart-type faible
                if data["count"] >= 3:
                    amounts = [t.get("montant", 0) for t in expenses if t.get("tiers", "").strip() == merchant]

                    if len(amounts) >= 3:
                        avg = sum(amounts) / len(amounts)
                        stdev = statistics.stdev(amounts)
                        variation_coeff = stdev / avg if avg > 0 else float('inf')

                        # Si le coefficient de variation est faible, c'est probablement récurrent
                        if variation_coeff < 0.2:  # 20% de variation
                            recurring_candidates.append(
                                {
                                    "merchant": merchant,
                                    "average_amount": avg,
                                    "frequency": data["count"],
                                    "variation_coeff": variation_coeff,
                                    "total": data["total"],
                                }
                            )

            # Trier par coefficient de variation (les plus stables d'abord)
            recurring_candidates.sort(key=lambda x: x["variation_coeff"])

            return {
                "time_range": time_range,
                "period": {"start": start_date.isoformat(), "end": today.isoformat()},
                "total_spend": total_spend,
                "transaction_count": len(expenses),
                "by_category": {
                    category: {
                        "total": data["total"],
                        "count": data["count"],
                        "average": data["average"],
                        "percentage": data["percentage"],
                        "max": data["max"],
                        "min": data["min"],
                    }
                    for category, data in by_category.items()
                },
                "by_day_of_week": by_day_named,
                "by_week_of_month": by_week_of_month,
                "top_merchants": {
                    "by_frequency": [
                        {"name": merchant, "count": data["count"], "total": data["total"], "average": data["average"]}
                        for merchant, data in top_merchants_by_frequency
                    ],
                    "by_amount": [
                        {"name": merchant, "count": data["count"], "total": data["total"], "average": data["average"]}
                        for merchant, data in top_merchants_by_amount
                    ],
                },
                "recurring_expenses": recurring_candidates,
            }

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des patterns de dépenses: {e}")
            return {"error": str(e), "time_range": time_range}

    def generate_financial_insights(self) -> list[dict[str, Any]]:
        """
        Génère des insights financiers personnalisés

        Returns:
            List[Dict[str, Any]]: Liste des insights
        """
        try:
            insights = []

            # 1. Analyser les dépenses récentes
            anomalies = self.generate_spending_anomalies()

            if anomalies:
                for anomaly in anomalies[:3]:  # Limiter à 3 insights d'anomalies
                    insights.append(
                        {
                            "type": "spending_anomaly",
                            "title": f"Dépenses inhabituelles en {anomaly['category']}",
                            "description": f"Vos dépenses en {anomaly['category']} sont {anomaly['deviation']:.1f}% plus élevées que d'habitude ce mois-ci.",
                            "severity": anomaly["severity"],
                            "data": anomaly,
                        }
                    )

            # 2. Analyser l'évolution de la santé financière
            health = get_financial_health()

            # Insight sur le fonds d'urgence
            reserve_urgence = health.get("reserve_urgence", 0)
            if reserve_urgence < 3:
                insights.append(
                    {
                        "type": "emergency_fund",
                        "title": "Fonds d'urgence insuffisant",
                        "description": f"Votre fonds d'urgence ne couvre que {reserve_urgence:.1f} mois de dépenses. L'idéal est de 3 à 6 mois.",
                        "severity": "high" if reserve_urgence < 1 else "medium",
                        "data": {
                            "reserve_urgence": reserve_urgence,
                            "recommendation": "Essayez d'épargner régulièrement pour atteindre au moins 3 mois de dépenses.",
                        },
                    }
                )

            # Insight sur le taux d'épargne
            taux_epargne = health.get("taux_epargne", 0)
            if taux_epargne < 10:
                insights.append(
                    {
                        "type": "savings_rate",
                        "title": "Taux d'épargne faible",
                        "description": f"Votre taux d'épargne est de {taux_epargne:.1f}%. Viser 20% de vos revenus est recommandé pour une bonne santé financière.",
                        "severity": "high" if taux_epargne < 0 else "medium",
                        "data": {
                            "taux_epargne": taux_epargne,
                            "recommendation": "Réduisez vos dépenses non essentielles et automatisez votre épargne.",
                        },
                    }
                )
            elif taux_epargne > 30:
                insights.append(
                    {
                        "type": "savings_rate",
                        "title": "Excellent taux d'épargne",
                        "description": f"Votre taux d'épargne de {taux_epargne:.1f}% est excellent ! Envisagez d'investir davantage.",
                        "severity": "low",
                        "data": {
                            "taux_epargne": taux_epargne,
                            "recommendation": "Considérez des investissements à long terme pour faire fructifier votre épargne.",
                        },
                    }
                )

            # 3. Vérifier les budgets du mois en cours
            today = datetime.date.today()
            budgets = get_budgets_par_categorie(today.month, today.year)

            # Identifier les budgets à risque de dépassement
            day_of_month = today.day
            days_in_month = calendar.monthrange(today.year, today.month)[1]
            month_progress = day_of_month / days_in_month

            for category, budget in budgets.items():
                if budget.get("montant", 0) > 0:
                    budget_progress = budget.get("pourcentage", 0) / 100

                    # Si la progression du budget dépasse la progression du mois de plus de 20%
                    if budget_progress > month_progress * 1.2 and budget_progress > 0.7:
                        insights.append(
                            {
                                "type": "budget_risk",
                                "title": f"Budget {category} à risque",
                                "description": f"Vous avez dépensé {budget_progress * 100:.1f}% de votre budget {category} alors que le mois est à {month_progress * 100:.1f}%.",
                                "severity": "high" if budget_progress > 0.9 else "medium",
                                "data": {
                                    "categorie": category,
                                    "budget": budget.get("montant", 0),
                                    "depense": budget.get("depense_actuelle", 0),
                                    "pourcentage": budget.get("pourcentage", 0),
                                    "month_progress": month_progress * 100,
                                },
                            }
                        )

            # 4. Vérifier les objectifs d'épargne
            goals = goal_service.get_goals(active_only=True)

            for goal in goals:
                stats = goal.get("stats", {})

                # Pour les objectifs presque échus
                if stats.get("days_remaining", 0) < 30 and stats.get("days_remaining", 0) > 0:
                    remaining_pct = (goal["montant_cible"] - goal["montant_actuel"]) / goal["montant_cible"] * 100

                    if remaining_pct > 20:
                        insights.append(
                            {
                                "type": "goal_deadline",
                                "title": f"Objectif {goal['nom']} bientôt échu",
                                "description": f"Votre objectif {goal['nom']} arrive à échéance dans {stats['days_remaining']} jours mais il vous reste encore {remaining_pct:.1f}% à atteindre.",
                                "severity": "high" if remaining_pct > 50 else "medium",
                                "data": {
                                    "goal_id": goal["id"],
                                    "nom": goal["nom"],
                                    "days_remaining": stats["days_remaining"],
                                    "remaining_amount": stats["remaining_amount"],
                                    "monthly_contribution": stats["monthly_contribution"],
                                },
                            }
                        )

            # 5. Vérifier les soldes de compte
            comptes = get_all_comptes()

            for compte in comptes:
                if compte.get("type_compte") == "courant" and compte.get("solde_actuel", 0) < 0:
                    decouvert = abs(compte.get("solde_actuel", 0))
                    decouvert_autorise = compte.get("montant_decouvert_autorise", 0)

                    if decouvert_autorise > 0 and decouvert > decouvert_autorise * 0.8:
                        insights.append(
                            {
                                "type": "account_overdraft",
                                "title": f"Découvert élevé sur {compte.get('nom', 'compte')}",
                                "description": f"Votre compte {compte.get('nom', '')} est à découvert de {decouvert:.2f} CHF, soit {decouvert / decouvert_autorise * 100:.1f}% du découvert autorisé.",
                                "severity": "high" if decouvert > decouvert_autorise else "medium",
                                "data": {
                                    "compte_id": compte.get("id"),
                                    "nom": compte.get("nom", ""),
                                    "solde": compte.get("solde_actuel", 0),
                                    "decouvert_autorise": decouvert_autorise,
                                },
                            }
                        )

            # 6. Vérifier les prévisions financières
            predictions = predire_depenses_futures(3)  # 3 mois

            if predictions and "error" not in predictions:
                revenus_predits = predictions.get("revenus_predits", [])
                depenses_predites = predictions.get("depenses_predites", [])

                if len(revenus_predits) >= 2 and len(depenses_predites) >= 2:
                    mois_1_net = revenus_predits[0] - depenses_predites[0]
                    mois_2_net = revenus_predits[1] - depenses_predites[1]

                    # Si le cash flow prévu est négatif pour les deux prochains mois
                    if mois_1_net < 0 and mois_2_net < 0:
                        insights.append(
                            {
                                "type": "future_cash_flow",
                                "title": "Cash flow négatif prévu",
                                "description": f"Nos prévisions indiquent un cash flow négatif pour les deux prochains mois, de {mois_1_net:.2f} CHF et {mois_2_net:.2f} CHF.",
                                "severity": "high",
                                "data": {
                                    "mois_1": {
                                        "revenus": revenus_predits[0],
                                        "depenses": depenses_predites[0],
                                        "net": mois_1_net,
                                    },
                                    "mois_2": {
                                        "revenus": revenus_predits[1],
                                        "depenses": depenses_predites[1],
                                        "net": mois_2_net,
                                    },
                                },
                            }
                        )

            # Trier les insights par sévérité
            severity_scores = {"high": 3, "medium": 2, "low": 1}
            insights.sort(key=lambda x: severity_scores.get(x.get("severity", "low"), 0), reverse=True)

            return insights

        except Exception as e:
            logger.error(f"Erreur lors de la génération des insights financiers: {e}")
            return [
                {
                    "type": "error",
                    "title": "Erreur lors de l'analyse",
                    "description": f"Une erreur est survenue lors de la génération des insights: {str(e)}",
                    "severity": "low",
                }
            ]

    def _generate_monthly_insights(
        self,
        total_expenses,
        total_incomes,
        net_cash_flow,
        expense_change_pct,
        income_change_pct,
        expense_by_category,
        budgets,
        budget_overruns,
    ) -> list[dict[str, Any]]:
        """
        Génère des insights pour le rapport mensuel

        Args:
            Divers paramètres d'analyse financière

        Returns:
            List[Dict[str, Any]]: Liste des insights
        """
        insights = []

        # Insight sur le cash flow
        if net_cash_flow < 0:
            insights.append(
                {
                    "type": "cash_flow",
                    "title": "Cash flow négatif",
                    "description": f"Vos dépenses ({total_expenses:.2f} CHF) ont dépassé vos revenus ({total_incomes:.2f} CHF) ce mois-ci, créant un déficit de {abs(net_cash_flow):.2f} CHF.",
                    "severity": "high",
                }
            )
        elif net_cash_flow > 0 and net_cash_flow > total_incomes * 0.2:
            insights.append(
                {
                    "type": "cash_flow",
                    "title": "Excellent cash flow",
                    "description": f"Vous avez épargné {net_cash_flow:.2f} CHF ce mois-ci, soit {(net_cash_flow / total_incomes * 100):.1f}% de vos revenus.",
                    "severity": "low",
                }
            )

        # Insight sur l'évolution des dépenses
        if expense_change_pct > 20:
            insights.append(
                {
                    "type": "expense_trend",
                    "title": "Augmentation des dépenses",
                    "description": f"Vos dépenses ont augmenté de {expense_change_pct:.1f}% par rapport au mois précédent.",
                    "severity": "medium",
                }
            )
        elif expense_change_pct < -15:
            insights.append(
                {
                    "type": "expense_trend",
                    "title": "Réduction significative des dépenses",
                    "description": f"Vous avez réduit vos dépenses de {abs(expense_change_pct):.1f}% par rapport au mois précédent.",
                    "severity": "low",
                }
            )

        # Insight sur les revenus
        if income_change_pct < -10:
            insights.append(
                {
                    "type": "income_trend",
                    "title": "Baisse des revenus",
                    "description": f"Vos revenus ont diminué de {abs(income_change_pct):.1f}% par rapport au mois précédent.",
                    "severity": "medium" if income_change_pct < -20 else "low",
                }
            )

        # Insight sur les dépassements de budget
        if budget_overruns:
            top_overrun = budget_overruns[0]
            insights.append(
                {
                    "type": "budget_overrun",
                    "title": f"Dépassement de budget: {top_overrun['categorie']}",
                    "description": f"Vous avez dépassé votre budget {top_overrun['categorie']} de {top_overrun['depassement']:.2f} CHF ({top_overrun['depassement_pct']:.1f}%).",
                    "severity": "high" if top_overrun['depassement_pct'] > 30 else "medium",
                }
            )

        # Insight sur les catégories de dépenses
        sorted_categories = sorted(expense_by_category.items(), key=lambda x: x[1], reverse=True)
        if sorted_categories:
            top_category, top_amount = sorted_categories[0]
            top_percentage = (top_amount / total_expenses * 100) if total_expenses > 0 else 0

            if top_percentage > 40:
                insights.append(
                    {
                        "type": "spending_distribution",
                        "title": f"Concentration des dépenses: {top_category}",
                        "description": f"La catégorie {top_category} représente {top_percentage:.1f}% de vos dépenses totales ce mois-ci.",
                        "severity": "medium" if top_percentage > 60 else "low",
                    }
                )

        return insights


# Singleton global
insights_service = InsightsService()
