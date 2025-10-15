from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path

# Configuration du logger
logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
        Tableau de bord interactif principal.

        Ce module implémente les fonctionnalités essentielles pour tableau de bord interactif principal.
        Il fournit une architecture robuste et évolutive intégrant les composants
        nécessaires au fonctionnement optimal du système. L'implémentation suit
    les principes de modularité et d'extensibilité pour faciliter l'évolution
    future du système.

    Le module gère l'initialisation, la configuration, le traitement des données,
    la communication inter-composants, et la persistance des états. Il s'intègre
    harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
    une séparation claire des responsabilités.

    L'architecture interne permet une évolution adaptative basée sur les interactions
    et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
    cohérente et authentique.
    """


def __init__(self) -> None:
    self.data_dir = Path(__file__).parent.parent / "data"
    self.output_dir = Path(__file__).parent.parent / "dashboard"
    self.output_file = self.output_dir / "jeffrey_dashboard.html"
    self.eco_scores = {}
    self.ia_registry = {}
    self.comparisons = []
    self.load_data()


def load_data(self):
    """Charge les données nécessaires pour le dashboard"""
    # Création du répertoire de sortie si nécessaire
    os.makedirs(self.output_dir, exist_ok=True)

    # Chargement des écoscores
    eco_scores_path = self.data_dir / "eco_scores.json"
    if eco_scores_path.exists():
        try:
            with open(eco_scores_path, encoding="utf-8") as f:
                self.eco_scores = json.load(f)
            logger.debug(f"Chargement des écoscores pour {len(self.eco_scores)} modèles")
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de format JSON dans le fichier d'écoscores: {e}", exc_info=True)
            self.eco_scores = {}
        except Exception as e:
            logger.error(f"Erreur lors du chargement des écoscores: {e}", exc_info=True)
            self.eco_scores = {}

    # Chargement des données IA
    registry_path = self.data_dir / "ia_registry.yaml"
    if registry_path.exists():
        try:
            import yaml

            with open(registry_path, encoding="utf-8") as f:
                self.ia_registry = yaml.safe_load(f)
            logger.debug(f"Chargement du registre IA réussi: {len(self.ia_registry.get('models', []))} modèles")
        except ImportError:
            logger.error("Module yaml non disponible. Installation nécessaire: pip install pyyaml")
            # Créer un registre vide au cas où
            self.ia_registry = {"models": []}
        except yaml.YAMLError as e:
            logger.error(f"Erreur dans le format YAML du registre IA: {e}", exc_info=True)
            self.ia_registry = {"models": []}
        except Exception as e:
            logger.error(f"Erreur lors du chargement du registre IA: {e}", exc_info=True)
            self.ia_registry = {"models": []}

    # Chargement des comparaisons
    comparisons_path = self.data_dir / "responses_comparison.json"
    if comparisons_path.exists():
        try:
            with open(comparisons_path, encoding="utf-8") as f:
                self.comparisons = json.load(f)
            logger.debug(f"Chargement de {len(self.comparisons)} comparaisons depuis {comparisons_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de format JSON dans le fichier de comparaisons: {e}", exc_info=True)
            self.comparisons = []
        except Exception as e:
            logger.error(f"Erreur lors du chargement des comparaisons: {e}", exc_info=True)
            self.comparisons = []


def calculate_avg_ecoscore(self):
    """Calcule le score moyen d'écologie des modèles"""
    if not self.eco_scores:
        return 0

    scores = [info.get("eco_score", 0) for info in self.eco_scores.values()]
    if not scores:
        return 0
    return sum(scores) / len(scores)


def generate_html(self):
    """Génère le contenu HTML du dashboard"""
    # Nombre de modèles et écoscore moyen
    num_models = len(self.eco_scores)
    avg_ecoscore = self.calculate_avg_ecoscore()
    avg_ecoscore_percentage = int(avg_ecoscore * 100)

    # Date de génération
    current_date = datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")

    # Formatage des données pour le graphique d'écoscore
    ecoscore_models = list(self.eco_scores.keys())
    ecoscore_values = [info.get("eco_score", 0) for model, info in self.eco_scores.items()]
    ecoscore_labels = [info.get("eco_label", "Non défini") for model, info in self.eco_scores.items()]

    # JSON pour les graphiques
    chart_data = {
        "models": ecoscore_models,
        "scores": ecoscore_values,
        "labels": ecoscore_labels,
    }

    # HTML
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jeffrey - Dashboard</title>

    <!-- Bootstrap 5 CDN -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">

    <!-- Chart.js CDN -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.css">

    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css">

    <!-- Poppins Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css">

    <style>
        :root {{
            --bg-color: #F4F6F7;
            --card-bg: #E8EEF1;
            --text-color: #202124;
            --accent-blue: #90CAF9;
            --accent-green: #A8D5BA;
            --shadow-sm: 0 2px 5px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 10px rgba(0,0,0,0.08);
            --border-radius: 12px;
            --transition: all 0.3s ease;
        }}

        body {{
            font-family: 'Poppins', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding-bottom: 2rem;
        }}

        .navbar {{
            background-color: white;
            box-shadow: var(--shadow-sm);
        }}

        .navbar-brand {{
            font-weight: 600;
            display: flex;
            align-items: center;
        }}

        .logo-placeholder {{
            width: 40px;
            height: 40px;
            background-color: var(--accent-blue);
            border-radius: 50%;
            margin-right: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}

        .card {{
            background-color: var(--card-bg);
            border: none;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-sm);
            transition: var(--transition);
            overflow: hidden;
            height: 100%;
        }}

        .card:hover {{
            box-shadow: var(--shadow-md);
            transform: translateY(-3px);
        }}

        .card-header {{
            background-color: white;
            border-bottom: 1px solid rgba(0,0,0,0.05);
            font-weight: 500;
            padding: 1rem;
        }}

        .card-body {{
            padding: 1.5rem;
        }}

        .btn-primary {{
            background-color: var(--accent-blue);
            border-color: var(--accent-blue);
        }}

        .btn-success {{
            background-color: var(--accent-green);
            border-color: var(--accent-green);
        }}

        .badge-eco {{
            background-color: var(--accent-green);
            color: var(--text-color);
            font-weight: 500;
        }}

        .progress-bar {{
            background-color: var(--accent-blue);
        }}

        .progress {{
            height: 10px;
            border-radius: 5px;
        }}

        .summary-icon {{
            width: 50px;
            height: 50px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            margin-right: 1rem;
        }}

        .summary-icon.models {{
            background-color: rgba(144, 202, 249, 0.2);
            color: var(--accent-blue);
        }}

        .summary-icon.eco {{
            background-color: rgba(168, 213, 186, 0.2);
            color: var(--accent-green);
        }}

        .slogan {{
            font-style: italic;
            color: #666;
            font-weight: 300;
        }}

        .welcomeSection {{
            background: linear-gradient(to right, rgba(144, 202, 249, 0.1), rgba(168, 213, 186, 0.1));
            border-radius: var(--border-radius);
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-sm);
        }}

        .chart-container {{
            position: relative;
            height: 300px;
            width: 100%;
        }}

        .filter-control {{
            padding: 0.5rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }}

        .dataTables_wrapper .dataTables_paginate .paginate_button.current {{
            background: var(--accent-blue) !important;
            color: white !important;
            border: none !important;
        }}

        .dataTables_wrapper .dataTables_paginate .paginate_button:hover {{
            background: var(--accent-green) !important;
            color: white !important;
            border: none !important;
        }}

        /* Animation subtile */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .animate {{
            animation: fadeIn 0.5s ease forwards;
        }}

        .delayed-1 {{ animation-delay: 0.1s; }}
        .delayed-2 {{ animation-delay: 0.2s; }}
        .delayed-3 {{ animation-delay: 0.3s; }}
        .delayed-4 {{ animation-delay: 0.4s; }}

        /* Footer */
        .footer {{
            padding: 1rem 0;
            text-align: center;
            margin-top: 2rem;
            font-size: 0.9rem;
            color: #666;
        }}

        /* Checkbox personnalisé */
        .form-check-input:checked {{
            background-color: var(--accent-blue);
            border-color: var(--accent-blue);
        }}

        /* Input group style */
        .input-group-text {{
            background-color: var(--accent-blue);
            color: white;
            border: none;
        }}

        .form-control {{
            border-radius: var(--border-radius);
            border: 1px solid rgba(0,0,0,0.1);
        }}

        .form-control:focus {{
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 0.2rem rgba(144, 202, 249, 0.25);
        }}

        /* Bouton d'exportation */
        .btn-export {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: var(--accent-green);
            color: white;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--shadow-md);
            transition: var(--transition);
            z-index: 100;
        }}

        .btn-export:hover {{
            transform: scale(1.1);
        }}

        /* Tooltip Personnalisé */
        .tooltip-inner {{
            background-color: var(--accent-blue);
            border-radius: 4px;
        }}

        .bs-tooltip-auto[data-popper-placement^=top] .tooltip-arrow::before,
        .bs-tooltip-top .tooltip-arrow::before {{
            border-top-color: var(--accent-blue);
        }}

        /* Tableaux améliorés */
        .table thead th {{
            background-color: rgba(144, 202, 249, 0.2);
            border-bottom: none;
            font-weight: 500;
        }}

        .table-striped tbody tr:nth-of-type(odd) {{
            background-color: rgba(144, 202, 249, 0.05);
        }}
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-light sticky-top mb-4">
        <div class="container">
            <a class="navbar-brand" href="#">
                <div class="logo-placeholder">
                    <i class="fas fa-user-tie"></i>
                </div>
                Jeffrey
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <span class="nav-link">
                            <i class="fas fa-calendar-alt me-1"></i> {current_date}
                        </span>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container">
        <!-- Section de bienvenue -->
        <div class="welcomeSection animate">
            <h1>Bienvenue sur votre Dashboard</h1>
            <p class="slogan mb-0">Jeffrey, l'intelligence qui apaise.</p>
        </div>

        <!-- Résumé de l'Orchestrateur -->
        <div class="row mb-4">
            <div class="col-md-6 animate delayed-1">
                <div class="card h-100">
                    <div class="card-header d-flex align-items-center">
                        <i class="fas fa-robot me-2"></i>
                        Modèles IA
                    </div>
                    <div class="card-body">
                        <div class="d-flex align-items-center">
                            <div class="summary-icon models">
                                <i class="fas fa-microchip"></i>
                            </div>
                            <div>
                                <h3 class="mb-0">{num_models}</h3>
                                <p class="text-muted mb-0">Modèles disponibles</p>
                            </div>
                        </div>
                        <div class="mt-3">
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <span class="text-muted small">Capacité</span>
                                <span class="badge bg-primary">Prêt</span>
                            </div>
                            <div class="progress">
                                <div class="progress-bar" role="progressbar" style="width: 85%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="col-md-6 animate delayed-2">
                <div class="card h-100">
                    <div class="card-header d-flex align-items-center">
                        <i class="fas fa-leaf me-2"></i>
                        Performance écologique
                    </div>
                    <div class="card-body">
                        <div class="d-flex align-items-center">
                            <div class="summary-icon eco">
                                <i class="fas fa-seedling"></i>
                            </div>
                            <div>
                                <h3 class="mb-0">{avg_ecoscore_percentage}%</h3>
                                <p class="text-muted mb-0">EcoScore Moyen</p>
                            </div>
                        </div>
                        <div class="mt-3">
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <span class="text-muted small">Efficacité</span>
                                <span class="badge badge-eco">Modérée +</span>
                            </div>
                            <div class="progress">
                                <div class="progress-bar bg-success" role="progressbar" style="width: {avg_ecoscore_percentage}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Visualisation des EcoScores -->
        <div class="row mb-4">
            <div class="col-lg-8 animate delayed-3">
                <div class="card h-100">
                    <div class="card-header d-flex align-items-center justify-content-between">
                        <div>
                            <i class="fas fa-chart-bar me-2"></i>
                            EcoScores par modèle
                        </div>
                        <div class="btn-group btn-group-sm">
                            <button type="button" class="btn btn-outline-primary active" id="viewBarChart">
                                <i class="fas fa-chart-bar"></i>
                            </button>
                            <button type="button" class="btn btn-outline-primary" id="viewRadarChart">
                                <i class="fas fa-chart-line"></i>
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="ecoscore-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <div class="col-lg-4 animate delayed-4">
                <div class="card h-100">
                    <div class="card-header d-flex align-items-center">
                        <i class="fas fa-filter me-2"></i>
                        Filtres
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">Score écologique minimal</label>
                            <input type="range" class="form-range filter-control" id="ecoScoreRange" min="0" max="1" step="0.1" value="0">
                            <div class="d-flex justify-content-between">
                                <span class="text-muted small">0%</span>
                                <span class="text-muted small">100%</span>
                            </div>
                        </div>

                        <div class="mb-3">
                            <label class="form-label">Modèles</label>
                            <div class="form-check">
                                <input class="form-check-input model-filter" type="checkbox" value="gpt-4" id="modelGPT" checked>
                                <label class="form-check-label" for="modelGPT">GPT-4</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input model-filter" type="checkbox" value="claude-3" id="modelClaude" checked>
                                <label class="form-check-label" for="modelClaude">Claude 3</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input model-filter" type="checkbox" value="grok" id="modelGrok" checked>
                                <label class="form-check-label" for="modelGrok">Grok</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input model-filter" type="checkbox" value="llama3" id="modelLlama" checked>
                                <label class="form-check-label" for="modelLlama">Llama 3</label>
                            </div>
                        </div>

                        <button class="btn btn-primary w-100">
                            <i class="fas fa-sync-alt me-2"></i>Appliquer
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Dernières comparaisons -->
        <div class="row mb-4 animate delayed-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex align-items-center justify-content-between">
                        <div>
                            <i class="fas fa-exchange-alt me-2"></i>
                            Dernières comparaisons IA
                        </div>
                        <div>
                            <button class="btn btn-sm btn-outline-primary">
                                <i class="fas fa-download me-1"></i>Export
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped" id="comparisons-table">
                                <thead>
                                    <tr>
                                        <th>Date</th>
                                        <th>Prompt</th>
                                        <th>Modèles</th>
                                        <th>Note</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Ces lignes seront remplies dynamiquement par JavaScript -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Gestion IA Rapide -->
        <div class="row animate delayed-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex align-items-center">
                        <i class="fas fa-bolt me-2"></i>
                        Test rapide
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">Sélectionner un modèle</label>
                            <select class="form-select" id="quick-test-model">
                                <option value="gpt-4">GPT-4</option>
                                <option value="claude-3">Claude 3</option>
                                <option value="grok">Grok</option>
                                <option value="llama3">Llama 3</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Votre prompt</label>
                            <textarea class="form-control" id="quick-test-prompt" rows="3" placeholder="Entrez votre prompt ici..."></textarea>
                        </div>
                        <button class="btn btn-primary" id="quick-test-button">
                            <i class="fas fa-paper-plane me-2"></i>Tester
                        </button>

                        <div class="mt-4 d-none" id="quick-test-result-container">
                            <label class="form-label">Réponse</label>
                            <div class="card bg-light">
                                <div class="card-body">
                                    <p id="quick-test-result" class="mb-0"></p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>© 2025 Jeffrey - Orchestrateur IA | Généré le {current_date}</p>
        </div>
    </div>

    <!-- Bouton d'exportation flottant -->
    <button class="btn btn-export" data-bs-toggle="tooltip" data-bs-placement="top" title="Exporter le rapport">
        <i class="fas fa-file-export"></i>
    </button>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>

    <script>
    // Initialisation des tooltips Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {{
        return new bootstrap.Tooltip(tooltipTriggerEl)
    }});

    // Données pour les graphiques
    const chartData = JSON.parse('{json.dumps(chart_data)}');

    // Fonction pour créer le graphique en barres des écoscores
    function createBarChart() {{
        const ctx = document.getElementById('ecoscore-chart').getContext('2d');

        // Détruire l'instance de graphique existante si elle existe
        if (window.ecoScoreChart) {{
            window.ecoScoreChart.destroy();
        }}

        window.ecoScoreChart = new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: chartData.models,
                datasets: [{{
                    label: 'EcoScore',
                    data: chartData.scores.map(score => score * 100), // Convertir en pourcentage
                    backgroundColor: [
                        'rgba(144, 202, 249, 0.6)',
                        'rgba(156, 204, 101, 0.6)',
                        'rgba(255, 167, 38, 0.6)',
                        'rgba(239, 83, 80, 0.6)'
                    ],
                    borderColor: [
                        'rgba(144, 202, 249, 1)',
                        'rgba(156, 204, 101, 1)',
                        'rgba(255, 167, 38, 1)',
                        'rgba(239, 83, 80, 1)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                let index = context.dataIndex;
        return `${{chartData.labels[index]}}: ${{context.parsed.y}}%`;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
        return value + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});
    }}

    // Fonction pour créer le graphique radar
    function createRadarChart() {{
        const ctx = document.getElementById('ecoscore-chart').getContext('2d');

        // Détruire l'instance de graphique existante si elle existe
        if (window.ecoScoreChart) {{
            window.ecoScoreChart.destroy();
        }}

        window.ecoScoreChart = new Chart(ctx, {{
            type: 'radar',
            data: {{
                labels: chartData.models,
                datasets: [{{
                    label: 'EcoScore',
                    data: chartData.scores.map(score => score * 100),
                    backgroundColor: 'rgba(144, 202, 249, 0.2)',
                    borderColor: 'rgba(144, 202, 249, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(144, 202, 249, 1)',
                    pointHoverRadius: 5
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        angleLines: {{
                            display: true
                        }},
                        suggestedMin: 0,
                        suggestedMax: 100
                    }}
                }}
            }}
        }});
    }}

    // Commutation entre les types de graphiques
    document.getElementById('viewBarChart').addEventListener('click', function() {{
        this.classList.add('active');
        document.getElementById('viewRadarChart').classList.remove('active');
        createBarChart();
    }});

    document.getElementById('viewRadarChart').addEventListener('click', function() {{
        this.classList.add('active');
        document.getElementById('viewBarChart').classList.remove('active');
        createRadarChart();
    }});

    // Chargement des comparaisons dans le tableau
    function loadComparisons() {{
        const tableBody = document.querySelector('#comparisons-table tbody');
        tableBody.innerHTML = '';

        // Utiliser les données de comparaison chargées dynamiquement
        const comparisons = {json.dumps(self.comparisons)};

        comparisons.forEach(comp => {{
            const row = document.createElement('tr');

            // Date formatée
            const date = new Date(comp.timestamp.replace(' ', 'T'));
            const formattedDate = date.toLocaleDateString('fr-FR', {{
                day: '2-digit',
                month: '2-digit',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            }});

            // Prompt tronqué
            const truncatedPrompt = comp.prompt.length > 50 ?
                comp.prompt.substring(0, 50) + '...' :
                comp.prompt;

            // Modèles utilisés
            const modelNames = Object.keys(comp.responses).join(', ');

            // Note moyenne si disponible
            let noteHtml = '<span class="text-muted">-</span>';
        if (comp.notes && Object.keys(comp.notes).length > 0) {{
                const notes = Object.values(comp.notes);
        if (notes.length > 0) {{
                    const avgNote = notes.reduce((a, b) => a + b, 0) / notes.length;
                    noteHtml = `<span class="badge bg-primary">${{avgNote.toFixed(1)}}/5</span>`;
                }}
            }}

            // Actions
            const actions = `
                <button class="btn btn-sm btn-outline-primary view-comparison" data-id="${{comp.timestamp}}">
                    <i class="fas fa-eye"></i>
                </button>
            `;

            row.innerHTML = `
                <td>${{formattedDate}}</td>
                <td>${{truncatedPrompt}}</td>
                <td>${{modelNames}}</td>
                <td>${{noteHtml}}</td>
                <td>${{actions}}</td>
            `;

            tableBody.appendChild(row);
        }});

        // Initialiser DataTables
        $('#comparisons-table').DataTable({{
            language: {{
                url: '//cdn.datatables.net/plug-ins/1.11.5/i18n/fr-FR.json'
            }},
            pageLength: 5,
            lengthMenu: [5, 10, 25, 50],
            order: [[0, 'desc']]
        }});
    }}

    // Simulation de test rapide
    document.getElementById('quick-test-button').addEventListener('click', function() {{
        const prompt = document.getElementById('quick-test-prompt').value;
        const model = document.getElementById('quick-test-model').value;

        if (!prompt) {{
            alert('Veuillez entrer un prompt.');
                                                                                                                    return;
        }}

        // Afficher un spinner pendant le "traitement"
        this.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Traitement...';
        this.disabled = true;

        // Simuler un délai de traitement
        setTimeout(() => {{
            // Réponses simulées selon le modèle
            let response = '';

            switch(model) {{
                case 'gpt-4':
                    response = "GPT says: J'ai analysé votre requête et voici ma réponse. [Réponse simulée de GPT-4]";
                                                                                                                        break;
                case 'claude-3':
                    response = "Claude says: J'ai considéré votre prompt et j'ai préparé cette réponse. [Réponse simulée de Claude 3]";
                                                                                                                            break;
                case 'grok':
                    response = "🎭 GROK POETICS 🎭\\n\\nInspired by your prompt, here's a creative response. [Réponse simulée de Grok]";
                                                                                                                                break;
                case 'llama3':
                    response = "Llama 🦙 says: Réponse open-source générée localement avec efficacité. [Réponse simulée de Llama 3]";
                                                                                                                                    break;
                default:
                    response = "Réponse générique du modèle.";
            }}

            // Afficher la réponse
            document.getElementById('quick-test-result').textContent = response;
            document.getElementById('quick-test-result-container').classList.remove('d-none');

            // Réinitialiser le bouton
            this.innerHTML = '<i class="fas fa-paper-plane me-2"></i>Tester';
            this.disabled = false;
        }}, 1500);
    }});

    // Fonction pour exporter le rapport
    document.querySelector('.btn-export').addEventListener('click', function() {{
        // En production, cela pourrait générer un PDF ou un rapport formaté
        alert('Fonctionnalité d\\'exportation : cette action générerait normalement un rapport complet en PDF.');
    }});

    // Initialisation au chargement de la page
    document.addEventListener('DOMContentLoaded', function() {{
        createBarChart();
        loadComparisons();

        // Ajouter des événements pour les filtres
        document.querySelectorAll('.model-filter').forEach(checkbox => {{
            checkbox.addEventListener('change', function() {{
                // Dans une implémentation réelle, ceci filtre les données
                console.log('Filter changed:', this.value, this.checked);
            }});
        }});

        document.getElementById('ecoScoreRange').addEventListener('input', function() {{
            // Dans une implémentation réelle, ceci filtre les données
            console.log('EcoScore minimum:', this.value);
        }});

        // Délégation d'événements pour les boutons de visualisation de comparaison
        document.querySelector('#comparisons-table').addEventListener('click', function(e) {{
        if (e.target.closest('.view-comparison')) {{
                const button = e.target.closest('.view-comparison');
                const id = button.getAttribute('data-id');
                alert(`Détails de la comparaison ${{id}} : cette action afficherait normalement un modal avec les détails complets.`);
            }}
        }});
    }});
    </script>
</body>
</html>
        """

    return html_content


def generate_dashboard(self):
    """Génère le dashboard et l'enregistre"""
    try:
        # Générer le contenu HTML
        html_content = self.generate_html()

        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(self.output_dir, exist_ok=True)

        # Écrire le fichier HTML
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Dashboard généré avec succès : {self.output_file}")
        return self.output_file

    except PermissionError as e:
        logger.error(f"Erreur de permission lors de l'écriture du dashboard: {e}", exc_info=True)
        raise RuntimeError(f"Impossible d'écrire le fichier de dashboard: {e}")

    except FileNotFoundError as e:
        logger.error(f"Dossier ou chemin invalide pour le dashboard: {e}", exc_info=True)
        raise RuntimeError(f"Chemin invalide pour le dashboard: {e}")

    except Exception as e:
        logger.error(f"Erreur lors de la génération du dashboard: {e}", exc_info=True)
        raise RuntimeError(f"Échec de génération du dashboard: {e}")


def generate_dashboard():
    """Fonction principale pour générer le dashboard"""
    generator = DashboardGenerator()
    return generator.generate_dashboard()


if __name__ == "__main__":
    generate_dashboard()
