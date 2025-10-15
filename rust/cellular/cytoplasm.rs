// Cytoplasm INTELLIGENT avec hot-reload et plugins

use super::base::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, VecDeque};

/// Cytoplasm - Milieu cellulaire intelligent
pub struct Cytoplasm {
    cells: Arc<RwLock<HashMap<String, Arc<RwLock<Box<dyn CellularMembrane>>>>>>,
    signal_queue: Arc<RwLock<VecDeque<CellularSignal>>>,
    metrics_collector: Arc<RwLock<MetricsCollector>>,
}

impl Cytoplasm {
    pub fn new() -> Self {
        Self {
            cells: Arc::new(RwLock::new(HashMap::new())),
            signal_queue: Arc::new(RwLock::new(VecDeque::new())),
            metrics_collector: Arc::new(RwLock::new(MetricsCollector::new())),
        }
    }

    /// Charger une cellule à chaud
    pub async fn hot_load_cell(&self, cell: Arc<RwLock<Box<dyn CellularMembrane>>>) -> Result<(), CellularError> {
        let dna = {
            let cell_guard = cell.read().await;
            cell_guard.dna().clone()
        };

        println!("🔬 Hot-loading cell: {}", dna.name);

        // Vérifier compatibilité
        self.check_compatibility(&dna).await?;

        // Initialiser la cellule
        {
            let mut cell_guard = cell.write().await;
            cell_guard.initialize().await?;
        }

        // Enregistrer
        {
            let mut cells = self.cells.write().await;
            cells.insert(dna.id.clone(), cell);
        }

        // Notifier les autres cellules
        self.broadcast_signal(CellularSignal {
            from: "cytoplasm".to_string(),
            to: "*".to_string(),
            signal_type: SignalType::Event,
            payload: serde_json::json!({
                "event": "cell_loaded",
                "cell_id": dna.id
            }),
            timestamp: chrono::Utc::now().timestamp() as u64,
        }).await;

        println!("✅ Cell {} loaded successfully", dna.id);
        Ok(())
    }

    /// Éjecter une cellule à chaud
    pub async fn hot_unload_cell(&self, cell_id: &str) -> Result<(), CellularError> {
        println!("🔬 Hot-unloading cell: {}", cell_id);

        let mut cells = self.cells.write().await;

        if let Some(cell) = cells.get(cell_id) {
            // Hibernation propre
            {
                let mut cell_guard = cell.write().await;
                cell_guard.hibernate().await?;
            }

            // Retirer de la liste
            cells.remove(cell_id);

            // Notifier
            self.broadcast_signal(CellularSignal {
                from: "cytoplasm".to_string(),
                to: "*".to_string(),
                signal_type: SignalType::Event,
                payload: serde_json::json!({
                    "event": "cell_unloaded",
                    "cell_id": cell_id
                }),
                timestamp: chrono::Utc::now().timestamp() as u64,
            }).await;

            println!("✅ Cell {} unloaded", cell_id);
        } else {
            return Err(CellularError::Unknown(format!("Cell {} not found", cell_id)));
        }

        Ok(())
    }

    /// Router un signal vers la cellule appropriée
    pub async fn route_signal(&self, signal: CellularSignal) -> Result<CellularResponse, CellularError> {
        let cells = self.cells.read().await;

        // Broadcast ou unicast
        if signal.to == "*" {
            // Broadcast à toutes les cellules
            for (id, cell) in cells.iter() {
                if id != &signal.from {
                    let mut cell_guard = cell.write().await;
                    let _ = cell_guard.receive_signal(signal.clone()).await;
                }
            }
            Ok(CellularResponse::default())
        } else {
            // Unicast à une cellule spécifique
            if let Some(cell) = cells.get(&signal.to) {
                let mut cell_guard = cell.write().await;
                Ok(cell_guard.receive_signal(signal).await)
            } else {
                Err(CellularError::CommunicationError(
                    format!("Target cell {} not found", signal.to)
                ))
            }
        }
    }

    /// Broadcast un signal à toutes les cellules
    pub async fn broadcast_signal(&self, signal: CellularSignal) {
        let cells = self.cells.read().await;
        for (id, cell) in cells.iter() {
            if id != &signal.from {
                let mut cell_guard = cell.write().await;
                let _ = cell_guard.receive_signal(signal.clone()).await;
            }
        }
    }

    /// Vérifier la compatibilité d'une cellule
    async fn check_compatibility(&self, dna: &CellularDNA) -> Result<(), CellularError> {
        // Vérifier version système
        if !is_version_compatible(&dna.compatible_since, SYSTEM_VERSION) {
            return Err(CellularError::CompatibilityError(
                format!("Cell requires version {} but system is {}",
                    dna.compatible_since, SYSTEM_VERSION)
            ));
        }

        // Vérifier dépendances
        let cells = self.cells.read().await;
        for dep in &dna.dependencies {
            if dep.required && !cells.contains_key(&dep.cell_id) {
                return Err(CellularError::DependencyError(
                    format!("Required dependency {} not found", dep.cell_id)
                ));
            }
        }

        Ok(())
    }

    /// Obtenir la liste des cellules actives
    pub async fn list_cells(&self) -> Vec<String> {
        let cells = self.cells.read().await;
        cells.keys().cloned().collect()
    }

    /// Obtenir le statut d'une cellule
    pub async fn get_cell_status(&self, cell_id: &str) -> Option<HealthStatus> {
        let cells = self.cells.read().await;
        if let Some(cell) = cells.get(cell_id) {
            let cell_guard = cell.read().await;
            Some(cell_guard.health_check().await)
        } else {
            None
        }
    }
}

/// Collecteur de métriques
pub struct MetricsCollector {
    metrics: HashMap<String, CellularMetrics>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    pub fn record(&mut self, cell_id: String, metrics: CellularMetrics) {
        self.metrics.insert(cell_id, metrics);
    }

    pub fn get_all(&self) -> &HashMap<String, CellularMetrics> {
        &self.metrics
    }
}
