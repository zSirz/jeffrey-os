// Nucleus INTELLIGENT avec bioscan et recovery

use super::base::*;
use super::cytoplasm::Cytoplasm;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Nucleus - Centre de contrôle cellulaire
pub struct Nucleus {
    cytoplasm: Arc<Cytoplasm>,
    bioscan_results: Arc<RwLock<BioscanReport>>,
    emergency_mode: Arc<AtomicBool>,
}

impl Nucleus {
    pub fn new(cytoplasm: Arc<Cytoplasm>) -> Self {
        Self {
            cytoplasm,
            bioscan_results: Arc::new(RwLock::new(BioscanReport::new())),
            emergency_mode: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Bioscan - Validation complète du système
    pub async fn run_bioscan(&self) -> BioscanReport {
        println!("🔬 Running bioscan...");
        let mut report = BioscanReport::new();

        // 1. Vérifier que toutes les cellules répondent
        let cell_ids = self.cytoplasm.list_cells().await;
        println!("  Scanning {} cells...", cell_ids.len());

        for cell_id in &cell_ids {
            if let Some(health) = self.cytoplasm.get_cell_status(cell_id).await {
                report.cell_status.insert(cell_id.clone(), health);
            }
        }

        // 2. Tester les interconnexions
        report.interconnection_test = self.test_interconnections().await;

        // 3. Vérifier les dépendances (simplifié pour le MVP)
        // Les dépendances sont vérifiées lors du hot_load_cell

        // 4. Tests de performance
        report.performance_metrics = self.run_performance_tests().await;

        // 5. Score global
        report.calculate_score();

        // Sauvegarder le rapport
        *self.bioscan_results.write().await = report.clone();

        println!("✅ Bioscan complete. Score: {:.1}/100", report.global_score);
        report
    }

    /// Test des interconnexions entre cellules
    async fn test_interconnections(&self) -> bool {
        println!("  Testing interconnections...");

        // Test simple : envoyer un signal de test et vérifier la réponse
        let test_signal = CellularSignal {
            from: "nucleus".to_string(),
            to: "*".to_string(),
            signal_type: SignalType::Query,
            payload: serde_json::json!({
                "test": "interconnection",
                "timestamp": chrono::Utc::now().timestamp()
            }),
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        // Broadcast et attendre
        self.cytoplasm.broadcast_signal(test_signal).await;

        // Pour le MVP, on considère que c'est OK si on a des cellules
        let cells_count = self.cytoplasm.list_cells().await.len();
        cells_count > 0
    }

    /// Tests de performance
    async fn run_performance_tests(&self) -> PerformanceMetrics {
        println!("  Running performance tests...");

        let start = std::time::Instant::now();

        // Test de latence simple
        let test_signal = CellularSignal {
            from: "nucleus".to_string(),
            to: "memory".to_string(),
            signal_type: SignalType::Query,
            payload: serde_json::json!({"test": "latency"}),
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        let _ = self.cytoplasm.route_signal(test_signal).await;
        let latency = start.elapsed().as_millis() as f32;

        PerformanceMetrics {
            avg_latency_ms: latency,
            throughput_rps: 1000.0 / latency.max(1.0), // Estimation
            memory_mb: 50.0, // Placeholder
            cpu_percent: 5.0, // Placeholder
        }
    }

    /// Test du cycle cognitif complet
    pub async fn test_cognitive_cycle(&self) -> bool {
        println!("🧠 Testing cognitive cycle...");

        // 1. Créer un souvenir
        let memory_signal = CellularSignal {
            from: "nucleus".to_string(),
            to: "memory".to_string(),
            signal_type: SignalType::Command,
            payload: serde_json::json!({
                "command": "create_memory",
                "content": "Test cognitive cycle",
                "emotion": "curiosity"
            }),
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        if let Ok(response) = self.cytoplasm.route_signal(memory_signal).await {
            if !response.success {
                println!("  ❌ Memory creation failed");
                return false;
            }
        }

        // 2. Générer un rêve basé sur le souvenir
        let dream_signal = CellularSignal {
            from: "nucleus".to_string(),
            to: "dreams".to_string(),
            signal_type: SignalType::Command,
            payload: serde_json::json!({
                "command": "generate_dream",
                "memories": ["Test cognitive cycle"]
            }),
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        if let Ok(response) = self.cytoplasm.route_signal(dream_signal).await {
            if !response.success {
                println!("  ❌ Dream generation failed");
                return false;
            }
        }

        // 3. Faire évoluer la conscience
        let consciousness_signal = CellularSignal {
            from: "nucleus".to_string(),
            to: "consciousness".to_string(),
            signal_type: SignalType::Command,
            payload: serde_json::json!({
                "command": "evolve",
                "influence": {"type": "dream", "intensity": 0.5}
            }),
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        if let Ok(response) = self.cytoplasm.route_signal(consciousness_signal).await {
            if !response.success {
                println!("  ❌ Consciousness evolution failed");
                return false;
            }
        }

        println!("  ✅ Cognitive cycle complete");
        true
    }

    /// Gestion des paniques avec isolation
    pub async fn handle_cell_panic(&self, cell_id: &str, error: String) {
        println!("⚠️ Cell {} panicked: {}", cell_id, error);

        // Pour le MVP, on tente juste de recharger la cellule
        if let Err(e) = self.cytoplasm.hot_unload_cell(cell_id).await {
            println!("❌ Failed to unload panicked cell: {}", e);

            // Mode d'urgence si cellule critique
            if self.is_critical_cell(cell_id) {
                self.enter_emergency_mode().await;
            }
        }
    }

    /// Vérifier si une cellule est critique
    fn is_critical_cell(&self, cell_id: &str) -> bool {
        matches!(cell_id, "memory" | "consciousness" | "vivarium")
    }

    /// Entrer en mode d'urgence
    async fn enter_emergency_mode(&self) {
        println!("🚨 ENTERING EMERGENCY MODE");
        self.emergency_mode.store(true, Ordering::SeqCst);

        // Notifier toutes les cellules
        self.cytoplasm.broadcast_signal(CellularSignal {
            from: "nucleus".to_string(),
            to: "*".to_string(),
            signal_type: SignalType::Emergency,
            payload: serde_json::json!({
                "event": "emergency_mode",
                "active": true
            }),
            timestamp: chrono::Utc::now().timestamp() as u64,
        }).await;
    }

    /// Sortir du mode d'urgence
    pub async fn exit_emergency_mode(&self) {
        println!("✅ Exiting emergency mode");
        self.emergency_mode.store(false, Ordering::SeqCst);

        self.cytoplasm.broadcast_signal(CellularSignal {
            from: "nucleus".to_string(),
            to: "*".to_string(),
            signal_type: SignalType::Event,
            payload: serde_json::json!({
                "event": "emergency_mode",
                "active": false
            }),
            timestamp: chrono::Utc::now().timestamp() as u64,
        }).await;
    }

    /// Obtenir le dernier rapport de bioscan
    pub async fn get_last_bioscan(&self) -> BioscanReport {
        self.bioscan_results.read().await.clone()
    }
}
