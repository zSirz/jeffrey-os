use anyhow::{anyhow, Result};
use std::collections::HashSet;
use wasmtime::{Config, Engine, Linker, Module, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};

use super::{SkillCapabilityContract, ValidationReport};

/// Membrane d'isolation pour une cellule WASM
/// Contrôle strictement les imports et exports autorisés
#[derive(Debug, Clone)]
pub struct Membrane {
    pub cell_id: uuid::Uuid,
    pub allowed_imports: HashSet<String>,
    pub blocked_imports: HashSet<String>,
    pub capability_token: Option<String>,
    pub wasi_config: WasiConfig,
}

#[derive(Debug, Clone)]
pub struct WasiConfig {
    pub allow_network: bool,
    pub allow_filesystem: bool,
    pub allowed_dirs: Vec<String>,
    pub env_vars: Vec<(String, String)>,
    pub max_memory_pages: u32,
}

impl Membrane {
    /// Créer une membrane depuis un SCC validé
    pub fn from_scc(scc: &SkillCapabilityContract, report: &ValidationReport) -> Result<Self> {
        let mut allowed_imports = HashSet::new();
        let mut blocked_imports = HashSet::new();

        // Par défaut, bloquer tout réseau et filesystem
        blocked_imports.insert("wasi_snapshot_preview1::sock_*".to_string());
        blocked_imports.insert("wasi_snapshot_preview1::path_*".to_string());
        blocked_imports.insert("wasi_snapshot_preview1::fd_write".to_string());
        blocked_imports.insert("wasi_snapshot_preview1::fd_read".to_string());

        // Autoriser seulement si explicitement permis ET consentement donné
        if scc.permissions.network && report.requires_consent {
            // Le token de capacité sera vérifié à l'exécution
            allowed_imports.insert("jeffrey::network::fetch".to_string());
        }

        // Autoriser les capteurs demandés
        for sensor in &scc.permissions.sensors {
            allowed_imports.insert(format!("jeffrey::sensors::{}", sensor));
        }

        // Autoriser les actions déclarées
        for action in &scc.actions {
            allowed_imports.insert(format!("jeffrey::actions::{}", action));
        }

        // Configuration WASI minimale
        let wasi_config = WasiConfig {
            allow_network: false, // Toujours false par défaut
            allow_filesystem: false,
            allowed_dirs: vec![],
            env_vars: vec![
                ("JEFFREY_SKILL_ID".to_string(), scc.id.clone()),
                ("JEFFREY_SKILL_VERSION".to_string(), scc.version.clone()),
            ],
            max_memory_pages: 256, // ~16MB max
        };

        Ok(Membrane {
            cell_id: uuid::Uuid::new_v4(),
            allowed_imports,
            blocked_imports,
            capability_token: None,
            wasi_config,
        })
    }

    /// Vérifier si un import est autorisé
    pub fn is_import_allowed(&self, import_name: &str) -> bool {
        // D'abord vérifier les blocages explicites
        for blocked_pattern in &self.blocked_imports {
            if import_name.starts_with(blocked_pattern.trim_end_matches('*')) {
                return false;
            }
        }

        // Ensuite vérifier les autorisations
        for allowed_pattern in &self.allowed_imports {
            if import_name == allowed_pattern ||
               (allowed_pattern.ends_with('*') &&
                import_name.starts_with(allowed_pattern.trim_end_matches('*'))) {
                return true;
            }
        }

        // Par défaut, refuser
        false
    }

    /// Créer un moteur WASM configuré avec les restrictions
    pub fn create_engine(&self) -> Result<Engine> {
        let mut config = Config::new();

        // Activer les protections
        config.consume_fuel(true);
        config.epoch_interruption(true);
        config.memory_guaranteed_dense_image_size(1 << 20); // 1MB dense
        config.memory_init_cow(false); // Pas de CoW pour isolation
        config.allocation_strategy(wasmtime::InstanceAllocationStrategy::pooling());

        // Limites strictes
        config.max_wasm_stack(512 * 1024); // 512KB stack

        Engine::new(&config).map_err(|e| anyhow!("Failed to create engine: {}", e))
    }

    /// Créer un contexte WASI avec les restrictions
    pub fn create_wasi_ctx(&self) -> Result<WasiCtx> {
        let mut builder = WasiCtxBuilder::new();

        // Ajouter les variables d'environnement
        for (key, value) in &self.wasi_config.env_vars {
            builder = builder.env(key, value)?;
        }

        // Pas d'accès réseau par défaut
        if !self.wasi_config.allow_network {
            builder = builder.inherit_network(false);
        }

        // Pas d'accès filesystem par défaut
        if !self.wasi_config.allow_filesystem {
            builder = builder.inherit_stdio(false);
        }

        Ok(builder.build())
    }

    /// Injecter un token de capacité après consentement
    pub fn inject_capability_token(&mut self, token: String) {
        self.capability_token = Some(token);
        tracing::info!("Capability token injected for cell {}", self.cell_id);
    }

    /// Vérifier si une opération nécessite un token
    pub fn check_capability(&self, operation: &str) -> Result<()> {
        let needs_token = matches!(operation,
            "network_access" | "sensor_access" | "data_write" | "notification_send"
        );

        if needs_token && self.capability_token.is_none() {
            return Err(anyhow!("ERR_NO_CAPABILITY: Operation '{}' requires capability token", operation));
        }

        Ok(())
    }
}

/// Export des permissions pour utilisation externe
pub use super::scc_validator::Permissions;
