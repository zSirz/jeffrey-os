use vaultrs::{
    client::{VaultClient, VaultClientSettings},
    auth::kubernetes,
    kv2,
};
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use std::time::Instant;
use tracing::{info, warn, error, debug};

use crate::config::VaultConfig;

#[derive(Debug, Clone)]
pub struct SecretValue {
    pub data: serde_json::Value,
    pub fetched_at: DateTime<Utc>,
    pub ttl: Duration,
}

impl SecretValue {
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.fetched_at + self.ttl
    }
}

pub struct VaultSecrets {
    client: VaultClient,
    cache: Arc<RwLock<HashMap<String, SecretValue>>>,
    throttle: Arc<Semaphore>,
    config: VaultConfig,
}

impl VaultSecrets {
    /// Initialize Vault client with Kubernetes auth
    pub async fn new(config: VaultConfig) -> Result<Self> {
        // Create client with address from config (externalized)
        let mut settings = VaultClientSettings::default();
        settings.address = config.address.clone();

        // Add CA cert if in Kubernetes
        if config.kubernetes_auth {
            if let Ok(ca_cert) = tokio::fs::read_to_string("/var/run/secrets/vault/ca.crt").await {
                settings.ca_certs = vec![ca_cert];
            }
        }

        let client = VaultClient::new(settings)?;

        // Authenticate
        if config.kubernetes_auth {
            // Read service account token
            let jwt = tokio::fs::read_to_string(
                "/var/run/secrets/kubernetes.io/serviceaccount/token"
            ).await?;

            // Login with Kubernetes auth
            let auth_info = kubernetes::login(
                &client,
                "kubernetes",
                &config.role,
                &jwt,
            ).await?;

            client.set_token(&auth_info.client_token);

            info!(
                role = %config.role,
                policies = ?auth_info.policies,
                "Successfully authenticated to Vault via Kubernetes"
            );
        } else {
            // Dev mode - use token from env
            if let Ok(token) = std::env::var("VAULT_TOKEN") {
                client.set_token(&token);
                info!("Using Vault token from environment (dev mode)");
            } else {
                return Err(anyhow!("No Vault authentication method available"));
            }
        }

        // Create semaphore for request throttling
        let throttle = Arc::new(Semaphore::new(config.max_concurrent_requests));

        let secrets = Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            throttle,
            config,
        };

        // Start background refresh task
        secrets.start_refresh_loop();

        Ok(secrets)
    }

    /// Get secret with caching and throttling
    pub async fn get(&self, path: &str) -> Result<serde_json::Value> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(value) = cache.get(path) {
                if !value.is_expired() {
                    metrics::counter!("vault.cache_hits").increment(1);
                    return Ok(value.data.clone());
                }
            }
        }

        metrics::counter!("vault.cache_misses").increment(1);

        // Acquire permit for throttling
        let _permit = self.throttle.acquire().await?;

        // Double-check cache after acquiring permit (another task might have fetched)
        {
            let cache = self.cache.read().await;
            if let Some(value) = cache.get(path) {
                if !value.is_expired() {
                    return Ok(value.data.clone());
                }
            }
        }

        // Fetch from Vault
        let start = Instant::now();

        let secret: HashMap<String, serde_json::Value> = kv2::read(
            &self.client,
            &self.config.mount_path,
            path,
        ).await.map_err(|e| anyhow!("Failed to read secret {}: {}", path, e))?;

        let data = serde_json::to_value(secret)?;

        metrics::histogram!("vault.fetch_duration_seconds")
            .record(start.elapsed().as_secs_f64());

        // Update cache
        let value = SecretValue {
            data: data.clone(),
            fetched_at: Utc::now(),
            ttl: Duration::seconds(self.config.refresh_interval_secs as i64),
        };

        {
            let mut cache = self.cache.write().await;
            cache.insert(path.to_string(), value);
        }

        debug!(path = %path, "Secret fetched and cached");

        Ok(data)
    }

    /// Set secret in Vault
    pub async fn set(&self, path: &str, data: serde_json::Value) -> Result<()> {
        // Acquire permit
        let _permit = self.throttle.acquire().await?;

        // Convert to HashMap for kv2
        let secret: HashMap<String, serde_json::Value> = match data {
            serde_json::Value::Object(map) => {
                map.into_iter().collect()
            },
            _ => {
                let mut map = HashMap::new();
                map.insert("value".to_string(), data);
                map
            }
        };

        // Write to Vault
        kv2::set(
            &self.client,
            &self.config.mount_path,
            path,
            &secret,
        ).await.map_err(|e| anyhow!("Failed to write secret {}: {}", path, e))?;

        // Invalidate cache
        {
            let mut cache = self.cache.write().await;
            cache.remove(path);
        }

        info!(path = %path, "Secret written to Vault");

        Ok(())
    }

    /// Delete secret from Vault
    pub async fn delete(&self, path: &str) -> Result<()> {
        // Acquire permit
        let _permit = self.throttle.acquire().await?;

        // Delete from Vault
        kv2::delete(
            &self.client,
            &self.config.mount_path,
            path,
        ).await.map_err(|e| anyhow!("Failed to delete secret {}: {}", path, e))?;

        // Remove from cache
        {
            let mut cache = self.cache.write().await;
            cache.remove(path);
        }

        info!(path = %path, "Secret deleted from Vault");

        Ok(())
    }

    /// Start background refresh loop for cached secrets
    fn start_refresh_loop(&self) {
        let cache = self.cache.clone();
        let client = self.client.clone();
        let config = self.config.clone();
        let throttle = self.throttle.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(config.refresh_interval_secs / 2)
            );

            loop {
                interval.tick().await;

                // Get list of paths to refresh
                let paths_to_refresh: Vec<String> = {
                    let cache_read = cache.read().await;
                    cache_read
                        .iter()
                        .filter(|(_, v)| {
                            // Refresh if expires in next interval
                            let expires_soon = v.fetched_at + v.ttl
                                < Utc::now() + Duration::seconds(config.refresh_interval_secs as i64);
                            expires_soon
                        })
                        .map(|(k, _)| k.clone())
                        .collect()
                };

                for path in paths_to_refresh {
                    // Acquire permit
                    let _permit = match throttle.acquire().await {
                        Ok(p) => p,
                        Err(e) => {
                            error!("Failed to acquire throttle permit: {}", e);
                            continue;
                        }
                    };

                    // Refresh secret
                    match kv2::read::<HashMap<String, serde_json::Value>>(
                        &client,
                        &config.mount_path,
                        &path,
                    ).await {
                        Ok(secret) => {
                            let data = serde_json::to_value(secret).unwrap();
                            let value = SecretValue {
                                data,
                                fetched_at: Utc::now(),
                                ttl: Duration::seconds(config.refresh_interval_secs as i64),
                            };

                            let mut cache_write = cache.write().await;
                            cache_write.insert(path.clone(), value);

                            debug!(path = %path, "Secret refreshed in background");
                        },
                        Err(e) => {
                            warn!(path = %path, error = %e, "Failed to refresh secret");
                        }
                    }
                }

                metrics::gauge!("vault.cache_size")
                    .set(cache.read().await.len() as f64);
            }
        });
    }

    /// Health check
    pub async fn health_check(&self) -> Result<bool> {
        let _permit = self.throttle.acquire().await?;

        match self.client.status().await {
            Ok(status) => {
                Ok(!status.sealed)
            },
            Err(e) => {
                error!("Vault health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

// Mock metrics
mod metrics {
    pub struct Counter;
    pub struct Histogram;
    pub struct Gauge;

    impl Counter {
        pub fn increment(&self, _: u64) {}
    }

    impl Histogram {
        pub fn record(&self, _: f64) {}
    }

    impl Gauge {
        pub fn set(&self, _: f64) {}
    }

    pub fn counter(_: &str) -> Counter { Counter }
    pub fn histogram(_: &str) -> Histogram { Histogram }
    pub fn gauge(_: &str) -> Gauge { Gauge }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_expiration() {
        let secret = SecretValue {
            data: serde_json::json!({"key": "value"}),
            fetched_at: Utc::now() - Duration::hours(2),
            ttl: Duration::hours(1),
        };

        assert!(secret.is_expired());

        let fresh_secret = SecretValue {
            data: serde_json::json!({"key": "value"}),
            fetched_at: Utc::now(),
            ttl: Duration::hours(1),
        };

        assert!(!fresh_secret.is_expired());
    }
}
