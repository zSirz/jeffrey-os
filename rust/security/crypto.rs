//! Cryptographic operations for envelope signing and verification
//!
//! This module implements Ed25519 signature operations with domain separation
//! for Jeffrey Bridge v0.2 envelopes. It provides both single and batch
//! verification capabilities with proper error handling.

use ed25519_dalek::{
    Keypair, PublicKey, SecretKey, Signature, Signer, Verifier,
    PUBLIC_KEY_LENGTH, SECRET_KEY_LENGTH, SIGNATURE_LENGTH,
};
use rand::rngs::OsRng;
use std::collections::HashMap;
use tracing::{debug, info, warn, error};
use zeroize::Zeroize;

use crate::envelope::{Envelope, JEFFREY_BRIDGE_DOMAIN};
use crate::envelope::keys::{KeyRing, KeyMetadata};

/// Context for signing envelopes with Ed25519
#[derive(Debug)]
pub struct SigningContext {
    keypair: Keypair,
    key_id: String,
    metadata: Option<KeyMetadata>,
}

/// Context for verifying envelope signatures
#[derive(Debug, Clone)]
pub struct VerificationContext {
    keys: HashMap<String, PublicKey>,
    primary_key: Option<String>,
}

/// Result of batch verification operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchVerificationReport {
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
    pub errors: Vec<BatchVerificationError>,
}

/// Comprehensive batch verify report with phase breakdown
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchVerifyReport {
    pub total: usize,
    pub ok: usize,
    pub failed: usize,
    pub pre_filter_failed: usize,
    pub sig_verify_failed: usize,
    pub replay_failed: usize,
}

/// Error details for failed batch verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchVerificationError {
    pub index: usize,
    pub envelope_id: uuid::Uuid,
    pub error: String,
}

/// Cryptographic errors
#[derive(thiserror::Error, Debug)]
pub enum CryptoError {
    #[error("Invalid key format: {0}")]
    InvalidKeyFormat(String),

    #[error("Key not found: {0}")]
    KeyNotFound(String),

    #[error("Signature verification failed")]
    SignatureVerificationFailed,

    #[error("Invalid signature format: {0}")]
    InvalidSignatureFormat(String),

    #[error("Random number generation failed: {0}")]
    RandomGenerationFailed(String),

    #[error("Key derivation failed: {0}")]
    KeyDerivationFailed(String),
}

impl From<crate::envelope::keys::KeyError> for CryptoError {
    fn from(error: crate::envelope::keys::KeyError) -> Self {
        CryptoError::KeyDerivationFailed(error.to_string())
    }
}

impl SigningContext {
    /// Create a new ephemeral signing context (for testing/development)
    pub fn new_ephemeral() -> Result<Self, CryptoError> {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let key_id = format!("ephemeral-{}", hex::encode(&keypair.public.to_bytes()[..8]));

        info!(key_id = %key_id, "Created ephemeral signing context");

        Ok(Self {
            keypair,
            key_id,
            metadata: None,
        })
    }

    /// Create signing context from a key ring
    pub fn from_keyring(key_ring: KeyRing) -> Result<Self, CryptoError> {
        let active_key = key_ring.get_active_key()
            .ok_or_else(|| CryptoError::KeyNotFound("No active key in keyring".to_string()))?;

        let metadata = key_ring.get_key_metadata(&active_key.id)?;
        let keypair = Self::keypair_from_key_data(&active_key.secret_key)?;

        info!(key_id = %active_key.id, "Created signing context from keyring");

        Ok(Self {
            keypair,
            key_id: active_key.id.clone(),
            metadata: Some(metadata),
        })
    }

    /// Create signing context from raw key material
    pub fn from_secret_key(secret_key: &[u8], key_id: String) -> Result<Self, CryptoError> {
        if secret_key.len() != SECRET_KEY_LENGTH {
            return Err(CryptoError::InvalidKeyFormat(
                format!("Expected {} bytes, got {}", SECRET_KEY_LENGTH, secret_key.len())
            ));
        }

        let secret = SecretKey::from_bytes(secret_key)
            .map_err(|e| CryptoError::InvalidKeyFormat(format!("Invalid secret key: {}", e)))?;

        let public = PublicKey::from(&secret);
        let keypair = Keypair { secret, public };

        info!(key_id = %key_id, "Created signing context from secret key");

        Ok(Self {
            keypair,
            key_id,
            metadata: None,
        })
    }

    /// Get the public key
    pub fn public_key(&self) -> &PublicKey {
        &self.keypair.public
    }

    /// Get the key identifier
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Get key metadata if available
    pub fn metadata(&self) -> Option<&KeyMetadata> {
        self.metadata.as_ref()
    }

    /// Sign an envelope
    #[tracing::instrument(skip(self, envelope), fields(key_id = %self.key_id, envelope_id = %envelope.header.id))]
    pub async fn sign_envelope(&self, envelope: &Envelope) -> Result<Vec<u8>, crate::error::BridgeError> {
        debug!("Signing envelope with key: {}", self.key_id);

        // Get canonical representation for signing
        let canonical_bytes = envelope.canonical_representation()?;

        // Create domain-separated message
        let message = self.create_domain_separated_message(&canonical_bytes);

        // Sign the message
        let signature = self.keypair.sign(&message);

        debug!(
            signature_len = signature.to_bytes().len(),
            message_len = message.len(),
            "Envelope signing completed"
        );

        Ok(signature.to_bytes().to_vec())
    }

    /// Create verification context from this signing context
    pub fn verification_context(&self) -> VerificationContext {
        let mut keys = HashMap::new();
        keys.insert(self.key_id.clone(), self.keypair.public);

        VerificationContext {
            keys,
            primary_key: Some(self.key_id.clone()),
        }
    }

    /// Export public key as bytes
    pub fn export_public_key(&self) -> [u8; PUBLIC_KEY_LENGTH] {
        self.keypair.public.to_bytes()
    }

    /// Helper to create domain-separated message
    fn create_domain_separated_message(&self, canonical_bytes: &[u8]) -> Vec<u8> {
        let mut message = Vec::with_capacity(JEFFREY_BRIDGE_DOMAIN.len() + canonical_bytes.len());
        message.extend_from_slice(JEFFREY_BRIDGE_DOMAIN);
        message.extend_from_slice(canonical_bytes);
        message
    }

    /// Helper to convert key data to keypair
    fn keypair_from_key_data(key_data: &[u8]) -> Result<Keypair, CryptoError> {
        if key_data.len() != SECRET_KEY_LENGTH {
            return Err(CryptoError::InvalidKeyFormat(
                format!("Expected {} bytes, got {}", SECRET_KEY_LENGTH, key_data.len())
            ));
        }

        let secret = SecretKey::from_bytes(key_data)
            .map_err(|e| CryptoError::InvalidKeyFormat(format!("Invalid secret key: {}", e)))?;

        let public = PublicKey::from(&secret);
        Ok(Keypair { secret, public })
    }
}

impl Drop for SigningContext {
    fn drop(&mut self) {
        // Zero out the secret key
        self.keypair.secret.zeroize();
    }
}

impl VerificationContext {
    /// Create a new empty verification context
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            primary_key: None,
        }
    }

    /// Add a public key to the verification context
    pub fn add_key(&mut self, key_id: String, public_key: PublicKey) {
        info!(key_id = %key_id, "Added public key to verification context");
        if self.primary_key.is_none() {
            self.primary_key = Some(key_id.clone());
        }
        self.keys.insert(key_id, public_key);
    }

    /// Add public key from bytes
    pub fn add_key_bytes(&mut self, key_id: String, public_key_bytes: &[u8]) -> Result<(), CryptoError> {
        if public_key_bytes.len() != PUBLIC_KEY_LENGTH {
            return Err(CryptoError::InvalidKeyFormat(
                format!("Expected {} bytes, got {}", PUBLIC_KEY_LENGTH, public_key_bytes.len())
            ));
        }

        let public_key = PublicKey::from_bytes(public_key_bytes)
            .map_err(|e| CryptoError::InvalidKeyFormat(format!("Invalid public key: {}", e)))?;

        self.add_key(key_id, public_key);
        Ok(())
    }

    /// Load keys from a key ring
    pub fn from_keyring(key_ring: &KeyRing) -> Result<Self, CryptoError> {
        let mut context = Self::new();

        for key in key_ring.get_all_keys() {
            let public_key = PublicKey::from_bytes(&key.public_key)
                .map_err(|e| CryptoError::InvalidKeyFormat(format!("Invalid public key for {}: {}", key.id, e)))?;
            context.add_key(key.id.clone(), public_key);
        }

        if let Some(active_key) = key_ring.get_active_key() {
            context.primary_key = Some(active_key.id.clone());
        }

        info!(key_count = context.keys.len(), "Created verification context from keyring");
        Ok(context)
    }

    /// Get available key IDs
    pub fn key_ids(&self) -> Vec<String> {
        self.keys.keys().cloned().collect()
    }

    /// Check if a key exists
    pub fn has_key(&self, key_id: &str) -> bool {
        self.keys.contains_key(key_id)
    }

    /// Verify an envelope signature
    #[tracing::instrument(skip(self, envelope), fields(envelope_id = %envelope.header.id))]
    pub async fn verify_envelope(&self, envelope: &Envelope) -> Result<(), crate::error::BridgeError> {
        debug!("Verifying envelope signature");

        // First check structural validity
        envelope.is_structurally_valid()?;

        // Parse signature
        if envelope.sig.len() != SIGNATURE_LENGTH {
            return Err(crate::error::BridgeError::CryptoError(
                format!("Invalid signature length: expected {}, got {}", SIGNATURE_LENGTH, envelope.sig.len())
            ));
        }

        let signature = Signature::from_bytes(&envelope.sig)
            .map_err(|e| crate::error::BridgeError::CryptoError(format!("Invalid signature format: {}", e)))?;

        // Get canonical representation
        let canonical_bytes = envelope.canonical_representation()?;

        // Create domain-separated message
        let message = self.create_domain_separated_message(&canonical_bytes);

        // Try verification with available keys
        let mut verification_attempted = false;

        // Try primary key first if available
        if let Some(primary_key_id) = &self.primary_key {
            if let Some(public_key) = self.keys.get(primary_key_id) {
                verification_attempted = true;
                if public_key.verify(&message, &signature).is_ok() {
                    debug!(key_id = %primary_key_id, "Envelope verified with primary key");
                    return Ok(());
                }
            }
        }

        // Try other keys
        for (key_id, public_key) in &self.keys {
            if Some(key_id) == self.primary_key.as_ref() {
                continue; // Already tried
            }

            verification_attempted = true;
            if public_key.verify(&message, &signature).is_ok() {
                debug!(key_id = %key_id, "Envelope verified with key");
                return Ok(());
            }
        }

        if !verification_attempted {
            error!("No keys available for verification");
            return Err(crate::error::BridgeError::CryptoError("No keys available for verification".to_string()));
        }

        warn!(envelope_id = %envelope.header.id, "Envelope signature verification failed");
        Err(crate::error::BridgeError::CryptoError("Signature verification failed".to_string()))
    }

    /// Basic batch verify for VerificationContext (simple sequential)
    #[tracing::instrument(skip(self, envelopes))]
    pub async fn batch_verify(&self, envelopes: &[Envelope]) -> BatchVerificationReport {
        info!(count = envelopes.len(), "Starting basic batch verification");

        let mut valid = 0;
        let mut errors = Vec::new();

        for (index, envelope) in envelopes.iter().enumerate() {
            match self.verify_envelope(envelope).await {
                Ok(()) => {
                    valid += 1;
                    debug!(index, envelope_id = %envelope.header.id, "Batch verification success");
                }
                Err(e) => {
                    let error = BatchVerificationError {
                        index,
                        envelope_id: envelope.header.id,
                        error: e.to_string(),
                    };
                    errors.push(error);
                    debug!(index, envelope_id = %envelope.header.id, error = %e, "Batch verification failed");
                }
            }
        }

        let report = BatchVerificationReport {
            total: envelopes.len(),
            valid,
            invalid: envelopes.len() - valid,
            errors,
        };

        info!(
            total = report.total,
            valid = report.valid,
            invalid = report.invalid,
            "Basic batch verification completed"
        );

        report
    }

    /// Batch verify signature data with domain separation (synchronous)
    pub fn batch_verify_signatures(&self, sig_data: &[(Vec<u8>, Vec<u8>, String, i64)]) -> Result<(), CryptoError> {
        // For simplicity, verify each signature individually
        // In production, you could use ed25519_dalek::verify_batch for true batch verification
        for (message, signature, kid, _timestamp) in sig_data {
            let public_key = self.keys.get(kid)
                .ok_or_else(|| CryptoError::KeyNotFound(kid.clone()))?;

            if signature.len() != 64 {
                return Err(CryptoError::InvalidSignatureFormat(
                    format!("Expected 64 bytes, got {}", signature.len())
                ));
            }

            let sig = ed25519_dalek::Signature::from_bytes(signature)
                .map_err(|e| CryptoError::InvalidSignatureFormat(e.to_string()))?;

            public_key.verify(message, &sig)
                .map_err(|_| CryptoError::SignatureVerificationFailed)?;
        }

        Ok(())
    }

    /// Helper to create domain-separated message
    fn create_domain_separated_message(&self, canonical_bytes: &[u8]) -> Vec<u8> {
        let mut message = Vec::with_capacity(JEFFREY_BRIDGE_DOMAIN.len() + canonical_bytes.len());
        message.extend_from_slice(JEFFREY_BRIDGE_DOMAIN);
        message.extend_from_slice(canonical_bytes);
        message
    }
}

impl Default for VerificationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Secure key generation utilities
pub struct KeyGenerator;

impl KeyGenerator {
    /// Generate a new Ed25519 keypair
    pub fn generate_keypair() -> (SecretKey, PublicKey) {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        (keypair.secret, keypair.public)
    }

    /// Generate a keypair with a derived key ID
    pub fn generate_keypair_with_id() -> (SecretKey, PublicKey, String) {
        let (secret, public) = Self::generate_keypair();
        let key_id = format!("key-{}", hex::encode(&public.to_bytes()[..8]));
        (secret, public, key_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{EnvelopeBuilder, EnvelopeHeader, SystemClock};
    use pretty_assertions::assert_eq;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_ephemeral_signing_context() {
        let signing_ctx = SigningContext::new_ephemeral().unwrap();
        assert!(!signing_ctx.key_id().is_empty());
        assert_eq!(signing_ctx.export_public_key().len(), PUBLIC_KEY_LENGTH);
    }

    #[tokio::test]
    async fn test_envelope_signing_and_verification() {
        use crate::envelope::clock::SystemClock;

        let signing_ctx = SigningContext::new_ephemeral().unwrap();
        let verification_ctx = signing_ctx.verification_context();
        let clock = SystemClock;

        let envelope = EnvelopeBuilder::new()
            .src("test-source")
            .dst("test-destination")
            .content_type("text/plain")
            .payload(b"Hello, Jeffrey Bridge!")
            .build_and_sign(&signing_ctx, &clock)
            .await
            .unwrap();

        // Verify the envelope
        verification_ctx.verify_envelope(&envelope).await.unwrap();

        // Check signature length
        assert_eq!(envelope.sig.len(), SIGNATURE_LENGTH);
    }

    #[tokio::test]
    async fn test_verification_with_wrong_key() {
        let clock = SystemClock;
        let signing_ctx1 = SigningContext::new_ephemeral().unwrap();
        let signing_ctx2 = SigningContext::new_ephemeral().unwrap();
        let wrong_verification_ctx = signing_ctx2.verification_context();

        let envelope = EnvelopeBuilder::new()
            .payload(b"Test payload")
            .build_and_sign(&signing_ctx1, &clock)
            .await
            .unwrap();

        // Verification should fail with wrong key
        let result = wrong_verification_ctx.verify_envelope(&envelope).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_batch_verification() {
        let clock = SystemClock;
        let signing_ctx = SigningContext::new_ephemeral().unwrap();
        let verification_ctx = signing_ctx.verification_context();

        let envelopes = vec![
            EnvelopeBuilder::new().payload(b"Test 1").build_and_sign(&signing_ctx, &clock).await.unwrap(),
            EnvelopeBuilder::new().payload(b"Test 2").build_and_sign(&signing_ctx, &clock).await.unwrap(),
            EnvelopeBuilder::new().payload(b"Test 3").build_and_sign(&signing_ctx, &clock).await.unwrap(),
        ];

        let report = verification_ctx.batch_verify(&envelopes).await;
        assert_eq!(report.total, 3);
        assert_eq!(report.valid, 3);
        assert_eq!(report.invalid, 0);
        assert!(report.errors.is_empty());
    }

    #[tokio::test]
    async fn test_batch_verification_with_failures() {
        let clock = SystemClock;
        let signing_ctx = SigningContext::new_ephemeral().unwrap();
        let verification_ctx = signing_ctx.verification_context();

        let mut valid_envelope = EnvelopeBuilder::new()
            .payload(b"Valid payload")
            .build_and_sign(&signing_ctx, &clock)
            .await
            .unwrap();

        let mut invalid_envelope = valid_envelope.clone();
        invalid_envelope.sig[0] ^= 1; // Corrupt signature

        let envelopes = vec![valid_envelope, invalid_envelope];
        let report = verification_ctx.batch_verify(&envelopes).await;

        assert_eq!(report.total, 2);
        assert_eq!(report.valid, 1);
        assert_eq!(report.invalid, 1);
        assert_eq!(report.errors.len(), 1);
    }

    #[test]
    fn test_key_generator() {
        let (secret, public) = KeyGenerator::generate_keypair();
        assert_eq!(secret.to_bytes().len(), SECRET_KEY_LENGTH);
        assert_eq!(public.to_bytes().len(), PUBLIC_KEY_LENGTH);

        let (_, _, key_id) = KeyGenerator::generate_keypair_with_id();
        assert!(key_id.starts_with("key-"));
    }

    #[test]
    fn test_verification_context_key_management() {
        let mut ctx = VerificationContext::new();
        let (_, public_key, key_id) = KeyGenerator::generate_keypair_with_id();

        ctx.add_key(key_id.clone(), public_key);
        assert!(ctx.has_key(&key_id));
        assert_eq!(ctx.key_ids().len(), 1);
    }

    #[tokio::test]
    async fn test_domain_separation() {
        // Create two identical envelopes
        let clock = SystemClock;
        let signing_ctx = SigningContext::new_ephemeral().unwrap();

        let envelope1 = EnvelopeBuilder::new()
            .payload(b"Same payload")
            .build_and_sign(&signing_ctx, &clock)
            .await
            .unwrap();

        let envelope2 = EnvelopeBuilder::new()
            .payload(b"Same payload")
            .build_and_sign(&signing_ctx, &clock)
            .await
            .unwrap();

        // Signatures should be different due to different envelope IDs and timestamps
        assert_ne!(envelope1.sig, envelope2.sig);
        assert_ne!(envelope1.header.id, envelope2.header.id);
    }
}
