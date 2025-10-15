#!/usr/bin/env python3
"""
Tests de validation P0 - Vérifie que l'infrastructure P2 est prête
Inclut tests de connexion, chaos basique et configuration
"""

import asyncio
import base64
import os
import time
from pathlib import Path

import argon2
import nats
import pytest
import redis
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

import docker

# Charger configuration
load_dotenv(".env.p2")


class P2Config(BaseSettings):
    """Configuration typée avec Pydantic"""

    # General
    environment: str = "development"
    debug: bool = True
    tenant_id: str = "default"

    # Security
    secret_key: str
    encryption_key: str
    jwt_secret_key: str

    # NATS
    nats_url: str = "nats://localhost:4222"
    nats_user: str = "jeffrey"
    nats_password: str = "jeffrey2024"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "jeffrey2024"

    # Observability
    otel_service_name: str = "jeffrey-core"
    metrics_port: int = 9091

    class Config:
        env_file = ".env.p2"

    def validate_encryption_key(self) -> bool:
        """Valide que la clé de chiffrement est au bon format"""
        try:
            key_bytes = base64.b64decode(self.encryption_key)
            return len(key_bytes) == 32  # AES-256 = 32 bytes
        except Exception:
            return False


config = P2Config()


class TestInfrastructure:
    """Tests de l'infrastructure Docker"""

    @pytest.fixture(scope="class")
    def docker_client(self):
        """Client Docker pour tests chaos"""
        return docker.from_env()

    def test_docker_services_running(self, docker_client):
        """Vérifie que tous les services Docker sont actifs"""
        required_services = [
            "jeffrey-nats",
            "jeffrey-redis",
            "jeffrey-jaeger",
            "jeffrey-nats-exporter",
        ]
        running_containers = {c.name for c in docker_client.containers.list()}

        for service in required_services:
            assert service in running_containers, f"Service {service} not running"

    @pytest.mark.asyncio
    async def test_nats_connection(self):
        """Test connexion NATS avec authentification"""
        nc = await nats.connect(config.nats_url, user=config.nats_user, password=config.nats_password)

        # Test JetStream
        js = nc.jetstream()

        # Créer un stream de test
        await js.add_stream(name="TEST_STREAM", subjects=["test.>"], max_msgs=1000)

        # Publier et consommer
        ack = await js.publish("test.validation", b"P2 ready")
        assert ack.seq > 0

        await nc.close()

    def test_redis_connection(self):
        """Test connexion Redis avec auth"""
        r = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password,
            decode_responses=True,
        )

        # Test basique
        r.set("test:p2:ready", "true", ex=60)
        value = r.get("test:p2:ready")
        assert value == "true"

        # Test cache pattern
        r.hset("cache:test", "key1", "value1")
        assert r.hget("cache:test", "key1") == "value1"

        r.close()

    def test_configuration_validity(self):
        """Valide la configuration P2"""
        # Vérifier que les clés ne sont pas par défaut
        assert config.secret_key != "CHANGE_ME_USE_OPENSSL_RAND_BASE64_32"
        assert config.encryption_key != "CHANGE_ME_USE_OPENSSL_RAND_BASE64_32"

        # Vérifier format des clés
        assert config.validate_encryption_key(), "Invalid encryption key format"

        # Vérifier variables critiques
        assert config.tenant_id
        assert config.otel_service_name

    def test_crypto_performance(self):
        """Test performance crypto < 15ms (ajusté pour machines lentes)"""
        # Test AES-256-GCM
        key = base64.b64decode(config.encryption_key)
        plaintext = b"Test data for encryption" * 100

        start = time.perf_counter()

        # Chiffrement
        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        _ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        encryption_time = (time.perf_counter() - start) * 1000

        # Test Argon2
        password = "test_password"
        ph = argon2.PasswordHasher(time_cost=1, memory_cost=64 * 1024, parallelism=2)

        start = time.perf_counter()
        _hash_result = ph.hash(password)
        hash_time = (time.perf_counter() - start) * 1000

        # Limites ajustées pour machines lentes / CI
        assert encryption_time < 15, f"Encryption too slow: {encryption_time}ms"
        assert hash_time < 150, f"Hashing too slow: {hash_time}ms"

    @pytest.mark.asyncio
    async def test_chaos_service_down(self, docker_client):
        """Test chaos : service NATS down et recovery"""
        # Arrêter NATS
        nats_container = docker_client.containers.get("jeffrey-nats")
        nats_container.pause()

        # Vérifier que la connexion échoue
        with pytest.raises(Exception):
            await nats.connect(config.nats_url, max_reconnect_attempts=1)

        # Redémarrer NATS
        nats_container.unpause()
        await asyncio.sleep(2)

        # Vérifier la reconnexion
        nc = await nats.connect(config.nats_url)
        assert nc.is_connected
        await nc.close()

    def test_structure_p2_created(self):
        """Vérifie que la structure P2 est en place"""
        required_dirs = [
            "src/jeffrey/core/bus",
            "src/jeffrey/core/kernel",
            "src/jeffrey/core/learning/kg",
            "src/jeffrey/infrastructure/nats",
            "src/jeffrey/avatars/api",
            "src/jeffrey/legacy",
            "tests/chaos",
        ]

        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"Missing directory: {dir_path}"

    def test_legacy_modules_preserved(self):
        """Vérifie que les modules P1 sont préservés"""
        legacy_modules = [
            "consciousness",
            "memory_manager",
            "emotional_core",
            "dream_engine",
            "symbiosis",
            "brain_kernel",
        ]

        legacy_path = Path("src/jeffrey/legacy")
        if legacy_path.exists():
            for module in legacy_modules:
                module_path = legacy_path / module
                if module_path.exists():
                    assert any(module_path.iterdir()), f"Legacy module {module} is empty"


class TestChaosAdvanced:
    """Tests chaos avancés avec network corruption"""

    @pytest.mark.chaos_tc
    @pytest.mark.skipif(os.getenv("CHAOS_MODE") != "tc", reason="TC chaos not enabled")
    def test_network_corruption_tc(self, docker_client):
        """Test avec traffic control pour simuler latence et perte de paquets"""
        import subprocess

        # Configuration tc
        delay = "100ms"
        jitter = "20ms"
        loss = "2%"

        try:
            # Appliquer corruption réseau sur NATS
            cmd = f"""docker exec jeffrey-nats sh -c '
                apk add --no-cache iproute2 2>/dev/null || true;
                tc qdisc add dev eth0 root netem delay {delay} {jitter} loss {loss};
                echo "Network corruption applied"
            '"""

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                pytest.skip(f"TC not available: {result.stderr}")

            # Test de connexion sous corruption
            import asyncio

            async def test_under_corruption():
                nc = await nats.connect(
                    config.nats_url,
                    user=config.nats_user,
                    password=config.nats_password,
                    max_reconnect_attempts=5,
                )

                # Devrait fonctionner malgré la corruption
                js = nc.jetstream()
                ack = await js.publish("test.chaos", b"Testing under network corruption")
                assert ack.seq > 0

                await nc.close()

            asyncio.run(test_under_corruption())

        finally:
            # CRITICAL: Toujours nettoyer tc
            cleanup_cmd = """docker exec jeffrey-nats sh -c 'tc qdisc del dev eth0 root 2>/dev/null || true'"""
            subprocess.run(cleanup_cmd, shell=True)
            print("✅ Network corruption cleaned up")


class TestCIReadiness:
    """Tests pour vérifier que le projet est prêt pour CI/CD"""

    def test_github_workflow_exists(self):
        """Vérifie la présence du workflow GitHub Actions"""
        workflow_path = Path(".github/workflows/p2-validation.yml")
        if not workflow_path.exists():
            pytest.skip("GitHub workflow not yet created")

        with open(workflow_path) as f:
            content = f.read()
            assert "pytest" in content
            assert "docker-compose" in content

    def test_makefile_exists(self):
        """Vérifie la présence d'un Makefile"""
        makefile = Path("Makefile")
        if not makefile.exists():
            pytest.skip("Makefile not yet created")

        with open(makefile) as f:
            content = f.read()
            assert "test:" in content
            assert "lint:" in content
            assert "up:" in content


if __name__ == "__main__":
    # Lancer les tests avec rapport détaillé
    pytest.main([__file__, "-v", "--tb=short"])
