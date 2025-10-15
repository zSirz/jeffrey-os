#!/usr/bin/env python3
"""
Gestionnaire NATS robuste avec tracking PID et health checks
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil


class NATSManager:
    """Gestionnaire NATS production-grade"""

    def __init__(self, data_dir: str = ".nats"):
        self.data_dir = Path(data_dir)
        self.pid_file = Path(".nats_pid")
        self.config_file = Path(".nats_config.json")
        self.process = None

    def start(self, namespace: str = None) -> bool:
        """Démarre NATS avec configuration robuste"""
        # Si déjà running, ne pas redémarrer
        if self.is_running():
            print("✅ NATS already running")
            return True

        # Vérifier si un NATS externe est déjà lancé
        if self.health_check():
            print("ℹ️  External NATS detected on port 4222, using existing instance")
            # Sauvegarder la config même pour un NATS externe
            if not namespace:
                namespace = f"soak_{int(time.time())}"
            config = {
                "namespace": namespace,
                "started_at": time.time(),
                "port": 4222,
                "external": True,
            }
            self.config_file.write_text(json.dumps(config, indent=2))
            return True

        # Créer le namespace pour cette session
        if not namespace:
            namespace = f"soak_{int(time.time())}"

        # Configuration NATS
        config = {
            "namespace": namespace,
            "started_at": time.time(),
            "port": 4222,
            "store_dir": str(self.data_dir),
            "external": False,
        }

        # Créer le répertoire de données
        self.data_dir.mkdir(exist_ok=True)

        # Lancer NATS avec options robustes
        cmd = [
            "nats-server",
            "-js",  # JetStream
            "--store_dir",
            str(self.data_dir),
            "--max_payload",
            "10MB",  # Pour gros messages ML
            "--max_connections",
            "1000",  # Pour tests de charge
            "--ping_interval",
            "10",
            "--ping_max",
            "3",
            "--write_deadline",
            "10s",
            "--max_control_line",
            "4KB",
        ]

        print(f"🚀 Starting NATS with namespace: {namespace}")

        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            # Sauvegarder PID et config
            self.pid_file.write_text(str(self.process.pid))
            self.config_file.write_text(json.dumps(config, indent=2))

            # Attendre que NATS soit prêt
            time.sleep(2)

            # Vérifier que le processus est bien lancé
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode()
                print(f"❌ NATS failed to start: {stderr}")
                return False

            # Health check
            if self.health_check():
                print(f"✅ NATS started (PID: {self.process.pid})")
                return True
            else:
                print("❌ NATS health check failed")
                self.stop()
                return False

        except Exception as e:
            print(f"❌ Failed to start NATS: {e}")
            # Vérifier si un NATS externe peut être utilisé
            if self.health_check():
                print("ℹ️  But external NATS is available on port 4222, using it")
                config["external"] = True
                self.config_file.write_text(json.dumps(config, indent=2))
                return True
            return False

    def stop(self, clean: bool = True) -> bool:
        """Arrête NATS proprement"""
        # Vérifier si c'est un NATS externe
        config = self.get_config()
        if config and config.get("external"):
            print("ℹ️  External NATS detected, not stopping")
            if clean:
                self.config_file.unlink(missing_ok=True)
            return True

        pid = self.get_pid()
        if not pid:
            print("⚠️  No NATS process to stop")
            return True

        try:
            # Arrêt gracieux d'abord
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)

            # Force kill si nécessaire
            if self.is_running():
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)

            print(f"✅ NATS stopped (PID: {pid})")

            # Cleanup
            if clean:
                self.pid_file.unlink(missing_ok=True)
                self.config_file.unlink(missing_ok=True)

            return True

        except ProcessLookupError:
            # Déjà arrêté
            return True
        except Exception as e:
            print(f"❌ Failed to stop NATS: {e}")
            return False

    def restart(self) -> bool:
        """Redémarre NATS en préservant le namespace"""
        config = self.get_config()
        namespace = config.get("namespace") if config else None

        self.stop(clean=False)
        time.sleep(1)
        return self.start(namespace=namespace)

    def get_pid(self) -> int | None:
        """Récupère le PID de NATS"""
        if self.pid_file.exists():
            try:
                return int(self.pid_file.read_text().strip())
            except:
                pass
        return None

    def get_config(self) -> dict[str, Any] | None:
        """Récupère la configuration"""
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text())
            except:
                pass
        return None

    def is_running(self) -> bool:
        """Vérifie si NATS est en cours d'exécution"""
        pid = self.get_pid()
        if pid:
            try:
                # Check if process exists
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                pass
        return False

    def health_check(self) -> bool:
        """Vérifie la santé de NATS"""
        # Check si un processus écoute sur le port 4222
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == 4222 and conn.status == "LISTEN":
                    return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            # Sur macOS, parfois besoin de droits admin pour voir toutes les connexions
            # Essayons une approche alternative
            try:
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(("127.0.0.1", 4222))
                sock.close()
                return result == 0
            except:
                pass

        return False

    def chaos_restart(self) -> bool:
        """Redémarrage chaos pour tests"""
        # Ne pas redémarrer un NATS externe
        config = self.get_config()
        if config and config.get("external"):
            print("🔥 CHAOS: External NATS, skipping restart")
            return True

        print("🔥 CHAOS: Restarting NATS...")
        return self.restart()


# CLI si exécuté directement
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NATS Manager")
    parser.add_argument("action", choices=["start", "stop", "restart", "status"])
    parser.add_argument("--namespace", help="Namespace for subjects")

    args = parser.parse_args()

    manager = NATSManager()

    if args.action == "start":
        manager.start(namespace=args.namespace)
    elif args.action == "stop":
        manager.stop()
    elif args.action == "restart":
        manager.restart()
    elif args.action == "status":
        if manager.is_running():
            config = manager.get_config()
            print(f"✅ NATS running (PID: {manager.get_pid()})")
            if config:
                print(f"   Namespace: {config.get('namespace')}")
                print(f"   Started: {config.get('started_at')}")
                if config.get("external"):
                    print("   Type: External NATS")
        else:
            # Vérifier si un NATS externe est disponible
            if manager.health_check():
                print("✅ External NATS available on port 4222")
            else:
                print("❌ NATS not running")
