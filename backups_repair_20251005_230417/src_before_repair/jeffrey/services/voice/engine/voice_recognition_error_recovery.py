#!/usr/bin/env python3

"""
Module de récupération d'erreurs pour la reconnaissance vocale de Jeffrey.
Permet de gérer et récupérer les erreurs courantes lors de l'activation du microphone
et de la reconnaissance vocale, améliorant la stabilité du système vocal.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import time
from typing import Any

# Configuration du logger
logger = logging.getLogger("jeffrey.voice.error_recovery")


class VoiceRecognitionErrorRecovery:
    """
    Système de récupération d'erreurs pour la reconnaissance vocale.
    Détecte et corrige les problèmes courants de microphone et de reconnaissance.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0, auto_repair: bool = True):
        """
        Initialise le système de récupération d'erreurs.

        Args:
            max_retries: Nombre maximum de tentatives de récupération
            retry_delay: Délai entre les tentatives en secondes
            auto_repair: Tenter automatiquement de réparer les problèmes
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_repair = auto_repair

        # État du système
        self.retries = 0
        self.last_error = None
        self.last_recovery_attempt = 0
        self.recovery_successful = False
        self.permission_status = None

        # Initialize platform-specific information
        self.is_macos = platform.system() == "Darwin"
        self.is_windows = platform.system() == "Windows"
        self.is_linux = platform.system() == "Linux"
        self.is_ios = False

        # Détection de Pythonista (environnement iOS)
        try:
            import motion

            self.is_ios = True
        except ImportError:
            pass

        logger.info(f"Système de récupération d'erreurs vocales initialisé (plateforme: {platform.system()})")

    def check_microphone_permissions(self) -> tuple[bool, str]:
        """
        Vérifie les permissions d'accès au microphone sur la plateforme actuelle.

        Returns:
            Tuple[bool, str]: (permission accordée, message d'état)
        """
        if self.is_macos:
            return self._check_macos_microphone_permissions()
        elif self.is_windows:
            return self._check_windows_microphone_permissions()
        elif self.is_linux:
            return self._check_linux_microphone_permissions()
        elif self.is_ios:
            return self._check_ios_microphone_permissions()
        else:
            logger.warning(f"Vérification des permissions non supportée sur {platform.system()}")
            return (True, "Non vérifiable sur cette plateforme")

    def _check_macos_microphone_permissions(self) -> tuple[bool, str]:
        """
        Vérifie les permissions microphone sur macOS.

        Returns:
            Tuple[bool, str]: (permission accordée, message d'état)
        """
        try:
            # Utiliser le framework TCC pour vérifier les permissions
            # Note: Cela nécessite des privilèges d'administrateur sous macOS
            # Cette approche fonctionne seulement en environnement de développement

            # Vérifier si le son est capturé via un test simple
            import pyaudio

            p = pyaudio.PyAudio()
            try:
                # Tenter d'ouvrir un flux pour vérifier l'accès au micro
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024,
                )
                stream.stop_stream()
                stream.close()
                p.terminate()
                return (True, "Accès au microphone confirmé")
            except OSError as e:
                if "Input overflowed" in str(e):
                    # Débordement d'entrée signifie qu'on a accès au micro
                    return (True, "Accès au microphone confirmé (débordement)")
                elif "Permission denied" in str(e) or "Resource busy" in str(e):
                    logger.warning(f"Accès au microphone refusé: {e}")
                    return (False, f"Accès refusé: {str(e)}")
                else:
                    logger.error(f"Erreur lors de l'accès au microphone: {e}")
                    return (False, f"Erreur d'accès: {str(e)}")
            finally:
                p.terminate()

        except Exception as e:
            logger.error(f"Erreur lors de la vérification des permissions microphone: {e}")
            return (False, f"Erreur de vérification: {str(e)}")

    def _check_windows_microphone_permissions(self) -> tuple[bool, str]:
        """
        Vérifie les permissions microphone sur Windows.

        Returns:
            Tuple[bool, str]: (permission accordée, message d'état)
        """
        try:
            # Sur Windows, tenter de capturer un échantillon audio
            import pyaudio

            p = pyaudio.PyAudio()
            try:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024,
                )
                stream.read(1024)  # Lire un échantillon
                stream.stop_stream()
                stream.close()
                return (True, "Accès au microphone confirmé")
            except OSError as e:
                logger.warning(f"Accès au microphone refusé: {e}")
                return (False, f"Accès refusé: {str(e)}")
            finally:
                p.terminate()
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des permissions microphone: {e}")
            return (False, f"Erreur de vérification: {str(e)}")

    def _check_linux_microphone_permissions(self) -> tuple[bool, str]:
        """
        Vérifie les permissions microphone sur Linux.

        Returns:
            Tuple[bool, str]: (permission accordée, message d'état)
        """
        try:
            # Vérifier si le périphérique est accessible
            import pyaudio

            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            p.terminate()

            if device_count > 0:
                # Tenter d'accéder au premier périphérique d'entrée
                p = pyaudio.PyAudio()
                for i in range(device_count):
                    device_info = p.get_device_info_by_index(i)
                    if device_info.get("maxInputChannels") > 0:
                        try:
                            stream = p.open(
                                format=pyaudio.paInt16,
                                channels=1,
                                rate=16000,
                                input=True,
                                input_device_index=i,
                                frames_per_buffer=1024,
                            )
                            stream.stop_stream()
                            stream.close()
                            p.terminate()
                            return (True, f"Accès au microphone confirmé (périphérique {i})")
                        except OSError as e:
                            logger.warning(f"Accès au périphérique {i} refusé: {e}")
                p.terminate()
                return (False, "Aucun périphérique d'entrée accessible")
            else:
                return (False, "Aucun périphérique audio détecté")
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des permissions microphone: {e}")
            return (False, f"Erreur de vérification: {str(e)}")

    def _check_ios_microphone_permissions(self) -> tuple[bool, str]:
        """
        Vérifie les permissions microphone sur iOS (via Pythonista).

        Returns:
            Tuple[bool, str]: (permission accordée, message d'état)
        """
        try:
            # Dans Pythonista, on utilise le framework AVFoundation
            import AVFoundation

            auth_status = AVFoundation.AVAudioSession.sharedInstance().recordPermission()

            if auth_status == 1:  # AVAudioSessionRecordPermissionGranted
                return (True, "Accès au microphone autorisé")
            elif auth_status == 0:  # AVAudioSessionRecordPermissionUndetermined
                return (False, "Permission non déterminée. Demande requise.")
            else:  # AVAudioSessionRecordPermissionDenied
                return (False, "Accès au microphone refusé. Vérifiez les paramètres.")
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des permissions iOS: {e}")
            return (False, f"Erreur de vérification: {str(e)}")

    def request_microphone_permissions(self) -> bool:
        """
        Demande les permissions d'accès au microphone si nécessaire.

        Returns:
            bool: True si les permissions sont obtenues
        """
        # Vérifier d'abord l'état actuel
        has_permission, status = self.check_microphone_permissions()

        if has_permission:
            logger.info("Permissions microphone déjà accordées")
            return True

        logger.info(f"Demande de permissions microphone ({status})")

        if self.is_macos:
            return self._request_macos_microphone_permissions()
        elif self.is_windows:
            return self._request_windows_microphone_permissions()
        elif self.is_linux:
            return self._request_linux_microphone_permissions()
        elif self.is_ios:
            return self._request_ios_microphone_permissions()
        else:
            logger.warning(f"Demande de permissions non supportée sur {platform.system()}")
            return False

    def _request_macos_microphone_permissions(self) -> bool:
        """
        Demande les permissions microphone sur macOS.

        Returns:
            bool: True si les permissions sont obtenues
        """
        try:
            # Sur macOS, afficher un dialogue d'information
            cmd = [
                "osascript",
                "-e",
                "display dialog \"Jeffrey a besoin d'accéder au microphone pour la reconnaissance vocale. "
                "Veuillez autoriser l'accès dans les paramètres système > Confidentialité et sécurité > Micro.\" "
                'buttons {"Ouvrir Paramètres", "Plus tard"} '
                'default button "Ouvrir Paramètres" '
                'with title "Permissions microphone requises"',
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            # Si l'utilisateur a cliqué sur "Ouvrir Paramètres"
            if "Ouvrir Paramètres" in stdout.decode("utf-8", errors="ignore"):
                # Ouvrir les paramètres du micro
                open_cmd = [
                    "open",
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
                ]
                subprocess.Popen(open_cmd)
                return True

            return False

        except Exception as e:
            logger.error(f"Erreur lors de la demande de permissions microphone: {e}")
            return False

    def _request_windows_microphone_permissions(self) -> bool:
        """
        Demande les permissions microphone sur Windows.

        Returns:
            bool: True si les permissions sont obtenues
        """
        try:
            # Sur Windows 10+, ouvrir les paramètres de confidentialité du microphone
            cmd = ["rundll32", "shell32.dll,OpenAs_RunDLL", "ms-settings:privacy-microphone"]
            subprocess.Popen(cmd)
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la demande de permissions microphone: {e}")
            return False

    def _request_linux_microphone_permissions(self) -> bool:
        """
        Demande les permissions microphone sur Linux.

        Returns:
            bool: True si les permissions sont obtenues
        """
        try:
            # Sur Linux, nous pouvons essayer d'ouvrir une boîte de dialogue
            # avec zenity si disponible
            cmd = [
                "zenity",
                "--info",
                '--title="Permissions microphone"',
                '--text="Jeffrey a besoin d\'accéder au microphone. Veuillez vérifier les permissions du périphérique audio."',
                '--ok-label="Compris"',
            ]

            try:
                subprocess.Popen(cmd)
                return True
            except FileNotFoundError:
                logger.warning("zenity n'est pas disponible pour afficher le dialogue")
                return False

        except Exception as e:
            logger.error(f"Erreur lors de la demande de permissions microphone: {e}")
            return False

    def _request_ios_microphone_permissions(self) -> bool:
        """
        Demande les permissions microphone sur iOS (via Pythonista).

        Returns:
            bool: True si les permissions sont obtenues
        """
        try:
            # Dans Pythonista, utiliser AVFoundation pour demander l'accès
            import AVFoundation
            import console

            console.alert(
                "Permissions microphone",
                "Jeffrey a besoin d'accéder au microphone pour la reconnaissance vocale.",
                "Compris",
            )

            # Demander l'accès au micro (cela affiche le popup système)
            AVFoundation.AVAudioSession.sharedInstance().requestRecordPermission_(lambda granted: None)

            # Vérifier si la permission a été accordée
            time.sleep(1)  # Attendre un peu le résultat
            return self.check_microphone_permissions()[0]

        except Exception as e:
            logger.error(f"Erreur lors de la demande de permissions iOS: {e}")
            return False

    def fix_microphone_issues(self) -> bool:
        """
        Tente de résoudre les problèmes courants de microphone.

        Returns:
            bool: True si les problèmes sont résolus
        """
        logger.info("Tentative de résolution des problèmes de microphone")

        # Vérifier les permissions
        has_permission, status = self.check_microphone_permissions()

        if not has_permission:
            # Demander les permissions
            if not self.request_microphone_permissions():
                logger.warning("Impossible d'obtenir les permissions microphone")
                return False

        # Essayer de résoudre les problèmes spécifiques à la plateforme
        if self.is_macos:
            return self._fix_macos_microphone_issues()
        elif self.is_windows:
            return self._fix_windows_microphone_issues()
        elif self.is_linux:
            return self._fix_linux_microphone_issues()
        elif self.is_ios:
            return self._fix_ios_microphone_issues()
        else:
            logger.warning(f"Résolution de problèmes non supportée sur {platform.system()}")
            return False

    def _fix_macos_microphone_issues(self) -> bool:
        """
        Résout les problèmes de microphone sur macOS.

        Returns:
            bool: True si les problèmes sont résolus
        """
        try:
            # Essayer de réinitialiser le système audio CoreAudio
            cmd = ["killall", "coreaudiod"]

            # Cette commande nécessite généralement sudo, mais tentons quand même
            try:
                subprocess.Popen(cmd)
                logger.info("Réinitialisation de CoreAudio tentée")
                time.sleep(2)  # Attendre que le service redémarre
                return True
            except Exception:
                logger.warning("Impossible de réinitialiser CoreAudio (privilèges insuffisants)")

            # Plan B: suggérer de reconnecter les périphériques audio
            return False

        except Exception as e:
            logger.error(f"Erreur lors de la résolution des problèmes macOS: {e}")
            return False

    def _fix_windows_microphone_issues(self) -> bool:
        """
        Résout les problèmes de microphone sur Windows.

        Returns:
            bool: True si les problèmes sont résolus
        """
        try:
            # Vérifier si le micro est désactivé dans le mixeur
            cmd = [
                "powershell",
                "-Command",
                '& {Get-WmiObject -Class Win32_SoundDevice | Where-Object {$_.Status -ne "OK"}}',
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if stdout:
                # Des périphériques audio ont des problèmes
                logger.warning(
                    f"Périphériques audio problématiques détectés: {stdout.decode('utf-8', errors='ignore')}"
                )

                # Tenter de réactiver les périphériques audio (nécessite des privilèges)
                restart_cmd = [
                    "powershell",
                    "-Command",
                    "& {Restart-Service -Name Audiosrv -Force}",
                ]

                try:
                    subprocess.Popen(restart_cmd)
                    logger.info("Service audio redémarré")
                    time.sleep(2)
                    return True
                except Exception:
                    logger.warning("Impossible de redémarrer le service audio (privilèges insuffisants)")

            return False

        except Exception as e:
            logger.error(f"Erreur lors de la résolution des problèmes Windows: {e}")
            return False

    def _fix_linux_microphone_issues(self) -> bool:
        """
        Résout les problèmes de microphone sur Linux.

        Returns:
            bool: True si les problèmes sont résolus
        """
        try:
            # Vérifier si pulseaudio est en cours d'exécution
            cmd = ["pulseaudio", "--check"]

            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process.communicate()

                if process.returncode != 0:
                    # Tenter de démarrer pulseaudio
                    restart_cmd = ["pulseaudio", "--start"]
                    subprocess.Popen(restart_cmd)
                    logger.info("PulseAudio redémarré")
                    time.sleep(2)
                return True
            except FileNotFoundError:
                logger.warning("PulseAudio non trouvé")

            return False

        except Exception as e:
            logger.error(f"Erreur lors de la résolution des problèmes Linux: {e}")
            return False

    def _fix_ios_microphone_issues(self) -> bool:
        """
        Résout les problèmes de microphone sur iOS (via Pythonista).

        Returns:
            bool: True si les problèmes sont résolus
        """
        try:
            # Sur iOS, nous avons peu d'options. Réinitialiser la session audio
            import AVFoundation
            import console

            # Réinitialiser la session audio
            AVFoundation.AVAudioSession.sharedInstance().setActive_error_(False, None)
            # TODO: Remplacer par asyncio.sleep ou threading.Event
            AVFoundation.AVAudioSession.sharedInstance().setActive_error_(True, None)

            # Informer l'utilisateur
            console.alert(
                "Réinitialisation audio",
                "Le système audio a été réinitialisé. Veuillez réessayer la reconnaissance vocale.",
                "OK",
            )

            return True

        except Exception as e:
            logger.error(f"Erreur lors de la résolution des problèmes iOS: {e}")
            return False

    def get_recovery_instructions(self) -> str:
        """
        Génère des instructions pour l'utilisateur afin de résoudre les problèmes de microphone.

        Returns:
            str: Instructions formatées
        """
        has_permission, status = self.check_microphone_permissions()

        instructions = "GUIDE DE DÉPANNAGE DU MICROPHONE\n\n"

        # Ajouter des informations sur les permissions
        instructions += f"État des permissions: {'✅ Accordées' if has_permission else '❌ Non accordées'}\n"
        instructions += f"Détails: {status}\n\n"

        # Instructions spécifiques à la plateforme
        if self.is_macos:
            instructions += "Pour macOS:\n"
            instructions += "1. Vérifiez que Jeffrey a accès au micro dans Paramètres Système > Confidentialité et sécurité > Micro\n"
            instructions += "2. Assurez-vous que votre micro n'est pas coupé physiquement\n"
            instructions += "3. Vérifiez qu'aucune autre app n'utilise le micro actuellement\n"
            instructions += "4. Ouvrez l'app 'Configuration audio et MIDI' pour vérifier si votre micro est détecté\n"
        elif self.is_windows:
            instructions += "Pour Windows:\n"
            instructions += "1. Vérifiez que Jeffrey a accès au micro dans Paramètres > Confidentialité > Microphone\n"
            instructions += "2. Cliquez-droit sur l'icône du son et vérifiez que votre micro est sélectionné comme périphérique d'enregistrement\n"
            instructions += "3. Assurez-vous que le niveau du micro n'est pas à zéro\n"
            instructions += "4. Redémarrez le service audio en tapant 'services.msc' dans le menu Démarrer\n"
        elif self.is_linux:
            instructions += "Pour Linux:\n"
            instructions += "1. Vérifiez le niveau du micro avec 'alsamixer' ou 'pavucontrol'\n"
            instructions += "2. Assurez-vous que le micro n'est pas coupé\n"
            instructions += "3. Redémarrez pulseaudio avec 'pulseaudio --kill && pulseaudio --start'\n"
        elif self.is_ios:
            instructions += "Pour iOS:\n"
            instructions += "1. Vérifiez que Pythonista a accès au micro dans Réglages > Confidentialité > Micro\n"
            instructions += "2. Redémarrez Pythonista\n"
            instructions += "3. Si le problème persiste, redémarrez votre appareil\n"

        return instructions

    def recommend_audio_devices(self) -> list[dict[str, Any]]:
        """
        Recommande les meilleurs périphériques audio à utiliser.

        Returns:
            List[Dict]: Liste de périphériques recommandés avec score
        """
        import speech_recognition as sr

        recommendations = []

        try:
            # Obtenir la liste des microphones
            devices = sr.Microphone.list_microphone_names()

            # Trier les périphériques selon des critères de qualité
            for i, device_name in enumerate(devices):
                score = 0
                device = {"index": i, "name": device_name, "score": 0, "recommendation": ""}

                # Périphériques externes généralement meilleurs
                if "airpods" in device_name.lower():
                    score += 5
                    device["recommendation"] = "Excellent pour la reconnaissance vocale"
                elif "headset" in device_name.lower() or "casque" in device_name.lower():
                    score += 4
                    device["recommendation"] = "Très bon pour la reconnaissance vocale"
                elif "external" in device_name.lower() or "externe" in device_name.lower():
                    score += 3
                    device["recommendation"] = "Bon pour la reconnaissance vocale"
                elif "built-in" in device_name.lower() or "intégré" in device_name.lower():
                    score += 1
                    device["recommendation"] = "Fonctionnel mais peut capter les bruits ambiants"
                else:
                    device["recommendation"] = "Qualité inconnue"

                device["score"] = score
                recommendations.append(device)

            # Trier par score descendant
            recommendations.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.error(f"Erreur lors de la recommandation de périphériques: {e}")

        return recommendations

    def handle_recognition_error(self, error, error_type=None) -> dict[str, Any]:
        """
        Gère une erreur de reconnaissance vocale et tente de la résoudre.

        Args:
            error: L'exception ou l'erreur rencontrée
            error_type: Type d'erreur pour classification (None = auto-détection)

        Returns:
            Dict: Résultat de la gestion d'erreur avec actions recommandées
        """
        error_str = str(error)
        self.last_error = error
        self.retries += 1

        # Déterminer le type d'erreur
        if error_type is None:
            if "access" in error_str.lower() or "permission" in error_str.lower():
                error_type = "permission"
            elif "device" in error_str.lower() or "resource" in error_str.lower():
                error_type = "device"
            elif "timeout" in error_str.lower():
                error_type = "timeout"
            elif "no match" in error_str.lower() or "not recognized" in error_str.lower():
                error_type = "recognition"
            else:
                error_type = "unknown"

        logger.warning(f"Erreur de reconnaissance vocale ({error_type}): {error_str}")

        # Initialiser le résultat
        result = {
            "error_type": error_type,
            "error_message": error_str,
            "retry_count": self.retries,
            "recovery_attempted": False,
            "recovery_successful": False,
            "user_action_required": False,
            "user_message": "",
            "technical_details": str(error),
            "recommended_actions": [],
        }

        # Vérifier si on a atteint le nombre max de tentatives
        if self.retries > self.max_retries:
            result["user_action_required"] = True
            result["user_message"] = "Trop de tentatives infructueuses. Une intervention manuelle est nécessaire."
            result["recommended_actions"].append(
                {
                    "action": "manual_intervention",
                    "description": "Demander à l'utilisateur de vérifier le microphone et les paramètres",
                }
            )
        return result

        # Vérifier si on doit attendre avant de réessayer
        current_time = time.time()
        if current_time - self.last_recovery_attempt < self.retry_delay:
            wait_time = self.retry_delay - (current_time - self.last_recovery_attempt)
            result["recommended_actions"].append(
                {
                    "action": "wait",
                    "description": f"Attendre {wait_time:.1f} secondes avant de réessayer",
                    "wait_time": wait_time,
                }
            )
        return result

        # Marquer le début de la tentative de récupération
        self.last_recovery_attempt = current_time
        result["recovery_attempted"] = True

        # Traiter selon le type d'erreur
        if error_type == "permission":
            # Problème de permission
            has_permission, status = self.check_microphone_permissions()
            result["technical_details"] = f"Permission status: {status}"

            if not has_permission and self.auto_repair:
                # Tenter d'obtenir les permissions
                permission_requested = self.request_microphone_permissions()
                result["recovery_successful"] = permission_requested

                if permission_requested:
                    result["user_message"] = "Veuillez autoriser l'accès au microphone dans les paramètres système."
                else:
                    result["user_message"] = (
                        "Jeffrey a besoin d'accéder à votre microphone pour la reconnaissance vocale."
                    )

                result["user_action_required"] = True
                result["recommended_actions"].append(
                    {
                        "action": "request_permission",
                        "description": "Demander l'autorisation d'accès au microphone",
                    }
                )
            else:
                result["user_message"] = "Jeffrey n'a pas la permission d'accéder au microphone."
                result["user_action_required"] = True

        elif error_type == "device":
            # Problème de périphérique
            if self.auto_repair:
                fixed = self.fix_microphone_issues()
                result["recovery_successful"] = fixed

                if fixed:
                    result["user_message"] = "Problème de microphone résolu automatiquement. Veuillez réessayer."
                    result["recommended_actions"].append(
                        {"action": "retry", "description": "Réessayer la reconnaissance"}
                    )
                else:
                    result["user_message"] = "Problème de microphone détecté. Vérifiez votre périphérique audio."
                    result["user_action_required"] = True

                    # Recommander de meilleurs périphériques
                    recommendations = self.recommend_audio_devices()
                    if recommendations:
                        top_devices = recommendations[:2]  # Les 2 meilleurs périphériques
                        for device in top_devices:
                            result["recommended_actions"].append(
                                {
                                    "action": "select_device",
                                    "description": f"Essayer le périphérique: {device['name']}",
                                    "device_index": device["index"],
                                }
                            )
            else:
                result["user_message"] = "Problème d'accès au microphone détecté."
                result["user_action_required"] = True

        elif error_type == "timeout":
            # Timeout d'écoute
            result["user_message"] = (
                "Jeffrey n'a pas entendu votre voix. Veuillez parler plus fort ou vérifier votre microphone."
            )
            result["recommended_actions"].append(
                {
                    "action": "retry",
                    "description": "Réessayer la reconnaissance avec une sensibilité plus élevée",
                }
            )
            result["recommended_actions"].append(
                {
                    "action": "adjust_sensitivity",
                    "description": "Augmenter la sensibilité du microphone",
                    "sensitivity_adjustment": 1.2,  # Augmenter de 20%
                }
            )

        elif error_type == "recognition":
            # Problème de reconnaissance
            result["user_message"] = "Jeffrey n'a pas compris ce que vous avez dit. Veuillez parler plus distinctement."
            result["recommended_actions"].append({"action": "retry", "description": "Réessayer la reconnaissance"})

        else:
            # Erreur inconnue
            result["user_message"] = "Un problème est survenu avec la reconnaissance vocale. Veuillez réessayer."
            result["user_action_required"] = True
            result["recommended_actions"].append(
                {"action": "retry", "description": "Réessayer la reconnaissance après une pause"}
            )

        return result

    def reset(self):
        """Réinitialise le système de récupération d'erreurs."""
        self.retries = 0
        self.last_error = None
        self.recovery_successful = False
        logger.info("Système de récupération d'erreurs réinitialisé")


# Instance singleton à utiliser dans d'autres modules
voice_error_recovery = VoiceRecognitionErrorRecovery()


def fix_microphone_issues():
    """
    Fonction d'aide pour résoudre les problèmes de microphone.
    À appeler depuis l'interface utilisateur ou la ligne de commande.

    Returns:
        bool: True si les problèmes sont résolus
    """
    return voice_error_recovery.fix_microphone_issues()


def check_mic_permissions():
    """
    Vérifie les permissions d'accès au microphone.
    À appeler avant de démarrer la reconnaissance vocale.

    Returns:
        Tuple[bool, str]: (permission accordée, message d'état)
    """
    return voice_error_recovery.check_microphone_permissions()


def show_mic_troubleshooting():
    """
    Affiche des instructions de dépannage pour le microphone.
    Utile à intégrer dans l'interface utilisateur.

    Returns:
        str: Instructions de dépannage
    """
    return voice_error_recovery.get_recovery_instructions()


def handle_recognition_error(error):
    """
    Gère une erreur de reconnaissance vocale et tente de la résoudre.
    À appeler depuis le module de reconnaissance vocale.

    Args:
        error: L'exception rencontrée

    Returns:
        Dict: Résultat de la gestion d'erreur avec actions recommandées
    """
    return voice_error_recovery.handle_recognition_error(error)


if __name__ == "__main__":
    # Configuration du logging pour les tests
    logging.basicConfig(level=logging.INFO)

    # Test simple du système de récupération d'erreurs
    print("=== Test du système de récupération d'erreurs pour la reconnaissance vocale ===")

    recovery = VoiceRecognitionErrorRecovery()

    # Vérifier les permissions
    has_permission, status = recovery.check_microphone_permissions()
    print(f"Permissions microphone: {has_permission} ({status})")

    # Afficher les instructions
    print("\nInstructions de dépannage:")
    print(recovery.get_recovery_instructions())

    # Recommander des périphériques
    print("\nPériphériques recommandés:")
    devices = recovery.recommend_audio_devices()
    for device in devices:
        print(f" - {device['name']} (score: {device['score']}): {device['recommendation']}")

    print("\nTest terminé.")
