#!/usr/bin/env python
"""
cognitive_cycle_engine.py – Moteur cognitif principal de Jeffrey

Ce module orchestre les cycles cognitifs de l'IA Jeffrey : perception, réflexion,
prise de décision et action. Il constitue le noyau du raisonnement autonome.

Ce moteur est extensible, traçable, et supporte un mode simulation, des hooks, une gestion d'erreurs et une évaluation par cycle.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime


class CognitiveCycleEngine:
    """
    Gère le cycle cognitif complet de Jeffrey :
    - collecte des perceptions
    - génération de pensées internes
    - prise de décision
    - déclenchement d'une action
    """

    def __init__(
        self,
        perception_module=None,
        thoughts_manager=None,
        decision_maker=None,
        action_executor=None,
        context_manager=None,
        simulate=False,
    ):
        self.perception_module = perception_module
        self.thoughts_manager = thoughts_manager
        self.decision_maker = decision_maker
        self.action_executor = action_executor
        self.context_manager = context_manager
        self.simulate = simulate
        self.history = []

    def run_cycle(self):
        """
        Exécute un cycle cognitif complet :
        1. Percevoir
        2. Penser
        3. Décider
        4. Agir
        """
        cycle_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        start = time.time()

        self.before_cycle()

        try:
            perceptions = self._perceive()
            thoughts = self._think(perceptions)
            decision = self._decide(thoughts)
            result = self._act(decision)
        except Exception as e:
            result = f"Erreur: {str(e)}"
            logging.exception(f"[CYCLE {cycle_id}] Exception during cycle execution")
            thoughts, decision = {}, None

        duration = round(time.time() - start, 3)
        score = self.evaluate_cycle(result)

        cycle_log = {
            "cycle_id": cycle_id,
            "timestamp": timestamp,
            "duration": duration,
            "perceptions": perceptions,
            "thoughts": thoughts,
            "decision": decision,
            "result": result,
            "score": score,
        }

        self.after_cycle(cycle_log)
        self.history.append(cycle_log)
        logging.info(f"[CYCLE {cycle_id}] Terminé en {duration}s – score: {score} – décision: {decision}")
        return cycle_log

    def _perceive(self):
        """Collecte les perceptions via le module de perception."""
        if self.perception_module:
            return self.perception_module.collect()
        return {}

    def _think(self, perceptions):
        """Génère des pensées à partir des perceptions et du contexte."""
        context = self.context_manager.get_context() if self.context_manager else {}
        if self.thoughts_manager:
            return self.thoughts_manager.generate(perceptions, context)
        return {}

    def _decide(self, thoughts):
        """Prend une décision à partir des pensées générées et du contexte."""
        context = self.context_manager.get_context() if self.context_manager else {}
        if self.decision_maker:
            return self.decision_maker.choose_action(thoughts, context)
        return None

    def _act(self, decision):
        """Exécute ou simule l'action décidée."""
        if not decision:
            return "Aucune action décidée"
        if self.simulate:
            return f"[SIMULATED ACTION] {decision}"
        if self.action_executor:
            return self.action_executor.execute(decision)
        return "Aucune action exécutée"

    def before_cycle(self):
        """Hook exécuté avant chaque cycle. Peut être surchargé."""
        pass

    def after_cycle(self, cycle_log):
        """Hook exécuté après chaque cycle. Peut être surchargé."""
        pass

    def evaluate_cycle(self, result):
        """
        Évalue le résultat d'un cycle pour générer un score (ex: renforcement futur).
        Par défaut, renvoie 1.0 si une action a été exécutée, sinon 0.0.
        """
        if isinstance(result, str) and result.startswith("Erreur"):
            return 0.0
        return 1.0 if result else 0.0
