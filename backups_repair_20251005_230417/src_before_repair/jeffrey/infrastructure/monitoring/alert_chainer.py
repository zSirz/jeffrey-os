"""
Alert Chainer - Configurable Rules with Control Theory
Jeffrey OS v0.6.2 - ROBUSTESSE ADAPTATIVE
"""

import asyncio
import json
import logging
import re
import smtplib
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import aiohttp

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ConditionOperator(Enum):
    """Condition evaluation operators"""

    GT = "greater_than"  # >
    GTE = "greater_than_equal"  # >=
    LT = "less_than"  # <
    LTE = "less_than_equal"  # <=
    EQ = "equals"  # ==
    NEQ = "not_equals"  # !=
    CONTAINS = "contains"  # in string
    REGEX = "regex_match"  # regex pattern
    RANGE = "in_range"  # between min/max
    TREND_UP = "trending_up"  # increasing trend
    TREND_DOWN = "trending_down"  # decreasing trend


class LogicalOperator(Enum):
    """Logical operators for combining conditions"""

    AND = "and"
    OR = "or"
    NOT = "not"


class ActionType(Enum):
    """Types of actions that can be triggered"""

    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSE = "circuit_breaker_close"
    WEBHOOK_SLACK = "webhook_slack"
    WEBHOOK_DISCORD = "webhook_discord"
    WEBHOOK_TEAMS = "webhook_teams"
    WEBHOOK_CUSTOM = "webhook_custom"
    EMAIL_ALERT = "email_alert"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    LOG_ROTATION = "log_rotation"
    RESTART_SERVICE = "restart_service"
    EXEC_COMMAND = "exec_command"
    UPDATE_CONFIG = "update_config"


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Condition:
    """Single condition in alert rule"""

    field: str  # Metric field to evaluate
    operator: ConditionOperator  # Comparison operator
    value: float | str | list[float]  # Threshold value(s)
    window_seconds: int = 60  # Time window for evaluation
    hysteresis_factor: float = 0.1  # Control theory hysteresis (10% default)
    description: str = ""


@dataclass
class AlertRule:
    """Configurable alert rule with multiple conditions"""

    rule_id: str
    name: str
    description: str

    # Conditions
    conditions: list[Condition]
    logical_operator: LogicalOperator  # How to combine conditions

    # Control theory parameters
    enable_hysteresis: bool = True
    hysteresis_upper_factor: float = 1.1  # 10% above threshold
    hysteresis_lower_factor: float = 0.9  # 10% below threshold

    # Actions
    actions: list["AlertAction"]

    # Rule configuration
    severity: AlertSeverity = AlertSeverity.MEDIUM
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes between identical actions
    max_triggers_per_hour: int = 10

    # Evaluation state
    current_state: bool = False  # Currently triggered
    last_trigger_time: float | None = None
    trigger_count_hour: int = 0
    evaluation_history: list[tuple[float, bool]] = None  # (timestamp, result)

    def __post_init__(self):
        if self.evaluation_history is None:
            self.evaluation_history = []


@dataclass
class AlertAction:
    """Action to execute when alert is triggered"""

    action_id: str
    action_type: ActionType
    parameters: dict[str, Any]

    # Execution control
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default
    max_retries: int = 3
    timeout_seconds: int = 30

    # Dependencies (chain actions)
    depends_on: list[str] = None  # List of action_ids that must succeed first
    failure_actions: list[str] = None  # Actions to trigger on failure

    # Execution state
    last_execution_time: float | None = None
    execution_count: int = 0
    last_result: bool | None = None
    last_error: str | None = None

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
        if self.failure_actions is None:
            self.failure_actions = []


@dataclass
class ActionExecution:
    """Record of action execution"""

    execution_id: str
    action_id: str
    rule_id: str
    timestamp: str

    # Execution details
    success: bool
    execution_time_ms: float
    error_message: str | None
    response_data: dict[str, Any] | None

    # Context
    triggered_by_conditions: list[str]
    metric_values: dict[str, float]
    test_mode: bool = False


class ControlTheoryEvaluator:
    """
    Control theory-based alert evaluation with hysteresis
    Prevents alert flapping and provides stable thresholds
    """

    def __init__(self):
        self.state_history: dict[str, list[tuple[float, float]]] = {}  # rule_id -> [(timestamp, value)]
        self.threshold_states: dict[str, dict[str, bool]] = {}  # rule_id -> {condition_id: current_state}

    def evaluate_condition_with_hysteresis(
        self, rule_id: str, condition: Condition, current_value: float, previous_state: bool
    ) -> bool:
        """
        Evaluate condition with hysteresis control theory

        Args:
            rule_id: Alert rule identifier
            condition: Condition to evaluate
            current_value: Current metric value
            previous_state: Previous condition state

        Returns:
            New condition state considering hysteresis
        """

        # Store value history
        condition_id = f"{condition.field}_{condition.operator.value}"
        if rule_id not in self.state_history:
            self.state_history[rule_id] = []

        current_time = time.time()
        self.state_history[rule_id].append((current_time, current_value))

        # Keep only recent history
        cutoff_time = current_time - condition.window_seconds
        self.state_history[rule_id] = [(t, v) for t, v in self.state_history[rule_id] if t > cutoff_time]

        # Base threshold evaluation
        base_result = self._evaluate_base_condition(condition, current_value)

        # Apply hysteresis if enabled
        if not condition.hysteresis_factor or condition.hysteresis_factor == 0:
            return base_result

        # Get threshold value
        if isinstance(condition.value, (int, float)):
            threshold = float(condition.value)
        else:
            return base_result  # Can't apply hysteresis to non-numeric thresholds

        # Calculate hysteresis bands
        hysteresis_delta = threshold * condition.hysteresis_factor

        if condition.operator in [ConditionOperator.GT, ConditionOperator.GTE]:
            # For "greater than" conditions
            upper_threshold = threshold
            lower_threshold = threshold - hysteresis_delta

            if previous_state:
                # Currently triggered - use lower threshold to stay triggered
                return current_value > lower_threshold
            else:
                # Not triggered - use upper threshold to trigger
                return current_value > upper_threshold

        elif condition.operator in [ConditionOperator.LT, ConditionOperator.LTE]:
            # For "less than" conditions
            lower_threshold = threshold
            upper_threshold = threshold + hysteresis_delta

            if previous_state:
                # Currently triggered - use upper threshold to stay triggered
                return current_value < upper_threshold
            else:
                # Not triggered - use lower threshold to trigger
                return current_value < lower_threshold

        # For other operators, return base evaluation
        return base_result

    def _evaluate_base_condition(self, condition: Condition, value: float) -> bool:
        """Evaluate condition without hysteresis"""

        if condition.operator == ConditionOperator.GT:
            return value > condition.value
        elif condition.operator == ConditionOperator.GTE:
            return value >= condition.value
        elif condition.operator == ConditionOperator.LT:
            return value < condition.value
        elif condition.operator == ConditionOperator.LTE:
            return value <= condition.value
        elif condition.operator == ConditionOperator.EQ:
            return abs(value - condition.value) < 0.001  # Float comparison
        elif condition.operator == ConditionOperator.NEQ:
            return abs(value - condition.value) >= 0.001
        elif condition.operator == ConditionOperator.RANGE:
            if isinstance(condition.value, list) and len(condition.value) == 2:
                return condition.value[0] <= value <= condition.value[1]

        return False

    def evaluate_trend_condition(self, rule_id: str, condition: Condition) -> bool:
        """Evaluate trend-based conditions"""

        if rule_id not in self.state_history or len(self.state_history[rule_id]) < 3:
            return False

        # Get recent values
        recent_values = [v for _, v in self.state_history[rule_id][-10:]]  # Last 10 values

        if len(recent_values) < 3:
            return False

        # Calculate trend (simple linear regression slope)
        if NUMPY_AVAILABLE:
            x = np.arange(len(recent_values))
            y = np.array(recent_values)
            slope = np.polyfit(x, y, 1)[0]
        else:
            # Fallback: simple slope calculation
            n = len(recent_values)
            x_mean = (n - 1) / 2
            y_mean = sum(recent_values) / n

            numerator = sum((i - x_mean) * (recent_values[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            slope = numerator / denominator if denominator != 0 else 0

        # Evaluate trend direction
        if condition.operator == ConditionOperator.TREND_UP:
            return slope > 0.01  # Positive trend threshold
        elif condition.operator == ConditionOperator.TREND_DOWN:
            return slope < -0.01  # Negative trend threshold

        return False


class AlertChainer:
    """
    Configurable alert chainer with control theory and action chains

    Features:
    - Multi-condition rules with logical operators
    - Control theory evaluation with hysteresis
    - Chained actions with dependencies
    - Configurable webhooks (Slack, Discord, Teams, custom)
    - Cooldown management
    - Test mode for validation
    """

    def __init__(self, config_file: str | None = None, test_mode: bool = False, webhook_timeout: int = 10):
        """
        Initialize alert chainer

        Args:
            config_file: Path to alert configuration file
            test_mode: Run in test mode (no actual actions)
            webhook_timeout: Timeout for webhook calls in seconds
        """
        self.test_mode = test_mode
        self.webhook_timeout = webhook_timeout

        # Alert configuration
        self.rules: dict[str, AlertRule] = {}
        self.actions: dict[str, AlertAction] = {}

        # Evaluation engine
        self.control_evaluator = ControlTheoryEvaluator()

        # Execution tracking
        self.execution_history: list[ActionExecution] = []
        self.action_cooldowns: dict[str, float] = {}  # action_id -> last_execution_time

        # Metrics storage
        self.current_metrics: dict[str, float] = {}
        self.metric_history: dict[str, list[tuple[float, float]]] = {}  # field -> [(timestamp, value)]

        # Webhook configurations
        self.webhook_configs = {
            "slack": {"url": None, "channel": None, "username": "Jeffrey OS"},
            "discord": {"url": None, "username": "Jeffrey OS"},
            "teams": {"url": None},
            "email": {
                "smtp_server": None,
                "smtp_port": 587,
                "username": None,
                "password": None,
                "from_addr": None,
            },
        }

        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "total_triggers": 0,
            "total_actions_executed": 0,
            "total_actions_failed": 0,
            "avg_evaluation_time_ms": 0.0,
            "avg_action_time_ms": 0.0,
        }

        # Thread safety
        self._lock = threading.Lock()
        self.running = False

        # Load configuration
        if config_file:
            self.load_configuration(config_file)

        logging.info(f"Alert Chainer initialized (test_mode: {test_mode})")

    def load_configuration(self, config_file: str):
        """Load alert rules and actions from configuration file"""

        try:
            with open(config_file) as f:
                config = json.load(f)

            # Load webhook configurations
            if "webhook_configs" in config:
                self.webhook_configs.update(config["webhook_configs"])

            # Load alert rules
            if "rules" in config:
                for rule_data in config["rules"]:
                    rule = self._deserialize_alert_rule(rule_data)
                    self.rules[rule.rule_id] = rule

            # Load actions
            if "actions" in config:
                for action_data in config["actions"]:
                    action = self._deserialize_alert_action(action_data)
                    self.actions[action.action_id] = action

            logging.info(f"Loaded {len(self.rules)} rules and {len(self.actions)} actions")

        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise

    def _deserialize_alert_rule(self, rule_data: dict[str, Any]) -> AlertRule:
        """Convert dict to AlertRule object"""

        conditions = []
        for cond_data in rule_data.get("conditions", []):
            condition = Condition(
                field=cond_data["field"],
                operator=ConditionOperator(cond_data["operator"]),
                value=cond_data["value"],
                window_seconds=cond_data.get("window_seconds", 60),
                hysteresis_factor=cond_data.get("hysteresis_factor", 0.1),
                description=cond_data.get("description", ""),
            )
            conditions.append(condition)

        actions = []
        for action_id in rule_data.get("actions", []):
            if action_id in self.actions:
                actions.append(self.actions[action_id])

        return AlertRule(
            rule_id=rule_data["rule_id"],
            name=rule_data["name"],
            description=rule_data["description"],
            conditions=conditions,
            logical_operator=LogicalOperator(rule_data.get("logical_operator", "and")),
            enable_hysteresis=rule_data.get("enable_hysteresis", True),
            hysteresis_upper_factor=rule_data.get("hysteresis_upper_factor", 1.1),
            hysteresis_lower_factor=rule_data.get("hysteresis_lower_factor", 0.9),
            actions=actions,
            severity=AlertSeverity(rule_data.get("severity", "medium")),
            enabled=rule_data.get("enabled", True),
            cooldown_seconds=rule_data.get("cooldown_seconds", 300),
            max_triggers_per_hour=rule_data.get("max_triggers_per_hour", 10),
        )

    def _deserialize_alert_action(self, action_data: dict[str, Any]) -> AlertAction:
        """Convert dict to AlertAction object"""

        return AlertAction(
            action_id=action_data["action_id"],
            action_type=ActionType(action_data["action_type"]),
            parameters=action_data.get("parameters", {}),
            enabled=action_data.get("enabled", True),
            cooldown_seconds=action_data.get("cooldown_seconds", 300),
            max_retries=action_data.get("max_retries", 3),
            timeout_seconds=action_data.get("timeout_seconds", 30),
            depends_on=action_data.get("depends_on", []),
            failure_actions=action_data.get("failure_actions", []),
        )

    async def start_monitoring(self):
        """Start alert monitoring loop"""

        if self.running:
            return

        self.running = True
        logging.info("Starting alert monitoring")

        try:
            while self.running:
                await self._evaluation_cycle()
                await asyncio.sleep(5)  # Evaluate every 5 seconds
        except Exception as e:
            logging.error(f"Alert monitoring error: {e}")
        finally:
            self.running = False

    async def stop_monitoring(self):
        """Stop alert monitoring"""
        self.running = False
        logging.info("Alert monitoring stopped")

    def update_metrics(self, metrics: dict[str, float]):
        """Update current metrics for evaluation"""

        current_time = time.time()

        with self._lock:
            self.current_metrics.update(metrics)

            # Store metric history
            for field, value in metrics.items():
                if field not in self.metric_history:
                    self.metric_history[field] = []

                self.metric_history[field].append((current_time, value))

                # Keep only recent history (1 hour)
                cutoff_time = current_time - 3600
                self.metric_history[field] = [(t, v) for t, v in self.metric_history[field] if t > cutoff_time]

    async def _evaluation_cycle(self):
        """Single evaluation cycle for all rules"""

        start_time = time.time()

        try:
            triggered_rules = []

            # Evaluate all enabled rules
            for rule_id, rule in self.rules.items():
                if not rule.enabled:
                    continue

                # Check cooldown
                if self._is_rule_in_cooldown(rule):
                    continue

                # Evaluate rule conditions
                rule_triggered = await self._evaluate_rule(rule)

                if rule_triggered and not rule.current_state:
                    # Rule newly triggered
                    triggered_rules.append(rule)
                    rule.current_state = True
                    rule.last_trigger_time = time.time()
                    rule.trigger_count_hour += 1

                    logging.info(f"Alert rule triggered: {rule.name}")

                elif not rule_triggered and rule.current_state:
                    # Rule no longer triggered
                    rule.current_state = False
                    logging.info(f"Alert rule cleared: {rule.name}")

            # Execute actions for triggered rules
            for rule in triggered_rules:
                await self._execute_rule_actions(rule)

            # Update statistics
            evaluation_time = (time.time() - start_time) * 1000
            with self._lock:
                self.stats["total_evaluations"] += 1
                self.stats["total_triggers"] += len(triggered_rules)

                current_avg = self.stats["avg_evaluation_time_ms"]
                total_evals = self.stats["total_evaluations"]
                self.stats["avg_evaluation_time_ms"] = (current_avg * (total_evals - 1) + evaluation_time) / total_evals

        except Exception as e:
            logging.error(f"Evaluation cycle failed: {e}")

    async def _evaluate_rule(self, rule: AlertRule) -> bool:
        """Evaluate alert rule with control theory"""

        condition_results = []

        # Evaluate each condition
        for condition in rule.conditions:
            # Get current metric value
            current_value = self.current_metrics.get(condition.field, 0.0)

            # Get previous state for hysteresis
            previous_state = self._get_condition_previous_state(rule.rule_id, condition)

            # Evaluate with hysteresis
            if condition.operator in [ConditionOperator.TREND_UP, ConditionOperator.TREND_DOWN]:
                result = self.control_evaluator.evaluate_trend_condition(rule.rule_id, condition)
            elif condition.operator in [ConditionOperator.CONTAINS, ConditionOperator.REGEX]:
                result = self._evaluate_string_condition(condition, current_value)
            else:
                result = self.control_evaluator.evaluate_condition_with_hysteresis(
                    rule.rule_id, condition, current_value, previous_state
                )

            condition_results.append(result)

            # Store condition state
            self._store_condition_state(rule.rule_id, condition, result)

        # Combine conditions using logical operator
        if rule.logical_operator == LogicalOperator.AND:
            final_result = all(condition_results)
        elif rule.logical_operator == LogicalOperator.OR:
            final_result = any(condition_results)
        else:  # NOT (apply to first condition)
            final_result = not condition_results[0] if condition_results else False

        # Store evaluation history
        current_time = time.time()
        rule.evaluation_history.append((current_time, final_result))

        # Keep only recent history
        rule.evaluation_history = rule.evaluation_history[-100:]

        return final_result

    def _evaluate_string_condition(self, condition: Condition, value: str | float) -> bool:
        """Evaluate string-based conditions"""

        str_value = str(value)

        if condition.operator == ConditionOperator.CONTAINS:
            return str(condition.value) in str_value
        elif condition.operator == ConditionOperator.REGEX:
            try:
                return bool(re.search(str(condition.value), str_value))
            except re.error:
                return False

        return False

    def _get_condition_previous_state(self, rule_id: str, condition: Condition) -> bool:
        """Get previous state of condition for hysteresis"""

        condition_key = f"{condition.field}_{condition.operator.value}"

        if rule_id not in self.control_evaluator.threshold_states:
            return False

        return self.control_evaluator.threshold_states[rule_id].get(condition_key, False)

    def _store_condition_state(self, rule_id: str, condition: Condition, state: bool):
        """Store current condition state"""

        condition_key = f"{condition.field}_{condition.operator.value}"

        if rule_id not in self.control_evaluator.threshold_states:
            self.control_evaluator.threshold_states[rule_id] = {}

        self.control_evaluator.threshold_states[rule_id][condition_key] = state

    def _is_rule_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period"""

        if not rule.last_trigger_time:
            return False

        time_since_trigger = time.time() - rule.last_trigger_time
        return time_since_trigger < rule.cooldown_seconds

    async def _execute_rule_actions(self, rule: AlertRule):
        """Execute all actions for triggered rule"""

        logging.info(f"Executing {len(rule.actions)} actions for rule: {rule.name}")

        # Group actions by dependencies
        action_groups = self._group_actions_by_dependencies(rule.actions)

        # Execute action groups in dependency order
        for group in action_groups:
            group_tasks = []

            for action in group:
                if self._can_execute_action(action):
                    task = self._execute_single_action(action, rule)
                    group_tasks.append(task)

            # Wait for all actions in group to complete
            if group_tasks:
                await asyncio.gather(*group_tasks, return_exceptions=True)

    def _group_actions_by_dependencies(self, actions: list[AlertAction]) -> list[list[AlertAction]]:
        """Group actions by dependency levels for sequential execution"""

        action_dict = {action.action_id: action for action in actions}
        groups = []
        processed = set()

        while len(processed) < len(actions):
            current_group = []

            for action in actions:
                if action.action_id in processed:
                    continue

                # Check if all dependencies are satisfied
                deps_satisfied = all(dep_id in processed for dep_id in action.depends_on)

                if deps_satisfied:
                    current_group.append(action)
                    processed.add(action.action_id)

            if current_group:
                groups.append(current_group)
            else:
                # Circular dependency or missing dependency - add remaining actions
                remaining = [action for action in actions if action.action_id not in processed]
                if remaining:
                    groups.append(remaining)
                    break

        return groups

    def _can_execute_action(self, action: AlertAction) -> bool:
        """Check if action can be executed (cooldown, limits, etc.)"""

        if not action.enabled:
            return False

        # Check cooldown
        if action.action_id in self.action_cooldowns:
            time_since_last = time.time() - self.action_cooldowns[action.action_id]
            if time_since_last < action.cooldown_seconds:
                return False

        return True

    async def _execute_single_action(self, action: AlertAction, rule: AlertRule):
        """Execute single action with retries and error handling"""

        execution_id = f"{action.action_id}_{int(time.time() * 1000000)}"
        start_time = time.time()

        success = False
        error_message = None
        response_data = None

        try:
            # Update cooldown
            self.action_cooldowns[action.action_id] = time.time()

            # Execute action based on type
            if action.action_type == ActionType.WEBHOOK_SLACK:
                success, response_data = await self._execute_slack_webhook(action)
            elif action.action_type == ActionType.WEBHOOK_DISCORD:
                success, response_data = await self._execute_discord_webhook(action)
            elif action.action_type == ActionType.WEBHOOK_TEAMS:
                success, response_data = await self._execute_teams_webhook(action)
            elif action.action_type == ActionType.WEBHOOK_CUSTOM:
                success, response_data = await self._execute_custom_webhook(action)
            elif action.action_type == ActionType.EMAIL_ALERT:
                success, response_data = await self._execute_email_alert(action, rule)
            elif action.action_type == ActionType.CIRCUIT_BREAKER_OPEN:
                success, response_data = await self._execute_circuit_breaker_action(action, True)
            elif action.action_type == ActionType.CIRCUIT_BREAKER_CLOSE:
                success, response_data = await self._execute_circuit_breaker_action(action, False)
            elif action.action_type == ActionType.SCALE_UP:
                success, response_data = await self._execute_scaling_action(action, "up")
            elif action.action_type == ActionType.SCALE_DOWN:
                success, response_data = await self._execute_scaling_action(action, "down")
            elif action.action_type == ActionType.EXEC_COMMAND:
                success, response_data = await self._execute_command_action(action)
            else:
                error_message = f"Unsupported action type: {action.action_type.value}"

            # Update action state
            action.last_execution_time = time.time()
            action.execution_count += 1
            action.last_result = success
            action.last_error = error_message

        except Exception as e:
            success = False
            error_message = str(e)
            logging.error(f"Action execution failed: {action.action_id}: {e}")

        # Create execution record
        execution_time = (time.time() - start_time) * 1000
        execution = ActionExecution(
            execution_id=execution_id,
            action_id=action.action_id,
            rule_id=rule.rule_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            success=success,
            execution_time_ms=execution_time,
            error_message=error_message,
            response_data=response_data,
            triggered_by_conditions=[cond.field for cond in rule.conditions],
            metric_values=self.current_metrics.copy(),
            test_mode=self.test_mode,
        )

        # Store execution record
        with self._lock:
            self.execution_history.append(execution)
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]

            # Update statistics
            if success:
                self.stats["total_actions_executed"] += 1
            else:
                self.stats["total_actions_failed"] += 1

            current_avg = self.stats["avg_action_time_ms"]
            total_actions = self.stats["total_actions_executed"] + self.stats["total_actions_failed"]
            if total_actions > 0:
                self.stats["avg_action_time_ms"] = (current_avg * (total_actions - 1) + execution_time) / total_actions

        # Execute failure actions if needed
        if not success and action.failure_actions:
            for failure_action_id in action.failure_actions:
                if failure_action_id in self.actions:
                    failure_action = self.actions[failure_action_id]
                    await self._execute_single_action(failure_action, rule)

        logging.info(
            f"Action {action.action_id} executed: {'success' if success else 'failed'} in {execution_time:.1f}ms"
        )

    async def _execute_slack_webhook(self, action: AlertAction) -> tuple[bool, dict[str, Any]]:
        """Execute Slack webhook"""

        if self.test_mode:
            return True, {"test_mode": True, "message": "Slack webhook (test)"}

        webhook_url = self.webhook_configs["slack"]["url"]
        if not webhook_url:
            raise ValueError("Slack webhook URL not configured")

        # Build Slack message
        message = {
            "username": self.webhook_configs["slack"]["username"],
            "text": action.parameters.get("message", "Jeffrey OS Alert"),
            "attachments": [
                {
                    "color": self._get_slack_color(action.parameters.get("severity", "medium")),
                    "fields": [
                        {
                            "title": "Alert",
                            "value": action.parameters.get("title", "System Alert"),
                            "short": True,
                        },
                        {
                            "title": "Time",
                            "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True,
                        },
                    ],
                }
            ],
        }

        # Add channel if specified
        if self.webhook_configs["slack"]["channel"]:
            message["channel"] = self.webhook_configs["slack"]["channel"]

        # Send webhook
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.webhook_timeout)) as session:
            async with session.post(webhook_url, json=message) as response:
                if response.status == 200:
                    return True, {"status": response.status, "response": "ok"}
                else:
                    response_text = await response.text()
                    return False, {"status": response.status, "error": response_text}

    async def _execute_discord_webhook(self, action: AlertAction) -> tuple[bool, dict[str, Any]]:
        """Execute Discord webhook"""

        if self.test_mode:
            return True, {"test_mode": True, "message": "Discord webhook (test)"}

        webhook_url = self.webhook_configs["discord"]["url"]
        if not webhook_url:
            raise ValueError("Discord webhook URL not configured")

        # Build Discord message
        message = {
            "username": self.webhook_configs["discord"]["username"],
            "content": action.parameters.get("message", "Jeffrey OS Alert"),
            "embeds": [
                {
                    "title": action.parameters.get("title", "System Alert"),
                    "description": action.parameters.get("description", ""),
                    "color": self._get_discord_color(action.parameters.get("severity", "medium")),
                    "timestamp": datetime.utcnow().isoformat(),
                    "footer": {"text": "Jeffrey OS v0.6.2"},
                }
            ],
        }

        # Send webhook
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.webhook_timeout)) as session:
            async with session.post(webhook_url, json=message) as response:
                if response.status in [200, 204]:
                    return True, {"status": response.status, "response": "ok"}
                else:
                    response_text = await response.text()
                    return False, {"status": response.status, "error": response_text}

    async def _execute_teams_webhook(self, action: AlertAction) -> tuple[bool, dict[str, Any]]:
        """Execute Microsoft Teams webhook"""

        if self.test_mode:
            return True, {"test_mode": True, "message": "Teams webhook (test)"}

        webhook_url = self.webhook_configs["teams"]["url"]
        if not webhook_url:
            raise ValueError("Teams webhook URL not configured")

        # Build Teams message (Adaptive Card format)
        message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": self._get_teams_color(action.parameters.get("severity", "medium")),
            "summary": action.parameters.get("title", "Jeffrey OS Alert"),
            "sections": [
                {
                    "activityTitle": action.parameters.get("title", "System Alert"),
                    "activitySubtitle": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "text": action.parameters.get("message", "Jeffrey OS Alert"),
                    "facts": [
                        {
                            "name": "Severity",
                            "value": action.parameters.get("severity", "medium").upper(),
                        },
                        {
                            "name": "Component",
                            "value": action.parameters.get("component", "Unknown"),
                        },
                    ],
                }
            ],
        }

        # Send webhook
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.webhook_timeout)) as session:
            async with session.post(webhook_url, json=message) as response:
                if response.status == 200:
                    return True, {"status": response.status, "response": "ok"}
                else:
                    response_text = await response.text()
                    return False, {"status": response.status, "error": response_text}

    async def _execute_custom_webhook(self, action: AlertAction) -> tuple[bool, dict[str, Any]]:
        """Execute custom webhook"""

        if self.test_mode:
            return True, {"test_mode": True, "message": "Custom webhook (test)"}

        webhook_url = action.parameters.get("url")
        if not webhook_url:
            raise ValueError("Custom webhook URL not specified")

        # Build custom payload
        payload = action.parameters.get("payload", {})
        method = action.parameters.get("method", "POST").upper()
        headers = action.parameters.get("headers", {"Content-Type": "application/json"})

        # Send webhook
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.webhook_timeout)) as session:
            if method == "GET":
                async with session.get(webhook_url, headers=headers) as response:
                    success = 200 <= response.status < 300
                    response_text = await response.text()
                    return success, {"status": response.status, "response": response_text}
            else:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    success = 200 <= response.status < 300
                    response_text = await response.text()
                    return success, {"status": response.status, "response": response_text}

    async def _execute_email_alert(self, action: AlertAction, rule: AlertRule) -> tuple[bool, dict[str, Any]]:
        """Execute email alert"""

        if self.test_mode:
            return True, {"test_mode": True, "message": "Email alert (test)"}

        # Check email configuration
        email_config = self.webhook_configs["email"]
        if not all([email_config["smtp_server"], email_config["username"], email_config["password"]]):
            raise ValueError("Email configuration incomplete")

        # Build email
        to_addresses = action.parameters.get("to", [])
        if not to_addresses:
            raise ValueError("No email recipients specified")

        subject = action.parameters.get("subject", f"Jeffrey OS Alert: {rule.name}")
        body = action.parameters.get("body", f"Alert triggered: {rule.description}")

        # Create email message
        msg = MIMEMultipart()
        msg["From"] = email_config["from_addr"]
        msg["To"] = ", ".join(to_addresses)
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        # Send email
        try:
            with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
                server.starttls()
                server.login(email_config["username"], email_config["password"])
                server.send_message(msg)

            return True, {"recipients": len(to_addresses), "subject": subject}

        except Exception as e:
            return False, {"error": str(e)}

    async def _execute_circuit_breaker_action(
        self, action: AlertAction, open_breaker: bool
    ) -> tuple[bool, dict[str, Any]]:
        """Execute circuit breaker action"""

        circuit_name = action.parameters.get("circuit_name", "default")

        if self.test_mode:
            return True, {
                "test_mode": True,
                "circuit": circuit_name,
                "action": "open" if open_breaker else "close",
            }

        # In production, integrate with actual circuit breaker system
        # For demo, simulate success
        action_name = "opened" if open_breaker else "closed"
        logging.info(f"Circuit breaker {circuit_name} {action_name}")

        return True, {"circuit": circuit_name, "action": action_name}

    async def _execute_scaling_action(self, action: AlertAction, direction: str) -> tuple[bool, dict[str, Any]]:
        """Execute scaling action"""

        component = action.parameters.get("component", "default")
        scale_factor = action.parameters.get("scale_factor", 1.5)

        if self.test_mode:
            return True, {
                "test_mode": True,
                "component": component,
                "direction": direction,
                "factor": scale_factor,
            }

        # In production, integrate with actual auto-scaler
        # For demo, simulate success
        logging.info(f"Scaling {component} {direction} by factor {scale_factor}")

        return True, {"component": component, "direction": direction, "scale_factor": scale_factor}

    async def _execute_command_action(self, action: AlertAction) -> tuple[bool, dict[str, Any]]:
        """Execute command action"""

        command = action.parameters.get("command")
        if not command:
            raise ValueError("Command not specified")

        if self.test_mode:
            return True, {"test_mode": True, "command": command}

        # In production, execute with proper security controls
        # For demo, simulate success
        logging.info(f"Executing command: {command}")

        return True, {"command": command, "executed": True}

    def _get_slack_color(self, severity: str) -> str:
        """Get Slack color for severity"""
        colors = {
            "low": "good",
            "medium": "warning",
            "high": "danger",
            "critical": "danger",
            "emergency": "danger",
        }
        return colors.get(severity, "warning")

    def _get_discord_color(self, severity: str) -> int:
        """Get Discord color for severity"""
        colors = {
            "low": 0x36A64F,  # Green
            "medium": 0xFFCC00,  # Yellow
            "high": 0xFF6600,  # Orange
            "critical": 0xFF0000,  # Red
            "emergency": 0x800080,  # Purple
        }
        return colors.get(severity, 0xFFCC00)

    def _get_teams_color(self, severity: str) -> str:
        """Get Teams color for severity"""
        colors = {
            "low": "28a745",  # Green
            "medium": "ffc107",  # Yellow
            "high": "fd7e14",  # Orange
            "critical": "dc3545",  # Red
            "emergency": "6f42c1",  # Purple
        }
        return colors.get(severity, "ffc107")

    def get_statistics(self) -> dict[str, Any]:
        """Get chainer statistics"""

        with self._lock:
            stats = self.stats.copy()

        stats.update(
            {
                "total_rules": len(self.rules),
                "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
                "total_actions": len(self.actions),
                "enabled_actions": len([a for a in self.actions.values() if a.enabled]),
                "active_cooldowns": len(self.action_cooldowns),
                "test_mode": self.test_mode,
            }
        )

        return stats

    def get_rule_status(self) -> list[dict[str, Any]]:
        """Get status of all rules"""

        status = []

        for rule in self.rules.values():
            rule_status = {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "enabled": rule.enabled,
                "current_state": rule.current_state,
                "last_trigger_time": rule.last_trigger_time,
                "trigger_count_hour": rule.trigger_count_hour,
                "cooldown_remaining": max(0, rule.cooldown_seconds - (time.time() - (rule.last_trigger_time or 0))),
                "conditions_count": len(rule.conditions),
                "actions_count": len(rule.actions),
            }
            status.append(rule_status)

        return status

    def get_execution_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent execution history"""

        with self._lock:
            recent_executions = self.execution_history[-limit:]

        return [asdict(execution) for execution in recent_executions]


# Demo and testing
async def main():
    """Demo alert chainer functionality"""
    print("🔗 Alert Chainer with Control Theory Demo")
    print("=" * 60)

    # Create alert chainer in test mode
    chainer = AlertChainer(test_mode=True)

    # Configure webhook URLs (dummy for demo)
    chainer.webhook_configs["slack"]["url"] = "https://hooks.slack.com/services/dummy"
    chainer.webhook_configs["discord"]["url"] = "https://discord.com/api/webhooks/dummy"

    print(f"Test mode: {chainer.test_mode}")
    print(f"Control theory evaluator: {type(chainer.control_evaluator).__name__}")

    try:
        # Create sample conditions
        cpu_condition = Condition(
            field="cpu_percent",
            operator=ConditionOperator.GT,
            value=80.0,
            window_seconds=60,
            hysteresis_factor=0.1,
            description="CPU usage above 80%",
        )

        memory_condition = Condition(
            field="memory_percent",
            operator=ConditionOperator.GT,
            value=90.0,
            window_seconds=60,
            hysteresis_factor=0.15,
            description="Memory usage above 90%",
        )

        # Create sample actions
        slack_action = AlertAction(
            action_id="slack_alert",
            action_type=ActionType.WEBHOOK_SLACK,
            parameters={
                "message": "High resource usage detected!",
                "title": "Resource Alert",
                "severity": "high",
            },
            cooldown_seconds=300,
        )

        scale_action = AlertAction(
            action_id="scale_up",
            action_type=ActionType.SCALE_UP,
            parameters={"component": "web_servers", "scale_factor": 1.5},
            depends_on=["slack_alert"],  # Chain after Slack notification
            cooldown_seconds=600,
        )

        circuit_action = AlertAction(
            action_id="circuit_breaker",
            action_type=ActionType.CIRCUIT_BREAKER_OPEN,
            parameters={"circuit_name": "database_pool"},
            depends_on=["scale_up"],  # Chain after scaling
            cooldown_seconds=900,
        )

        # Create alert rule
        resource_rule = AlertRule(
            rule_id="high_resource_usage",
            name="High Resource Usage Alert",
            description="Triggered when CPU > 80% AND Memory > 90%",
            conditions=[cpu_condition, memory_condition],
            logical_operator=LogicalOperator.AND,
            actions=[slack_action, scale_action, circuit_action],
            severity=AlertSeverity.HIGH,
            cooldown_seconds=300,
        )

        # Add to chainer
        chainer.rules[resource_rule.rule_id] = resource_rule
        chainer.actions[slack_action.action_id] = slack_action
        chainer.actions[scale_action.action_id] = scale_action
        chainer.actions[circuit_action.action_id] = circuit_action

        print(f"\n📋 Created alert rule: {resource_rule.name}")
        print(f"   Conditions: {len(resource_rule.conditions)} (AND logic)")
        print(f"   Actions: {len(resource_rule.actions)} (chained)")
        print(f"   Hysteresis: {cpu_condition.hysteresis_factor:.1%} / {memory_condition.hysteresis_factor:.1%}")

        # Test hysteresis evaluation
        print("\n🔬 Testing control theory with hysteresis...")

        # Simulate metric updates that should trigger hysteresis
        test_scenarios = [
            # CPU scenarios (threshold: 80%, hysteresis: 10%)
            ({"cpu_percent": 75.0, "memory_percent": 85.0}, "Below thresholds"),
            ({"cpu_percent": 82.0, "memory_percent": 92.0}, "Above thresholds - should trigger"),
            (
                {"cpu_percent": 78.0, "memory_percent": 92.0},
                "CPU drops but within hysteresis - should stay triggered",
            ),
            (
                {"cpu_percent": 70.0, "memory_percent": 85.0},
                "Both drop below hysteresis - should clear",
            ),
        ]

        for i, (metrics, description) in enumerate(test_scenarios):
            print(f"\n   Scenario {i + 1}: {description}")
            print(f"   Metrics: CPU={metrics['cpu_percent']:.1f}%, Memory={metrics['memory_percent']:.1f}%")

            # Update metrics
            chainer.update_metrics(metrics)

            # Evaluate rule
            rule_triggered = await chainer._evaluate_rule(resource_rule)

            print(f"   Rule triggered: {'YES' if rule_triggered else 'NO'}")

            # Show hysteresis details
            for condition in resource_rule.conditions:
                current_value = metrics[condition.field]
                threshold = condition.value
                hysteresis_delta = threshold * condition.hysteresis_factor

                if condition.operator == ConditionOperator.GT:
                    upper_threshold = threshold
                    lower_threshold = threshold - hysteresis_delta
                    print(
                        f"   {condition.field}: {current_value:.1f} "
                        f"(thresholds: {lower_threshold:.1f} - {upper_threshold:.1f})"
                    )

        # Test action chaining
        print("\n⛓️  Testing action chaining...")

        # Trigger the rule
        chainer.update_metrics({"cpu_percent": 85.0, "memory_percent": 95.0})

        # Simulate rule trigger and action execution
        resource_rule.current_state = False  # Reset state
        rule_triggered = await chainer._evaluate_rule(resource_rule)

        if rule_triggered:
            print("   Rule triggered! Executing chained actions...")

            # Execute actions manually for demo
            await chainer._execute_rule_actions(resource_rule)

            # Show execution results
            executions = chainer.get_execution_history(10)
            print(f"   Executed {len(executions)} actions:")

            for execution in executions:
                status = "✅ SUCCESS" if execution["success"] else "❌ FAILED"
                print(f"     {execution['action_id']}: {status} ({execution['execution_time_ms']:.1f}ms)")

        # Show statistics
        print("\n📊 Chainer Statistics:")
        stats = chainer.get_statistics()

        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Show rule status
        print("\n📋 Rule Status:")
        rule_status = chainer.get_rule_status()

        for status in rule_status:
            print(f"   {status['name']}: {'ENABLED' if status['enabled'] else 'DISABLED'}")
            print(f"     Current state: {'TRIGGERED' if status['current_state'] else 'NORMAL'}")
            print(f"     Conditions: {status['conditions_count']}, Actions: {status['actions_count']}")

            if status["cooldown_remaining"] > 0:
                print(f"     Cooldown: {status['cooldown_remaining']:.0f}s remaining")

        print("\n✅ Alert chainer demo complete!")
        print("\n🎯 KEY FEATURES DEMONSTRATED:")
        print("   • Multi-condition rules with logical operators (AND/OR)")
        print("   • Control theory evaluation with hysteresis (10-15%)")
        print("   • Action chaining with dependencies (Slack → Scale → Circuit)")
        print("   • Configurable webhooks (Slack, Discord, Teams, custom)")
        print("   • Cooldown management between identical actions")
        print("   • Test mode for safe validation")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
