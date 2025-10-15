"""
Security Scanner - Automated Dependency Security Audit
Jeffrey OS v0.6.2 - ROBUSTESSE ADAPTATIVE
"""

import asyncio
import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import packaging.version as pkg_version

    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False


class VulnerabilitySeverity(Enum):
    """CVE severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ScanTrigger(Enum):
    """What triggered the security scan"""

    MANUAL = "manual"
    BUILD_HOOK = "build_hook"
    SCHEDULED = "scheduled"
    CI_CD = "ci_cd"
    DEPENDENCY_CHANGE = "dependency_change"


class RemediationStrategy(Enum):
    """Strategies for fixing vulnerabilities"""

    UPGRADE = "upgrade"
    PATCH = "patch"
    REPLACE = "replace"
    IGNORE = "ignore"
    MONITOR = "monitor"


@dataclass
class Vulnerability:
    """Individual vulnerability record"""

    cve_id: str
    package_name: str
    package_version: str
    severity: VulnerabilitySeverity

    # Vulnerability details
    title: str
    description: str
    published_date: str
    updated_date: str

    # Impact assessment
    cvss_score: float | None
    cvss_vector: str | None
    affected_versions: list[str]
    fixed_versions: list[str]

    # Remediation
    remediation_strategy: RemediationStrategy
    suggested_version: str | None
    workaround: str | None

    # Context
    direct_dependency: bool
    dependency_chain: list[str]
    exploitability: str  # "functional", "poc", "unproven", "high"


@dataclass
class SecurityScanResult:
    """Complete security scan result"""

    scan_id: str
    timestamp: str
    trigger: ScanTrigger

    # Scan metadata
    scan_duration_seconds: float
    total_packages_scanned: int
    packages_with_vulnerabilities: int

    # Vulnerability summary
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    total_vulnerabilities: int

    # Vulnerabilities
    vulnerabilities: list[Vulnerability]

    # Recommendations
    immediate_actions: list[str]
    upgrade_recommendations: list[tuple[str, str, str]]  # (package, current, recommended)
    policy_violations: list[str]

    # Risk assessment
    overall_risk_score: float  # 0-100
    business_impact: str
    remediation_priority: str

    # Compliance
    security_policy_compliant: bool
    regulatory_impact: list[str]


@dataclass
class SecurityPolicy:
    """Security scanning policy configuration"""

    # Severity thresholds
    max_critical_vulnerabilities: int = 0
    max_high_vulnerabilities: int = 2
    max_medium_vulnerabilities: int = 10
    max_low_vulnerabilities: int = 50

    # Age thresholds (days)
    max_vulnerability_age_critical: int = 1
    max_vulnerability_age_high: int = 7
    max_vulnerability_age_medium: int = 30
    max_vulnerability_age_low: int = 90

    # Package policies
    allowed_packages: set[str] = None
    blocked_packages: set[str] = None
    required_package_versions: dict[str, str] = None

    # Scanning configuration
    scan_on_build: bool = True
    scan_schedule_hours: int = 24
    fail_build_on_critical: bool = True
    fail_build_on_high: bool = False

    # Notification settings
    notify_on_new_vulnerabilities: bool = True
    notify_on_policy_violations: bool = True
    notification_channels: list[str] = None

    def __post_init__(self):
        if self.allowed_packages is None:
            self.allowed_packages = set()
        if self.blocked_packages is None:
            self.blocked_packages = set()
        if self.required_package_versions is None:
            self.required_package_versions = {}
        if self.notification_channels is None:
            self.notification_channels = ["email", "slack"]


class PipAuditScanner:
    """
    pip-audit based security scanner with enhanced features
    """

    def __init__(self):
        self.pip_audit_available = False
        self._check_pip_audit_availability()

    def _check_pip_audit_availability(self):
        """Check if pip-audit is available"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip_audit", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                self.pip_audit_available = True
                logging.info("pip-audit is available")
            else:
                logging.warning("pip-audit not available - installing...")
                self._install_pip_audit()
        except Exception as e:
            logging.warning(f"pip-audit check failed: {e}")
            self._install_pip_audit()

    def _install_pip_audit(self):
        """Install pip-audit if not available"""
        try:
            logging.info("Installing pip-audit...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "pip-audit"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                self.pip_audit_available = True
                logging.info("pip-audit installed successfully")
            else:
                logging.error(f"Failed to install pip-audit: {result.stderr}")
        except Exception as e:
            logging.error(f"pip-audit installation failed: {e}")

    async def scan_dependencies(self, requirements_file: str | None = None) -> dict[str, Any]:
        """
        Scan dependencies using pip-audit

        Args:
            requirements_file: Path to requirements.txt file

        Returns:
            Raw pip-audit results
        """

        if not self.pip_audit_available:
            raise RuntimeError("pip-audit not available")

        try:
            # Build pip-audit command
            cmd = [sys.executable, "-m", "pip_audit", "--format", "json", "--desc"]

            if requirements_file and Path(requirements_file).exists():
                cmd.extend(["--requirement", requirements_file])

            # Run pip-audit
            logging.info("Running pip-audit scan...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Parse JSON output
                if result.stdout:
                    return json.loads(result.stdout)
                else:
                    return {"vulnerabilities": []}
            else:
                # pip-audit returns non-zero if vulnerabilities found
                if result.stdout:
                    return json.loads(result.stdout)
                else:
                    raise RuntimeError(f"pip-audit failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("pip-audit scan timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse pip-audit output: {e}")
        except Exception as e:
            raise RuntimeError(f"pip-audit scan failed: {e}")

    async def get_package_info(self, package_name: str) -> dict[str, Any]:
        """Get package information from PyPI"""

        if not REQUESTS_AVAILABLE:
            return {}

        try:
            url = f"https://pypi.org/pypi/{package_name}/json"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                return {}

        except Exception as e:
            logging.warning(f"Failed to get package info for {package_name}: {e}")
            return {}


class SecurityScanner:
    """
    Comprehensive security scanner with pip-audit integration
    Provides automated dependency scanning with alerts and remediation
    """

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        alert_chainer=None,
        requirements_file: str = "requirements.txt",
    ):
        """
        Initialize security scanner

        Args:
            policy: Security scanning policy
            alert_chainer: Alert chainer for notifications
            requirements_file: Path to requirements.txt
        """
        self.policy = policy or SecurityPolicy()
        self.alert_chainer = alert_chainer
        self.requirements_file = requirements_file

        # Scanner components
        self.pip_scanner = PipAuditScanner()

        # State tracking
        self.scan_history: list[SecurityScanResult] = []
        self.last_scan_time: float | None = None
        self.current_vulnerabilities: dict[str, Vulnerability] = {}

        # Known vulnerability database (cached)
        self.vulnerability_cache: dict[str, dict[str, Any]] = {}
        self.cache_expiry = 3600  # 1 hour

        # Statistics
        self.scan_stats = {
            "total_scans": 0,
            "vulnerabilities_found": 0,
            "vulnerabilities_fixed": 0,
            "policy_violations": 0,
            "avg_scan_time": 0.0,
        }

        # Thread safety
        self._lock = threading.Lock()
        self.running = False

        logging.info("Security Scanner initialized")

    async def start_scheduled_scanning(self):
        """Start scheduled security scanning"""

        if self.running:
            return

        self.running = True
        logging.info(f"Starting scheduled security scanning (every {self.policy.scan_schedule_hours}h)")

        try:
            while self.running:
                await self._perform_scheduled_scan()
                await asyncio.sleep(self.policy.scan_schedule_hours * 3600)
        except Exception as e:
            logging.error(f"Scheduled scanning error: {e}")
        finally:
            self.running = False

    async def stop_scheduled_scanning(self):
        """Stop scheduled scanning"""
        self.running = False
        logging.info("Scheduled security scanning stopped")

    async def scan_now(self, trigger: ScanTrigger = ScanTrigger.MANUAL) -> SecurityScanResult:
        """
        Perform immediate security scan

        Args:
            trigger: What triggered this scan

        Returns:
            Security scan result
        """

        scan_id = f"scan_{int(time.time() * 1000000)}"
        start_time = time.time()

        logging.info(f"Starting security scan: {scan_id} (trigger: {trigger.value})")

        try:
            # Run pip-audit scan
            raw_results = await self.pip_scanner.scan_dependencies(self.requirements_file)

            # Process results
            vulnerabilities = await self._process_vulnerabilities(raw_results)

            # Calculate summary statistics
            severity_counts = self._calculate_severity_counts(vulnerabilities)

            # Assess risk and generate recommendations
            risk_assessment = await self._assess_risk(vulnerabilities)
            recommendations = await self._generate_recommendations(vulnerabilities)

            # Check policy compliance
            policy_compliant, policy_violations = self._check_policy_compliance(vulnerabilities, severity_counts)

            # Create scan result
            scan_duration = time.time() - start_time

            scan_result = SecurityScanResult(
                scan_id=scan_id,
                timestamp=datetime.utcnow().isoformat() + "Z",
                trigger=trigger,
                scan_duration_seconds=scan_duration,
                total_packages_scanned=len(set(v.package_name for v in vulnerabilities)) if vulnerabilities else 0,
                packages_with_vulnerabilities=len(set(v.package_name for v in vulnerabilities))
                if vulnerabilities
                else 0,
                critical_count=severity_counts.get(VulnerabilitySeverity.CRITICAL, 0),
                high_count=severity_counts.get(VulnerabilitySeverity.HIGH, 0),
                medium_count=severity_counts.get(VulnerabilitySeverity.MEDIUM, 0),
                low_count=severity_counts.get(VulnerabilitySeverity.LOW, 0),
                total_vulnerabilities=len(vulnerabilities),
                vulnerabilities=vulnerabilities,
                immediate_actions=recommendations["immediate_actions"],
                upgrade_recommendations=recommendations["upgrade_recommendations"],
                policy_violations=policy_violations,
                overall_risk_score=risk_assessment["risk_score"],
                business_impact=risk_assessment["business_impact"],
                remediation_priority=risk_assessment["remediation_priority"],
                security_policy_compliant=policy_compliant,
                regulatory_impact=risk_assessment.get("regulatory_impact", []),
            )

            # Store scan result
            with self._lock:
                self.scan_history.append(scan_result)
                if len(self.scan_history) > 100:  # Keep last 100 scans
                    self.scan_history = self.scan_history[-100:]

                self.last_scan_time = time.time()

                # Update statistics
                self.scan_stats["total_scans"] += 1
                self.scan_stats["vulnerabilities_found"] += len(vulnerabilities)

                current_avg = self.scan_stats["avg_scan_time"]
                total_scans = self.scan_stats["total_scans"]
                self.scan_stats["avg_scan_time"] = (current_avg * (total_scans - 1) + scan_duration) / total_scans

            # Send alerts if needed
            await self._send_security_alerts(scan_result)

            logging.info(
                f"Security scan completed: {len(vulnerabilities)} vulnerabilities found in {scan_duration:.1f}s"
            )

            return scan_result

        except Exception as e:
            logging.error(f"Security scan failed: {e}")
            # Create failure result
            return SecurityScanResult(
                scan_id=scan_id,
                timestamp=datetime.utcnow().isoformat() + "Z",
                trigger=trigger,
                scan_duration_seconds=time.time() - start_time,
                total_packages_scanned=0,
                packages_with_vulnerabilities=0,
                critical_count=0,
                high_count=0,
                medium_count=0,
                low_count=0,
                total_vulnerabilities=0,
                vulnerabilities=[],
                immediate_actions=[f"Fix scan error: {str(e)}"],
                upgrade_recommendations=[],
                policy_violations=[f"Scan failed: {str(e)}"],
                overall_risk_score=100.0,  # Maximum risk on scan failure
                business_impact="unknown",
                remediation_priority="high",
                security_policy_compliant=False,
                regulatory_impact=["scan_failure"],
            )

    async def _perform_scheduled_scan(self):
        """Perform scheduled security scan"""
        try:
            await self.scan_now(ScanTrigger.SCHEDULED)
        except Exception as e:
            logging.error(f"Scheduled scan failed: {e}")

    async def _process_vulnerabilities(self, raw_results: dict[str, Any]) -> list[Vulnerability]:
        """Process raw pip-audit results into structured vulnerabilities"""

        vulnerabilities = []

        if not raw_results or "vulnerabilities" not in raw_results:
            return vulnerabilities

        for vuln_data in raw_results["vulnerabilities"]:
            try:
                # Extract basic info
                package_name = vuln_data.get("package", "unknown")
                package_version = vuln_data.get("installed_version", "unknown")

                # CVE information
                cve_id = vuln_data.get("id", "UNKNOWN")
                title = vuln_data.get("summary", "No summary available")
                description = vuln_data.get("description", "No description available")

                # Severity mapping
                severity_str = vuln_data.get("severity", "unknown").lower()
                severity = self._map_severity(severity_str)

                # Version information
                affected_versions = vuln_data.get("affected_versions", [])
                fixed_versions = vuln_data.get("fixed_versions", [])

                # Determine remediation strategy
                remediation_strategy, suggested_version = await self._determine_remediation(
                    package_name, package_version, fixed_versions
                )

                # Assess dependency chain
                direct_dependency = await self._is_direct_dependency(package_name)
                dependency_chain = await self._get_dependency_chain(package_name)

                vulnerability = Vulnerability(
                    cve_id=cve_id,
                    package_name=package_name,
                    package_version=package_version,
                    severity=severity,
                    title=title,
                    description=description,
                    published_date=vuln_data.get("published", "unknown"),
                    updated_date=vuln_data.get("updated", "unknown"),
                    cvss_score=vuln_data.get("cvss_score"),
                    cvss_vector=vuln_data.get("cvss_vector"),
                    affected_versions=affected_versions if isinstance(affected_versions, list) else [affected_versions],
                    fixed_versions=fixed_versions if isinstance(fixed_versions, list) else [fixed_versions],
                    remediation_strategy=remediation_strategy,
                    suggested_version=suggested_version,
                    workaround=vuln_data.get("workaround"),
                    direct_dependency=direct_dependency,
                    dependency_chain=dependency_chain,
                    exploitability=vuln_data.get("exploitability", "unknown"),
                )

                vulnerabilities.append(vulnerability)

            except Exception as e:
                logging.warning(f"Failed to process vulnerability: {e}")
                continue

        return vulnerabilities

    def _map_severity(self, severity_str: str) -> VulnerabilitySeverity:
        """Map severity string to enum"""

        severity_map = {
            "critical": VulnerabilitySeverity.CRITICAL,
            "high": VulnerabilitySeverity.HIGH,
            "medium": VulnerabilitySeverity.MEDIUM,
            "moderate": VulnerabilitySeverity.MEDIUM,
            "low": VulnerabilitySeverity.LOW,
            "unknown": VulnerabilitySeverity.UNKNOWN,
        }

        return severity_map.get(severity_str.lower(), VulnerabilitySeverity.UNKNOWN)

    async def _determine_remediation(
        self, package_name: str, current_version: str, fixed_versions: list[str]
    ) -> tuple[RemediationStrategy, str | None]:
        """Determine remediation strategy and suggested version"""

        if not fixed_versions:
            return RemediationStrategy.MONITOR, None

        # Find the lowest fixed version that's higher than current
        if PACKAGING_AVAILABLE:
            try:
                current_ver = pkg_version.Version(current_version)

                # Filter and sort fixed versions
                valid_fixed = []
                for fixed_ver_str in fixed_versions:
                    try:
                        fixed_ver = pkg_version.Version(fixed_ver_str)
                        if fixed_ver > current_ver:
                            valid_fixed.append(fixed_ver)
                    except:
                        continue

                if valid_fixed:
                    suggested_version = str(min(valid_fixed))
                    return RemediationStrategy.UPGRADE, suggested_version

            except Exception:
                pass

        # Fallback: suggest first fixed version
        if fixed_versions:
            return RemediationStrategy.UPGRADE, fixed_versions[0]

        return RemediationStrategy.MONITOR, None

    async def _is_direct_dependency(self, package_name: str) -> bool:
        """Check if package is a direct dependency"""

        if not Path(self.requirements_file).exists():
            return False

        try:
            with open(self.requirements_file) as f:
                requirements = f.read().lower()
                return package_name.lower() in requirements
        except Exception:
            return False

    async def _get_dependency_chain(self, package_name: str) -> list[str]:
        """Get dependency chain for package"""

        # Simplified dependency chain (would use pip show in production)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package_name],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Parse dependencies from pip show output
                lines = result.stdout.split("\n")
                for line in lines:
                    if line.startswith("Required-by:"):
                        required_by = line.split(":", 1)[1].strip()
                        if required_by:
                            return [pkg.strip() for pkg in required_by.split(",")]
                        else:
                            return []  # Direct dependency

        except Exception:
            pass

        return []  # Unknown chain

    def _calculate_severity_counts(self, vulnerabilities: list[Vulnerability]) -> dict[VulnerabilitySeverity, int]:
        """Calculate vulnerability counts by severity"""

        counts = {}
        for vuln in vulnerabilities:
            counts[vuln.severity] = counts.get(vuln.severity, 0) + 1

        return counts

    async def _assess_risk(self, vulnerabilities: list[Vulnerability]) -> dict[str, Any]:
        """Assess overall security risk"""

        if not vulnerabilities:
            return {
                "risk_score": 0.0,
                "business_impact": "low",
                "remediation_priority": "low",
                "regulatory_impact": [],
            }

        # Calculate weighted risk score
        severity_weights = {
            VulnerabilitySeverity.CRITICAL: 40,
            VulnerabilitySeverity.HIGH: 20,
            VulnerabilitySeverity.MEDIUM: 5,
            VulnerabilitySeverity.LOW: 1,
            VulnerabilitySeverity.UNKNOWN: 2,
        }

        total_weight = 0
        for vuln in vulnerabilities:
            weight = severity_weights.get(vuln.severity, 1)

            # Increase weight for direct dependencies
            if vuln.direct_dependency:
                weight *= 1.5

            # Increase weight based on exploitability
            if vuln.exploitability in ["functional", "high"]:
                weight *= 1.3
            elif vuln.exploitability == "poc":
                weight *= 1.1

            total_weight += weight

        # Normalize risk score (0-100)
        risk_score = min(100.0, total_weight)

        # Determine business impact
        if risk_score >= 80:
            business_impact = "critical"
            remediation_priority = "immediate"
        elif risk_score >= 60:
            business_impact = "high"
            remediation_priority = "urgent"
        elif risk_score >= 30:
            business_impact = "medium"
            remediation_priority = "planned"
        else:
            business_impact = "low"
            remediation_priority = "routine"

        # Assess regulatory impact
        regulatory_impact = []
        critical_count = len([v for v in vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL])
        if critical_count > 0:
            regulatory_impact.extend(["gdpr_breach_risk", "data_protection_violation"])

        high_count = len([v for v in vulnerabilities if v.severity == VulnerabilitySeverity.HIGH])
        if high_count > 5:
            regulatory_impact.append("security_framework_non_compliance")

        return {
            "risk_score": risk_score,
            "business_impact": business_impact,
            "remediation_priority": remediation_priority,
            "regulatory_impact": regulatory_impact,
        }

    async def _generate_recommendations(self, vulnerabilities: list[Vulnerability]) -> dict[str, Any]:
        """Generate remediation recommendations"""

        immediate_actions = []
        upgrade_recommendations = []

        # Group vulnerabilities by package
        package_vulns = {}
        for vuln in vulnerabilities:
            if vuln.package_name not in package_vulns:
                package_vulns[vuln.package_name] = []
            package_vulns[vuln.package_name].append(vuln)

        # Generate recommendations per package
        for package_name, vulns in package_vulns.items():
            # Find highest severity
            max_severity = max((v.severity for v in vulns), default=VulnerabilitySeverity.LOW)

            # Critical/High vulnerabilities need immediate action
            if max_severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]:
                immediate_actions.append(
                    f"URGENT: Update {package_name} - {len(vulns)} {max_severity.value} vulnerabilities"
                )

            # Find best upgrade recommendation
            suggested_versions = [v.suggested_version for v in vulns if v.suggested_version]
            if suggested_versions and PACKAGING_AVAILABLE:
                try:
                    # Choose highest suggested version
                    versions = [pkg_version.Version(v) for v in suggested_versions]
                    best_version = str(max(versions))
                    current_version = vulns[0].package_version

                    upgrade_recommendations.append((package_name, current_version, best_version))
                except:
                    # Fallback to first suggestion
                    upgrade_recommendations.append((package_name, vulns[0].package_version, suggested_versions[0]))

        # Add general recommendations
        if len(vulnerabilities) > 10:
            immediate_actions.append("Consider security review of dependency management practices")

        if any(v.direct_dependency for v in vulnerabilities):
            immediate_actions.append("Prioritize updates for direct dependencies")

        return {
            "immediate_actions": immediate_actions,
            "upgrade_recommendations": upgrade_recommendations,
        }

    def _check_policy_compliance(
        self,
        vulnerabilities: list[Vulnerability],
        severity_counts: dict[VulnerabilitySeverity, int],
    ) -> tuple[bool, list[str]]:
        """Check compliance with security policy"""

        violations = []

        # Check severity thresholds
        critical_count = severity_counts.get(VulnerabilitySeverity.CRITICAL, 0)
        if critical_count > self.policy.max_critical_vulnerabilities:
            violations.append(
                f"Critical vulnerabilities exceed limit: {critical_count} > {self.policy.max_critical_vulnerabilities}"
            )

        high_count = severity_counts.get(VulnerabilitySeverity.HIGH, 0)
        if high_count > self.policy.max_high_vulnerabilities:
            violations.append(
                f"High vulnerabilities exceed limit: {high_count} > {self.policy.max_high_vulnerabilities}"
            )

        medium_count = severity_counts.get(VulnerabilitySeverity.MEDIUM, 0)
        if medium_count > self.policy.max_medium_vulnerabilities:
            violations.append(
                f"Medium vulnerabilities exceed limit: {medium_count} > {self.policy.max_medium_vulnerabilities}"
            )

        # Check blocked packages
        for vuln in vulnerabilities:
            if vuln.package_name in self.policy.blocked_packages:
                violations.append(f"Blocked package found: {vuln.package_name}")

        # Check required versions
        for vuln in vulnerabilities:
            required_version = self.policy.required_package_versions.get(vuln.package_name)
            if required_version and vuln.package_version != required_version:
                violations.append(
                    f"Package {vuln.package_name} version {vuln.package_version} != required {required_version}"
                )

        return len(violations) == 0, violations

    async def _send_security_alerts(self, scan_result: SecurityScanResult):
        """Send security alerts via alert chainer"""

        if not self.alert_chainer:
            return

        # Determine if alerts should be sent
        should_alert = False
        alert_reasons = []

        if scan_result.critical_count > 0:
            should_alert = True
            alert_reasons.append(f"{scan_result.critical_count} critical vulnerabilities")

        if scan_result.high_count > self.policy.max_high_vulnerabilities:
            should_alert = True
            alert_reasons.append(
                f"{scan_result.high_count} high vulnerabilities (limit: {self.policy.max_high_vulnerabilities})"
            )

        if not scan_result.security_policy_compliant:
            should_alert = True
            alert_reasons.append("Security policy violations")

        if should_alert:
            # Update alert chainer metrics
            security_metrics = {
                "security_vulnerabilities_total": scan_result.total_vulnerabilities,
                "security_vulnerabilities_critical": scan_result.critical_count,
                "security_vulnerabilities_high": scan_result.high_count,
                "security_risk_score": scan_result.overall_risk_score,
                "security_policy_compliant": 1.0 if scan_result.security_policy_compliant else 0.0,
            }

            self.alert_chainer.update_metrics(security_metrics)

            logging.info(f"Security alerts triggered: {', '.join(alert_reasons)}")

    async def scan_on_build(self) -> bool:
        """
        Scan dependencies during build process

        Returns:
            True if build should continue, False if build should fail
        """

        if not self.policy.scan_on_build:
            return True

        try:
            scan_result = await self.scan_now(ScanTrigger.BUILD_HOOK)

            # Check if build should fail
            if self.policy.fail_build_on_critical and scan_result.critical_count > 0:
                logging.error(f"BUILD FAILED: {scan_result.critical_count} critical vulnerabilities found")
                return False

            if self.policy.fail_build_on_high and scan_result.high_count > 0:
                logging.error(f"BUILD FAILED: {scan_result.high_count} high vulnerabilities found")
                return False

            return True

        except Exception as e:
            logging.error(f"Build security scan failed: {e}")
            return not self.policy.fail_build_on_critical  # Fail safe

    def get_scan_statistics(self) -> dict[str, Any]:
        """Get security scan statistics"""

        with self._lock:
            stats = self.scan_stats.copy()

        # Add recent scan info
        if self.scan_history:
            latest_scan = self.scan_history[-1]
            stats.update(
                {
                    "last_scan_timestamp": latest_scan.timestamp,
                    "last_scan_vulnerabilities": latest_scan.total_vulnerabilities,
                    "last_scan_risk_score": latest_scan.overall_risk_score,
                    "last_scan_compliant": latest_scan.security_policy_compliant,
                }
            )

        return stats

    def get_current_vulnerabilities(self) -> list[Vulnerability]:
        """Get current active vulnerabilities"""

        if self.scan_history:
            return self.scan_history[-1].vulnerabilities
        else:
            return []

    def generate_security_report(self) -> dict[str, Any]:
        """Generate security report for delta analyzer integration"""

        if not self.scan_history:
            return {
                "security_scanning_enabled": True,
                "last_scan": None,
                "current_risk_level": "unknown",
                "vulnerability_summary": {},
                "compliance_status": "unknown",
            }

        latest_scan = self.scan_history[-1]

        return {
            "security_scanning_enabled": True,
            "last_scan": {
                "timestamp": latest_scan.timestamp,
                "scan_id": latest_scan.scan_id,
                "total_vulnerabilities": latest_scan.total_vulnerabilities,
                "critical": latest_scan.critical_count,
                "high": latest_scan.high_count,
                "medium": latest_scan.medium_count,
                "low": latest_scan.low_count,
            },
            "current_risk_level": latest_scan.business_impact,
            "overall_risk_score": latest_scan.overall_risk_score,
            "vulnerability_summary": {
                "packages_affected": latest_scan.packages_with_vulnerabilities,
                "direct_dependencies_affected": len([v for v in latest_scan.vulnerabilities if v.direct_dependency]),
                "upgrade_recommendations": len(latest_scan.upgrade_recommendations),
                "immediate_actions_required": len(latest_scan.immediate_actions),
            },
            "compliance_status": {
                "policy_compliant": latest_scan.security_policy_compliant,
                "policy_violations": len(latest_scan.policy_violations),
                "regulatory_impact": latest_scan.regulatory_impact,
            },
            "scan_statistics": self.get_scan_statistics(),
        }


# Demo and testing
async def main():
    """Demo security scanner functionality"""
    print("🔒 Security Scanner with pip-audit Demo")
    print("=" * 60)

    # Create security policy
    policy = SecurityPolicy(
        max_critical_vulnerabilities=0,
        max_high_vulnerabilities=2,
        scan_on_build=True,
        fail_build_on_critical=True,
        notification_channels=["slack", "email"],
    )

    # Create security scanner
    scanner = SecurityScanner(
        policy=policy,
        requirements_file="requirements.txt",  # Use existing requirements.txt
    )

    print(f"pip-audit available: {scanner.pip_scanner.pip_audit_available}")
    print(f"Policy: Critical={policy.max_critical_vulnerabilities}, High={policy.max_high_vulnerabilities}")

    try:
        # Test build scan
        print("\n🔨 Testing build-time security scan...")

        build_should_continue = await scanner.scan_on_build()
        print(f"   Build result: {'PASS' if build_should_continue else 'FAIL'}")

        # Perform manual scan
        print("\n🔍 Performing manual security scan...")

        scan_result = await scanner.scan_now(ScanTrigger.MANUAL)

        print(f"   Scan ID: {scan_result.scan_id}")
        print(f"   Duration: {scan_result.scan_duration_seconds:.1f}s")
        print(f"   Packages scanned: {scan_result.total_packages_scanned}")
        print(f"   Total vulnerabilities: {scan_result.total_vulnerabilities}")

        # Show vulnerability breakdown
        if scan_result.total_vulnerabilities > 0:
            print("\n🚨 Vulnerability Breakdown:")
            print(f"   Critical: {scan_result.critical_count}")
            print(f"   High: {scan_result.high_count}")
            print(f"   Medium: {scan_result.medium_count}")
            print(f"   Low: {scan_result.low_count}")

            print("\n🔧 Sample Vulnerabilities:")
            for i, vuln in enumerate(scan_result.vulnerabilities[:3]):  # Show first 3
                print(f"   {i + 1}. {vuln.package_name} v{vuln.package_version}")
                print(f"      CVE: {vuln.cve_id}")
                print(f"      Severity: {vuln.severity.value.upper()}")
                print(f"      Direct dependency: {'Yes' if vuln.direct_dependency else 'No'}")
                if vuln.suggested_version:
                    print(f"      Suggested version: {vuln.suggested_version}")
                print(f"      Title: {vuln.title[:60]}...")

            # Show upgrade recommendations
            if scan_result.upgrade_recommendations:
                print("\n📦 Upgrade Recommendations:")
                for package, current, recommended in scan_result.upgrade_recommendations[:5]:
                    print(f"   {package}: {current} → {recommended}")

            # Show immediate actions
            if scan_result.immediate_actions:
                print("\n⚡ Immediate Actions Required:")
                for action in scan_result.immediate_actions:
                    print(f"   • {action}")
        else:
            print("\n✅ No vulnerabilities found!")

        # Show risk assessment
        print("\n📊 Risk Assessment:")
        print(f"   Overall risk score: {scan_result.overall_risk_score:.1f}/100")
        print(f"   Business impact: {scan_result.business_impact.upper()}")
        print(f"   Remediation priority: {scan_result.remediation_priority.upper()}")
        print(f"   Policy compliant: {'Yes' if scan_result.security_policy_compliant else 'No'}")

        # Show policy violations
        if scan_result.policy_violations:
            print("\n⚠️ Policy Violations:")
            for violation in scan_result.policy_violations:
                print(f"   • {violation}")

        # Show statistics
        print("\n📈 Scanner Statistics:")
        stats = scanner.get_scan_statistics()

        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Generate security report for delta analyzer
        print("\n📋 Security Report for Delta Analyzer:")
        security_report = scanner.generate_security_report()

        print(f"   Risk level: {security_report['current_risk_level']}")
        print(f"   Packages affected: {security_report['vulnerability_summary']['packages_affected']}")
        print(f"   Compliance status: {security_report['compliance_status']['policy_compliant']}")

        if security_report["last_scan"]:
            last_scan = security_report["last_scan"]
            print(f"   Last scan: {last_scan['total_vulnerabilities']} vulnerabilities")
            print(
                f"   Breakdown: C={last_scan['critical']}, H={last_scan['high']}, M={last_scan['medium']}, L={last_scan['low']}"
            )

        print("\n✅ Security scanner demo complete!")
        print("\n🎯 KEY FEATURES DEMONSTRATED:")
        print("   • Automated pip-audit integration with CVE detection")
        print("   • Build-time security scanning with failure policies")
        print("   • Vulnerability severity classification and risk scoring")
        print("   • Upgrade recommendations with version suggestions")
        print("   • Policy compliance checking with configurable thresholds")
        print("   • Integration ready for alert_chainer notifications")
        print("   • Security reporting for delta_analyzer ROI calculation")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
