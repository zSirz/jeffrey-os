"""
Threat Assessor - Real-time web search for quantum threat assessment
Jeffrey OS v0.6.2 - ROBUSTESSE ADAPTATIVE
"""

import asyncio
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

try:
    from textblob import TextBlob

    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ThreatLevel(Enum):
    """Enhanced threat levels for v0.6.2"""

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    QUANTUM_IMMINENT = "quantum_imminent"


class ThreatSource(Enum):
    """Sources of threat intelligence"""

    NIST_QUANTUM = "nist_quantum"
    ARXIV_PAPERS = "arxiv_papers"
    CVE_DATABASE = "cve_database"
    SECURITY_ADVISORIES = "security_advisories"
    QUANTUM_NEWS = "quantum_news"
    CRYPTOGRAPHY_FORUMS = "crypto_forums"


@dataclass
class ThreatIntelligence:
    """Real-time threat intelligence data"""

    source: ThreatSource
    timestamp: str
    threat_indicators: list[str]
    quantum_progress_score: float  # 0.0-1.0
    urgency_score: float  # 0.0-1.0
    credibility_score: float  # 0.0-1.0
    content_summary: str
    raw_url: str | None = None


@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment result"""

    timestamp: str
    overall_threat_level: ThreatLevel
    confidence_score: float
    quantum_threat_score: float
    conventional_threat_score: float
    recommended_crypto_level: str

    # Intelligence sources
    intelligence_sources: list[ThreatIntelligence]
    assessment_reasoning: str
    time_to_quantum_years: float | None

    # Adaptive recommendations
    immediate_actions: list[str]
    monitoring_frequency: float  # hours
    next_assessment_time: str


class RealTimeThreatAssessor:
    """
    Real-time threat assessor with web search capabilities
    Monitors quantum computing progress and crypto threats
    """

    def __init__(
        self,
        assessment_interval: float = 24.0,  # hours
        cache_duration: float = 6.0,  # hours
        max_concurrent_searches: int = 5,
    ):
        """
        Initialize threat assessor

        Args:
            assessment_interval: How often to perform full assessment
            cache_duration: How long to cache web search results
            max_concurrent_searches: Max simultaneous web searches
        """
        self.assessment_interval = assessment_interval
        self.cache_duration = cache_duration
        self.max_concurrent_searches = max_concurrent_searches

        # Threat state
        self.current_assessment: ThreatAssessment | None = None
        self.last_full_assessment = 0
        self.assessment_history: list[ThreatAssessment] = []

        # Web search cache
        self.search_cache: dict[str, tuple[float, list[dict]]] = {}
        self.search_semaphore = asyncio.Semaphore(max_concurrent_searches)

        # Quantum progress tracking
        self.quantum_milestones = {
            "logical_qubits": {"threshold": 1000, "weight": 0.4},
            "error_correction": {"threshold": 0.001, "weight": 0.3},
            "coherence_time": {"threshold": 100, "weight": 0.2},  # microseconds
            "quantum_volume": {"threshold": 1000000, "weight": 0.1},
        }

        # Threat keywords for monitoring
        self.threat_keywords = {
            ThreatSource.NIST_QUANTUM: [
                "post-quantum cryptography",
                "NIST standardization",
                "quantum computer breakthrough",
                "cryptographic migration",
            ],
            ThreatSource.ARXIV_PAPERS: [
                "quantum algorithm",
                "Shor's algorithm",
                "cryptography",
                "quantum supremacy",
                "fault-tolerant quantum",
            ],
            ThreatSource.CVE_DATABASE: [
                "cryptographic vulnerability",
                "RSA weakness",
                "ECDSA attack",
                "encryption bypass",
                "key extraction",
            ],
            ThreatSource.SECURITY_ADVISORIES: [
                "cryptographic library",
                "security update",
                "encryption flaw",
                "certificate authority",
                "TLS vulnerability",
            ],
        }

        # Search endpoints (mock for demo - replace with real APIs)
        self.search_endpoints = {
            ThreatSource.NIST_QUANTUM: "https://www.nist.gov/news-events/news/search",
            ThreatSource.ARXIV_PAPERS: "https://export.arxiv.org/api/query",
            ThreatSource.CVE_DATABASE: "https://cve.mitre.org/cgi-bin/cvekey.cgi",
            ThreatSource.SECURITY_ADVISORIES: "https://www.cisa.gov/news-events/cybersecurity-advisories",
        }

        # Threading
        self._lock = threading.Lock()
        self.running = False

        logging.info("Real-time Threat Assessor initialized")

    async def start_continuous_assessment(self):
        """Start continuous threat assessment"""
        if self.running:
            return

        self.running = True
        logging.info("Starting continuous threat assessment")

        try:
            while self.running:
                await self._perform_threat_assessment()
                await asyncio.sleep(self.assessment_interval * 3600)  # Convert to seconds

        except Exception as e:
            logging.error(f"Threat assessment error: {e}")

        finally:
            self.running = False

    async def stop_assessment(self):
        """Stop continuous assessment"""
        self.running = False
        logging.info("Threat assessment stopped")

    async def get_current_threat_level(
        self, event_type: str = "", content: dict[str, Any] = None, force_refresh: bool = False
    ) -> ThreatAssessment:
        """
        Get current threat assessment, with optional forced refresh

        Args:
            event_type: Type of event being assessed
            content: Event content for context
            force_refresh: Force new web search assessment

        Returns:
            Current threat assessment
        """
        current_time = time.time()

        # Check if we need a fresh assessment
        if (
            force_refresh
            or not self.current_assessment
            or current_time - self.last_full_assessment > (self.assessment_interval * 3600)
        ):
            await self._perform_threat_assessment()

        # Adjust assessment based on event context
        if content:
            return await self._contextualize_assessment(self.current_assessment, event_type, content)

        return self.current_assessment

    async def _perform_threat_assessment(self):
        """Perform comprehensive threat assessment with web search"""
        logging.info("Starting comprehensive threat assessment")

        try:
            # Gather intelligence from multiple sources
            intelligence_tasks = [self._search_threat_source(source) for source in ThreatSource]

            intelligence_results = await asyncio.gather(*intelligence_tasks, return_exceptions=True)

            # Filter successful results
            all_intelligence = []
            for result in intelligence_results:
                if isinstance(result, list):
                    all_intelligence.extend(result)
                elif isinstance(result, Exception):
                    logging.warning(f"Intelligence gathering failed: {result}")

            # Analyze intelligence and create assessment
            assessment = await self._analyze_intelligence(all_intelligence)

            with self._lock:
                self.current_assessment = assessment
                self.last_full_assessment = time.time()
                self.assessment_history.append(assessment)

                # Keep only last 100 assessments
                if len(self.assessment_history) > 100:
                    self.assessment_history.pop(0)

            logging.info(f"Threat assessment complete: {assessment.overall_threat_level.value}")

        except Exception as e:
            logging.error(f"Threat assessment failed: {e}")

            # Fallback to conservative assessment
            fallback = ThreatAssessment(
                timestamp=datetime.utcnow().isoformat() + "Z",
                overall_threat_level=ThreatLevel.HIGH,
                confidence_score=0.3,
                quantum_threat_score=0.6,
                conventional_threat_score=0.4,
                recommended_crypto_level="HYBRID",
                intelligence_sources=[],
                assessment_reasoning="Assessment failed - using conservative fallback",
                time_to_quantum_years=5.0,
                immediate_actions=["Switch to post-quantum crypto", "Increase monitoring"],
                monitoring_frequency=1.0,
                next_assessment_time=(datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
            )

            with self._lock:
                self.current_assessment = fallback

    async def _search_threat_source(self, source: ThreatSource) -> list[ThreatIntelligence]:
        """Search specific threat intelligence source"""

        async with self.search_semaphore:
            cache_key = f"{source.value}_{int(time.time() // (self.cache_duration * 3600))}"

            # Check cache first
            if cache_key in self.search_cache:
                cache_time, cached_results = self.search_cache[cache_key]
                if time.time() - cache_time < (self.cache_duration * 3600):
                    return [self._dict_to_intelligence(result) for result in cached_results]

            try:
                # Simulate web search (replace with real implementation)
                search_results = await self._simulate_web_search(source)

                # Process results into threat intelligence
                intelligence_list = []
                for result in search_results:
                    intelligence = await self._process_search_result(source, result)
                    if intelligence:
                        intelligence_list.append(intelligence)

                # Cache results
                self.search_cache[cache_key] = (
                    time.time(),
                    [asdict(intel) for intel in intelligence_list],
                )

                return intelligence_list

            except Exception as e:
                logging.error(f"Search failed for {source.value}: {e}")
                return []

    async def _simulate_web_search(self, source: ThreatSource) -> list[dict[str, Any]]:
        """
        Simulate web search (replace with real search implementation)
        In production, this would use actual search APIs
        """

        # Simulate realistic search results based on source
        mock_results = {
            ThreatSource.NIST_QUANTUM: [
                {
                    "title": "NIST Finalizes Post-Quantum Cryptography Standards",
                    "content": "NIST has finalized three post-quantum cryptographic algorithms designed to withstand attacks from quantum computers. Organizations should begin transitioning to these standards.",
                    "url": "https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards",
                    "date": "2024-08-13",
                    "quantum_indicators": ["post-quantum", "standardization", "transition"],
                },
                {
                    "title": "Quantum Computing Progress Report",
                    "content": "Recent advances in error correction bring practical quantum computers closer. Current systems achieving 1000+ logical qubits expected within 3-5 years.",
                    "url": "https://www.nist.gov/quantum-computing",
                    "date": "2024-12-01",
                    "quantum_indicators": [
                        "logical qubits",
                        "error correction",
                        "practical quantum",
                    ],
                },
            ],
            ThreatSource.ARXIV_PAPERS: [
                {
                    "title": "Improved Quantum Algorithms for Cryptanalysis",
                    "content": "Novel quantum algorithm variants show 40% improvement in attacking current RSA implementations, reducing theoretical break time from centuries to decades.",
                    "url": "https://arxiv.org/abs/2312.XXXX",
                    "date": "2024-12-15",
                    "quantum_indicators": [
                        "quantum algorithm",
                        "RSA",
                        "cryptanalysis",
                        "break time",
                    ],
                },
                {
                    "title": "Fault-Tolerant Quantum Computer Architecture",
                    "content": "New architecture demonstrates coherence times exceeding 100 microseconds with error rates below 0.001%, marking significant progress toward practical quantum computing.",
                    "url": "https://arxiv.org/abs/2401.XXXX",
                    "date": "2024-01-10",
                    "quantum_indicators": [
                        "fault-tolerant",
                        "coherence time",
                        "error rate",
                        "practical quantum",
                    ],
                },
            ],
            ThreatSource.CVE_DATABASE: [
                {
                    "title": "CVE-2024-XXXX: OpenSSL ECDSA Implementation Weakness",
                    "content": "Critical vulnerability in OpenSSL ECDSA signature verification allows key recovery through side-channel analysis. Immediate patching required.",
                    "url": "https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-XXXX",
                    "date": "2024-11-20",
                    "quantum_indicators": [
                        "ECDSA",
                        "key recovery",
                        "side-channel",
                        "vulnerability",
                    ],
                }
            ],
            ThreatSource.SECURITY_ADVISORIES: [
                {
                    "title": "CISA Alert: Prepare for Post-Quantum Transition",
                    "content": "Federal agencies must begin inventory of cryptographic systems and develop migration plans. Quantum threat timeline accelerated.",
                    "url": "https://www.cisa.gov/news-events/alerts/2024/09/15/prepare-post-quantum-cryptography-transition",
                    "date": "2024-09-15",
                    "quantum_indicators": [
                        "post-quantum transition",
                        "migration plans",
                        "quantum threat",
                    ],
                }
            ],
        }

        # Add some randomness and current date awareness
        results = mock_results.get(source, [])

        # Simulate search delay
        await asyncio.sleep(0.5)

        return results[:3]  # Limit results

    async def _process_search_result(self, source: ThreatSource, result: dict[str, Any]) -> ThreatIntelligence | None:
        """Process individual search result into threat intelligence"""

        try:
            content = result.get("content", "")
            quantum_indicators = result.get("quantum_indicators", [])

            # Calculate quantum progress score
            quantum_score = await self._calculate_quantum_progress_score(content, quantum_indicators)

            # Calculate urgency score
            urgency_score = await self._calculate_urgency_score(content, result.get("date", ""))

            # Calculate credibility score
            credibility_score = await self._calculate_credibility_score(source, result)

            # Extract threat indicators
            threat_indicators = await self._extract_threat_indicators(content)

            # Generate content summary
            summary = await self._generate_content_summary(content)

            return ThreatIntelligence(
                source=source,
                timestamp=datetime.utcnow().isoformat() + "Z",
                threat_indicators=threat_indicators,
                quantum_progress_score=quantum_score,
                urgency_score=urgency_score,
                credibility_score=credibility_score,
                content_summary=summary,
                raw_url=result.get("url"),
            )

        except Exception as e:
            logging.error(f"Failed to process search result: {e}")
            return None

    async def _calculate_quantum_progress_score(self, content: str, indicators: list[str]) -> float:
        """Calculate quantum computing progress score"""

        score = 0.0

        # Check for milestone keywords
        content_lower = content.lower()

        # Logical qubits progress
        if any(term in content_lower for term in ["logical qubit", "1000 qubit", "fault-tolerant"]):
            score += 0.3

        # Error correction progress
        if any(term in content_lower for term in ["error correction", "error rate", "0.001%"]):
            score += 0.25

        # Coherence time improvements
        if any(term in content_lower for term in ["coherence time", "100 microsecond", "stability"]):
            score += 0.2

        # Practical quantum computing claims
        if any(term in content_lower for term in ["practical quantum", "commercial quantum", "production ready"]):
            score += 0.25

        # Timeline indicators
        if any(term in content_lower for term in ["3-5 years", "near-term", "within decade"]):
            score += 0.15
        elif any(term in content_lower for term in ["breakthrough", "major advance", "significant progress"]):
            score += 0.1

        return min(score, 1.0)

    async def _calculate_urgency_score(self, content: str, date_str: str) -> float:
        """Calculate urgency score based on content and recency"""

        urgency = 0.0

        # Recency factor
        try:
            if date_str:
                result_date = datetime.strptime(date_str, "%Y-%m-%d")
                days_old = (datetime.now() - result_date).days
                recency_factor = max(0, 1.0 - (days_old / 365))  # Decay over year
                urgency += recency_factor * 0.3
        except:
            pass

        # Urgency keywords
        content_lower = content.lower()

        if any(term in content_lower for term in ["immediate", "urgent", "critical", "emergency"]):
            urgency += 0.4
        elif any(term in content_lower for term in ["soon", "near-term", "imminent"]):
            urgency += 0.3
        elif any(term in content_lower for term in ["transition", "migration", "prepare"]):
            urgency += 0.2

        # Threat level indicators
        if any(term in content_lower for term in ["vulnerability", "exploit", "attack", "breach"]):
            urgency += 0.3

        return min(urgency, 1.0)

    async def _calculate_credibility_score(self, source: ThreatSource, result: dict[str, Any]) -> float:
        """Calculate source credibility score"""

        # Base credibility by source type
        base_credibility = {
            ThreatSource.NIST_QUANTUM: 0.95,
            ThreatSource.ARXIV_PAPERS: 0.85,
            ThreatSource.CVE_DATABASE: 0.90,
            ThreatSource.SECURITY_ADVISORIES: 0.88,
            ThreatSource.QUANTUM_NEWS: 0.65,
            ThreatSource.CRYPTOGRAPHY_FORUMS: 0.55,
        }

        credibility = base_credibility.get(source, 0.5)

        # Adjust based on URL/source indicators
        url = result.get("url", "")
        if "nist.gov" in url or "cisa.gov" in url:
            credibility += 0.05
        elif "arxiv.org" in url:
            credibility += 0.03
        elif "mitre.org" in url:
            credibility += 0.04

        return min(credibility, 1.0)

    async def _extract_threat_indicators(self, content: str) -> list[str]:
        """Extract specific threat indicators from content"""

        indicators = []
        content_lower = content.lower()

        # Quantum computing indicators
        quantum_terms = [
            "quantum computer",
            "quantum algorithm",
            "Shor's algorithm",
            "logical qubits",
            "error correction",
            "quantum supremacy",
            "post-quantum cryptography",
            "quantum threat",
        ]

        for term in quantum_terms:
            if term in content_lower:
                indicators.append(term)

        # Cryptographic vulnerability indicators
        crypto_terms = [
            "RSA vulnerability",
            "ECDSA weakness",
            "key recovery",
            "side-channel attack",
            "cryptographic flaw",
            "encryption bypass",
        ]

        for term in crypto_terms:
            if term in content_lower:
                indicators.append(term)

        # Timeline indicators
        timeline_patterns = [
            r"(\d+)[-\s]*years?",
            r"within (\d+) years",
            r"by (\d{4})",
            r"(\d+)[-\s]*months?",
            r"near[- ]?term",
            r"long[- ]?term",
        ]

        for pattern in timeline_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                indicators.append(f"timeline: {match}")

        return list(set(indicators))  # Remove duplicates

    async def _generate_content_summary(self, content: str) -> str:
        """Generate concise summary of content"""

        # Simple extractive summary (first sentences + key points)
        sentences = content.split(". ")

        # Take first sentence and any sentence with key terms
        key_terms = [
            "quantum",
            "cryptography",
            "vulnerability",
            "breakthrough",
            "algorithm",
            "threat",
            "security",
            "encryption",
        ]

        summary_sentences = [sentences[0]] if sentences else []

        for sentence in sentences[1:]:
            if any(term in sentence.lower() for term in key_terms):
                summary_sentences.append(sentence)
                if len(summary_sentences) >= 3:
                    break

        summary = ". ".join(summary_sentences)

        # Truncate if too long
        if len(summary) > 200:
            summary = summary[:197] + "..."

        return summary

    async def _analyze_intelligence(self, intelligence_sources: list[ThreatIntelligence]) -> ThreatAssessment:
        """Analyze collected intelligence and generate threat assessment"""

        if not intelligence_sources:
            # No intelligence available
            return ThreatAssessment(
                timestamp=datetime.utcnow().isoformat() + "Z",
                overall_threat_level=ThreatLevel.MODERATE,
                confidence_score=0.2,
                quantum_threat_score=0.5,
                conventional_threat_score=0.3,
                recommended_crypto_level="ECDSA",
                intelligence_sources=[],
                assessment_reasoning="No intelligence sources available - using baseline assessment",
                time_to_quantum_years=10.0,
                immediate_actions=["Monitor situation", "Maintain current crypto"],
                monitoring_frequency=12.0,
                next_assessment_time=(datetime.utcnow() + timedelta(hours=12)).isoformat() + "Z",
            )

        # Calculate aggregate scores
        quantum_scores = [intel.quantum_progress_score for intel in intelligence_sources]
        urgency_scores = [intel.urgency_score for intel in intelligence_sources]
        credibility_scores = [intel.credibility_score for intel in intelligence_sources]

        # Weighted averages
        total_credibility = sum(credibility_scores)
        if total_credibility > 0:
            weighted_quantum = sum(q * c for q, c in zip(quantum_scores, credibility_scores)) / total_credibility
            weighted_urgency = sum(u * c for u, c in zip(urgency_scores, credibility_scores)) / total_credibility
        else:
            weighted_quantum = sum(quantum_scores) / len(quantum_scores) if quantum_scores else 0.5
            weighted_urgency = sum(urgency_scores) / len(urgency_scores) if urgency_scores else 0.3

        # Determine threat level
        threat_level = self._determine_threat_level(weighted_quantum, weighted_urgency)

        # Calculate conventional threat score
        conventional_threat = self._calculate_conventional_threats(intelligence_sources)

        # Overall confidence
        confidence = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5

        # Time to quantum estimation
        time_to_quantum = self._estimate_time_to_quantum(weighted_quantum, intelligence_sources)

        # Generate recommendations
        crypto_recommendation = self._recommend_crypto_level(threat_level, weighted_quantum)
        immediate_actions = self._generate_immediate_actions(threat_level, weighted_quantum, weighted_urgency)

        # Monitoring frequency based on threat level
        monitoring_frequency = self._calculate_monitoring_frequency(threat_level, weighted_urgency)

        # Assessment reasoning
        reasoning = self._generate_assessment_reasoning(
            threat_level, weighted_quantum, weighted_urgency, intelligence_sources
        )

        return ThreatAssessment(
            timestamp=datetime.utcnow().isoformat() + "Z",
            overall_threat_level=threat_level,
            confidence_score=confidence,
            quantum_threat_score=weighted_quantum,
            conventional_threat_score=conventional_threat,
            recommended_crypto_level=crypto_recommendation,
            intelligence_sources=intelligence_sources,
            assessment_reasoning=reasoning,
            time_to_quantum_years=time_to_quantum,
            immediate_actions=immediate_actions,
            monitoring_frequency=monitoring_frequency,
            next_assessment_time=(datetime.utcnow() + timedelta(hours=monitoring_frequency)).isoformat() + "Z",
        )

    def _determine_threat_level(self, quantum_score: float, urgency_score: float) -> ThreatLevel:
        """Determine overall threat level"""

        combined_score = (quantum_score * 0.7) + (urgency_score * 0.3)

        if combined_score >= 0.9:
            return ThreatLevel.QUANTUM_IMMINENT
        elif combined_score >= 0.75:
            return ThreatLevel.CRITICAL
        elif combined_score >= 0.6:
            return ThreatLevel.HIGH
        elif combined_score >= 0.4:
            return ThreatLevel.MODERATE
        elif combined_score >= 0.2:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.MINIMAL

    def _calculate_conventional_threats(self, intelligence_sources: list[ThreatIntelligence]) -> float:
        """Calculate conventional (non-quantum) threat score"""

        conventional_score = 0.0

        for intel in intelligence_sources:
            # Look for conventional crypto vulnerabilities
            indicators = [indicator.lower() for indicator in intel.threat_indicators]

            if any(term in " ".join(indicators) for term in ["vulnerability", "weakness", "attack", "exploit"]):
                conventional_score += 0.3 * intel.credibility_score

            if intel.source == ThreatSource.CVE_DATABASE:
                conventional_score += 0.2 * intel.credibility_score

        return min(conventional_score, 1.0)

    def _estimate_time_to_quantum(self, quantum_score: float, intelligence_sources: list[ThreatIntelligence]) -> float:
        """Estimate years until practical quantum threat"""

        # Base estimate: 10 years
        base_years = 10.0

        # Adjust based on quantum progress score
        adjustment_factor = 1.0 - quantum_score
        estimated_years = base_years * adjustment_factor

        # Look for specific timeline mentions in intelligence
        for intel in intelligence_sources:
            for indicator in intel.threat_indicators:
                if "timeline:" in indicator:
                    try:
                        # Extract numeric timeline
                        import re

                        numbers = re.findall(r"\d+", indicator)
                        if numbers:
                            mentioned_years = float(numbers[0])
                            if mentioned_years < estimated_years:
                                estimated_years = mentioned_years * intel.credibility_score + estimated_years * (
                                    1 - intel.credibility_score
                                )
                    except:
                        pass

        return max(0.5, estimated_years)  # Minimum 6 months

    def _recommend_crypto_level(self, threat_level: ThreatLevel, quantum_score: float) -> str:
        """Recommend cryptographic protection level"""

        if threat_level in [ThreatLevel.QUANTUM_IMMINENT, ThreatLevel.CRITICAL]:
            return "HYBRID_POSTQUANTUM"
        elif threat_level == ThreatLevel.HIGH or quantum_score > 0.6:
            return "DILITHIUM"
        elif threat_level == ThreatLevel.MODERATE:
            return "HYBRID_ECDSA"
        else:
            return "ECDSA"

    def _generate_immediate_actions(
        self, threat_level: ThreatLevel, quantum_score: float, urgency_score: float
    ) -> list[str]:
        """Generate immediate action recommendations"""

        actions = []

        if threat_level in [ThreatLevel.QUANTUM_IMMINENT, ThreatLevel.CRITICAL]:
            actions.extend(
                [
                    "Switch to post-quantum cryptography immediately",
                    "Audit all cryptographic implementations",
                    "Prepare for emergency crypto migration",
                    "Increase monitoring to continuous",
                ]
            )
        elif threat_level == ThreatLevel.HIGH:
            actions.extend(
                [
                    "Begin post-quantum crypto testing",
                    "Inventory all cryptographic systems",
                    "Develop migration timeline",
                    "Increase monitoring frequency",
                ]
            )
        elif threat_level == ThreatLevel.MODERATE:
            actions.extend(
                [
                    "Monitor quantum computing developments",
                    "Evaluate post-quantum alternatives",
                    "Update crypto migration plan",
                ]
            )
        else:
            actions.extend(["Continue standard monitoring", "Review crypto policies quarterly"])

        if urgency_score > 0.7:
            actions.append("Escalate to security leadership")

        return actions

    def _calculate_monitoring_frequency(self, threat_level: ThreatLevel, urgency_score: float) -> float:
        """Calculate recommended monitoring frequency in hours"""

        base_frequency = {
            ThreatLevel.QUANTUM_IMMINENT: 0.5,  # 30 minutes
            ThreatLevel.CRITICAL: 1.0,  # 1 hour
            ThreatLevel.HIGH: 4.0,  # 4 hours
            ThreatLevel.MODERATE: 12.0,  # 12 hours
            ThreatLevel.LOW: 24.0,  # 24 hours
            ThreatLevel.MINIMAL: 72.0,  # 72 hours
        }

        frequency = base_frequency.get(threat_level, 24.0)

        # Adjust for urgency
        if urgency_score > 0.8:
            frequency *= 0.5
        elif urgency_score > 0.6:
            frequency *= 0.75

        return max(0.25, frequency)  # Minimum 15 minutes

    def _generate_assessment_reasoning(
        self,
        threat_level: ThreatLevel,
        quantum_score: float,
        urgency_score: float,
        intelligence_sources: list[ThreatIntelligence],
    ) -> str:
        """Generate human-readable assessment reasoning"""

        reasoning_parts = []

        # Threat level explanation
        reasoning_parts.append(f"Threat level: {threat_level.value.upper()}")

        # Quantum progress analysis
        if quantum_score > 0.8:
            reasoning_parts.append("High quantum computing progress detected")
        elif quantum_score > 0.6:
            reasoning_parts.append("Significant quantum computing advances")
        elif quantum_score > 0.4:
            reasoning_parts.append("Moderate quantum computing progress")
        else:
            reasoning_parts.append("Limited quantum computing threats")

        # Source summary
        source_count = len(set(intel.source for intel in intelligence_sources))
        reasoning_parts.append(f"Based on {len(intelligence_sources)} intelligence reports from {source_count} sources")

        # Key findings
        high_credibility_sources = [intel for intel in intelligence_sources if intel.credibility_score > 0.8]
        if high_credibility_sources:
            reasoning_parts.append(f"{len(high_credibility_sources)} high-credibility sources confirmed threats")

        # Urgency factors
        if urgency_score > 0.7:
            reasoning_parts.append("High urgency indicators present")
        elif urgency_score > 0.5:
            reasoning_parts.append("Moderate urgency factors detected")

        return ". ".join(reasoning_parts) + "."

    async def _contextualize_assessment(
        self, assessment: ThreatAssessment, event_type: str, content: dict[str, Any]
    ) -> ThreatAssessment:
        """Adjust assessment based on specific event context"""

        # Create copy for contextualized assessment
        context_assessment = ThreatAssessment(
            timestamp=assessment.timestamp,
            overall_threat_level=assessment.overall_threat_level,
            confidence_score=assessment.confidence_score,
            quantum_threat_score=assessment.quantum_threat_score,
            conventional_threat_score=assessment.conventional_threat_score,
            recommended_crypto_level=assessment.recommended_crypto_level,
            intelligence_sources=assessment.intelligence_sources,
            assessment_reasoning=assessment.assessment_reasoning,
            time_to_quantum_years=assessment.time_to_quantum_years,
            immediate_actions=assessment.immediate_actions.copy(),
            monitoring_frequency=assessment.monitoring_frequency,
            next_assessment_time=assessment.next_assessment_time,
        )

        # Adjust based on event type
        if event_type in ["security_alert", "decision_made"]:
            # Increase threat level for security-critical events
            if context_assessment.overall_threat_level == ThreatLevel.MODERATE:
                context_assessment.overall_threat_level = ThreatLevel.HIGH
                context_assessment.recommended_crypto_level = "HYBRID_ECDSA"

        # Adjust based on content
        if content and isinstance(content, dict):
            threat_indicators = content.get("threat_indicators", [])
            if any("quantum" in str(indicator).lower() for indicator in threat_indicators):
                context_assessment.quantum_threat_score = min(1.0, context_assessment.quantum_threat_score + 0.1)

        return context_assessment

    def _dict_to_intelligence(self, data: dict[str, Any]) -> ThreatIntelligence:
        """Convert dictionary back to ThreatIntelligence object"""
        return ThreatIntelligence(
            source=ThreatSource(data["source"]),
            timestamp=data["timestamp"],
            threat_indicators=data["threat_indicators"],
            quantum_progress_score=data["quantum_progress_score"],
            urgency_score=data["urgency_score"],
            credibility_score=data["credibility_score"],
            content_summary=data["content_summary"],
            raw_url=data.get("raw_url"),
        )

    def get_assessment_history(self, limit: int = 10) -> list[ThreatAssessment]:
        """Get recent assessment history"""
        with self._lock:
            return self.assessment_history[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get threat assessor statistics"""
        with self._lock:
            if not self.assessment_history:
                return {"message": "No assessments performed yet"}

            recent_assessments = self.assessment_history[-10:]

            return {
                "total_assessments": len(self.assessment_history),
                "current_threat_level": self.current_assessment.overall_threat_level.value
                if self.current_assessment
                else None,
                "average_quantum_score": sum(a.quantum_threat_score for a in recent_assessments)
                / len(recent_assessments),
                "average_confidence": sum(a.confidence_score for a in recent_assessments) / len(recent_assessments),
                "cache_size": len(self.search_cache),
                "last_assessment": self.last_full_assessment,
                "next_assessment": time.time() + (self.assessment_interval * 3600),
            }


# Demo and testing
async def main():
    """Demo threat assessor functionality"""
    print("🔍 Real-time Threat Assessor Demo")
    print("=" * 40)

    # Create threat assessor
    assessor = RealTimeThreatAssessor(
        assessment_interval=0.1,
        cache_duration=0.05,  # 6 minutes for demo  # 3 minutes cache
    )

    print("🚀 Starting threat assessment...")

    try:
        # Get initial assessment
        assessment = await assessor.get_current_threat_level(
            event_type="security_test",
            content={"threat_indicators": ["quantum", "vulnerability"]},
            force_refresh=True,
        )

        print("\n📊 THREAT ASSESSMENT RESULTS:")
        print(f"   Threat Level: {assessment.overall_threat_level.value.upper()}")
        print(f"   Confidence: {assessment.confidence_score:.1%}")
        print(f"   Quantum Score: {assessment.quantum_threat_score:.1%}")
        print(f"   Recommended Crypto: {assessment.recommended_crypto_level}")
        print(f"   Time to Quantum: {assessment.time_to_quantum_years:.1f} years")
        print(f"   Intelligence Sources: {len(assessment.intelligence_sources)}")

        print("\n🎯 IMMEDIATE ACTIONS:")
        for action in assessment.immediate_actions:
            print(f"   • {action}")

        print("\n📈 ASSESSMENT REASONING:")
        print(f"   {assessment.assessment_reasoning}")

        # Show intelligence sources
        print("\n🔍 INTELLIGENCE SOURCES:")
        for intel in assessment.intelligence_sources:
            print(f"   • {intel.source.value}: {intel.content_summary[:60]}...")
            print(f"     Quantum Score: {intel.quantum_progress_score:.1%}, Credibility: {intel.credibility_score:.1%}")

        print("\n✅ Threat assessment complete!")

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
