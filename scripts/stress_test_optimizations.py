#!/usr/bin/env python3
"""
Comprehensive Stress Test for Jeffrey OS Optimizations
Validates all performance improvements implemented in the optimization phase
"""
import asyncio
import aiohttp
import time
import json
import sys
import os
from typing import Dict, List, Any
from statistics import mean, median
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationStressTest:
    """
    Comprehensive stress test to validate all optimizations:
    1. Idempotence fix validation
    2. Backpressure and queue management
    3. Cognitive concurrency improvements
    4. ML adapter cache and semaphore
    5. Auto-debug monitoring functionality
    """

    def __init__(self, base_url: str = "http://localhost:8000", max_concurrent: int = 50):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.results: Dict[str, Any] = {}

        # Test data for variety (to test cache effectiveness)
        self.test_texts = [
            "I feel absolutely amazing today!",
            "This makes me so sad and disappointed",
            "I'm really angry about this situation",
            "I feel calm and peaceful right now",
            "This is extremely frustrating to deal with",
            "I'm surprised by this unexpected news",
            "I feel neutral about this topic",
            "This brings me so much joy and happiness",
            "I'm feeling quite anxious about tomorrow",
            "I'm excited about the upcoming project"
        ]

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all stress tests and return comprehensive results"""
        logger.info("🚀 Starting Comprehensive Optimization Stress Test")

        # Test 1: Baseline performance
        await self._test_baseline_performance()

        # Test 2: Idempotence validation (duplicate detection)
        await self._test_idempotence_fix()

        # Test 3: Backpressure and queue saturation
        await self._test_backpressure_handling()

        # Test 4: High concurrency performance
        await self._test_high_concurrency()

        # Test 5: ML adapter cache effectiveness
        await self._test_ml_cache_effectiveness()

        # Test 6: Auto-debug monitoring
        await self._test_auto_debug_monitoring()

        # Test 7: System stability under load
        await self._test_system_stability()

        # Final health check
        await self._final_health_assessment()

        logger.info("✅ Comprehensive Stress Test Completed")
        return self.results

    async def _test_baseline_performance(self):
        """Test baseline performance metrics"""
        logger.info("📊 Testing baseline performance...")

        start_time = time.perf_counter()
        latencies = []
        success_count = 0
        error_count = 0

        async with aiohttp.ClientSession() as session:
            # Send 100 requests sequentially for baseline
            for i in range(100):
                request_start = time.perf_counter()
                try:
                    text = random.choice(self.test_texts)
                    async with session.post(
                        f"{self.base_url}/api/v1/emotion/detect",
                        json={"text": text},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            success_count += 1
                        else:
                            error_count += 1
                        latencies.append((time.perf_counter() - request_start) * 1000)

                except Exception as e:
                    error_count += 1
                    logger.debug(f"Request {i} failed: {e}")

        total_time = time.perf_counter() - start_time

        self.results["baseline_performance"] = {
            "total_requests": 100,
            "success_count": success_count,
            "error_count": error_count,
            "error_rate_pct": (error_count / 100) * 100,
            "total_time_sec": total_time,
            "requests_per_sec": 100 / total_time,
            "avg_latency_ms": mean(latencies) if latencies else 0,
            "median_latency_ms": median(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        }

        logger.info(f"✅ Baseline: {success_count}/100 success, {mean(latencies):.1f}ms avg latency")

    async def _test_idempotence_fix(self):
        """Test that idempotence fix prevents incorrect duplicate detection"""
        logger.info("🔄 Testing idempotence fix...")

        # Send same text multiple times rapidly (should NOT be deduplicated)
        same_text = "Testing idempotence fix with unique timestamp"

        start_time = time.perf_counter()
        success_count = 0

        async with aiohttp.ClientSession() as session:
            tasks = []
            # Send 30 identical requests concurrently (old bug would dedupe most)
            for i in range(30):
                task = self._send_emotion_request(session, same_text)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if not isinstance(result, Exception) and result.get("status") == 200:
                    success_count += 1

        total_time = time.perf_counter() - start_time

        # Before fix: would expect ~1-3 successes due to false deduplication
        # After fix: should get close to 30 successes
        success_rate = (success_count / 30) * 100

        self.results["idempotence_fix"] = {
            "total_requests": 30,
            "success_count": success_count,
            "success_rate_pct": success_rate,
            "total_time_sec": total_time,
            "idempotence_working": success_rate > 80  # Should be >80% with fix
        }

        if success_rate > 80:
            logger.info(f"✅ Idempotence fix working: {success_count}/30 processed ({success_rate:.1f}%)")
        else:
            logger.warning(f"⚠️ Idempotence issue detected: only {success_count}/30 processed")

    async def _test_backpressure_handling(self):
        """Test backpressure handling under queue saturation"""
        logger.info("🚫 Testing backpressure handling...")

        # Send a burst of requests to trigger backpressure
        start_time = time.perf_counter()
        response_codes = []
        backpressure_count = 0

        async with aiohttp.ClientSession() as session:
            tasks = []
            # Send 100 requests rapidly to saturate queue
            for i in range(100):
                text = f"Backpressure test {i} {random.choice(self.test_texts)}"
                task = self._send_emotion_request_with_status(session, text)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if not isinstance(result, Exception):
                    status = result.get("status", 0)
                    response_codes.append(status)
                    if status == 503:  # Service Unavailable (backpressure)
                        backpressure_count += 1

        total_time = time.perf_counter() - start_time
        success_count = sum(1 for code in response_codes if code == 200)

        self.results["backpressure_handling"] = {
            "total_requests": 100,
            "success_count": success_count,
            "backpressure_rejections": backpressure_count,
            "other_errors": len(response_codes) - success_count - backpressure_count,
            "total_time_sec": total_time,
            "backpressure_triggered": backpressure_count > 0
        }

        logger.info(f"✅ Backpressure: {success_count} success, {backpressure_count} backpressure rejections")

    async def _test_high_concurrency(self):
        """Test high concurrency performance with increased limits"""
        logger.info("⚡ Testing high concurrency performance...")

        start_time = time.perf_counter()
        latencies = []
        success_count = 0
        error_count = 0

        async with aiohttp.ClientSession() as session:
            tasks = []
            # Test with higher concurrency (should handle better with optimizations)
            for i in range(self.max_concurrent):
                text = f"Concurrency test {i}: {random.choice(self.test_texts)}"
                task = self._send_emotion_request(session, text)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                else:
                    if result.get("status") == 200:
                        success_count += 1
                        latency = result.get("latency_ms", 0)
                        if latency > 0:  # Only add valid latencies
                            latencies.append(latency)
                    else:
                        error_count += 1

        total_time = time.perf_counter() - start_time

        self.results["high_concurrency"] = {
            "concurrent_requests": self.max_concurrent,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate_pct": (success_count / self.max_concurrent) * 100,
            "total_time_sec": total_time,
            "avg_latency_ms": mean(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        }

        logger.info(f"✅ High concurrency: {success_count}/{self.max_concurrent} success, {mean(latencies):.1f}ms avg")

    async def _test_ml_cache_effectiveness(self):
        """Test ML adapter cache effectiveness"""
        logger.info("💾 Testing ML cache effectiveness...")

        # First, send requests to populate cache
        cache_test_text = "This is a cache effectiveness test message"

        async with aiohttp.ClientSession() as session:
            # Send same request multiple times to test cache
            first_request_time = None
            cached_request_times = []

            # First request (cache miss)
            start = time.perf_counter()
            result = await self._send_emotion_request(session, cache_test_text)
            first_request_time = (time.perf_counter() - start) * 1000

            # Multiple cached requests
            for i in range(10):
                start = time.perf_counter()
                await self._send_emotion_request(session, cache_test_text)
                cached_request_times.append((time.perf_counter() - start) * 1000)

        # Get cache stats
        cache_stats = await self._get_ml_stats()

        avg_cached_time = mean(cached_request_times) if cached_request_times else 0
        cache_speedup = first_request_time / avg_cached_time if avg_cached_time > 0 else 1

        self.results["ml_cache_effectiveness"] = {
            "first_request_latency_ms": first_request_time,
            "avg_cached_latency_ms": avg_cached_time,
            "cache_speedup_factor": cache_speedup,
            "cache_hit_rate_pct": cache_stats.get("cache_hit_rate", 0),
            "cache_enabled": cache_stats.get("cache_enabled", False),
            "cache_size": cache_stats.get("cache_size", 0)
        }

        logger.info(f"✅ Cache: {cache_speedup:.1f}x speedup, {cache_stats.get('cache_hit_rate', 0):.1f}% hit rate")

    async def _test_auto_debug_monitoring(self):
        """Test auto-debug monitoring system"""
        logger.info("🤖 Testing auto-debug monitoring...")

        try:
            async with aiohttp.ClientSession() as session:
                # Test auto-debug endpoint
                async with session.get(f"{self.base_url}/api/v1/brain/auto-debug") as response:
                    if response.status == 200:
                        debug_report = await response.json()
                    else:
                        debug_report = {"error": "Failed to fetch"}

                # Test issues endpoint
                async with session.get(f"{self.base_url}/api/v1/brain/auto-debug/issues") as response:
                    if response.status == 200:
                        issues_report = await response.json()
                    else:
                        issues_report = {"error": "Failed to fetch"}

        except Exception as e:
            debug_report = {"error": str(e)}
            issues_report = {"error": str(e)}

        self.results["auto_debug_monitoring"] = {
            "debug_endpoint_working": "error" not in debug_report,
            "issues_endpoint_working": "error" not in issues_report,
            "health_score": debug_report.get("health_score", 0),
            "active_issues_count": len(debug_report.get("active_issues", {})),
            "critical_issues": issues_report.get("total_critical", 0),
            "warning_issues": issues_report.get("total_warning", 0),
            "monitoring_active": debug_report.get("monitoring_stats", {}).get("components_monitored", [])
        }

        logger.info(f"✅ Auto-debug: Health score {debug_report.get('health_score', 0):.0f}%, "
                   f"{issues_report.get('total_critical', 0)} critical issues")

    async def _test_system_stability(self):
        """Test overall system stability under sustained load"""
        logger.info("🏠 Testing system stability under sustained load...")

        start_time = time.perf_counter()
        total_requests = 0
        total_success = 0
        total_errors = 0
        latencies = []

        # Run sustained load for 60 seconds
        end_time = start_time + 60

        async with aiohttp.ClientSession() as session:
            while time.perf_counter() < end_time:
                batch_start = time.perf_counter()

                # Send batch of 10 requests
                tasks = []
                for i in range(10):
                    text = f"Stability test {total_requests + i}: {random.choice(self.test_texts)}"
                    task = self._send_emotion_request(session, text)
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    total_requests += 1
                    if isinstance(result, Exception):
                        total_errors += 1
                    elif result.get("status") == 200:
                        total_success += 1
                        latency = result.get("latency_ms", 0)
                        if latency > 0:  # Only add valid latencies
                            latencies.append(latency)
                    else:
                        total_errors += 1

                # Small delay between batches
                await asyncio.sleep(0.1)

        total_time = time.perf_counter() - start_time

        self.results["system_stability"] = {
            "test_duration_sec": total_time,
            "total_requests": total_requests,
            "total_success": total_success,
            "total_errors": total_errors,
            "success_rate_pct": (total_success / max(1, total_requests)) * 100,
            "requests_per_sec": total_requests / total_time,
            "avg_latency_ms": mean(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "stable": total_errors / max(1, total_requests) < 0.05  # <5% error rate
        }

        logger.info(f"✅ Stability: {total_success}/{total_requests} success over {total_time:.0f}s "
                   f"({total_requests/total_time:.1f} req/s)")

    async def _final_health_assessment(self):
        """Final health assessment of all systems"""
        logger.info("🏥 Final health assessment...")

        try:
            async with aiohttp.ClientSession() as session:
                # Get all monitoring endpoints
                endpoints = [
                    "/api/v1/brain/health",
                    "/api/v1/brain/fortress",
                    "/api/v1/emotion/stats",
                    "/api/v1/brain/auto-debug"
                ]

                health_data = {}
                for endpoint in endpoints:
                    try:
                        async with session.get(f"{self.base_url}{endpoint}") as response:
                            if response.status == 200:
                                health_data[endpoint] = await response.json()
                            else:
                                health_data[endpoint] = {"status": "error", "code": response.status}
                    except Exception as e:
                        health_data[endpoint] = {"status": "error", "error": str(e)}

        except Exception as e:
            health_data = {"error": str(e)}

        self.results["final_health"] = health_data

        # Calculate overall optimization success score
        score = self._calculate_optimization_score()
        self.results["optimization_score"] = score

        logger.info(f"✅ Final health check complete - Optimization Score: {score:.1f}%")

    def _calculate_optimization_score(self) -> float:
        """Calculate overall optimization success score"""
        score = 0.0
        max_score = 100.0

        # Baseline performance (20 points)
        baseline = self.results.get("baseline_performance", {})
        if baseline.get("error_rate_pct", 100) < 5:
            score += 20
        elif baseline.get("error_rate_pct", 100) < 10:
            score += 10

        # Idempotence fix (20 points)
        idempotence = self.results.get("idempotence_fix", {})
        if idempotence.get("idempotence_working", False):
            score += 20
        elif idempotence.get("success_rate_pct", 0) > 50:
            score += 10

        # Backpressure handling (15 points)
        backpressure = self.results.get("backpressure_handling", {})
        if backpressure.get("backpressure_triggered", False):
            score += 15  # Good that backpressure activated

        # High concurrency (15 points)
        concurrency = self.results.get("high_concurrency", {})
        if concurrency.get("success_rate_pct", 0) > 80:
            score += 15
        elif concurrency.get("success_rate_pct", 0) > 60:
            score += 8

        # ML cache effectiveness (15 points)
        cache = self.results.get("ml_cache_effectiveness", {})
        if cache.get("cache_speedup_factor", 1) > 2:
            score += 15
        elif cache.get("cache_speedup_factor", 1) > 1.5:
            score += 8

        # Auto-debug monitoring (10 points)
        autodebug = self.results.get("auto_debug_monitoring", {})
        if autodebug.get("debug_endpoint_working", False) and autodebug.get("issues_endpoint_working", False):
            score += 10
        elif autodebug.get("debug_endpoint_working", False) or autodebug.get("issues_endpoint_working", False):
            score += 5

        # System stability (5 points)
        stability = self.results.get("system_stability", {})
        if stability.get("stable", False):
            score += 5

        return min(score, max_score)

    async def _send_emotion_request(self, session: aiohttp.ClientSession, text: str) -> Dict[str, Any]:
        """Send emotion detection request and return result with timing"""
        start_time = time.perf_counter()
        try:
            async with session.post(
                f"{self.base_url}/api/v1/emotion/detect",
                json={"text": text},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                latency_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "status": response.status,
                    "latency_ms": latency_ms,
                    "data": await response.json() if response.status == 200 else None
                }
        except Exception as e:
            return {"status": 0, "error": str(e), "latency_ms": 0}

    async def _send_emotion_request_with_status(self, session: aiohttp.ClientSession, text: str) -> Dict[str, Any]:
        """Send emotion detection request and return status info"""
        try:
            async with session.post(
                f"{self.base_url}/api/v1/emotion/detect",
                json={"text": text},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return {"status": response.status}
        except Exception as e:
            return {"status": 0, "error": str(e)}

    async def _get_ml_stats(self) -> Dict[str, Any]:
        """Get ML adapter statistics"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v1/emotion/stats") as response:
                    if response.status == 200:
                        return await response.json()
                    return {}
        except Exception:
            return {}

    def print_results(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 80)
        print("🚀 JEFFREY OS OPTIMIZATION VALIDATION RESULTS")
        print("=" * 80)

        score = self.results.get("optimization_score", 0)
        if score >= 90:
            status = "🟢 EXCELLENT"
        elif score >= 75:
            status = "🟡 GOOD"
        elif score >= 50:
            status = "🟠 NEEDS IMPROVEMENT"
        else:
            status = "🔴 CRITICAL ISSUES"

        print(f"\nOVERALL OPTIMIZATION SCORE: {score:.1f}% {status}")
        print("-" * 80)

        # Detailed results
        for test_name, results in self.results.items():
            if test_name == "optimization_score":
                continue

            print(f"\n📊 {test_name.upper().replace('_', ' ')}")
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")

        print("\n" + "=" * 80)

async def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"

    if len(sys.argv) > 2:
        max_concurrent = int(sys.argv[2])
    else:
        max_concurrent = 50

    print(f"🧪 Starting stress test against {base_url}")
    print(f"📈 Max concurrency: {max_concurrent}")

    test = OptimizationStressTest(base_url, max_concurrent)

    try:
        results = await test.run_comprehensive_test()

        # Print results
        test.print_results()

        # Save results to file
        timestamp = int(time.time())
        results_file = f"stress_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_file}")

        # Exit with appropriate code
        score = results.get("optimization_score", 0)
        if score >= 75:
            print("✅ All optimizations validated successfully!")
            sys.exit(0)
        else:
            print("⚠️ Some optimizations need attention")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())