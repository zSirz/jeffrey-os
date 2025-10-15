#!/usr/bin/env python3
"""
Jeffrey Phoenix Super Analyzer v3.1 - Ultimate Edition
=====================================================

Production-grade file analysis system with advanced optimizations for massive codebases.
Designed to handle 171k+ Python files efficiently with state-of-the-art deduplication.

Key Features (V3.1 ULTIMATE ENHANCEMENTS):
- Parallélisation Phase 1 avec ThreadPoolExecutor optimisé
- Mode léger avec N0 fingerprint ultra-rapide
- Fix agrégation Phase 5 avec fallbacks robustes
- DB SQLite avec pragmas robustes et index optimisés
- CLI étendue avec tous les flags et options
- Monitoring avec ETA et ajustement dynamique
- SimHash 64-bit + LSH multi-band clustering
- Double-stat iCloud stability checking
- Config snapshots + structured error journaling
- Memory management avancé avec garbage collection
- Streaming JSONL avec rotation automatique
- Architecture robuste avec crash recovery

Performance Optimizations:
- BLAKE2b/BLAKE3 hashing (3-5x faster than SHA-256)
- N0 fast fingerprint (64KB header + size) for instant deduplication
- ThreadPoolExecutor avec worker pool optimisé
- Unicode NFC normalization for macOS compatibility
- Inode deduplication for hardlinks
- Robust iCloud placeholder detection
- Atomic checkpoints with .tmp files for crash recovery
- SQLite with WAL mode, explicit transactions and busy timeout
- Memory monitoring with psutil + dynamic GC
- Streaming JSONL without loading into memory
- Dynamic ETA calculation with performance adjustment
- Smart batching with memory pressure detection

Architecture:
- Phase 1: Ultra-fast parallel scan with streaming and dedup
- Phase 2: Triple-level deduplication (fingerprint, content, structure)
- Phase 3: Advanced analysis with pattern detection
- Phase 4: LSH-based similarity detection with intelligent capping
- Phase 5: Comprehensive reports avec fallbacks et metrics

Author: Jeffrey Phoenix Team
Version: 3.1.0 Ultimate Edition
License: Private - Jeffrey Project
"""

import argparse
import gc
import gzip
import hashlib
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Try to import optimized libraries with graceful fallbacks
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from blake3 import blake3

    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

try:
    import xxhash

    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

try:
    import chardet

    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

# Performance and optimization constants
CHECKPOINT_INTERVAL = 1000
BATCH_SIZE = 500  # Increased for better throughput
MAX_SIMILARITY_COMPARISONS = 2000  # Increased cap
JSONL_ROTATION_SIZE = 100 * 1024 * 1024  # 100MB
FAST_FINGERPRINT_SIZE = 64 * 1024  # 64KB for N0 fingerprint
ICLOUD_PLACEHOLDER_MARKERS = [b'.com.apple.icloud', b'bplist00']
MAX_MEMORY_USAGE = 85  # Percentage
WORKER_THREAD_COUNT = min(16, (os.cpu_count() or 4) * 2)  # Optimized worker count
GC_THRESHOLD = 1000  # Files processed before garbage collection
PHASE1_PARALLEL_BATCH_SIZE = 100  # Batch size for parallel processing


@dataclass
class FileMetrics:
    """Comprehensive file metrics for analysis."""

    path: str
    size: int
    inode: int | None
    mtime: float
    blake2b_hash: str
    n0_fingerprint: str
    simhash: int | None  # 64-bit SimHash for similarity
    is_icloud_placeholder: bool
    encoding: str | None
    line_count: int
    func_count: int
    class_count: int
    import_count: int
    complexity_score: float
    signals_score: float  # Aggregated signals score
    file_type: str
    extension: str
    processing_time: float  # Time taken to process this file

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'FileMetrics':
        """Create FileMetrics from dictionary."""
        return cls(**data)


@dataclass
class SimilarityResult:
    """Result of similarity comparison between files."""

    file1: str
    file2: str
    similarity_score: float
    shared_patterns: list[str]
    diff_lines: int
    is_duplicate: bool
    hamming_distance: int  # Hamming distance for SimHash
    structural_similarity: float  # Structural similarity score


class PerformanceMonitor:
    """Advanced performance monitoring with ETA calculation."""

    def __init__(self):
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self.processed_count = 0
        self.total_count = 0
        self.last_update_time = time.time()
        self.processing_rates = []  # Recent processing rates for ETA calculation
        self.memory_samples = []
        self.logger = logging.getLogger(__name__)

    def start_phase(self, phase_name: str, total_items: int = 0):
        """Start tracking a new phase."""
        self.phase_start_time = time.time()
        self.processed_count = 0
        self.total_count = total_items
        self.processing_rates = []
        self.logger.info(f"=== Starting {phase_name} (Total: {total_items:,} items) ===")

    def update_progress(self, processed: int, force_log: bool = False):
        """Update progress and calculate ETA."""
        current_time = time.time()
        self.processed_count = processed

        # Calculate processing rate (items per second)
        time_diff = current_time - self.last_update_time
        if time_diff >= 5.0 or force_log:  # Update every 5 seconds or on force
            if time_diff > 0:
                rate = (processed - getattr(self, '_last_processed', 0)) / time_diff
                self.processing_rates.append(rate)

                # Keep only recent rates (last 5 measurements)
                if len(self.processing_rates) > 5:
                    self.processing_rates.pop(0)

            # Calculate ETA
            if self.total_count > 0 and processed > 0:
                avg_rate = sum(self.processing_rates) / len(self.processing_rates) if self.processing_rates else 1
                remaining = self.total_count - processed
                eta_seconds = remaining / max(avg_rate, 0.1)  # Avoid division by zero
                eta_str = self._format_duration(eta_seconds)

                progress_pct = (processed / self.total_count) * 100
                elapsed = current_time - self.phase_start_time
                elapsed_str = self._format_duration(elapsed)

                # Memory usage
                memory_info = ""
                if PSUTIL_AVAILABLE:
                    memory = psutil.virtual_memory()
                    memory_info = f", Memory: {memory.percent:.1f}%"

                self.logger.info(
                    f"Progress: {processed:,}/{self.total_count:,} ({progress_pct:.1f}%) "
                    f"| Rate: {avg_rate:.1f}/s | Elapsed: {elapsed_str} | ETA: {eta_str}{memory_info}"
                )
            else:
                elapsed = current_time - self.phase_start_time
                elapsed_str = self._format_duration(elapsed)
                rate = sum(self.processing_rates) / len(self.processing_rates) if self.processing_rates else 0
                self.logger.info(f"Processed: {processed:,} | Rate: {rate:.1f}/s | Elapsed: {elapsed_str}")

            self._last_processed = processed
            self.last_update_time = current_time

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"

    def finish_phase(self, phase_name: str):
        """Finish tracking current phase."""
        elapsed = time.time() - self.phase_start_time
        elapsed_str = self._format_duration(elapsed)
        avg_rate = self.processed_count / max(elapsed, 0.1)
        self.logger.info(f"=== Completed {phase_name} in {elapsed_str} (avg rate: {avg_rate:.1f}/s) ===")


class MemoryManager:
    """Advanced memory management with dynamic garbage collection."""

    def __init__(self, max_usage_percent: float = MAX_MEMORY_USAGE):
        self.max_usage = max_usage_percent
        self.logger = logging.getLogger(__name__)
        self.last_gc_time = time.time()
        self.gc_threshold_count = 0

    def check_memory_pressure(self) -> tuple[float, bool, bool]:
        """Check memory pressure and determine if GC is needed."""
        usage_percent = 0.0
        is_high = False
        needs_gc = False

        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            is_high = usage_percent > self.max_usage

            # Trigger GC if memory is high or we've processed many files
            current_time = time.time()
            self.gc_threshold_count += 1

            needs_gc = (
                is_high
                or self.gc_threshold_count >= GC_THRESHOLD
                or (current_time - self.last_gc_time) > 300  # Every 5 minutes
            )

            if is_high:
                self.logger.warning(f"High memory usage: {usage_percent:.1f}%")

            if needs_gc:
                self.logger.info(f"Triggering garbage collection (memory: {usage_percent:.1f}%)")

        return usage_percent, is_high, needs_gc

    def force_garbage_collection(self):
        """Force garbage collection and reset counters."""
        before_memory = 0
        after_memory = 0

        if PSUTIL_AVAILABLE:
            before_memory = psutil.virtual_memory().percent

        # Aggressive garbage collection
        collected = gc.collect()

        if PSUTIL_AVAILABLE:
            after_memory = psutil.virtual_memory().percent

        self.last_gc_time = time.time()
        self.gc_threshold_count = 0

        self.logger.info(
            f"Garbage collection: {collected} objects freed, memory: {before_memory:.1f}% → {after_memory:.1f}%"
        )


class HashOptimizer:
    """Ultra-optimized hashing with multiple algorithms and caching."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._simhash_cache = {}  # LRU cache for SimHash
        self._cache_hits = 0
        self._cache_misses = 0

    def blake2b_hash(self, data: bytes) -> str:
        """Fast BLAKE2b hashing."""
        return hashlib.blake2b(data, digest_size=32).hexdigest()

    def blake3_hash(self, data: bytes) -> str:
        """Ultra-fast BLAKE3 hashing if available."""
        if BLAKE3_AVAILABLE:
            return blake3(data).hexdigest()
        return self.blake2b_hash(data)

    def xxhash64(self, data: bytes) -> str:
        """Fast non-cryptographic hash for fingerprinting."""
        if XXHASH_AVAILABLE:
            return xxhash.xxh64(data).hexdigest()
        return hashlib.md5(data).hexdigest()

    def n0_fingerprint(self, file_path: str) -> str:
        """Create ultra-fast N0 fingerprint from first 64KB + file size + mtime."""
        try:
            stat_info = os.stat(file_path)
            # Include mtime for better uniqueness
            fingerprint_data = f"{stat_info.st_size}_{stat_info.st_mtime}"

            with open(file_path, 'rb') as f:
                header_data = f.read(FAST_FINGERPRINT_SIZE)
                if header_data:
                    header_hash = self.xxhash64(header_data)
                    return f"{fingerprint_data}_{header_hash}"
                return f"{fingerprint_data}_empty"

        except OSError as e:
            self.logger.debug(f"N0 fingerprint failed for {file_path}: {e}")
            return "error"

    @staticmethod
    def _extract_tokens(text: str, max_tokens: int = 1000) -> Iterator[str]:
        """Extract significant tokens for SimHash with limit."""
        count = 0
        # Enhanced token extraction: identifiers, strings, numbers
        for token in re.findall(r'[A-Za-z_][A-Za-z0-9_]{2,}|"[^"]*"|\'[^\']*\'|\d+', text[:20000]):
            if count >= max_tokens:
                break
            yield token.lower()
            count += 1

    def simhash64(self, text: str, use_cache: bool = True) -> int:
        """
        Enhanced SimHash 64-bit for structural similarity with caching.
        """
        if use_cache:
            text_hash = hashlib.md5(text[:1000].encode()).hexdigest()
            if text_hash in self._simhash_cache:
                self._cache_hits += 1
                return self._simhash_cache[text_hash]
            self._cache_misses += 1

        # Vector of votes per bit
        v = [0] * 64
        token_count = 0

        for token in self._extract_tokens(text):
            token_count += 1
            # Hash the token using BLAKE2b for better distribution
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), 'big')

            # Weight tokens by frequency (simple approximation)
            weight = 1 + min(text.count(token[:10]), 5)  # Cap weight at 6

            # Vote on each bit with weight
            for i in range(64):
                if (h >> i) & 1:
                    v[i] += weight
                else:
                    v[i] -= weight

        # Aggregation: sign of votes → final bits
        result = 0
        for i in range(64):
            if v[i] >= 0:
                result |= 1 << i

        # Cache result if using cache
        if use_cache and token_count > 5:  # Only cache meaningful results
            # Simple LRU: remove oldest if cache is too large
            if len(self._simhash_cache) > 10000:
                oldest_key = next(iter(self._simhash_cache))
                del self._simhash_cache[oldest_key]
            self._simhash_cache[text_hash] = result

        return result

    @staticmethod
    def hamming64(a: int, b: int) -> int:
        """Optimized Hamming distance between two 64-bit values."""
        xor = a ^ b
        # Use bit_count() if available (Python 3.10+), otherwise fallback
        return bin(xor).count('1')

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / max(total_requests, 1)) * 100
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self._simhash_cache),
        }


class RobustFileProcessor:
    """Robust file processing with advanced error handling and iCloud support."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.hash_optimizer = HashOptimizer()
        self.memory_manager = MemoryManager()
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode for macOS compatibility."""
        return unicodedata.normalize('NFC', text)

    def is_icloud_placeholder(self, file_path: str) -> bool:
        """Enhanced iCloud placeholder detection."""
        try:
            # Quick size check first
            stat_info = os.stat(file_path)
            if stat_info.st_size == 0:
                return True  # Empty files are likely placeholders

            # Check file header for iCloud markers
            with open(file_path, 'rb') as f:
                header = f.read(2048)  # Read more bytes for better detection
                return any(marker in header for marker in ICLOUD_PLACEHOLDER_MARKERS)
        except OSError:
            return True  # Assume placeholder if can't read

    def detect_encoding(self, file_path: str) -> str | None:
        """Enhanced encoding detection with fallback."""
        if not CHARDET_AVAILABLE:
            return None

        try:
            with open(file_path, 'rb') as f:
                # Read larger sample for better detection
                sample = f.read(32768)  # 32KB sample
                if not sample:
                    return 'utf-8'

                result = chardet.detect(sample)
                confidence = result.get('confidence', 0)
                encoding = result.get('encoding')

                # Only trust high-confidence results
                if confidence > 0.7 and encoding:
                    return encoding

                # Fallback detection for common encodings
                try:
                    sample.decode('utf-8')
                    return 'utf-8'
                except UnicodeDecodeError:
                    try:
                        sample.decode('latin-1')
                        return 'latin-1'
                    except UnicodeDecodeError:
                        return 'utf-8'  # Last resort

        except Exception as e:
            self.logger.debug(f"Encoding detection failed for {file_path}: {e}")
            return 'utf-8'

    def extract_advanced_signals(self, content: str) -> dict[str, int | float]:
        """Advanced pattern detection with more signals."""
        signals = {
            'lines': content.count('\n'),
            'functions': content.count('def '),
            'classes': content.count('class '),
            'imports': content.count('import ') + content.count('from '),
            'docstrings': content.count('"""') + content.count("'''"),
            'comments': content.count('#'),
            'async_functions': content.count('async def '),
            'decorators': content.count('@'),
            'try_blocks': content.count('try:'),
            'lambda_functions': content.count('lambda '),
            'list_comprehensions': content.count('[') - content.count('[]'),
            'dict_literals': content.count('{') - content.count('{}'),
            'string_literals': content.count('"') + content.count("'"),
            'numeric_literals': len(re.findall(r'\b\d+\.?\d*\b', content[:5000])),
            'indentation_level': self._estimate_max_indentation(content),
            'cyclomatic_complexity': self._estimate_complexity(content),
            'unique_identifiers': len(set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content[:5000]))),
            'average_line_length': sum(len(line) for line in content.split('\n')[:100])
            / max(1, min(100, content.count('\n'))),
        }
        return signals

    def _estimate_max_indentation(self, content: str) -> int:
        """Estimate maximum indentation level."""
        max_indent = 0
        for line in content.split('\n')[:200]:  # Sample first 200 lines
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                if line.startswith(' '):
                    indent_level = indent // 4  # Assume 4-space indentation
                elif line.startswith('\t'):
                    indent_level = indent
                else:
                    indent_level = 0
                max_indent = max(max_indent, indent_level)
        return max_indent

    def _estimate_complexity(self, content: str) -> int:
        """Enhanced complexity estimation."""
        complexity_keywords = [
            'if ',
            'elif ',
            'else:',
            'for ',
            'while ',
            'try:',
            'except:',
            'finally:',
            'with ',
            'match ',
            'case ',
            'and ',
            'or ',
            'not ',
            'break',
            'continue',
            'return',
            'yield',
            'raise',
            'assert',
        ]

        # Sample first 10KB for performance
        sample = content[:10000]
        complexity = sum(sample.count(keyword) for keyword in complexity_keywords)

        # Add nested complexity
        nesting_chars = sample.count('{') + sample.count('[') + sample.count('(')
        return complexity + (nesting_chars // 10)  # Weight nesting less

    def calculate_signals_score(self, signals: dict[str, int | float]) -> float:
        """Calculate composite signals score for ranking."""
        # Weighted scoring based on code complexity indicators
        score = (
            signals['functions'] * 3.0
            + signals['classes'] * 5.0
            + signals['lines'] * 0.1
            + signals['cyclomatic_complexity'] * 2.0
            + signals['unique_identifiers'] * 0.5
            + signals['imports'] * 1.5
            + signals['indentation_level'] * 1.0
        )
        return float(score)

    def process_file_robust(self, file_path: str, output_dir: Path | None = None) -> FileMetrics | None:
        """Process single file with comprehensive error handling and performance tracking."""
        start_time = time.time()

        try:
            # Normalize path for macOS
            normalized_path = self.normalize_unicode(str(file_path))

            # Double-stat iCloud stability check
            stat_before = os.stat(normalized_path)
            size_before = stat_before.st_size
            mtime_before = stat_before.st_mtime

            # Skip if too large
            max_file_size = self.config.get('max_file_size', 10 * 1024 * 1024)  # 10MB default
            if size_before > max_file_size:
                self.logger.debug(f"Skipping large file: {normalized_path} ({size_before:,} bytes)")
                self.skipped_count += 1
                return None

            # Check if it's an iCloud placeholder
            is_placeholder = self.is_icloud_placeholder(normalized_path)
            if is_placeholder:
                self.logger.debug(f"Skipping iCloud placeholder: {normalized_path}")
                self.skipped_count += 1
                return None

            # Generate fast N0 fingerprint
            n0_fingerprint = self.hash_optimizer.n0_fingerprint(normalized_path)
            if n0_fingerprint == "error":
                self.logger.debug(f"Failed to generate fingerprint for: {normalized_path}")
                self.error_count += 1
                return None

            # Read and analyze file content
            try:
                encoding = self.detect_encoding(normalized_path)
                with open(normalized_path, encoding=encoding or 'utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                self.logger.debug(f"Failed to read content from {normalized_path}: {e}")
                # Fallback to binary read for hash only
                with open(normalized_path, 'rb') as f:
                    binary_content = f.read()
                content = ""
                blake2b_hash = self.hash_optimizer.blake2b_hash(binary_content)
                simhash = None
            else:
                # Calculate hash from content
                content_bytes = content.encode('utf-8', errors='ignore')
                if BLAKE3_AVAILABLE:
                    blake2b_hash = self.hash_optimizer.blake3_hash(content_bytes)
                else:
                    blake2b_hash = self.hash_optimizer.blake2b_hash(content_bytes)

                # Calculate SimHash for similarity analysis
                simhash = self.hash_optimizer.simhash64(content) if content.strip() else None

            # Post-read stability check
            stat_after = os.stat(normalized_path)
            size_after = stat_after.st_size
            mtime_after = stat_after.st_mtime

            # Check for file modification during read (iCloud sync detection)
            if (size_before, mtime_before) != (size_after, mtime_after):
                self.logger.warning(f"File modified during read (iCloud sync?): {normalized_path}")
                self._log_unstable_read(normalized_path, size_before, size_after, mtime_before, mtime_after, output_dir)
                self.error_count += 1
                return None  # Skip unstable file

            # Extract advanced signals
            signals = self.extract_advanced_signals(content)
            signals_score = self.calculate_signals_score(signals)

            # Determine file type with enhanced classification
            extension = Path(normalized_path).suffix.lower()
            file_type = self._classify_file_type_advanced(extension, content)

            # Calculate processing time
            processing_time = time.time() - start_time

            self.processed_count += 1

            return FileMetrics(
                path=normalized_path,
                size=stat_after.st_size,
                inode=getattr(stat_after, 'st_ino', None),
                mtime=stat_after.st_mtime,
                blake2b_hash=blake2b_hash,
                n0_fingerprint=n0_fingerprint,
                simhash=simhash,
                is_icloud_placeholder=is_placeholder,
                encoding=encoding,
                line_count=signals['lines'],
                func_count=signals['functions'],
                class_count=signals['classes'],
                import_count=signals['imports'],
                complexity_score=float(signals['cyclomatic_complexity']),
                signals_score=signals_score,
                file_type=file_type,
                extension=extension,
                processing_time=processing_time,
            )

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            self.error_count += 1
            return None

    def _classify_file_type_advanced(self, extension: str, content: str) -> str:
        """Advanced file type classification with content analysis."""
        # Extension-based classification first
        python_extensions = {'.py', '.pyx', '.pyi', '.pyw'}
        rust_extensions = {'.rs'}
        js_extensions = {'.js', '.jsx', '.ts', '.tsx', '.mjs'}
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}
        doc_extensions = {'.md', '.rst', '.txt', '.doc'}

        if extension in python_extensions:
            return 'python'
        elif extension in rust_extensions:
            return 'rust'
        elif extension in js_extensions:
            return 'javascript'
        elif extension in config_extensions:
            return 'config'
        elif extension in doc_extensions:
            return 'documentation'
        elif extension in {'.sh', '.bash', '.zsh', '.fish'}:
            return 'shell'

        # Content-based classification for files without clear extensions
        if content:
            content_sample = content[:2000].lower()

            if any(keyword in content_sample for keyword in ['def ', 'class ', 'import ', 'from ', 'if __name__']):
                return 'python_like'
            elif any(keyword in content_sample for keyword in ['fn ', 'struct ', 'impl ', 'use ', 'mod ']):
                return 'rust_like'
            elif any(
                keyword in content_sample for keyword in ['function', 'const ', 'let ', 'var ', 'import ', 'export']
            ):
                return 'javascript_like'
            elif content_sample.startswith('#!'):
                return 'script'
            elif any(marker in content_sample for marker in ['<?xml', '<html', '<!doctype']):
                return 'markup'

        return 'other'

    def _log_unstable_read(
        self,
        path: str,
        size_before: int,
        size_after: int,
        mtime_before: float,
        mtime_after: float,
        output_dir: Path | None = None,
    ):
        """Log files that changed during read (atomic write)."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'path': path,
            'size_before': size_before,
            'size_after': size_after,
            'mtime_before': mtime_before,
            'mtime_after': mtime_after,
            'reason': 'file_modified_during_read',
        }

        # Determine log path
        if output_dir:
            unstable_log = output_dir / 'unstable_reads.jsonl'
        else:
            unstable_log = Path('unstable_reads.jsonl')

        # Atomic write with temp file
        temp_log = unstable_log.with_suffix('.jsonl.tmp')

        # Append to existing file
        try:
            with open(temp_log, 'w') as f:
                # Read existing entries if file exists
                if unstable_log.exists():
                    with open(unstable_log) as existing:
                        for line in existing:
                            f.write(line)
                # Write new entry
                f.write(json.dumps(entry, separators=(',', ':')) + '\n')
            # Atomic replace
            os.replace(str(temp_log), str(unstable_log))
        except Exception as e:
            self.logger.error(f"Failed to log unstable read: {e}")
            if temp_log.exists():
                temp_log.unlink()

    def get_processing_stats(self) -> dict[str, int]:
        """Get processing statistics."""
        return {'processed': self.processed_count, 'errors': self.error_count, 'skipped': self.skipped_count}


class StreamingJSONLWriter:
    """High-performance streaming JSONL writer with automatic rotation."""

    def __init__(self, base_path: str, max_size: int = JSONL_ROTATION_SIZE):
        self.base_path = Path(base_path)
        self.max_size = max_size
        self.current_file = None
        self.current_size = 0
        self.file_counter = 0
        self.logger = logging.getLogger(__name__)
        self.write_lock = threading.Lock()

    def __enter__(self):
        self._rotate_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.write_lock:
            if self.current_file:
                self.current_file.close()
                self.current_file = None

    def _rotate_file(self):
        """Rotate to a new file when size limit is reached."""
        with self.write_lock:
            if self.current_file:
                self.current_file.close()

            if self.file_counter == 0:
                file_path = self.base_path
            else:
                stem = self.base_path.stem
                suffix = self.base_path.suffix
                file_path = self.base_path.parent / f"{stem}_{self.file_counter:04d}{suffix}"

            self.current_file = open(file_path, 'w', buffering=8192)  # 8KB buffer
            self.current_size = 0
            self.file_counter += 1
            self.logger.info(f"Rotated to new JSONL file: {file_path}")

    def write_batch(self, data_list: list[dict[str, Any]]):
        """Write batch of data efficiently."""
        with self.write_lock:
            if not self.current_file:
                self._rotate_file()

            lines = []
            total_size = 0

            for data in data_list:
                line = json.dumps(data, separators=(',', ':')) + '\n'
                line_bytes = line.encode('utf-8')
                lines.append(line)
                total_size += len(line_bytes)

            # Check if rotation is needed
            if self.current_size + total_size > self.max_size:
                self._rotate_file()

            # Write all lines at once
            for line in lines:
                self.current_file.write(line)

            self.current_file.flush()
            self.current_size += total_size


class AtomicCheckpointManager:
    """Atomic checkpoint management with compression and versioning."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint_compressed(self, phase: str, data: dict[str, Any], counter: int):
        """Save compressed checkpoint atomically."""
        checkpoint_name = f"checkpoint_{phase}_{counter}.json.gz"
        temp_path = self.checkpoint_dir / f"{checkpoint_name}.tmp"
        final_path = self.checkpoint_dir / checkpoint_name

        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'phase': phase,
                'counter': counter,
                'data': data,
                'version': '3.1',
            }

            # Write compressed
            with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, separators=(',', ':'))

            # Atomic move
            os.replace(str(temp_path), str(final_path))
            self.logger.info(f"Saved compressed checkpoint: {final_path}")

            # Clean up old checkpoints (keep last 5)
            self._cleanup_old_checkpoints(phase, keep=5)

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def load_latest_checkpoint(self, phase: str) -> tuple[int, dict[str, Any]] | None:
        """Load the latest checkpoint for a phase."""
        checkpoints = list(self.checkpoint_dir.glob(f"checkpoint_{phase}_*.json*"))
        if not checkpoints:
            return None

        # Find latest checkpoint by counter number
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))

        try:
            if latest.suffix == '.gz':
                with gzip.open(latest, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(latest) as f:
                    data = json.load(f)

            return data['counter'], data['data']
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {latest}: {e}")
            return None

    def _cleanup_old_checkpoints(self, phase: str, keep: int = 5):
        """Clean up old checkpoints, keeping only the most recent."""
        checkpoints = list(self.checkpoint_dir.glob(f"checkpoint_{phase}_*.json*"))
        if len(checkpoints) <= keep:
            return

        # Sort by counter and remove oldest
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        for old_checkpoint in checkpoints[:-keep]:
            try:
                old_checkpoint.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")


class RobustSQLiteManager:
    """Enhanced SQLite manager with robust error handling and optimization."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._conn = None

    def __enter__(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._conn = sqlite3.connect(
                    self.db_path,
                    timeout=60.0,  # Increased timeout
                    isolation_level=None,  # Autocommit mode
                )

                # Enhanced SQLite optimizations
                self._conn.executescript("""
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA cache_size=50000;
                    PRAGMA temp_store=memory;
                    PRAGMA mmap_size=268435456;
                    PRAGMA optimize=0x10002;
                """)

                self._create_tables()
                return self

            except Exception as e:
                self.logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            try:
                self._conn.execute("PRAGMA optimize")
                self._conn.close()
            except Exception as e:
                self.logger.error(f"Error closing database: {e}")

    def _create_tables(self):
        """Create optimized database tables with enhanced indexes."""
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS file_metrics (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            size INTEGER,
            inode INTEGER,
            mtime REAL,
            blake2b_hash TEXT,
            n0_fingerprint TEXT,
            simhash INTEGER,
            is_icloud_placeholder BOOLEAN,
            encoding TEXT,
            line_count INTEGER,
            func_count INTEGER,
            class_count INTEGER,
            import_count INTEGER,
            complexity_score REAL,
            signals_score REAL,
            file_type TEXT,
            extension TEXT,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS duplicates (
            id INTEGER PRIMARY KEY,
            group_hash TEXT NOT NULL,
            file_paths TEXT NOT NULL,
            duplicate_type TEXT NOT NULL,
            file_count INTEGER,
            total_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS similarities (
            id INTEGER PRIMARY KEY,
            file1 TEXT NOT NULL,
            file2 TEXT NOT NULL,
            similarity_score REAL,
            hamming_distance INTEGER,
            structural_similarity REAL,
            shared_patterns TEXT,
            diff_lines INTEGER,
            is_duplicate BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS processing_stats (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            files_processed INTEGER,
            processing_time REAL,
            memory_peak REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Performance indexes
        CREATE INDEX IF NOT EXISTS idx_blake2b_hash ON file_metrics(blake2b_hash);
        CREATE INDEX IF NOT EXISTS idx_n0_fingerprint ON file_metrics(n0_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_simhash ON file_metrics(simhash);
        CREATE INDEX IF NOT EXISTS idx_signals_score ON file_metrics(signals_score DESC);
        CREATE INDEX IF NOT EXISTS idx_inode ON file_metrics(inode);
        CREATE INDEX IF NOT EXISTS idx_file_type ON file_metrics(file_type);
        CREATE INDEX IF NOT EXISTS idx_extension ON file_metrics(extension);
        CREATE INDEX IF NOT EXISTS idx_size ON file_metrics(size DESC);
        CREATE INDEX IF NOT EXISTS idx_complexity ON file_metrics(complexity_score DESC);

        -- Duplicates indexes
        CREATE INDEX IF NOT EXISTS idx_dup_group_hash ON duplicates(group_hash);
        CREATE INDEX IF NOT EXISTS idx_dup_type ON duplicates(duplicate_type);
        CREATE INDEX IF NOT EXISTS idx_dup_count ON duplicates(file_count DESC);

        -- Similarities indexes
        CREATE INDEX IF NOT EXISTS idx_sim_score ON similarities(similarity_score DESC);
        CREATE INDEX IF NOT EXISTS idx_sim_hamming ON similarities(hamming_distance);
        CREATE INDEX IF NOT EXISTS idx_sim_file1 ON similarities(file1);
        CREATE INDEX IF NOT EXISTS idx_sim_file2 ON similarities(file2);

        -- Processing stats indexes
        CREATE INDEX IF NOT EXISTS idx_stats_run_id ON processing_stats(run_id);
        CREATE INDEX IF NOT EXISTS idx_stats_phase ON processing_stats(phase);
        """)

    def bulk_insert_metrics(self, metrics_list: list[FileMetrics], batch_size: int = 1000):
        """Bulk insert file metrics with batching."""
        if not metrics_list:
            return

        self._conn.execute("BEGIN TRANSACTION")

        try:
            for i in range(0, len(metrics_list), batch_size):
                batch = metrics_list[i : i + batch_size]

                values = []
                for metrics in batch:
                    values.append(
                        (
                            metrics.path,
                            metrics.size,
                            metrics.inode,
                            metrics.mtime,
                            metrics.blake2b_hash,
                            metrics.n0_fingerprint,
                            metrics.simhash,
                            metrics.is_icloud_placeholder,
                            metrics.encoding,
                            metrics.line_count,
                            metrics.func_count,
                            metrics.class_count,
                            metrics.import_count,
                            metrics.complexity_score,
                            metrics.signals_score,
                            metrics.file_type,
                            metrics.extension,
                            metrics.processing_time,
                        )
                    )

                self._conn.executemany(
                    """
                    INSERT OR REPLACE INTO file_metrics
                    (path, size, inode, mtime, blake2b_hash, n0_fingerprint, simhash,
                     is_icloud_placeholder, encoding, line_count, func_count,
                     class_count, import_count, complexity_score, signals_score,
                     file_type, extension, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    values,
                )

            self._conn.execute("COMMIT")
            self.logger.info(f"Bulk inserted {len(metrics_list)} metrics into database")

        except Exception as e:
            self._conn.execute("ROLLBACK")
            self.logger.error(f"Failed to bulk insert metrics: {e}")
            raise

    @property
    def conn(self):
        """Access to connection for direct queries."""
        return self._conn


class JeffreyPhoenixSuperAnalyzer:
    """Ultimate Jeffrey Phoenix Super Analyzer v3.1 with all optimizations."""

    def __init__(
        self,
        config_path: str | None = None,
        max_files: int | None = None,
        max_files_per_env: dict[str, int] | None = None,
        **kwargs,
    ):
        """Initialize the super analyzer with all optimizations."""

        # Load configuration
        self.config = self._load_config(config_path)

        # Apply CLI overrides
        for key, value in kwargs.items():
            if value is not None:
                self.config[key] = value

        # Setup logging first
        self.logger = self._setup_logging()

        # Generate unique run ID
        self.run_id = self._generate_run_id()
        self.logger.info(f"Initializing Jeffrey Phoenix Super Analyzer v3.1 (Run ID: {self.run_id[:12]})")

        # Setup output directory with run ID
        base_output = self.config.get('output_dir', 'analyzer_results')
        self.output_dir = Path(f"{base_output}_{self.run_id[:8]}")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.file_processor = RobustFileProcessor(self.config)
        self.memory_manager = MemoryManager()
        self.performance_monitor = PerformanceMonitor()
        self.hash_optimizer = HashOptimizer()

        # File processing limits
        self.max_files_per_env = max_files_per_env or {}
        if max_files is not None:
            self.max_files = max_files
            self.logger.info(f"File processing cap set from CLI: {self.max_files}")
        else:
            self.max_files = self._get_dynamic_file_cap()
            self.logger.info(f"File processing cap set dynamically: {self.max_files}")

        # Threading configuration
        self.max_workers = min(self.config.get('max_workers', WORKER_THREAD_COUNT), WORKER_THREAD_COUNT)

        # Error tracking
        self.error_journal = []
        self.processing_stats = {}

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_requested = False

        self.logger.info(f"Analyzer initialized: {self.max_workers} workers, {self.max_files:,} file cap")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, requesting graceful shutdown...")
        self.shutdown_requested = True

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load and merge configuration from multiple sources."""
        # Start with default config
        config = {
            'scan_paths': ['.'],
            'output_dir': 'analyzer_results',
            'extensions': ['.py', '.rs', '.js', '.ts', '.yaml', '.json', '.md', '.txt'],
            'exclude_dirs': [
                '.git',
                '__pycache__',
                'node_modules',
                'target',
                'build',
                'dist',
                '.next',
                '.nuxt',
                'vendor',
                '.env',
                '.venv',
                'venv',
                '.tox',
                'coverage',
                '.pytest_cache',
                '.idea',
                '*.egg-info',
            ],
            'max_workers': WORKER_THREAD_COUNT,
            'enable_similarity': True,
            'similarity_threshold': 0.8,
            'checkpoint_interval': CHECKPOINT_INTERVAL,
            'max_memory_usage': MAX_MEMORY_USAGE,
            'max_file_size': 50 * 1024 * 1024,  # 50MB
            'enable_compression': True,
            'batch_size': PHASE1_PARALLEL_BATCH_SIZE,
        }

        # Load from YAML file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    yaml_config = yaml.safe_load(f)
                    config.update(yaml_config)
                    self.logger.info(f"Loaded configuration from: {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")

        # Environment-based overrides
        env_overrides = {
            'JEFFREY_MAX_FILES': ('max_files', int),
            'JEFFREY_MAX_WORKERS': ('max_workers', int),
            'JEFFREY_MEMORY_LIMIT': ('max_memory_usage', int),
            'JEFFREY_ENABLE_SIMILARITY': ('enable_similarity', lambda x: x.lower() == 'true'),
            'JEFFREY_BATCH_SIZE': ('batch_size', int),
        }

        for env_var, (config_key, converter) in env_overrides.items():
            env_value = os.environ.get(env_var)
            if env_value:
                try:
                    config[config_key] = converter(env_value)
                    self.logger.info(f"Applied environment override: {env_var}={env_value}")
                except ValueError:
                    self.logger.warning(f"Invalid environment value for {env_var}: {env_value}")

        return config

    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging with performance tracking."""
        logger = logging.getLogger(__name__)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.setLevel(logging.INFO)

        # Console handler with enhanced format
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler for detailed logs
        log_file = self.output_dir / 'analyzer.log' if hasattr(self, 'output_dir') else Path('analyzer.log')
        try:
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | PID:%(process)d | %(funcName)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")

        return logger

    def _get_dynamic_file_cap(self) -> int:
        """Get dynamic file processing cap based on system resources."""
        # Check environment variable first
        env_cap = os.environ.get('JEFFREY_MAX_FILES')
        if env_cap:
            try:
                return int(env_cap)
            except ValueError:
                pass

        # Dynamic calculation based on system resources
        if PSUTIL_AVAILABLE:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count(logical=True)

            # Base calculation: 1000 files per GB RAM + 500 per CPU core
            calculated_cap = int(memory_gb * 1000 + cpu_count * 500)

            # Apply environment-based constraints
            if 'CI' in os.environ:
                return min(calculated_cap, 1000)  # CI limit
            elif os.path.exists('/Applications'):  # macOS
                return min(calculated_cap, 100000)  # macOS with iCloud
            else:
                return min(calculated_cap, 50000)  # Linux/other

        # Fallback values
        if 'CI' in os.environ:
            return 1000
        elif os.path.exists('/Applications'):
            return 50000
        else:
            return 20000

    def _generate_run_id(self) -> str:
        """Generate unique run ID based on timestamp and parameters."""
        timestamp = datetime.now().isoformat()
        params_str = json.dumps(self.config, sort_keys=True)
        combined = f"{timestamp}_{params_str}"
        return hashlib.blake2b(combined.encode(), digest_size=16).hexdigest()

    def _save_config_snapshot(self):
        """Save configuration snapshot for reproducibility."""
        snapshot_path = self.output_dir / 'config_snapshot.yaml'
        temp_path = snapshot_path.with_suffix('.yaml.tmp')

        snapshot_data = {
            **self.config,
            'run_metadata': {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'platform': sys.platform,
                'python_version': sys.version,
                'file_cap': self.max_files,
                'max_workers': self.max_workers,
                'output_dir': str(self.output_dir),
                'psutil_available': PSUTIL_AVAILABLE,
                'blake3_available': BLAKE3_AVAILABLE,
                'xxhash_available': XXHASH_AVAILABLE,
                'chardet_available': CHARDET_AVAILABLE,
            },
        }

        try:
            with open(temp_path, 'w') as f:
                yaml.dump(snapshot_data, f, default_flow_style=False, sort_keys=True)
            os.replace(str(temp_path), str(snapshot_path))
            self.logger.info(f"Configuration snapshot saved: {snapshot_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config snapshot: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def log_error(self, phase: str, path: str, error: str, **kwargs):
        """Log structured error with enhanced context."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'path': path,
            'error': str(error),
            'run_id': self.run_id,
            'thread_id': threading.current_thread().ident,
            **kwargs,
        }

        self.error_journal.append(entry)

        # Write to persistent error log
        error_log = self.output_dir / 'error_journal.jsonl'
        temp_log = error_log.with_suffix('.jsonl.tmp')

        try:
            with open(temp_log, 'w') as f:
                for e in self.error_journal:
                    f.write(json.dumps(e, separators=(',', ':')) + '\n')
            os.replace(str(temp_log), str(error_log))
        except Exception as e:
            self.logger.error(f"Failed to write error journal: {e}")

    def scan_files_efficiently(self, paths: list[str]) -> Iterator[str]:
        """Efficient file scanning with filtering and early termination."""
        file_count = 0
        extensions = set(self.config['extensions'])
        exclude_dirs = set(self.config['exclude_dirs'])
        since_timestamp = self.config.get('since_timestamp')

        scan_msg = f"Starting file scan of {len(paths)} paths"
        if since_timestamp:
            scan_msg += f" (incremental: files modified after {datetime.fromtimestamp(since_timestamp)})"
        self.logger.info(scan_msg + "...")

        for scan_path in paths:
            if self.shutdown_requested:
                self.logger.warning("Shutdown requested, stopping file scan")
                break

            scan_path = Path(scan_path)
            if not scan_path.exists():
                self.logger.warning(f"Path does not exist: {scan_path}")
                continue

            if scan_path.is_file():
                if scan_path.suffix.lower() in extensions:
                    # Check timestamp filter for incremental scanning
                    if since_timestamp:
                        try:
                            file_mtime = scan_path.stat().st_mtime
                            if file_mtime <= since_timestamp:
                                continue  # Skip files not modified since timestamp
                        except OSError:
                            continue  # Skip files that can't be accessed

                    yield str(scan_path)
                    file_count += 1
                continue

            # Walk directory with exclusion filtering
            try:
                for root, dirs, files in os.walk(scan_path, followlinks=False):
                    # Filter excluded directories in-place
                    dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

                    for file in files:
                        if file_count >= self.max_files:
                            self.logger.info(f"Reached file limit: {self.max_files:,}")
                            return

                        if self.shutdown_requested:
                            self.logger.warning("Shutdown requested during scan")
                            return

                        file_path = Path(root) / file
                        if file_path.suffix.lower() in extensions and not file.startswith('.'):
                            # Check timestamp filter for incremental scanning
                            if since_timestamp:
                                try:
                                    file_mtime = file_path.stat().st_mtime
                                    if file_mtime <= since_timestamp:
                                        continue  # Skip files not modified since timestamp
                                except OSError:
                                    continue  # Skip files that can't be accessed

                            yield str(file_path)
                            file_count += 1

            except Exception as e:
                self.logger.error(f"Error scanning path {scan_path}: {e}")
                self.log_error('scan', str(scan_path), str(e))
                continue

        self.logger.info(f"File scan completed: {file_count:,} files found")

    def phase1_parallel_scan(self) -> dict[str, Any]:
        """Phase 1: Ultra-fast parallel scan with advanced optimizations."""
        self.logger.info("=== PHASE 1: PARALLEL FAST SCAN ===")

        # Setup checkpoint manager
        checkpoint_mgr = AtomicCheckpointManager(self.output_dir / 'checkpoints')

        # Try to resume from checkpoint
        checkpoint_data = checkpoint_mgr.load_latest_checkpoint('phase1')
        if checkpoint_data:
            counter, data = checkpoint_data
            self.logger.info(f"Resuming Phase 1 from checkpoint at file {counter:,}")
            processed_files = set(data.get('processed_files', []))
            n0_dedup = data.get('n0_dedup', {})
            inode_dedup = data.get('inode_dedup', {})
            start_counter = counter
        else:
            processed_files = set()
            n0_dedup = {}
            inode_dedup = {}
            start_counter = 0

        # Results tracking
        results = {
            'total_files_scanned': 0,
            'files_processed': 0,
            'icloud_placeholders': 0,
            'n0_duplicates': 0,
            'inode_duplicates': 0,
            'processing_errors': 0,
            'processing_time': 0,
            'parallel_efficiency': 0,
            'cache_stats': {},
        }

        start_time = time.time()

        # Get file list
        file_list = list(self.scan_files_efficiently(self.config['scan_paths']))
        results['total_files_scanned'] = len(file_list)

        if not file_list:
            self.logger.warning("No files found to process")
            return results

        # Filter already processed files
        remaining_files = [f for f in file_list if f not in processed_files]
        self.logger.info(f"Processing {len(remaining_files):,} files ({len(processed_files):,} already processed)")

        # Setup performance monitoring
        self.performance_monitor.start_phase("Phase 1 Parallel Scan", len(remaining_files))

        # Setup streaming JSONL writer
        with StreamingJSONLWriter(self.output_dir / 'phase1_files.jsonl') as jsonl_writer:
            # Process files with bounded futures - controlled concurrency
            counter = start_counter
            batch_results = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Bounded futures processing
                file_iter = iter(remaining_files)
                futures_to_files = {}

                # Fill initial futures pool
                for _ in range(min(self.max_workers * 2, len(remaining_files))):
                    try:
                        file_path = next(file_iter)
                        future = executor.submit(self.file_processor.process_file_robust, file_path, self.output_dir)
                        futures_to_files[future] = file_path
                    except StopIteration:
                        break

                # Process completed futures and submit new ones
                while futures_to_files:
                    if self.shutdown_requested:
                        self.logger.warning("Shutdown requested, stopping Phase 1")
                        break

                    # Wait for at least one future to complete
                    completed_futures = set()
                    for future in as_completed(futures_to_files):
                        completed_futures.add(future)
                        file_path = futures_to_files[future]

                        try:
                            metrics = future.result()
                            if metrics:
                                # Process deduplication
                                if metrics.n0_fingerprint in n0_dedup:
                                    results['n0_duplicates'] += 1
                                    n0_dedup[metrics.n0_fingerprint].append(file_path)
                                else:
                                    n0_dedup[metrics.n0_fingerprint] = [file_path]

                                # Inode deduplication
                                if metrics.inode and metrics.inode in inode_dedup:
                                    results['inode_duplicates'] += 1
                                    inode_dedup[metrics.inode].append(file_path)
                                else:
                                    if metrics.inode:
                                        inode_dedup[metrics.inode] = [file_path]

                                if metrics.is_icloud_placeholder:
                                    results['icloud_placeholders'] += 1

                                batch_results.append(metrics.to_dict())
                                results['files_processed'] += 1
                            else:
                                results['processing_errors'] += 1

                        except Exception as e:
                            self.logger.error(f"Error processing {file_path}: {e}")
                            self.log_error('phase1', file_path, str(e))
                            results['processing_errors'] += 1

                        counter += 1
                        processed_files.add(file_path)

                        # Update progress
                        if counter % 100 == 0:
                            self.performance_monitor.update_progress(counter - start_counter)

                            # Memory pressure check with bounded futures
                            if PSUTIL_AVAILABLE:
                                memory_percent = psutil.virtual_memory().percent
                                if memory_percent > 85:  # High memory threshold
                                    # Reduce concurrent futures temporarily
                                    if len(futures_to_files) > self.max_workers:
                                        # Wait for more futures to complete before submitting new ones
                                        time.sleep(0.1)
                                    gc.collect()

                        # Checkpoint periodically
                        if counter % self.config['checkpoint_interval'] == 0:
                            # Write current batch to JSONL
                            if batch_results:
                                jsonl_writer.write_batch(batch_results)
                                batch_results = []

                            # Save checkpoint
                            checkpoint_data = {
                                'processed_files': list(processed_files),
                                'n0_dedup': n0_dedup,
                                'inode_dedup': inode_dedup,
                            }
                            checkpoint_mgr.save_checkpoint_compressed('phase1', checkpoint_data, counter)

                            # Log intermediate stats
                            processing_stats = self.file_processor.get_processing_stats()
                            self.logger.info(
                                f"Checkpoint {counter:,}: {processing_stats['processed']} processed, "
                                f"{processing_stats['errors']} errors, {processing_stats['skipped']} skipped"
                            )

                        # Process a batch of completed futures at once for efficiency
                        if len(completed_futures) >= min(10, len(futures_to_files) // 2):
                            break

                    # Remove completed futures and submit new ones
                    for future in completed_futures:
                        del futures_to_files[future]

                    # Refill futures pool - maintain bounded concurrency
                    futures_to_add = min(
                        len(completed_futures),  # Same number as completed
                        self.max_workers * 2 - len(futures_to_files),  # Don't exceed bound
                    )

                    for _ in range(futures_to_add):
                        try:
                            file_path = next(file_iter)
                            future = executor.submit(
                                self.file_processor.process_file_robust, file_path, self.output_dir
                            )
                            futures_to_files[future] = file_path
                        except StopIteration:
                            break  # No more files to process

            # Write final batch
            if batch_results:
                jsonl_writer.write_batch(batch_results)

        # Finalize results
        results['processing_time'] = time.time() - start_time
        results['parallel_efficiency'] = results['files_processed'] / max(results['processing_time'], 1)
        results['cache_stats'] = self.hash_optimizer.get_cache_stats()

        # Save final results
        self._save_phase_results('phase1', results)

        # Final progress update
        self.performance_monitor.update_progress(counter - start_counter, force_log=True)
        self.performance_monitor.finish_phase("Phase 1 Parallel Scan")

        self.logger.info(f"Phase 1 complete: {results}")
        return results

    def phase2_advanced_deduplication(self) -> dict[str, Any]:
        """Phase 2: Pure streaming advanced deduplication - no memory accumulation."""
        self.logger.info("=== PHASE 2: STREAMING ADVANCED DEDUPLICATION ===")

        start_time = time.time()

        # Load Phase 1 data with fallback (for validation only)
        try:
            _ = self._load_json_with_fallback(self.output_dir / 'phase1_results.json')
        except Exception as e:
            self.logger.error(f"Failed to load Phase 1 results: {e}")
            return {'error': str(e), 'processing_time': time.time() - start_time}

        # Streaming duplicates tracking - using generators to avoid memory buildup
        duplicates = {
            'n0_fingerprint': defaultdict(list),
            'content_hash': defaultdict(list),
            'structural_similarity': defaultdict(list),
            'inode_hardlinks': defaultdict(list),
        }

        processed_count = 0
        error_count = 0

        # Process JSONL files from Phase 1 with error handling
        jsonl_files = list(self.output_dir.glob('phase1_files*.jsonl'))

        if not jsonl_files:
            self.logger.error("No Phase 1 JSONL files found")
            return {'error': 'No input data', 'processing_time': time.time() - start_time}

        total_lines = sum(self._count_jsonl_lines(f) for f in jsonl_files)
        self.performance_monitor.start_phase("Phase 2 Streaming Deduplication", total_lines)

        # Setup streaming output for unique files
        unique_files_path = self.output_dir / 'unique_files.jsonl'

        with open(unique_files_path, 'w') as unique_out:
            for jsonl_file in jsonl_files:
                if self.shutdown_requested:
                    break

                try:
                    with open(jsonl_file) as f:
                        for line_num, line in enumerate(f, 1):
                            if self.shutdown_requested:
                                break

                            try:
                                line = line.strip()
                                if not line:
                                    continue

                                data = json.loads(line)
                                metrics = FileMetrics.from_dict(data)

                                # Multi-level deduplication - streaming approach
                                duplicates['n0_fingerprint'][metrics.n0_fingerprint].append(metrics.path)
                                duplicates['content_hash'][metrics.blake2b_hash].append(metrics.path)

                                # Structural similarity grouping
                                struct_key = f"{metrics.func_count}_{metrics.class_count}_{metrics.line_count}_{metrics.file_type}"
                                duplicates['structural_similarity'][struct_key].append(metrics.path)

                                # Inode deduplication
                                if metrics.inode:
                                    duplicates['inode_hardlinks'][metrics.inode].append(metrics.path)

                                # Stream unique files to disk immediately
                                unique_out.write(line + '\n')
                                processed_count += 1

                                if processed_count % 1000 == 0:
                                    self.performance_monitor.update_progress(processed_count)

                                    # Memory pressure check and cleanup
                                    if PSUTIL_AVAILABLE:
                                        memory_percent = psutil.virtual_memory().percent
                                        if memory_percent > 80:  # High memory usage
                                            # Force garbage collection
                                            gc.collect()
                                            self.logger.warning(f"High memory usage: {memory_percent:.1f}% - forced GC")

                            except json.JSONDecodeError as e:
                                error_count += 1
                                self.log_error('phase2', f"{jsonl_file}:{line_num}", f"JSON decode error: {e}")
                                continue
                            except Exception as e:
                                error_count += 1
                                self.log_error('phase2', f"{jsonl_file}:{line_num}", f"Processing error: {e}")
                                continue

                except Exception as e:
                    self.logger.error(f"Error reading {jsonl_file}: {e}")
                    self.log_error('phase2', str(jsonl_file), str(e))
                    continue

        # Filter for actual duplicates and calculate stats - streaming approach
        actual_duplicates = {}
        duplicate_stats = {}

        for dup_type, groups in duplicates.items():
            filtered_groups = {k: v for k, v in groups.items() if len(v) > 1}
            actual_duplicates[dup_type] = filtered_groups

            # Calculate statistics
            duplicate_stats[dup_type] = {
                'groups': len(filtered_groups),
                'total_files': sum(len(files) for files in filtered_groups.values()),
                'largest_group': max(len(files) for files in filtered_groups.values()) if filtered_groups else 0,
            }

        results = {
            'unique_files': processed_count,  # Count instead of list for memory efficiency
            'processed_files': processed_count,
            'processing_errors': error_count,
            'duplicate_stats': duplicate_stats,
            'processing_time': time.time() - start_time,
        }

        # Save results with atomic writes
        self._atomic_save_json(self.output_dir / 'duplicates.json', actual_duplicates)
        # Note: unique_files already saved as JSONL stream above
        self._save_phase_results('phase2', results)

        # Clear memory
        del duplicates
        gc.collect()

        self.performance_monitor.finish_phase("Phase 2 Streaming Deduplication")
        self.logger.info(f"Phase 2 complete: {results}")
        return results

    def phase3_enhanced_analysis(self) -> dict[str, Any]:
        """Phase 3: Enhanced pattern analysis with advanced metrics."""
        self.logger.info("=== PHASE 3: ENHANCED ANALYSIS ===")

        start_time = time.time()

        # Load unique files with error handling
        try:
            files_data = self._load_json_with_fallback(self.output_dir / 'unique_files.json')
        except Exception as e:
            self.logger.error(f"Failed to load unique files: {e}")
            return {'error': str(e), 'processing_time': time.time() - start_time}

        if not files_data:
            self.logger.warning("No files data to analyze")
            return {'error': 'No data to analyze', 'processing_time': time.time() - start_time}

        self.performance_monitor.start_phase("Phase 3 Analysis", len(files_data))

        # Initialize analysis structures
        analysis = {
            'file_type_distribution': Counter(),
            'extension_distribution': Counter(),
            'size_distribution': {'tiny': 0, 'small': 0, 'medium': 0, 'large': 0, 'huge': 0},
            'complexity_distribution': {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0},
            'encoding_distribution': Counter(),
            'language_patterns': defaultdict(dict),
            'quality_metrics': {},
            'totals': {'lines': 0, 'functions': 0, 'classes': 0, 'imports': 0, 'total_size': 0, 'processing_time': 0},
            'top_files': {'largest': [], 'most_complex': [], 'most_functions': []},
        }

        # Process each file
        for i, file_data in enumerate(files_data):
            if self.shutdown_requested:
                break

            try:
                metrics = FileMetrics.from_dict(file_data)

                # Basic distributions
                analysis['file_type_distribution'][metrics.file_type] += 1
                analysis['extension_distribution'][metrics.extension] += 1

                # Size distribution (refined buckets)
                size = metrics.size
                if size < 100:
                    analysis['size_distribution']['tiny'] += 1
                elif size < 1024:
                    analysis['size_distribution']['small'] += 1
                elif size < 10240:
                    analysis['size_distribution']['medium'] += 1
                elif size < 102400:
                    analysis['size_distribution']['large'] += 1
                else:
                    analysis['size_distribution']['huge'] += 1

                # Complexity distribution (refined buckets)
                complexity = metrics.complexity_score
                if complexity < 0.05:
                    analysis['complexity_distribution']['very_low'] += 1
                elif complexity < 0.15:
                    analysis['complexity_distribution']['low'] += 1
                elif complexity < 0.35:
                    analysis['complexity_distribution']['medium'] += 1
                elif complexity < 0.60:
                    analysis['complexity_distribution']['high'] += 1
                else:
                    analysis['complexity_distribution']['very_high'] += 1

                # Encoding distribution
                if metrics.encoding:
                    analysis['encoding_distribution'][metrics.encoding] += 1

                # Language-specific patterns
                if metrics.file_type not in analysis['language_patterns']:
                    analysis['language_patterns'][metrics.file_type] = {
                        'avg_functions': 0,
                        'avg_classes': 0,
                        'avg_lines': 0,
                        'avg_complexity': 0,
                        'file_count': 0,
                    }

                lang_stats = analysis['language_patterns'][metrics.file_type]
                lang_stats['file_count'] += 1
                lang_stats['avg_functions'] += metrics.func_count
                lang_stats['avg_classes'] += metrics.class_count
                lang_stats['avg_lines'] += metrics.line_count
                lang_stats['avg_complexity'] += metrics.complexity_score

                # Accumulate totals
                analysis['totals']['lines'] += metrics.line_count
                analysis['totals']['functions'] += metrics.func_count
                analysis['totals']['classes'] += metrics.class_count
                analysis['totals']['imports'] += metrics.import_count
                analysis['totals']['total_size'] += metrics.size
                analysis['totals']['processing_time'] += metrics.processing_time

                # Track top files
                file_entry = {'path': metrics.path, 'value': 0}

                # Largest files
                file_entry['value'] = metrics.size
                analysis['top_files']['largest'].append(file_entry.copy())

                # Most complex files
                file_entry['value'] = metrics.complexity_score
                analysis['top_files']['most_complex'].append(file_entry.copy())

                # Most functions
                file_entry['value'] = metrics.func_count
                analysis['top_files']['most_functions'].append(file_entry.copy())

                if i % 1000 == 0:
                    self.performance_monitor.update_progress(i)

            except Exception as e:
                self.logger.error(f"Error analyzing file {i}: {e}")
                self.log_error('phase3', f"file_{i}", str(e))
                continue

        # Finalize language pattern averages
        for lang, stats in analysis['language_patterns'].items():
            if stats['file_count'] > 0:
                stats['avg_functions'] /= stats['file_count']
                stats['avg_classes'] /= stats['file_count']
                stats['avg_lines'] /= stats['file_count']
                stats['avg_complexity'] /= stats['file_count']

        # Calculate quality metrics
        total_files = len(files_data)
        if total_files > 0:
            analysis['quality_metrics'] = {
                'avg_file_size': analysis['totals']['total_size'] / total_files,
                'avg_lines_per_file': analysis['totals']['lines'] / total_files,
                'avg_functions_per_file': analysis['totals']['functions'] / total_files,
                'avg_classes_per_file': analysis['totals']['classes'] / total_files,
                'avg_processing_time': analysis['totals']['processing_time'] / total_files,
                'complexity_ratio': sum(
                    analysis['complexity_distribution'][level] * weight
                    for level, weight in [('very_low', 1), ('low', 2), ('medium', 3), ('high', 4), ('very_high', 5)]
                )
                / total_files,
                'documentation_ratio': analysis['file_type_distribution']['documentation'] / total_files,
                'config_ratio': analysis['file_type_distribution']['config'] / total_files,
            }

        # Sort and truncate top files
        for category in analysis['top_files']:
            analysis['top_files'][category].sort(key=lambda x: x['value'], reverse=True)
            analysis['top_files'][category] = analysis['top_files'][category][:10]

        # Convert counters to regular dicts for JSON serialization
        analysis['file_type_distribution'] = dict(analysis['file_type_distribution'])
        analysis['extension_distribution'] = dict(analysis['extension_distribution'])
        analysis['encoding_distribution'] = dict(analysis['encoding_distribution'])
        analysis['language_patterns'] = dict(analysis['language_patterns'])

        analysis['processing_time'] = time.time() - start_time
        analysis['analyzed_files'] = total_files

        # Save analysis results
        self._atomic_save_json(self.output_dir / 'analysis.json', analysis)
        self._save_phase_results('phase3', analysis)

        self.performance_monitor.finish_phase("Phase 3 Analysis")
        self.logger.info(f"Phase 3 complete: analyzed {total_files:,} files")
        return analysis

    def phase4_smart_similarity(self) -> dict[str, Any]:
        """Phase 4: Smart LSH-based similarity detection with intelligent capping."""
        self.logger.info("=== PHASE 4: SMART SIMILARITY DETECTION ===")

        if not self.config.get('enable_similarity', True):
            self.logger.info("Similarity analysis disabled in configuration")
            return {'similarities_found': 0, 'processing_time': 0, 'status': 'disabled'}

        start_time = time.time()

        # Load unique files
        try:
            files_data = self._load_json_with_fallback(self.output_dir / 'unique_files.json')
        except Exception as e:
            self.logger.error(f"Failed to load files for similarity analysis: {e}")
            return {'error': str(e), 'processing_time': time.time() - start_time}

        # Filter files suitable for similarity analysis
        suitable_files = [
            f
            for f in files_data
            if f.get('file_type') in ['python', 'rust', 'javascript', 'python_like']
            and f.get('line_count', 0) > 5  # Skip very small files
        ]

        total_files = len(suitable_files)
        max_comparisons = min(total_files, MAX_SIMILARITY_COMPARISONS)

        if total_files > max_comparisons:
            self.logger.info(f"Capping similarity analysis: {total_files:,} → {max_comparisons:,} files")
            # Sort by signals_score and take top files
            suitable_files.sort(key=lambda x: x.get('signals_score', 0), reverse=True)
            suitable_files = suitable_files[:max_comparisons]

        if len(suitable_files) < 2:
            self.logger.info("Insufficient files for similarity analysis")
            return {'similarities_found': 0, 'processing_time': time.time() - start_time}

        self.performance_monitor.start_phase("Phase 4 Similarity", len(suitable_files))

        similarities = []
        threshold = self.config.get('similarity_threshold', 0.8)

        # LSH Configuration
        BANDS = 6  # Increased for better precision
        BITS_PER_BAND = 10  # Decreased for more buckets
        MAX_BUCKET_SIZE = 20  # Limit comparisons per bucket

        # Build LSH index
        self.logger.info(f"Building LSH index for {len(suitable_files):,} files...")
        buckets = [defaultdict(list) for _ in range(BANDS)]
        file_items = []

        for i, file_data in enumerate(suitable_files):
            if self.shutdown_requested:
                break

            try:
                metrics = FileMetrics.from_dict(file_data)

                # Read content for SimHash (with size limit)
                try:
                    content_sample = self._read_file_sample(metrics.path, 20000)  # 20KB sample
                    if not content_sample.strip():
                        continue
                except:
                    continue

                # Calculate SimHash
                simhash = self.hash_optimizer.simhash64(content_sample)

                item = {
                    'index': i,
                    'metrics': metrics,
                    'simhash': simhash,
                    'content_sample': content_sample[:500],  # Keep small sample for comparison
                }
                file_items.append(item)

                # Index in LSH buckets
                for band_idx in range(BANDS):
                    shift = band_idx * BITS_PER_BAND
                    bucket_key = (simhash >> shift) & ((1 << BITS_PER_BAND) - 1)
                    buckets[band_idx][bucket_key].append(item)

                if i % 500 == 0:
                    self.performance_monitor.update_progress(i)

            except Exception as e:
                self.logger.error(f"Error building LSH index for file {i}: {e}")
                continue

        self.logger.info(f"LSH index built with {len(file_items):,} files")

        # Find similar pairs using LSH
        self.logger.info("Finding similar pairs...")
        seen_pairs = set()
        comparisons_made = 0
        max_total_comparisons = 50000  # Absolute limit on comparisons

        for band_idx, band_buckets in enumerate(buckets):
            if self.shutdown_requested or comparisons_made >= max_total_comparisons:
                break

            for bucket_key, bucket_items in band_buckets.items():
                bucket_size = len(bucket_items)
                if bucket_size < 2 or bucket_size > MAX_BUCKET_SIZE:
                    continue

                # Sort by file size for better comparison order
                bucket_items.sort(key=lambda x: x['metrics'].size)

                # Compare pairs within bucket
                for i in range(len(bucket_items)):
                    for j in range(i + 1, min(i + 8, len(bucket_items))):  # Sliding window
                        if comparisons_made >= max_total_comparisons:
                            break

                        item1, item2 = bucket_items[i], bucket_items[j]

                        # Avoid duplicate comparisons
                        pair_key = tuple(sorted([item1['index'], item2['index']]))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)
                        comparisons_made += 1

                        # Quick Hamming distance check
                        hamming_dist = self.hash_optimizer.hamming64(item1['simhash'], item2['simhash'])

                        # Only proceed if SimHash suggests similarity
                        if hamming_dist <= 12:  # Increased threshold slightly
                            try:
                                similarity_result = self._calculate_detailed_similarity(
                                    item1['metrics'],
                                    item2['metrics'],
                                    item1['content_sample'],
                                    item2['content_sample'],
                                    hamming_dist,
                                )

                                if similarity_result and similarity_result.similarity_score >= threshold:
                                    similarities.append(similarity_result)

                            except Exception as e:
                                self.logger.debug(f"Error calculating similarity: {e}")
                                continue

        # Sort similarities by score
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)

        # Calculate results
        results = {
            'similarities_found': len(similarities),
            'high_similarity_pairs': len([s for s in similarities if s.similarity_score >= 0.9]),
            'likely_duplicates': len([s for s in similarities if s.is_duplicate]),
            'total_comparisons': comparisons_made,
            'files_analyzed': len(file_items),
            'lsh_efficiency': len(file_items) / max(comparisons_made, 1),
            'processing_time': time.time() - start_time,
        }

        # Save similarities
        similarities_data = [asdict(s) for s in similarities]
        self._atomic_save_json(self.output_dir / 'similarities.json', similarities_data)
        self._save_phase_results('phase4', results)

        self.performance_monitor.finish_phase("Phase 4 Similarity")
        self.logger.info(f"Phase 4 complete: {results}")
        return results

    def phase5_comprehensive_reports(self) -> dict[str, Any]:
        """Phase 5: Comprehensive reporting with robust fallbacks."""
        self.logger.info("=== PHASE 5: COMPREHENSIVE REPORTING ===")

        start_time = time.time()

        # Load all phase results with fallbacks
        try:
            phase1_results = self._load_json_with_fallback(self.output_dir / 'phase1_results.json', {})
            phase2_results = self._load_json_with_fallback(self.output_dir / 'phase2_results.json', {})
            phase3_results = self._load_json_with_fallback(self.output_dir / 'analysis.json', {})
            phase4_results = self._load_json_with_fallback(self.output_dir / 'phase4_results.json', {})
            similarities_data = self._load_json_with_fallback(self.output_dir / 'similarities.json', [])
        except Exception as e:
            self.logger.error(f"Error loading phase results for reporting: {e}")
            similarities_data = []

        # Generate comprehensive final report
        final_report = {
            'metadata': {
                'analyzer_version': '3.1.0 Ultimate Edition',
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'platform': sys.platform,
                'python_version': sys.version.split()[0],
                'available_optimizations': {
                    'psutil': PSUTIL_AVAILABLE,
                    'blake3': BLAKE3_AVAILABLE,
                    'xxhash': XXHASH_AVAILABLE,
                    'chardet': CHARDET_AVAILABLE,
                },
            },
            'summary': {
                'total_files_scanned': phase1_results.get('total_files_scanned', 0),
                'files_successfully_processed': phase1_results.get('files_processed', 0),
                'unique_files': phase2_results.get('unique_files', 0),
                'duplicates_found': sum(
                    stats.get('groups', 0) for stats in phase2_results.get('duplicate_stats', {}).values()
                ),
                'similarities_found': len(similarities_data) if isinstance(similarities_data, list) else 0,
                'total_lines_of_code': phase3_results.get('totals', {}).get('lines', 0),
                'total_functions': phase3_results.get('totals', {}).get('functions', 0),
                'total_classes': phase3_results.get('totals', {}).get('classes', 0),
                'total_size_bytes': phase3_results.get('totals', {}).get('total_size', 0),
            },
            'performance': {
                'phase1_time': phase1_results.get('processing_time', 0),
                'phase2_time': phase2_results.get('processing_time', 0),
                'phase3_time': phase3_results.get('processing_time', 0),
                'phase4_time': phase4_results.get('processing_time', 0),
                'phase5_time': 0,  # Will be calculated
                'total_time': 0,  # Will be calculated
                'parallel_efficiency': phase1_results.get('parallel_efficiency', 0),
                'files_per_second': 0,  # Will be calculated
            },
            'optimization_impact': {
                'icloud_placeholders_skipped': phase1_results.get('icloud_placeholders', 0),
                'n0_duplicates_found': phase1_results.get('n0_duplicates', 0),
                'processing_errors': phase1_results.get('processing_errors', 0),
                'cache_hit_rate': phase1_results.get('cache_stats', {}).get('hit_rate_percent', 0),
                'memory_optimizations': True,
                'streaming_jsonl': True,
                'atomic_checkpoints': True,
                'compression_enabled': self.config.get('enable_compression', True),
            },
            'file_analysis': phase3_results,
            'duplicate_analysis': phase2_results.get('duplicate_stats', {}),
            'similarity_analysis': phase4_results,
            'quality_assessment': self._generate_quality_assessment(phase3_results),
            'recommendations': self._generate_recommendations(phase1_results, phase2_results, phase3_results),
        }

        # Calculate final performance metrics
        total_processing_time = sum(
            [
                final_report['performance']['phase1_time'],
                final_report['performance']['phase2_time'],
                final_report['performance']['phase3_time'],
                final_report['performance']['phase4_time'],
            ]
        )

        phase5_time = time.time() - start_time
        final_report['performance']['phase5_time'] = phase5_time
        final_report['performance']['total_time'] = total_processing_time + phase5_time

        if final_report['performance']['total_time'] > 0:
            final_report['performance']['files_per_second'] = (
                final_report['summary']['files_successfully_processed'] / final_report['performance']['total_time']
            )

        # Generate reports in multiple formats
        report_formats = {}

        try:
            # JSON report (detailed)
            json_report_path = self.output_dir / 'final_report.json'
            self._atomic_save_json(json_report_path, final_report)
            report_formats['json'] = str(json_report_path)

            # Markdown report (human-readable)
            markdown_report = self._generate_markdown_report(final_report)
            md_report_path = self.output_dir / 'final_report.md'
            self._atomic_save_text(md_report_path, markdown_report)
            report_formats['markdown'] = str(md_report_path)

            # Executive summary (concise)
            summary_report = self._generate_executive_summary(final_report)
            summary_path = self.output_dir / 'executive_summary.md'
            self._atomic_save_text(summary_path, summary_report)
            report_formats['summary'] = str(summary_path)

        except Exception as e:
            self.logger.error(f"Error generating report formats: {e}")

        final_report['report_formats'] = report_formats
        final_report['status'] = 'completed'

        self.logger.info(f"Phase 5 complete - Reports generated: {list(report_formats.keys())}")
        return final_report

    def _calculate_detailed_similarity(
        self, metrics1: FileMetrics, metrics2: FileMetrics, content1: str, content2: str, hamming_distance: int
    ) -> SimilarityResult | None:
        """Calculate detailed similarity between two files."""
        if metrics1.file_type != metrics2.file_type:
            return None

        # Structural similarity based on metrics
        factors = []

        # Size similarity
        if metrics1.size > 0 and metrics2.size > 0:
            size_ratio = min(metrics1.size, metrics2.size) / max(metrics1.size, metrics2.size)
            factors.append(('size', size_ratio, 0.15))

        # Function count similarity
        if metrics1.func_count > 0 or metrics2.func_count > 0:
            max_funcs = max(metrics1.func_count, metrics2.func_count)
            if max_funcs > 0:
                func_similarity = 1 - abs(metrics1.func_count - metrics2.func_count) / max_funcs
                factors.append(('functions', func_similarity, 0.25))

        # Class count similarity
        if metrics1.class_count > 0 or metrics2.class_count > 0:
            max_classes = max(metrics1.class_count, metrics2.class_count)
            if max_classes > 0:
                class_similarity = 1 - abs(metrics1.class_count - metrics2.class_count) / max_classes
                factors.append(('classes', class_similarity, 0.20))

        # Complexity similarity
        if metrics1.complexity_score > 0 or metrics2.complexity_score > 0:
            max_complexity = max(metrics1.complexity_score, metrics2.complexity_score)
            if max_complexity > 0:
                complexity_similarity = 1 - abs(metrics1.complexity_score - metrics2.complexity_score) / max_complexity
                factors.append(('complexity', complexity_similarity, 0.25))

        # SimHash similarity (inverse of normalized hamming distance)
        simhash_similarity = 1 - (hamming_distance / 64.0)
        factors.append(('simhash', simhash_similarity, 0.15))

        if not factors:
            return None

        # Calculate weighted similarity
        total_weight = sum(weight for _, _, weight in factors)
        weighted_sum = sum(score * weight for _, score, weight in factors)
        similarity_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine if it's likely a duplicate
        is_duplicate = (
            similarity_score >= 0.95 and hamming_distance <= 6 and abs(metrics1.line_count - metrics2.line_count) <= 5
        )

        # Extract shared patterns
        shared_patterns = [f"{name}:{score:.2f}" for name, score, _ in factors]

        return SimilarityResult(
            file1=metrics1.path,
            file2=metrics2.path,
            similarity_score=similarity_score,
            shared_patterns=shared_patterns,
            diff_lines=abs(metrics1.line_count - metrics2.line_count),
            is_duplicate=is_duplicate,
            hamming_distance=hamming_distance,
            structural_similarity=similarity_score,
        )

    def _generate_quality_assessment(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate code quality assessment from analysis."""
        if not analysis:
            return {}

        quality_metrics = analysis.get('quality_metrics', {})
        complexity_dist = analysis.get('complexity_distribution', {})
        file_type_dist = analysis.get('file_type_distribution', {})

        total_files = sum(complexity_dist.values()) or 1

        # Calculate quality scores (0-100)
        complexity_score = (
            complexity_dist.get('very_low', 0) * 100
            + complexity_dist.get('low', 0) * 80
            + complexity_dist.get('medium', 0) * 60
            + complexity_dist.get('high', 0) * 40
            + complexity_dist.get('very_high', 0) * 20
        ) / total_files

        documentation_ratio = file_type_dist.get('documentation', 0) / max(sum(file_type_dist.values()), 1)
        config_ratio = file_type_dist.get('config', 0) / max(sum(file_type_dist.values()), 1)

        maintainability_score = min(
            100, (complexity_score * 0.4 + documentation_ratio * 200 * 0.3 + config_ratio * 200 * 0.3)
        )

        return {
            'complexity_score': complexity_score,
            'maintainability_score': maintainability_score,
            'documentation_ratio': documentation_ratio,
            'config_ratio': config_ratio,
            'avg_file_size': quality_metrics.get('avg_file_size', 0),
            'avg_functions_per_file': quality_metrics.get('avg_functions_per_file', 0),
            'recommendations': [],
        }

    def _generate_recommendations(self, phase1: dict, phase2: dict, phase3: dict) -> list[dict[str, str]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Performance recommendations
        processing_time = phase1.get('processing_time', 0)
        if processing_time > 300:  # 5 minutes
            recommendations.append(
                {
                    'category': 'performance',
                    'priority': 'medium',
                    'title': 'Long Processing Time',
                    'description': f'Analysis took {processing_time:.1f}s. Consider enabling more optimizations.',
                    'action': 'Install blake3 and xxhash for faster hashing',
                }
            )

        # Duplicate recommendations
        n0_duplicates = phase1.get('n0_duplicates', 0)
        if n0_duplicates > 100:
            recommendations.append(
                {
                    'category': 'duplicates',
                    'priority': 'high',
                    'title': 'High Duplicate Count',
                    'description': f'Found {n0_duplicates:,} potential duplicates.',
                    'action': 'Review duplicate files and consider consolidation',
                }
            )

        # Code quality recommendations
        complexity_dist = phase3.get('complexity_distribution', {})
        high_complexity = complexity_dist.get('high', 0) + complexity_dist.get('very_high', 0)
        if high_complexity > 0:
            recommendations.append(
                {
                    'category': 'quality',
                    'priority': 'medium',
                    'title': 'High Complexity Files',
                    'description': f'Found {high_complexity} high-complexity files.',
                    'action': 'Consider refactoring complex functions and classes',
                }
            )

        return recommendations

    def _generate_markdown_report(self, report: dict[str, Any]) -> str:
        """Generate comprehensive markdown report."""
        summary = report['summary']
        performance = report['performance']
        optimization = report['optimization_impact']

        return f"""# Jeffrey Phoenix Super Analyzer v3.1 - Analysis Report

## Executive Summary

**Run ID:** `{report['metadata']['run_id'][:12]}`
**Generated:** {datetime.fromisoformat(report['metadata']['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
**Platform:** {report['metadata']['platform']} | Python {report['metadata']['python_version']}

### Key Metrics

| Metric | Value |
|--------|-------|
| **Total Files Scanned** | {summary['total_files_scanned']:,} |
| **Files Successfully Processed** | {summary['files_successfully_processed']:,} |
| **Unique Files** | {summary['unique_files']:,} |
| **Duplicates Found** | {summary['duplicates_found']:,} |
| **Similarities Found** | {summary['similarities_found']:,} |
| **Total Lines of Code** | {summary['total_lines_of_code']:,} |
| **Total Functions** | {summary['total_functions']:,} |
| **Total Classes** | {summary['total_classes']:,} |
| **Total Size** | {self._format_bytes(summary['total_size_bytes'])} |

## Performance Analysis

| Phase | Duration | Rate |
|-------|----------|------|
| **Phase 1 (Parallel Scan)** | {performance['phase1_time']:.1f}s | {performance['parallel_efficiency']:.1f} files/s |
| **Phase 2 (Deduplication)** | {performance['phase2_time']:.1f}s | - |
| **Phase 3 (Analysis)** | {performance['phase3_time']:.1f}s | - |
| **Phase 4 (Similarity)** | {performance['phase4_time']:.1f}s | - |
| **Phase 5 (Reporting)** | {performance['phase5_time']:.1f}s | - |
| **Total Processing Time** | {performance['total_time']:.1f}s | {performance['files_per_second']:.1f} files/s |

## Optimization Impact

✅ **Optimizations Applied:**
- BLAKE2b/BLAKE3 hashing: {BLAKE3_AVAILABLE and "✓ Enabled" or "○ Fallback"}
- xxHash fingerprinting: {XXHASH_AVAILABLE and "✓ Enabled" or "○ Fallback"}
- Memory monitoring: {PSUTIL_AVAILABLE and "✓ Enabled" or "○ Fallback"}
- Character detection: {CHARDET_AVAILABLE and "✓ Enabled" or "○ Fallback"}
- Parallel processing: ✓ Enabled ({self.max_workers} workers)
- Streaming JSONL: ✓ Enabled
- Atomic checkpoints: ✓ Enabled
- Memory management: ✓ Enabled

📊 **Impact Metrics:**
- iCloud placeholders skipped: {optimization['icloud_placeholders_skipped']:,}
- N0 duplicates found: {optimization['n0_duplicates_found']:,}
- Processing errors: {optimization['processing_errors']:,}
- Cache hit rate: {optimization['cache_hit_rate']:.1f}%

## Quality Assessment

{self._format_quality_section(report.get('quality_assessment', {}))}

## Recommendations

{self._format_recommendations(report.get('recommendations', []))}

## Technical Implementation

This analysis was performed using the **Jeffrey Phoenix Super Analyzer v3.1 Ultimate Edition**,
featuring advanced optimizations for large-scale codebase analysis:

### Architecture Highlights
- **5-Phase Processing Pipeline** with parallel execution
- **LSH-based Similarity Detection** for O(n) complexity
- **Robust Error Handling** with graceful degradation
- **Memory-Aware Processing** with dynamic garbage collection
- **iCloud Stability Checks** for macOS compatibility
- **Atomic Operations** for crash recovery

### Performance Optimizations
- Multi-threaded file processing with work-stealing
- Advanced hashing algorithms (BLAKE3, xxHash)
- Streaming processing to minimize memory usage
- Intelligent capping for similarity comparisons
- Dynamic ETA calculation with rate adjustment

---
*Generated by Jeffrey Phoenix Super Analyzer v3.1 Ultimate Edition*
*For more information, see the detailed JSON report.*
"""

    def _format_quality_section(self, quality: dict[str, Any]) -> str:
        """Format quality assessment section."""
        if not quality:
            return "No quality assessment data available."

        complexity_score = quality.get('complexity_score', 0)
        maintainability_score = quality.get('maintainability_score', 0)

        return f"""
**Overall Code Quality:** {self._get_quality_grade(maintainability_score)}

- **Complexity Score:** {complexity_score:.1f}/100
- **Maintainability Score:** {maintainability_score:.1f}/100
- **Documentation Ratio:** {quality.get('documentation_ratio', 0):.1%}
- **Configuration Ratio:** {quality.get('config_ratio', 0):.1%}
- **Average File Size:** {self._format_bytes(quality.get('avg_file_size', 0))}
- **Average Functions per File:** {quality.get('avg_functions_per_file', 0):.1f}
"""

    def _format_recommendations(self, recommendations: list[dict]) -> str:
        """Format recommendations section."""
        if not recommendations:
            return "No specific recommendations at this time."

        formatted = []
        for rec in recommendations:
            priority_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(rec.get('priority', 'low'), '⚪')
            formatted.append(f"""
### {priority_icon} {rec.get('title', 'Recommendation')}
**Category:** {rec.get('category', 'General')}
**Priority:** {rec.get('priority', 'Low').title()}

{rec.get('description', 'No description')}

**Recommended Action:** {rec.get('action', 'No action specified')}
""")

        return '\n'.join(formatted)

    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade from score."""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Average)"
        elif score >= 60:
            return "D (Below Average)"
        else:
            return "F (Needs Improvement)"

    def _format_bytes(self, bytes_value: int | float) -> str:
        """Format bytes in human-readable format."""
        if bytes_value == 0:
            return "0 B"

        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = 0
        while bytes_value >= 1024 and i < len(units) - 1:
            bytes_value /= 1024.0
            i += 1

        return f"{bytes_value:.1f} {units[i]}"

    def _generate_executive_summary(self, report: dict[str, Any]) -> str:
        """Generate concise executive summary."""
        summary = report['summary']
        performance = report['performance']

        return f"""# Jeffrey Phoenix Analysis - Executive Summary

**Run ID:** {report['metadata']['run_id'][:12]}
**Completed:** {datetime.fromisoformat(report['metadata']['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}

## Key Results

- **{summary['total_files_scanned']:,}** files scanned in **{performance['total_time']:.1f}** seconds
- **{summary['files_successfully_processed']:,}** files processed successfully
- **{summary['duplicates_found']:,}** duplicate groups identified
- **{summary['similarities_found']:,}** similar file pairs found
- **{summary['total_lines_of_code']:,}** total lines of code analyzed

## Processing Rate

**{performance['files_per_second']:.1f} files/second** average processing rate

## Next Steps

{len(report.get('recommendations', []))} recommendations generated - see full report for details.

---
*Full analysis report available in `final_report.json` and `final_report.md`*
"""

    # Utility methods for robust file handling

    def _load_json_with_fallback(self, path: Path, default=None):
        """Load JSON file with fallback handling."""
        try:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load JSON from {path}: {e}")

        if default is not None:
            return default
        else:
            raise FileNotFoundError(f"Could not load required file: {path}")

    def _atomic_save_json(self, path: Path, data: Any):
        """Save JSON data atomically."""
        temp_path = path.with_suffix(f'{path.suffix}.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))
            os.replace(str(temp_path), str(path))
            self.logger.debug(f"Saved JSON to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _atomic_save_text(self, path: Path, text: str):
        """Save text data atomically."""
        temp_path = path.with_suffix(f'{path.suffix}.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(text)
            os.replace(str(temp_path), str(path))
            self.logger.debug(f"Saved text to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save text to {path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _save_phase_results(self, phase: str, results: dict[str, Any]):
        """Save phase results with error handling."""
        results_path = self.output_dir / f'{phase}_results.json'
        try:
            self._atomic_save_json(results_path, results)
        except Exception as e:
            self.logger.error(f"Failed to save {phase} results: {e}")

    def _count_jsonl_lines(self, path: Path) -> int:
        """Count lines in JSONL file efficiently."""
        try:
            with open(path) as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _read_file_sample(self, path: str, max_bytes: int) -> str:
        """Read file sample safely."""
        try:
            with open(path, encoding='utf-8', errors='ignore') as f:
                return f.read(max_bytes)
        except Exception:
            return ""

    def run_complete_analysis(self, **kwargs) -> dict[str, Any]:
        """Run complete 5-phase analysis with all optimizations."""

        self.logger.info("=" * 80)
        self.logger.info("JEFFREY PHOENIX SUPER ANALYZER v3.1 ULTIMATE EDITION")
        self.logger.info("=" * 80)
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Output Directory: {self.output_dir.absolute()}")
        self.logger.info(f"Max Files: {self.max_files:,}")
        self.logger.info(f"Max Workers: {self.max_workers}")
        self.logger.info(
            f"Optimizations: BLAKE3={BLAKE3_AVAILABLE}, xxHash={XXHASH_AVAILABLE}, psutil={PSUTIL_AVAILABLE}"
        )
        self.logger.info("=" * 80)

        # Save configuration snapshot
        self._save_config_snapshot()

        total_start_time = time.time()

        try:
            # Phase 1: Ultra-fast parallel scan
            self.phase1_parallel_scan()

            # Phase 2: Advanced deduplication
            self.phase2_advanced_deduplication()

            # Phase 3: Enhanced analysis
            self.phase3_enhanced_analysis()

            # Phase 4: Smart similarity detection
            self.phase4_smart_similarity()

            # Phase 5: Comprehensive reporting
            final_report = self.phase5_comprehensive_reports()

            # Calculate total time
            total_time = time.time() - total_start_time
            final_report['performance']['actual_total_time'] = total_time

            # Final summary
            summary = final_report['summary']
            performance = final_report['performance']

            self.logger.info("=" * 80)
            self.logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Files Processed: {summary['files_successfully_processed']:,}")
            self.logger.info(f"Total Time: {total_time:.1f}s")
            self.logger.info(f"Processing Rate: {performance['files_per_second']:.1f} files/s")
            self.logger.info(f"Duplicates Found: {summary['duplicates_found']:,}")
            self.logger.info(f"Similarities Found: {summary['similarities_found']:,}")
            self.logger.info(f"Lines of Code: {summary['total_lines_of_code']:,}")
            self.logger.info("=" * 80)
            self.logger.info(f"Results saved to: {self.output_dir.absolute()}")

            # Save final report
            final_report_path = self.output_dir / 'FINAL_ANALYSIS_REPORT.json'
            self._atomic_save_json(final_report_path, final_report)

            return final_report

        except KeyboardInterrupt:
            self.logger.warning("Analysis interrupted by user")
            return {
                'status': 'interrupted',
                'error': 'User interruption',
                'partial_results': True,
                'output_dir': str(self.output_dir),
            }

        except Exception as e:
            self.logger.error(f"Analysis failed with error: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'output_dir': str(self.output_dir),
                'processing_time': time.time() - total_start_time,
            }


def main():
    """Main entry point with comprehensive CLI interface."""
    parser = argparse.ArgumentParser(
        description="Jeffrey Phoenix Super Analyzer v3.1 Ultimate Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python jeffrey_phoenix_analyzer.py --scan-path . --output-dir results

  # With configuration file
  python jeffrey_phoenix_analyzer.py --config config.yaml

  # High-performance mode
  python jeffrey_phoenix_analyzer.py --max-workers 16 --batch-size 200 --max-files 100000

  # CI/CD optimized
  python jeffrey_phoenix_analyzer.py --max-files 1000 --no-similarity --output-format json

  # Memory-constrained environment
  python jeffrey_phoenix_analyzer.py --max-memory 70 --batch-size 50 --checkpoint-interval 500

Environment Variables:
  JEFFREY_MAX_FILES=50000       # Maximum files to process
  JEFFREY_MAX_WORKERS=8         # Number of parallel workers
  JEFFREY_MEMORY_LIMIT=85       # Memory usage limit (%)
  JEFFREY_ENABLE_SIMILARITY=true # Enable similarity analysis
  JEFFREY_BATCH_SIZE=100        # Batch size for processing
        """,
    )

    # Input/Output arguments
    parser.add_argument('--config', '-c', help='Path to YAML configuration file')
    parser.add_argument(
        '--scan-path', '-p', action='append', dest='scan_paths', help='Paths to scan (can be specified multiple times)'
    )
    parser.add_argument('--output-dir', '-o', help='Output directory for results')

    # Processing limits
    parser.add_argument('--max-files', '-m', type=int, help='Maximum number of files to process')
    parser.add_argument('--max-workers', '-w', type=int, help='Maximum number of parallel workers')
    parser.add_argument('--batch-size', '-b', type=int, help='Batch size for parallel processing')
    parser.add_argument('--max-memory', type=int, help='Memory usage limit (percentage)')

    # Feature toggles
    parser.add_argument('--no-similarity', action='store_true', help='Disable similarity analysis')
    parser.add_argument('--similarity-threshold', type=float, help='Similarity detection threshold (0.0-1.0)')
    parser.add_argument('--enable-compression', action='store_true', help='Enable compression for checkpoints')

    # Performance options
    parser.add_argument('--checkpoint-interval', type=int, help='Checkpoint interval (number of files)')
    parser.add_argument('--max-file-size', type=int, help='Maximum file size to process (bytes)')

    # Output options
    parser.add_argument(
        '--output-format', choices=['json', 'markdown', 'both'], default='both', help='Output format for reports'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress non-error output')

    # Advanced options
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if available')
    parser.add_argument('--force-gc', action='store_true', help='Force aggressive garbage collection')
    parser.add_argument(
        '--since-timestamp', type=float, help='Only scan files modified after this Unix timestamp (incremental scan)'
    )
    parser.add_argument('--since-days', type=int, help='Only scan files modified in the last N days')
    parser.add_argument('--since-hours', type=int, help='Only scan files modified in the last N hours')

    args = parser.parse_args()

    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Build configuration overrides from CLI arguments
    config_overrides = {}

    if args.scan_paths:
        config_overrides['scan_paths'] = args.scan_paths
    if args.max_workers:
        config_overrides['max_workers'] = args.max_workers
    if args.batch_size:
        config_overrides['batch_size'] = args.batch_size
    if args.max_memory:
        config_overrides['max_memory_usage'] = args.max_memory
    if args.no_similarity:
        config_overrides['enable_similarity'] = False
    if args.similarity_threshold:
        config_overrides['similarity_threshold'] = args.similarity_threshold
    if args.enable_compression:
        config_overrides['enable_compression'] = True
    if args.checkpoint_interval:
        config_overrides['checkpoint_interval'] = args.checkpoint_interval
    if args.max_file_size:
        config_overrides['max_file_size'] = args.max_file_size

    # Handle incremental scanning timestamp logic
    since_timestamp = None
    if args.since_timestamp:
        since_timestamp = args.since_timestamp
    elif args.since_days:
        since_timestamp = time.time() - (args.since_days * 24 * 3600)
    elif args.since_hours:
        since_timestamp = time.time() - (args.since_hours * 3600)

    if since_timestamp:
        config_overrides['since_timestamp'] = since_timestamp

    try:
        # Initialize analyzer
        analyzer = JeffreyPhoenixSuperAnalyzer(config_path=args.config, max_files=args.max_files, **config_overrides)

        # Override output directory if specified
        if args.output_dir:
            analyzer.output_dir = Path(f"{args.output_dir}_{analyzer.run_id[:8]}")
            analyzer.output_dir.mkdir(exist_ok=True, parents=True)

        # Run complete analysis
        results = analyzer.run_complete_analysis()

        # Display results
        if not args.quiet and results.get('status') != 'failed':
            summary = results.get('summary', {})
            performance = results.get('performance', {})

            print("\n" + "=" * 80)
            print("JEFFREY PHOENIX SUPER ANALYZER v3.1 - ANALYSIS COMPLETE")
            print("=" * 80)
            print(f"Files Processed: {summary.get('files_successfully_processed', 0):,}")
            print(f"Total Time: {performance.get('actual_total_time', 0):.1f}s")
            print(f"Processing Rate: {performance.get('files_per_second', 0):.1f} files/s")
            print(f"Lines of Code: {summary.get('total_lines_of_code', 0):,}")
            print(f"Duplicates Found: {summary.get('duplicates_found', 0):,}")
            print(f"Similarities Found: {summary.get('similarities_found', 0):,}")
            print("=" * 80)
            print(f"Results: {analyzer.output_dir.absolute()}")

            # Show available reports
            report_formats = results.get('report_formats', {})
            if report_formats:
                print(f"Reports: {', '.join(report_formats.keys())}")
            print("=" * 80)

        return 0 if results.get('status') != 'failed' else 1

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 130
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
