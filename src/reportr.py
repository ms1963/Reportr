#!/usr/bin/env python3
"""
Reportr - Agentic Research System - Complete Production Version
===============================================================

Full-featured multi-agent research system with:
- Multiple LLM providers (Ollama, OpenAI, Anthropic, Local)
- Async parallel downloads and processing
- Thread-safe operations with proper locking
- RAG with ChromaDB
- Structured report management
- Interactive mode with full report display
- All search sources (arXiv, Web, Scholar, Medium)
- Comprehensive error handling
- Production-grade logging and monitoring

Author:  Michael Stal, February/2026
Version: 3.0.0
Date: 2026-02-25
License: MIT
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import sys
import logging
import traceback
import json
import asyncio
import time
import hashlib
import re
import threading
import signal
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from functools import lru_cache, wraps
from collections import defaultdict, deque
from queue import Queue, Empty
from threading import Lock, RLock, Event, Semaphore
from contextlib import contextmanager
import atexit

# Platform-specific imports
if sys.platform != 'win32':
    import fcntl  # File locking (Unix/Linux/Mac)

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================
try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: beautifulsoup4 not installed. Run: pip install beautifulsoup4")
    sys.exit(1)

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)

try:
    import arxiv
except ImportError:
    print("ERROR: arxiv not installed. Run: pip install arxiv")
    sys.exit(1)

try:
    from scholarly import scholarly
except ImportError:
    print("WARNING: scholarly not installed. Google Scholar search disabled. Run: pip install scholarly")
    scholarly = None

try:
    from ddgs import DDGS  # ✅ CORRECTED IMPORT
except ImportError:
    print("WARNING: ddgs not installed. Web search disabled. Run: pip install ddgs")
    DDGS = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    print("ERROR: chromadb not installed. Run: pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

# Optional imports with flags
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("WARNING: PyPDF2 not installed. PDF extraction disabled. Run: pip install PyPDF2")

try:
    from openai import OpenAI
    OPENAI_SUPPORT = True
except ImportError:
    OPENAI_SUPPORT = False
    print("WARNING: openai package not installed. Ollama/OpenAI support disabled. Run: pip install openai")

try:
    import anthropic
    ANTHROPIC_SUPPORT = True
except ImportError:
    ANTHROPIC_SUPPORT = False
    print("WARNING: anthropic not installed. Claude support disabled. Run: pip install anthropic")

try:
    from llama_cpp import Llama  # ✅ ADDED - Was missing
    LLAMA_CPP_SUPPORT = True
except ImportError:
    LLAMA_CPP_SUPPORT = False
    print("WARNING: llama-cpp-python not installed. Local LLM support disabled. Run: pip install llama-cpp-python")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("WARNING: tqdm not installed. Progress bars disabled. Run: pip install tqdm")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
VERSION = "3.0.0"
USER_AGENT = f"AgenticResearchSystem/{VERSION} (Python {sys.version.split()[0]})"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0

# ============================================================================
# THREAD-SAFE UTILITIES
# ============================================================================

class ThreadSafeCounter:
    """Thread-safe counter with lock."""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= amount
            return self._value
    
    @property
    def value(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class ThreadSafeSet:
    """Thread-safe set with lock."""
    
    def __init__(self):
        self._set: Set[Any] = set()
        self._lock = Lock()
    
    def add(self, item: Any) -> None:
        """Add item to set."""
        with self._lock:
            self._set.add(item)
    
    def remove(self, item: Any) -> None:
        """Remove item from set."""
        with self._lock:
            self._set.discard(item)
    
    def contains(self, item: Any) -> bool:
        """Check if item is in set."""
        with self._lock:
            return item in self._set
    
    def __len__(self) -> int:
        """Get set size."""
        with self._lock:
            return len(self._set)
    
    def to_list(self) -> List[Any]:
        """Convert to list."""
        with self._lock:
            return list(self._set)
    
    def clear(self) -> None:
        """Clear set."""
        with self._lock:
            self._set.clear()


class ThreadSafeList:
    """Thread-safe list with lock."""
    
    def __init__(self):
        self._list: List[Any] = []
        self._lock = Lock()
    
    def append(self, item: Any) -> None:
        """Append item to list."""
        with self._lock:
            self._list.append(item)
    
    def extend(self, items: List[Any]) -> None:
        """Extend list with items."""
        with self._lock:
            self._list.extend(items)
    
    def __len__(self) -> int:
        """Get list size."""
        with self._lock:
            return len(self._list)
    
    def to_list(self) -> List[Any]:
        """Get copy of list."""
        with self._lock:
            return self._list.copy()
    
    def clear(self) -> None:
        """Clear list."""
        with self._lock:
            self._list.clear()
    
    def __getitem__(self, index: int) -> Any:
        """Get item by index."""
        with self._lock:
            return self._list[index]


class ThreadSafeDict:
    """Thread-safe dictionary with lock."""
    
    def __init__(self):
        self._dict: Dict[Any, Any] = {}
        self._lock = Lock()
    
    def set(self, key: Any, value: Any) -> None:
        """Set key-value pair."""
        with self._lock:
            self._dict[key] = value
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get value by key."""
        with self._lock:
            return self._dict.get(key, default)
    
    def delete(self, key: Any) -> None:
        """Delete key."""
        with self._lock:
            self._dict.pop(key, None)
    
    def keys(self) -> List[Any]:
        """Get all keys."""
        with self._lock:
            return list(self._dict.keys())
    
    def values(self) -> List[Any]:
        """Get all values."""
        with self._lock:
            return list(self._dict.values())
    
    def items(self) -> List[Tuple[Any, Any]]:
        """Get all items."""
        with self._lock:
            return list(self._dict.items())
    
    def __len__(self) -> int:
        """Get dict size."""
        with self._lock:
            return len(self._dict)
    
    def to_dict(self) -> Dict[Any, Any]:
        """Get copy of dict."""
        with self._lock:
            return self._dict.copy()


class ProgressTracker:
    """Thread-safe progress tracker with optional tqdm."""
    
    def __init__(self, total: int, description: str = "Progress", use_tqdm: bool = True):
        self.total = total
        self.description = description
        self._completed = ThreadSafeCounter(0)
        self._failed = ThreadSafeCounter(0)
        self._lock = Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._start_time = time.time()
        
        # Initialize tqdm if available and requested
        self._pbar = None
        if use_tqdm and TQDM_AVAILABLE:
            self._pbar = tqdm(total=total, desc=description, unit="item")
    
    def update(self, success: bool = True, n: int = 1) -> None:
        """Update progress."""
        if success:
            completed = self._completed.increment(n)
        else:
            self._failed.increment(n)
            completed = self._completed.value
        
        # Update tqdm
        if self._pbar:
            self._pbar.update(n)
        
        # Log every 10% or every 5 items
        if completed % max(1, self.total // 10) == 0 or completed % 5 == 0:
            elapsed = time.time() - self._start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (self.total - completed) / rate if rate > 0 else 0
            
            self.logger.info(
                f"{self.description}: {completed}/{self.total} "
                f"({completed/self.total*100:.1f}%) "
                f"[{rate:.1f} items/s, ETA: {eta:.1f}s]"
            )
    
    def finish(self) -> None:
        """Finish progress tracking."""
        if self._pbar:
            self._pbar.close()
        
        elapsed = time.time() - self._start_time
        self.logger.info(
            f"{self.description} completed: {self._completed.value}/{self.total} successful, "
            f"{self._failed.value} failed in {elapsed:.2f}s"
        )


# ============================================================================
# FILE LOCKING UTILITIES
# ============================================================================

@contextmanager
def file_lock(filepath: Path, timeout: float = 10.0):
    """
    Context manager for file locking (Unix/Linux/Mac only).
    On Windows, this is a no-op.
    """
    if sys.platform == 'win32':
        # Windows doesn't have fcntl, just yield
        yield
        return
    
    lock_file = Path(str(filepath) + '.lock')
    lock_fd = None
    
    try:
        lock_fd = open(lock_file, 'w')
        start_time = time.time()
        
        while True:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Could not acquire lock on {filepath}")
                time.sleep(0.1)
        
        yield
        
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                lock_fd.close()
                lock_file.unlink()
            except:
                pass


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class ConfigManager:
    """Manages system configuration with validation."""
    
    DEFAULT_CONFIG = {
        'llm_config': {
            'provider': 'ollama',
            'model_name': 'mistral:latest',
            'api_base': 'http://localhost:11434/v1',
            'api_key': None,
            'temperature': 0.1,
            'max_tokens': 2048,
            'context_window': 8192,
            'fallback_providers': ['ollama', 'local']
        },
        'search_sources': [
            {
                'name': 'arxiv',
                'enabled': True,
                'max_results': 10,
                'fetch_full_text': True
            },
            {
                'name': 'web_search',
                'enabled': True,
                'max_results': 10,
                'fetch_full_text': True
            },
            {
                'name': 'google_scholar',
                'enabled': True,
                'max_results': 10,
                'fetch_full_text': False
            },
            {
                'name': 'medium',
                'enabled': True,
                'max_results': 5,
                'fetch_full_text': True
            }
        ],
        'report_config': {
            'max_items': 10,
            'relevance_threshold': 0.6,
            'deduplication_similarity': 0.85,
            'display_in_terminal': True,
            'save_to_file': True
        },
        'rag_config': {
            'vector_store': 'chromadb',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'collection_name': 'research_documents',
            'persist_directory': './data/vectorstore',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'top_k': 5,
            'batch_size': 32
        },
        'system_config': {
            'log_level': 'INFO',
            'log_file': './logs/agentic_system.log',
            'max_concurrent_searches': 4,
            'max_concurrent_downloads': 10,
            'max_concurrent_processing': 4,
            'request_timeout': 30,
            'retry_attempts': 3,
            'retry_delay': 2,
            'cache_enabled': True,
            'cache_ttl': 3600,
            'cache_dir': './cache',
            'rate_limit_calls': 10,
            'rate_limit_window': 60,
            'reports_dir': './reports',
            'scheduled_reports_dir': './reports/scheduled',
            'single_reports_dir': './reports/single'
        },
        'scheduled_tasks': []
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path or 'config.yaml'
        self._lock = RLock()
        self.config = self._load_config()
        self._validate_config()
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        with self._lock:
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r') as f:
                        user_config = yaml.safe_load(f)
                    
                    config = self._deep_merge(self.DEFAULT_CONFIG.copy(), user_config)
                    self.logger.info(f"Configuration loaded from {self.config_path}")
                    return config
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
                    self.logger.info("Using default configuration")
                    return self.DEFAULT_CONFIG.copy()
            else:
                self.logger.info(f"Config file not found at {self.config_path}, using defaults")
                self._save_default_config()
                return self.DEFAULT_CONFIG.copy()
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        required_sections = ['llm_config', 'search_sources', 'system_config']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        dirs = [
            self.config['system_config']['cache_dir'],
            self.config['system_config']['reports_dir'],
            self.config['system_config']['scheduled_reports_dir'],
            self.config['system_config']['single_reports_dir'],
            self.config['rag_config']['persist_directory'],
            os.path.dirname(self.config['system_config']['log_file'])
        ]
        
        for dir_path in dirs:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
    
    def _save_default_config(self) -> None:
        """Save default configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Default configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save default config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (thread-safe)."""
        with self._lock:
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
                if value is None:
                    return default
            return value
    
    def save(self) -> None:
        """Save current configuration to file (thread-safe)."""
        with self._lock:
            try:
                with open(self.config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                self.logger.info(f"Configuration saved to {self.config_path}")
            except Exception as e:
                self.logger.error(f"Failed to save configuration: {e}")


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_level = config.get('system_config', {}).get('log_level', 'INFO')
    log_file = config.get('system_config', {}).get('log_file', './logs/agentic_system.log')
    
    os.makedirs(os.path.dirname(log_file) or './logs', exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.WARNING)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SearchResult:
    """Represents a search result from any source."""
    title: str
    authors: List[str]
    abstract: str
    url: str
    source: str
    published_date: Optional[datetime] = None
    full_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """✅ Convert to dictionary with proper datetime handling."""
        data = asdict(self)
        if self.published_date:
            try:
                # Handle both timezone-aware and naive datetimes
                if self.published_date.tzinfo is not None:
                    # Convert to UTC and make naive
                    data['published_date'] = self.published_date.replace(tzinfo=None).isoformat()
                else:
                    data['published_date'] = self.published_date.isoformat()
            except Exception:
                data['published_date'] = str(self.published_date)
        return data
    
    def get_content(self) -> str:
        """Get the best available content."""
        return self.full_text or self.abstract or self.title


@dataclass
class ResearchReport:
    """Represents a generated research report."""
    topic: str
    query: str
    generated_at: datetime
    results: List[SearchResult]
    summary: str
    key_findings: List[str]
    trends: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    report_type: str = 'single'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'topic': self.topic,
            'query': self.query,
            'generated_at': self.generated_at.isoformat(),
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary,
            'key_findings': self.key_findings,
            'trends': self.trends,
            'metadata': self.metadata,
            'report_type': self.report_type
        }


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Represents a scheduled research task."""
    task_id: str
    topic: str
    schedule: str
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    Removes apostrophes, quotes, and special characters.
    """
    if not filename:
        return 'unnamed'
    
    # Remove all types of apostrophes and quotes
    filename = filename.replace("'", "")  # ✅ Removes apostrophes
    filename = filename.replace('"', "")
    filename = filename.replace('`', "")
    filename = filename.replace(''', "")
    filename = filename.replace(''', "")
    filename = filename.replace('"', "")
    filename = filename.replace('"', "")
    
    # Replace file system reserved characters with underscore
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Replace other special characters
    filename = re.sub(r'[^\w\s.\-]', '', filename)
    
    # Normalize whitespace
    filename = re.sub(r'\s+', ' ', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Replace multiple underscores with single underscore
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing dots, spaces, and underscores
    filename = filename.strip('. _')
    
    # Limit length to 200 characters
    if len(filename) > 200:
        filename = filename[:200].rstrip('. _')
    
    # Ensure we have a valid filename
    if not filename or filename == '_':
        return 'unnamed'
    
    return filename

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def retry_on_failure(max_attempts: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    
                    logger = logging.getLogger(func.__module__)
                    logger.warning(f"Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator


def get_requests_session(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    """Get requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': USER_AGENT})
    return session


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Thread-safe rate limiter."""
    
    def __init__(self, max_calls: int, window: int):
        self.max_calls = max_calls
        self.window = window
        self.calls = deque()
        self._lock = Lock()
    
    def acquire(self) -> bool:
        """Acquire permission to make a call (thread-safe)."""
        with self._lock:
            now = time.time()
            
            # Remove old calls outside the window
            while self.calls and now - self.calls[0] > self.window:
                self.calls.popleft()
            
            # Check if we can make a new call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit is exceeded."""
        while not self.acquire():
            time.sleep(0.1)


# ============================================================================
# CACHE
# ============================================================================

class Cache:
    """Thread-safe file-based cache with locking."""
    
    def __init__(self, cache_dir: str, ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.logger = logging.getLogger(self.__class__.__name__)
        self._locks: Dict[str, Lock] = {}
        self._locks_lock = Lock()
    
    def _get_lock(self, key: str) -> Lock:
        """Get or create lock for key."""
        with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = Lock()
            return self._locks[key]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (thread-safe)."""
        lock = self._get_lock(key)
        with lock:
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                return None
            
            try:
                # Check if cache is expired
                if time.time() - cache_path.stat().st_mtime > self.ttl:
                    cache_path.unlink()
                    return None
                
                with file_lock(cache_path):
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                        return data['value']
            except Exception as e:
                self.logger.debug(f"Cache read failed: {e}")
                return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache (thread-safe)."""
        lock = self._get_lock(key)
        with lock:
            cache_path = self._get_cache_path(key)
            
            try:
                with file_lock(cache_path):
                    with open(cache_path, 'w') as f:
                        json.dump({'value': value, 'timestamp': time.time()}, f)
            except Exception as e:
                self.logger.debug(f"Cache write failed: {e}")
    
    def clear(self) -> None:
        """Clear all cache."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.debug(f"Failed to delete cache file {cache_file}: {e}")


# ============================================================================
# LLM PROVIDERS - COMPLETE SET
# ============================================================================

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    def generate_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        """Generate with retry logic."""
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        
        raise RuntimeError("All generation attempts failed")
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if provider is healthy."""
        pass


class OllamaProvider(BaseLLMProvider):
    """✅ Ollama LLM provider using OpenAI-compatible API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not OPENAI_SUPPORT:
            raise ImportError("OpenAI package required for Ollama. Install with: pip install openai")
        
        self.model_name = config.get('model_name', 'mistral:latest')
        if ':' not in self.model_name:
            self.model_name = f"{self.model_name}:latest"
        
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 2048)
        self.api_base = config.get('api_base', 'http://localhost:11434/v1')
        
        self.client = OpenAI(
            base_url=self.api_base,
            api_key='ollama'
        )
        
        self.logger.info(f"Ollama provider initialized: {self.model_name} at {self.api_base}")
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            session = get_requests_session()
            response = session.get(f"{self.api_base.replace('/v1', '')}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                self.logger.info(f"Available Ollama models: {available_models}")
                
                if self.model_name not in available_models:
                    self.logger.warning(
                        f"Model '{self.model_name}' not found. Available: {available_models}\n"
                        f"Run: ollama pull {self.model_name.split(':')[0]}"
                    )
            
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            self.logger.info(f"✅ Successfully connected to Ollama: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            session = get_requests_session()
            response = session.get(f"{self.api_base.replace('/v1', '')}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class OpenAIProvider(BaseLLMProvider):
    """✅ OpenAI LLM provider (GPT-4, GPT-3.5)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not OPENAI_SUPPORT:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set in config or OPENAI_API_KEY env var")
        
        self.model_name = config.get('model_name', 'gpt-4-turbo-preview')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 2048)
        
        self.client = OpenAI(api_key=self.api_key)
        
        self.logger.info(f"OpenAI provider initialized: {self.model_name}")
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to OpenAI."""
        try:
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            self.logger.info(f"✅ Successfully connected to OpenAI: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to OpenAI: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if OpenAI is healthy."""
        try:
            self.client.models.list()
            return True
        except:
            return False


class AnthropicProvider(BaseLLMProvider):
    """✅ Anthropic LLM provider (Claude)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not ANTHROPIC_SUPPORT:
            raise ImportError("Anthropic package required. Install with: pip install anthropic")
        
        self.api_key = config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set in config or ANTHROPIC_API_KEY env var")
        
        self.model_name = config.get('model_name', 'claude-3-opus-20240229')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 2048)
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        self.logger.info(f"Anthropic provider initialized: {self.model_name}")
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Anthropic."""
        try:
            self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            self.logger.info(f"✅ Successfully connected to Anthropic: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Anthropic: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic Claude."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Anthropic is healthy."""
        try:
            self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except:
            return False


class LocalLLMProvider(BaseLLMProvider):
    """✅ Local LLM provider using llama.cpp."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not LLAMA_CPP_SUPPORT:
            raise ImportError("llama-cpp-python required. Install with: pip install llama-cpp-python")
        
        model_path = config.get('model_path') or config.get('model_name')
        if not model_path:
            raise ValueError("model_path required for local LLM")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 2048)
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=config.get('n_ctx', 4096),
            n_threads=config.get('n_threads', 4),
            n_gpu_layers=config.get('n_gpu_layers', 0),
            verbose=False
        )
        
        self.logger.info(f"Local LLM initialized: {model_path}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using llama.cpp."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "Human:", "User:"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            self.logger.error(f"Local LLM generation failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if local LLM is healthy."""
        try:
            self.llm("test", max_tokens=5)
            return True
        except:
            return False


class LLMFactory:
    """✅ Factory for creating LLM providers with fallback chain."""
    
    @staticmethod
    def create_provider(config: Dict[str, Any]) -> BaseLLMProvider:
        """Create LLM provider with fallback support."""
        logger = logging.getLogger('LLMFactory')
        provider_type = config.get('provider', 'ollama').lower()
        fallback_providers = config.get('fallback_providers', [])
        
        # Try primary provider
        try:
            return LLMFactory._create_single_provider(provider_type, config)
        except Exception as e:
            logger.warning(f"Primary provider '{provider_type}' failed: {e}")
            
            # Try fallback providers
            for fallback in fallback_providers:
                if fallback == provider_type:
                    continue
                
                try:
                    logger.info(f"Trying fallback provider: {fallback}")
                    return LLMFactory._create_single_provider(fallback, config)
                except Exception as e2:
                    logger.warning(f"Fallback provider '{fallback}' failed: {e2}")
            
            raise RuntimeError(f"All LLM providers failed. Tried: {provider_type}, {fallback_providers}")
    
    @staticmethod
    def _create_single_provider(provider_type: str, config: Dict[str, Any]) -> BaseLLMProvider:
        """Create a single provider instance."""
        if provider_type == 'ollama':
            return OllamaProvider(config)
        elif provider_type == 'openai':
            return OpenAIProvider(config)
        elif provider_type == 'anthropic':
            return AnthropicProvider(config)
        elif provider_type == 'local':
            return LocalLLMProvider(config)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_type}")


# ============================================================================
# SEARCH AGENTS - COMPLETE IMPLEMENTATIONS
# ============================================================================

class BaseSearchAgent(ABC):
    """Base class for search agents with async support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_results = config.get('max_results', 10)
        self.fetch_full_text = config.get('fetch_full_text', False)
        self.max_concurrent_downloads = config.get('max_concurrent_downloads', 10)
        self.timeout = config.get('request_timeout', 30)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = get_requests_session()
    
    @abstractmethod
    def search(self, query: str) -> List[SearchResult]:
        """Perform search and return results."""
        pass
    
    async def fetch_full_texts_async(self, results: List[SearchResult]) -> List[SearchResult]:
        """✅ Fetch full texts asynchronously in parallel."""
        if not self.fetch_full_text or not results:
            return results
        
        self.logger.info(f"Fetching full text for {len(results)} results asynchronously...")
        start_time = time.time()
        
        progress = ProgressTracker(len(results), f"{self.__class__.__name__} downloads", use_tqdm=False)
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                self._fetch_single_async(session, result, progress) 
                for result in results
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        progress.finish()
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r.full_text)
        self.logger.info(
            f"Async full-text fetch completed in {elapsed:.2f}s "
            f"({successful}/{len(results)} successful)"
        )
        
        return results
    
    async def _fetch_single_async(self, session: aiohttp.ClientSession, result: SearchResult, progress: ProgressTracker) -> None:
        """Fetch full text for a single result asynchronously."""
        try:
            if not result.url:
                progress.update(False)
                return
            
            if result.url.endswith('.pdf'):
                loop = asyncio.get_event_loop()
                result.full_text = await loop.run_in_executor(
                    None, 
                    self._fetch_pdf_content_sync, 
                    result.url
                )
            elif 'arxiv.org/abs/' in result.url:
                arxiv_id = result.url.split('/')[-1]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                loop = asyncio.get_event_loop()
                result.full_text = await loop.run_in_executor(
                    None, 
                    self._fetch_pdf_content_sync, 
                    pdf_url
                )
            else:
                result.full_text = await self._fetch_web_content_async(session, result.url)
            
            if result.full_text:
                result.metadata['full_text_fetched'] = True
                result.metadata['full_text_length'] = len(result.full_text)
                progress.update(True)
            else:
                progress.update(False)
        
        except Exception as e:
            self.logger.debug(f"Failed to fetch {result.url}: {e}")
            progress.update(False)
    
    async def _fetch_web_content_async(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch and extract text from web page asynchronously."""
        try:
            headers = {
                'User-Agent': USER_AGENT
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return ""
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:50000]
        
        except Exception as e:
            self.logger.debug(f"Web content extraction failed for {url}: {e}")
            return ""
    
    def _fetch_pdf_content_sync(self, url: str) -> str:
        """Fetch and extract text from PDF (synchronous)."""
        if not PDF_SUPPORT:
            return ""
        
        try:
            headers = {
                'User-Agent': USER_AGENT
            }
            
            response = self.session.get(url, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            
            from io import BytesIO
            pdf_file = BytesIO(response.content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in reader.pages[:50]:
                text += page.extract_text()
            
            return text[:100000]
        
        except Exception as e:
            self.logger.debug(f"PDF extraction failed for {url}: {e}")
            return ""
    
    def _fetch_parallel_threaded(self, results: List[SearchResult]) -> List[SearchResult]:
        """Fallback to threaded parallel fetch."""
        progress = ProgressTracker(len(results), f"{self.__class__.__name__} downloads", use_tqdm=False)
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
            futures = {
                executor.submit(self._fetch_single_sync, result): result 
                for result in results
            }
            
            for future in as_completed(futures):
                try:
                    updated_result = future.result()
                    if updated_result.full_text:
                        progress.update(True)
                    else:
                        progress.update(False)
                except Exception as e:
                    self.logger.debug(f"Download failed: {e}")
                    progress.update(False)
        
        progress.finish()
        return results
    
    def _fetch_single_sync(self, result: SearchResult) -> SearchResult:
        """Fetch single document synchronously."""
        try:
            if result.url.endswith('.pdf') or 'arxiv.org/abs/' in result.url:
                if 'arxiv.org/abs/' in result.url:
                    arxiv_id = result.url.split('/')[-1]
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                else:
                    pdf_url = result.url
                
                result.full_text = self._fetch_pdf_content_sync(pdf_url)
            else:
                result.full_text = self._fetch_web_content_sync(result.url)
            
            if result.full_text:
                result.metadata['full_text_fetched'] = True
        except Exception as e:
            self.logger.debug(f"Failed to fetch: {e}")
        
        return result
    
    def _fetch_web_content_sync(self, url: str) -> str:
        """Fetch web content synchronously."""
        try:
            headers = {
                'User-Agent': USER_AGENT
            }
            
            response = self.session.get(url, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:50000]
        
        except Exception as e:
            self.logger.debug(f"Web fetch failed for {url}: {e}")
            return ""


class ArxivSearchAgent(BaseSearchAgent):
    """✅ Agent for searching arXiv with parallel PDF downloads."""
    
    @retry_on_failure(max_attempts=3, delay=2.0)
    def search(self, query: str) -> List[SearchResult]:
        """Search arXiv and fetch full texts in parallel."""
        results = []
        
        try:
            self.logger.info(f"Searching arXiv for: {query}")
            
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                result = SearchResult(
                    title=paper.title,
                    authors=[author.name for author in paper.authors],
                    abstract=paper.summary,
                    url=paper.entry_id,
                    source='arxiv',
                    published_date=paper.published,
                    metadata={
                        'arxiv_id': paper.entry_id.split('/')[-1],
                        'pdf_url': paper.pdf_url,
                        'categories': paper.categories,
                        'doi': paper.doi
                    }
                )
                results.append(result)
            
            self.logger.info(f"Found {len(results)} arXiv results")
            
            if self.fetch_full_text and results:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        results = self._fetch_parallel_threaded(results)
                    else:
                        results = loop.run_until_complete(self.fetch_full_texts_async(results))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(self.fetch_full_texts_async(results))
                    finally:
                        loop.close()
        
        except Exception as e:
            self.logger.error(f"arXiv search failed: {e}")
            self.logger.debug(traceback.format_exc())
        
        return results


class WebSearchAgent(BaseSearchAgent):
    """✅ Agent for web search using DuckDuckGo (corrected import)."""
    
    @retry_on_failure(max_attempts=3, delay=2.0)
    def search(self, query: str) -> List[SearchResult]:
        """Search the web using DuckDuckGo."""
        results = []
        
        if not DDGS:
            self.logger.warning("DDGS not available, skipping web search")
            return results
        
        try:
            self.logger.info(f"Searching web for: {query}")
            
            try:
                ddgs = DDGS()
                search_results = ddgs.text(
                    query, 
                    max_results=self.max_results,
                    backend="api"
                )
                
                for item in search_results:
                    result = SearchResult(
                        title=item.get('title', 'No title'),
                        authors=[],
                        abstract=item.get('body', ''),
                        url=item.get('href', ''),
                        source='web_search',
                        published_date=None,
                        metadata={
                            'snippet': item.get('body', ''),
                            'backend': 'ddgs_api'
                        }
                    )
                    results.append(result)
                
                self.logger.info(f"Found {len(results)} web results using API backend")
            
            except Exception as api_error:
                self.logger.warning(f"API backend failed: {api_error}, trying lite backend")
                
                try:
                    ddgs = DDGS()
                    search_results = ddgs.text(
                        query, 
                        max_results=self.max_results,
                        backend="lite"
                    )
                    
                    for item in search_results:
                        result = SearchResult(
                            title=item.get('title', 'No title'),
                            authors=[],
                            abstract=item.get('body', ''),
                            url=item.get('href', ''),
                            source='web_search',
                            published_date=None,
                            metadata={
                                'snippet': item.get('body', ''),
                                'backend': 'ddgs_lite'
                            }
                        )
                        results.append(result)
                    
                    self.logger.info(f"Found {len(results)} web results using lite backend")
                
                except Exception as lite_error:
                    self.logger.error(f"Both API and lite backends failed: {lite_error}")
            
            if self.fetch_full_text and results:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        results = self._fetch_parallel_threaded(results)
                    else:
                        results = loop.run_until_complete(self.fetch_full_texts_async(results))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(self.fetch_full_texts_async(results))
                    finally:
                        loop.close()
        
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            self.logger.debug(traceback.format_exc())
        
        return results


class GoogleScholarAgent(BaseSearchAgent):
    """✅ Agent for searching Google Scholar - COMPLETE IMPLEMENTATION."""
    
    @retry_on_failure(max_attempts=3, delay=5.0)
    def search(self, query: str) -> List[SearchResult]:
        """Search Google Scholar."""
        results = []
        
        if not scholarly:
            self.logger.warning("scholarly not available, skipping Google Scholar search")
            return results
        
        try:
            self.logger.info(f"Searching Google Scholar for: {query}")
            
            search_query = scholarly.search_pubs(query)
            
            count = 0
            for pub in search_query:
                if count >= self.max_results:
                    break
                
                try:
                    bib = pub.get('bib', {})
                    
                    # Extract publication year
                    pub_date = None
                    if 'pub_year' in bib:
                        try:
                            pub_date = datetime(int(bib['pub_year']), 1, 1)
                        except:
                            pass
                    
                    result = SearchResult(
                        title=bib.get('title', 'No title'),
                        authors=bib.get('author', []) if isinstance(bib.get('author'), list) else [bib.get('author', 'Unknown')],
                        abstract=bib.get('abstract', ''),
                        url=pub.get('pub_url', pub.get('eprint_url', '')),
                        source='google_scholar',
                        published_date=pub_date,
                        metadata={
                            'citations': pub.get('num_citations', 0),
                            'year': bib.get('pub_year'),
                            'venue': bib.get('venue', ''),
                            'publisher': bib.get('publisher', '')
                        }
                    )
                    results.append(result)
                    count += 1
                    
                    # Rate limiting
                    time.sleep(1)
                
                except Exception as e:
                    self.logger.debug(f"Failed to parse Scholar result: {e}")
                    continue
            
            self.logger.info(f"Found {len(results)} Google Scholar results")
        
        except Exception as e:
            self.logger.error(f"Google Scholar search failed: {e}")
            self.logger.debug(traceback.format_exc())
        
        return results


class MediumSearchAgent(BaseSearchAgent):
    """✅ Agent for searching Medium articles - COMPLETE IMPLEMENTATION."""
    
    @retry_on_failure(max_attempts=3, delay=2.0)
    def search(self, query: str) -> List[SearchResult]:
        """Search Medium via web scraping."""
        results = []
        
        try:
            self.logger.info(f"Searching Medium for: {query}")
            
            search_url = f"https://medium.com/search?q={query.replace(' ', '%20')}"
            
            headers = {
                'User-Agent': USER_AGENT
            }
            
            response = self.session.get(search_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Medium uses dynamic loading, try to find article links
            articles = soup.find_all('article', limit=self.max_results * 2)  # Get more to filter
            
            if not articles:
                # Try alternative selectors
                articles = soup.find_all('div', {'class': re.compile(r'postArticle')}, limit=self.max_results * 2)
            
            for article in articles[:self.max_results]:
                try:
                    # Try to extract title
                    title_elem = article.find('h2') or article.find('h3') or article.find('h1')
                    title = title_elem.get_text().strip() if title_elem else 'No title'
                    
                    # Try to extract link
                    link_elem = article.find('a', href=True)
                    url = link_elem['href'] if link_elem else ''
                    if url and not url.startswith('http'):
                        url = f"https://medium.com{url}"
                    
                    # Try to extract snippet
                    snippet_elem = article.find('p') or article.find('div', {'class': re.compile(r'subtitle')})
                    snippet = snippet_elem.get_text().strip() if snippet_elem else ''
                    
                    # Try to extract author
                    author_elem = article.find('a', {'data-action': 'show-user-card'}) or article.find('a', {'rel': 'author'})
                    author = author_elem.get_text().strip() if author_elem else 'Unknown'
                    
                    if title != 'No title' and url:
                        result = SearchResult(
                            title=title,
                            authors=[author],
                            abstract=snippet,
                            url=url,
                            source='medium',
                            published_date=None,
                            metadata={'platform': 'medium'}
                        )
                        results.append(result)
                
                except Exception as e:
                    self.logger.debug(f"Failed to parse Medium article: {e}")
                    continue
            
            self.logger.info(f"Found {len(results)} Medium results")
            
            if self.fetch_full_text and results:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        results = self._fetch_parallel_threaded(results)
                    else:
                        results = loop.run_until_complete(self.fetch_full_texts_async(results))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(self.fetch_full_texts_async(results))
                    finally:
                        loop.close()
        
        except Exception as e:
            self.logger.error(f"Medium search failed: {e}")
            self.logger.debug(traceback.format_exc())
        
        return results


# ============================================================================
# SEARCH COORDINATOR
# ============================================================================

class SearchCoordinator:
    """✅ Coordinates searches across multiple sources in parallel with thread safety."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_concurrent_searches = config.get('system_config', {}).get('max_concurrent_searches', 4)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents = self._initialize_agents()
        self._results_lock = Lock()
    
    def _initialize_agents(self) -> Dict[str, BaseSearchAgent]:
        """Initialize search agents based on configuration."""
        agents = {}
        
        search_sources = self.config.get('search_sources', [])
        system_config = self.config.get('system_config', {})
        
        for source_config in search_sources:
            if not source_config.get('enabled', True):
                continue
            
            name = source_config['name']
            agent_config = {**system_config, **source_config}
            
            try:
                if name == 'arxiv':
                    agents[name] = ArxivSearchAgent(agent_config)
                elif name == 'web_search':
                    agents[name] = WebSearchAgent(agent_config)
                elif name == 'google_scholar':
                    agents[name] = GoogleScholarAgent(agent_config)
                elif name == 'medium':
                    agents[name] = MediumSearchAgent(agent_config)
                else:
                    self.logger.warning(f"Unknown search source: {name}")
            
            except Exception as e:
                self.logger.error(f"Failed to initialize {name} agent: {e}")
        
        self.logger.info(f"Initialized {len(agents)} search agents: {list(agents.keys())}")
        return agents
    
    def search_all(self, query: str) -> List[SearchResult]:
        """✅ Search all sources in parallel with thread-safe result aggregation."""
        if not self.agents:
            self.logger.warning("No search agents available")
            return []
        
        self.logger.info(f"Starting parallel search across {len(self.agents)} sources for: {query}")
        start_time = time.time()
        
        all_results = ThreadSafeList()
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_searches) as executor:
            future_to_agent = {
                executor.submit(self._search_single_agent, agent_name, agent, query): agent_name
                for agent_name, agent in self.agents.items()
            }
            
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    self.logger.info(f"{agent_name} returned {len(results)} results")
                except Exception as e:
                    self.logger.error(f"{agent_name} search failed: {e}")
        
        final_results = all_results.to_list()
        elapsed = time.time() - start_time
        
        self.logger.info(
            f"Parallel search completed in {elapsed:.2f}s. "
            f"Total results: {len(final_results)} from {len(self.agents)} sources"
        )
        
        return final_results
    
    def _search_single_agent(self, agent_name: str, agent: BaseSearchAgent, query: str) -> List[SearchResult]:
        """Search using a single agent."""
        try:
            return agent.search(query)
        except Exception as e:
            self.logger.error(f"{agent_name} search failed: {e}")
            return []


# ============================================================================
# RAG SYSTEM
# ============================================================================

class RAGSystem:
    """✅ RAG system with parallel document processing and thread-safe operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_config = config.get('rag_config', {})
        self.chunk_size = self.rag_config.get('chunk_size', 1000)
        self.chunk_overlap = self.rag_config.get('chunk_overlap', 200)
        self.max_concurrent_processing = config.get('system_config', {}).get('max_concurrent_processing', 4)
        self.batch_size = self.rag_config.get('batch_size', 32)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = RLock()
        
        self._initialize_vector_store()
        self._initialize_embeddings()
    
    def _initialize_vector_store(self) -> None:
        """Initialize ChromaDB vector store."""
        persist_directory = self.rag_config.get('persist_directory', './data/vectorstore')
        collection_name = self.rag_config.get('collection_name', 'research_documents')
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Vector store initialized: {collection_name} at {persist_directory}")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _initialize_embeddings(self) -> None:
        """Initialize embedding model."""
        model_name = self.rag_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Embedding model initialized: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def add_documents_parallel(self, search_results: List[SearchResult]) -> None:
        """✅ Add documents to vector store in parallel with thread-safe operations."""
        if not search_results:
            return
        
        self.logger.info(f"Processing {len(search_results)} documents in parallel...")
        start_time = time.time()
        
        progress = ProgressTracker(len(search_results), "Document processing", use_tqdm=False)
        
        all_chunks = ThreadSafeList()
        all_metadatas = ThreadSafeList()
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_processing) as executor:
            future_to_result = {
                executor.submit(self._process_single_document, result): result 
                for result in search_results
            }
            
            for future in as_completed(future_to_result):
                try:
                    chunks, metadatas = future.result()
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)
                    progress.update(True)
                except Exception as e:
                    self.logger.warning(f"Failed to process document: {e}")
                    progress.update(False)
        
        progress.finish()
        
        chunks_list = all_chunks.to_list()
        metadatas_list = all_metadatas.to_list()
        
        if chunks_list:
            self._add_chunks_in_batches(chunks_list, metadatas_list)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Document processing completed in {elapsed:.2f}s. Added {len(chunks_list)} chunks")
    
    def _process_single_document(self, result: SearchResult) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a single document into chunks."""
        chunks = []
        metadatas = []
        
        try:
            text = result.full_text or result.abstract or ""
            
            if not text:
                return chunks, metadatas
            
            doc_chunks = self._split_text(text)
            
            for i, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                metadatas.append({
                    'title': result.title,
                    'source': result.source,
                    'url': result.url,
                    'chunk_index': i,
                    'total_chunks': len(doc_chunks),
                    'authors': ', '.join(result.authors[:3])
                })
        
        except Exception as e:
            self.logger.debug(f"Failed to process document {result.title}: {e}")
        
        return chunks, metadatas
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _add_chunks_in_batches(self, chunks: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """✅ Add chunks to vector store in batches with thread-safe operations."""
        with self._lock:
            total_chunks = len(chunks)
            self.logger.info(f"Adding {total_chunks} chunks to vector store in batches of {self.batch_size}...")
            
            for i in range(0, total_chunks, self.batch_size):
                batch_chunks = chunks[i:i + self.batch_size]
                batch_metadatas = metadatas[i:i + self.batch_size]
                
                embeddings = self.embedding_model.encode(batch_chunks, show_progress_bar=False)
                
                ids = [f"chunk_{int(time.time() * 1000000)}_{j}" for j in range(len(batch_chunks))]
                
                self.collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    embeddings=embeddings.tolist(),
                    ids=ids
                )
                
                if (i + self.batch_size) % (self.batch_size * 10) == 0:
                    self.logger.info(f"Progress: {min(i + self.batch_size, total_chunks)}/{total_chunks} chunks added")
            
            self.logger.info(f"Successfully added {total_chunks} chunks to vector store")
    
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """✅ Query vector store with thread-safe operations."""
        with self._lock:
            try:
                query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
                
                results = self.collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=top_k
                )
                
                formatted_results = []
                if results['documents']:
                    for i in range(len(results['documents'][0])):
                        formatted_results.append({
                            'text': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i] if 'distances' in results else None
                        })
                
                return formatted_results
            
            except Exception as e:
                self.logger.error(f"Vector store query failed: {e}")
                return []
    
    def clear(self) -> None:
        """✅ Clear vector store with thread-safe operations."""
        with self._lock:
            try:
                self.client.delete_collection(self.collection.name)
                self.collection = self.client.create_collection(
                    name=self.collection.name,
                    metadata={"hnsw:space": "cosine"}
                )
                self.logger.info("Vector store cleared")
            except Exception as e:
                self.logger.error(f"Failed to clear vector store: {e}")



# ============================================================================
# REPORT AGENT
# ============================================================================

class ReportAgent:
    """Agent for generating research reports with fixed datetime handling."""
    
    def __init__(self, config: Dict[str, Any], llm_provider: BaseLLMProvider):
        self.config = config
        self.llm = llm_provider
        self.report_config = config.get('report_config', {})
        self.max_items = self.report_config.get('max_items', 10)
        self.relevance_threshold = self.report_config.get('relevance_threshold', 0.6)
        self.dedup_similarity = self.report_config.get('deduplication_similarity', 0.85)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_report(self, topic: str, results: List[SearchResult]) -> ResearchReport:
        """Generate a comprehensive research report."""
        self.logger.info(f"Generating report for: {topic}")
        
        # Deduplicate results
        results = self._deduplicate_results(results)
        
        # ✅ FIX: Normalize all datetimes before sorting
        results = self._normalize_datetimes(results)
        
        # Sort by relevance and date
        results = sorted(
            results,
            key=lambda x: (
                x.relevance_score,
                x.published_date if x.published_date else datetime.min.replace(tzinfo=None)
            ),
            reverse=True
        )
        
        # Limit to max items
        results = results[:self.max_items]
        
        # Generate summary and insights using LLM
        summary = self._generate_summary(topic, results)
        key_findings = self._extract_key_findings(results)
        trends = self._identify_trends(results)
        
        report = ResearchReport(
            topic=topic,
            query=topic,
            generated_at=datetime.now(),
            results=results,
            summary=summary,
            key_findings=key_findings,
            trends=trends,
            metadata={
                'total_sources': len(results),
                'sources_breakdown': self._get_sources_breakdown(results)
            }
        )
        
        self.logger.info(f"Report generated with {len(results)} results")
        return report
    
    def _normalize_datetimes(self, results: List[SearchResult]) -> List[SearchResult]:
        """✅ Normalize all datetime objects to be timezone-naive."""
        for result in results:
            if result.published_date:
                # If datetime is timezone-aware, convert to naive
                if result.published_date.tzinfo is not None:
                    result.published_date = result.published_date.replace(tzinfo=None)
        return results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on similarity."""
        if not results:
            return results
        
        unique_results = []
        seen_titles = set()
        
        for result in results:
            # Check exact title match
            if result.title.lower() in seen_titles:
                continue
            
            # Check similarity with existing results
            is_duplicate = False
            for existing in unique_results:
                similarity = calculate_similarity(result.title, existing.title)
                if similarity > self.dedup_similarity:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_titles.add(result.title.lower())
        
        removed = len(results) - len(unique_results)
        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate results")
        
        return unique_results
    
    def _generate_summary(self, topic: str, results: List[SearchResult]) -> str:
        """Generate summary using LLM."""
        if not results:
            return f"No results found for topic: {topic}"
        
        try:
            # Prepare context from results
            context = "\n\n".join([
                f"Title: {r.title}\nSource: {r.source}\nSummary: {r.abstract[:500]}"
                for r in results[:5]
            ])
            
            prompt = f"""Based on the following research results about "{topic}", provide a comprehensive summary (2-3 paragraphs):

{context}

Summary:"""
            
            summary = self.llm.generate_with_retry(prompt, max_tokens=500)
            return summary.strip()
        
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return f"Summary generation failed. Found {len(results)} results about {topic}."
    
    def _extract_key_findings(self, results: List[SearchResult]) -> List[str]:
        """Extract key findings from results."""
        findings = []
        
        for result in results[:5]:
            finding = f"{result.title} ({result.source})"
            findings.append(finding)
        
        return findings
    
    def _identify_trends(self, results: List[SearchResult]) -> List[str]:
        """Identify trends from results."""
        try:
            # Extract keywords from titles
            all_words = []
            for result in results:
                words = result.title.lower().split()
                all_words.extend([w for w in words if len(w) > 4])
            
            # Count word frequency
            word_freq = {}
            for word in all_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top trends
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            trends = [word.capitalize() for word, count in sorted_words[:5] if count > 1]
            
            return trends if trends else ["Emerging research area"]
        
        except Exception as e:
            self.logger.debug(f"Failed to identify trends: {e}")
            return ["Various topics"]
    
    def _get_sources_breakdown(self, results: List[SearchResult]) -> Dict[str, int]:
        """Get breakdown of results by source."""
        breakdown = {}
        for result in results:
            breakdown[result.source] = breakdown.get(result.source, 0) + 1
        return breakdown
    
    def render_report_markdown(self, report: ResearchReport) -> str:
        """✅ Render report as markdown for display and saving."""
        md = []
        
        md.append(f"# {report.topic}")
        md.append("")
        md.append(f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"**Query:** {report.query}")
        md.append(f"**Total Results:** {len(report.results)}")
        md.append(f"**Report Type:** {report.report_type}")
        md.append("")
        
        md.append("## Summary")
        md.append("")
        md.append(report.summary)
        md.append("")
        
        if report.trends:
            md.append("## Key Trends")
            md.append("")
            for trend in report.trends:
                md.append(f"- {trend}")
            md.append("")
        
        md.append(f"## Top {len(report.results)} Research Findings")
        md.append("")
        
        for i, result in enumerate(report.results, 1):
            md.append(f"### {i}. {result.title}")
            md.append("")
            md.append(f"**Source:** {result.source}")
            
            if result.authors:
                md.append(f"**Authors:** {', '.join(result.authors[:5])}")
            
            # ✅ FIX: Handle both timezone-aware and naive datetimes
            if result.published_date:
                try:
                    if result.published_date.tzinfo is not None:
                        # Timezone-aware
                        md.append(f"**Published:** {result.published_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    else:
                        # Timezone-naive
                        md.append(f"**Published:** {result.published_date.strftime('%Y-%m-%d')}")
                except Exception as e:
                    self.logger.debug(f"Failed to format date: {e}")
                    md.append(f"**Published:** {str(result.published_date)[:10]}")
            
            md.append("")
            md.append(result.abstract[:500] + ("..." if len(result.abstract) > 500 else ""))
            md.append("")
            md.append(f"**URL:** {result.url}")
            md.append("")
            
            if result.metadata.get('full_text_fetched'):
                md.append(f"*Full text was fetched and analyzed ({result.metadata.get('full_text_length', 0)} characters).*")
                md.append("")
            
            md.append("---")
            md.append("")
        
        md.append("## Report Metadata")
        md.append("")
        md.append(f"- Total sources searched: {sum(report.metadata.get('sources_breakdown', {}).values())}")
        md.append(f"- Results by source:")
        for source, count in report.metadata.get('sources_breakdown', {}).items():
            md.append(f"  - {source}: {count}")
        md.append("")
        
        return "\n".join(md)

# ============================================================================
# ORCHESTRATOR AGENT
# ============================================================================

class OrchestratorAgent:
    """✅ Main orchestrator with enhanced report management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.llm = LLMFactory.create_provider(config['llm_config'])
        self.search_coordinator = SearchCoordinator(config)
        self.rag_system = RAGSystem(config)
        self.report_agent = ReportAgent(config, self.llm)
        
        self.reports_dir = Path(config.get('system_config', {}).get('reports_dir', './reports'))
        self.scheduled_reports_dir = Path(config.get('system_config', {}).get('scheduled_reports_dir', './reports/scheduled'))
        self.single_reports_dir = Path(config.get('system_config', {}).get('single_reports_dir', './reports/single'))
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.scheduled_reports_dir.mkdir(parents=True, exist_ok=True)
        self.single_reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Orchestrator initialized")
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process user query and route to appropriate handler."""
        intent = self._classify_intent(user_input)
        
        if intent['type'] == 'research':
            return self._handle_one_time_research(intent)
        elif intent['type'] == 'schedule':
            return self._handle_schedule_task(intent)
        elif intent['type'] == 'query_rag':
            return self._handle_rag_query(intent)
        else:
            return {
                'status': 'error',
                'message': 'Could not understand the request'
            }
    
    def _classify_intent(self, user_input: str) -> Dict[str, Any]:
        """Classify user intent."""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['research', 'report', 'find', 'search', 'analyze']):
            topic = user_input
            for prefix in ['research', 'create a report on', 'find information about', 'search for']:
                if prefix in user_input_lower:
                    topic = user_input[user_input_lower.index(prefix) + len(prefix):].strip()
                    break
            
            return {
                'type': 'research',
                'topic': topic
            }
        
        elif any(word in user_input_lower for word in ['schedule', 'recurring', 'periodic']):
            return {
                'type': 'schedule',
                'details': user_input
            }
        
        elif any(word in user_input_lower for word in ['ask', 'question', 'what', 'how', 'why']):
            return {
                'type': 'query_rag',
                'question': user_input
            }
        
        return {
            'type': 'research',
            'topic': user_input
        }
    
    def _handle_one_time_research(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """✅ Handle one-time research with report saving and display."""
        topic = intent.get('topic', '')
        
        if not topic:
            return {
                'status': 'error',
                'message': 'No research topic specified'
            }
        
        self.logger.info(f"Starting one-time research for: {topic}")
        
        try:
            search_results = self.search_coordinator.search_all(topic)
            
            if not search_results:
                return {
                    'status': 'error',
                    'message': f'No results found for topic: {topic}'
                }
            
            self.rag_system.add_documents_parallel(search_results)
            
            report = self.report_agent.generate_report(topic, search_results)
            report.report_type = 'single'
            
            markdown = self.report_agent.render_report_markdown(report)
            
            self._save_report(topic, report, markdown, report_type='single')
            
            if self.config.get('report_config', {}).get('display_in_terminal', True):
                self._display_report(markdown)
            
            return {
                'status': 'success',
                'message': f'Research report generated for: {topic}',
                'report': report,
                'markdown': markdown
            }
        
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            self.logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Research failed: {str(e)}'
            }
    
    def _handle_schedule_task(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task scheduling."""
        return {
            'status': 'info',
            'message': 'Task scheduling not yet implemented'
        }
    
    def _handle_rag_query(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG-based query."""
        question = intent.get('question', '')
        
        if not question:
            return {
                'status': 'error',
                'message': 'No question specified'
            }
        
        try:
            context_results = self.rag_system.query(question, top_k=5)
            
            if not context_results:
                return {
                    'status': 'info',
                    'message': 'No relevant information found in knowledge base. Try running a research query first.'
                }
            
            context = "\n\n".join([
                f"Source: {r['metadata'].get('title', 'Unknown')}\n{r['text']}"
                for r in context_results
            ])
            
            prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {question}

Answer:"""
            
            answer = self.llm.generate_with_retry(prompt, max_tokens=500)
            
            return {
                'status': 'success',
                'answer': answer,
                'sources': [r['metadata'] for r in context_results]
            }
        
        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            return {
                'status': 'error',
                'message': f'Query failed: {str(e)}'
            }
    
    def _save_report(self, topic: str, report: ResearchReport, markdown: str, report_type: str = 'single') -> None:
        """✅ Save report to appropriate directory with subdirectory organization."""
        try:
            if report_type == 'scheduled':
                base_dir = self.scheduled_reports_dir
            else:
                base_dir = self.single_reports_dir
            
            safe_topic = sanitize_filename(topic)
            topic_dir = base_dir / safe_topic
            topic_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_topic}_{timestamp}.md"
            filepath = topic_dir / filename
            
            with file_lock(filepath):
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown)
            
            json_filepath = topic_dir / f"{safe_topic}_{timestamp}.json"
            with file_lock(json_filepath):
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(report.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Report saved to: {filepath}")
            self.logger.info(f"Report JSON saved to: {json_filepath}")
        
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def _display_report(self, markdown: str) -> None:
        """✅ Display report in terminal with formatting."""
        print("\n" + "=" * 80)
        print("RESEARCH REPORT")
        print("=" * 80 + "\n")
        print(markdown)
        print("\n" + "=" * 80 + "\n")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

class InteractiveMode:
    """✅ Interactive mode with full report display."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestrator = OrchestratorAgent(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.running = True
        
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nShutting down gracefully...")
        self.running = False
        sys.exit(0)
    
    def run(self) -> None:
        """Run interactive mode."""
        print("\n" + "=" * 80)
        print("REPORTR - AGENTIC RESEARCH SYSTEM - Interactive Mode")
        print("=" * 80)
        print("\nCommands:")
        print("  - Type your research query (e.g., 'Research quantum computing')")
        print("  - Type 'quit' or 'exit' to quit")
        print("  - Type 'help' for more information")
        print("=" * 80 + "\n")
        
        while self.running:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                print("\nProcessing your request...\n")
                result = self.orchestrator.process_query(user_input)
                
                if result['status'] == 'success':
                    print(f"\n✅ {result['message']}\n")
                elif result['status'] == 'error':
                    print(f"\n❌ Error: {result['message']}\n")
                elif result['status'] == 'info':
                    print(f"\nℹ️  {result['message']}\n")
            
            except KeyboardInterrupt:
                print("\n\nShutting down gracefully...")
                break
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
                print(f"\n❌ An error occurred: {e}\n")
    
    def _show_help(self) -> None:
        """Show help information."""
        print("\n" + "=" * 80)
        print("HELP")
        print("=" * 80)
        print("\nExample queries:")
        print("  - Research quantum computing")
        print("  - Find information about machine learning")
        print("  - Create a report on climate change")
        print("\nFeatures:")
        print("  - Searches multiple sources (arXiv, Web, Scholar, Medium)")
        print("  - Downloads and analyzes full papers")
        print("  - Generates comprehensive reports")
        print("  - Saves reports to ./reports/single/")
        print("  - Displays reports in terminal")
        print("=" * 80 + "\n")


# ============================================================================
# CLI MODE
# ============================================================================

def cli_mode(config: Dict[str, Any], query: str) -> None:
    """✅ CLI mode for single query execution."""
    orchestrator = OrchestratorAgent(config)
    
    print(f"\nProcessing query: {query}\n")
    
    result = orchestrator.process_query(query)
    
    if result['status'] == 'success':
        print(f"\n✅ {result['message']}\n")
    elif result['status'] == 'error':
        print(f"\n❌ Error: {result['message']}\n")
    else:
        print(f"\nℹ️  {result['message']}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Agentic Research System - Multi-agent research assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python reportr.py --interactive
  
  # Single query
  python reportr.py --query "quantum computing"
  
  # Custom config
  python reportr.py --config my_config.yaml --interactive
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Single research query to execute'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'Agentic Research System v{VERSION}'
    )
    
    args = parser.parse_args()
    
    try:
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        
        setup_logging(config)
        
        logger = logging.getLogger('Main')
        logger.info("=" * 80)
        logger.info(f"REPORTR - AGENTIC RESEARCH SYSTEM v{VERSION} STARTING")
        logger.info("=" * 80)
        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"LLM Provider: {config['llm_config']['provider']}")
        logger.info(f"Enabled search sources: {[s['name'] for s in config['search_sources'] if s.get('enabled')]}")
        logger.info("=" * 80)
        
        if args.interactive:
            interactive = InteractiveMode(config)
            interactive.run()
        elif args.query:
            cli_mode(config, args.query)
        else:
            parser.print_help()
            print("\n❌ Error: Please specify either --interactive or --query")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger = logging.getLogger('Main')
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        logger = logging.getLogger('Main')
        logger.info("System shutdown complete")


if __name__ == "__main__":
    main()
