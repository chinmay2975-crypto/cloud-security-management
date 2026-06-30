#!/usr/bin/env python3
"""
Cloud Log Simulator with ML Batch Consumer & LLM Alert Explainer
A real-time cloud log generator using Python's asyncio library.
Simulates thousands of log entries from various cloud services concurrently,
batches them asynchronously, converts features to NumPy, simulates
model.predict_on_batch() inference, and pushes critical alerts to a
background LLM explanation worker.
"""

import asyncio
import random
import time
import math
import csv
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────
# 1. Configuration & Data
# ──────────────────────────────────────────────────────────────

# Log levels with associated weights (probability of occurrence)
LOG_LEVELS: Dict[str, float] = {
    "INFO": 0.60,     # 60% of logs are informational
    "WARN": 0.20,     # 20% are warnings
    "ERROR": 0.12,    # 12% are errors
    "DEBUG": 0.05,    # 5% are debug messages
    "CRITICAL": 0.03, # 3% are critical
}

# Numeric mapping for log levels (feature encoding)
LEVEL_TO_SEVERITY: Dict[str, int] = {
    "DEBUG": 0,
    "INFO": 1,
    "WARN": 2,
    "ERROR": 3,
    "CRITICAL": 4,
}

# Numeric mapping for services (feature encoding)
SERVICE_TO_INDEX: Dict[str, int] = {
    "api-gateway": 0,
    "auth-service": 1,
    "database-cluster": 2,
    "load-balancer": 3,
    "cdn-edge": 4,
    "storage-bucket": 5,
    "message-queue": 6,
    "container-orchestrator": 7,
    "monitoring-agent": 8,
    "serverless-function": 9,
}

# Cloud services being simulated
SERVICES: List[str] = list(SERVICE_TO_INDEX.keys())

# Sample log message templates per service
LOG_MESSAGES: Dict[str, List[str]] = {
    "api-gateway": [
        "Request {method} {path} processed in {duration}ms | Status: {status}",
        "Rate limit exceeded for IP {ip} — {requests} requests in window",
        "SSL handshake completed for client {ip} using TLSv1.3",
        "Upstream connection pool exhausted, queuing request",
    ],
    "auth-service": [
        "User {user} authenticated via OAuth2 provider {provider}",
        "Token refresh succeeded for user {user}",
        "Invalid credentials attempt from IP {ip}",
        "JWT signature verification failed — token expired",
        "MFA challenge completed for user {user}",
    ],
    "database-cluster": [
        "Query executed in {duration}ms | Rows returned: {rows}",
        "Connection pool at {percent}% capacity ({active}/{max} connections)",
        "Replica lag detected: {lag}s behind primary",
        "Table '{table}' — sequential scan performed (missing index)",
        "Deadlock detected between transactions {tx1} and {tx2}",
    ],
    "load-balancer": [
        "Routing request to backend {backend} (health: OK, load: {load}%)",
        "Backend {backend} marked as UNHEALTHY — removing from pool",
        "Sticky session cookie set for client {ip}",
        "SSL certificate for {domain} expires in {days} days",
    ],
    "cdn-edge": [
        "Cache HIT for {path} (TTL: {ttl}s, served from {location})",
        "Cache MISS for {path} — fetching from origin",
        "Purging cache for path pattern '{pattern}' ({files} files)",
        "Edge worker execution completed in {duration}ms",
    ],
    "storage-bucket": [
        "Object '{key}' uploaded ({size} bytes) by user {user}",
        "Pre-signed URL generated for '{key}' (expires in {expiry}s)",
        "Lifecycle policy applied: archived {count} objects to cold storage",
        "Bucket '{bucket}' — {method} request from IP {ip}",
    ],
    "message-queue": [
        "Message published to topic '{topic}' (size: {size} bytes)",
        "Consumer {consumer} acknowledged message ID {msg_id}",
        "Queue depth for '{queue}' reached {depth} — scaling consumers",
        "Dead-letter message '{msg_id}' — exceeded max retries ({retries})",
    ],
    "container-orchestrator": [
        "Pod '{pod}' scheduled on node '{node}' (resources: {cpu}c/{mem}Mi)",
        "Container '{container}' in pod '{pod}' restarted {count} times",
        "Node '{node}' at {cpu}% CPU / {mem}% memory — consider scaling",
        "Deployment '{deployment}' rolled out successfully ({replicas} replicas)",
    ],
    "monitoring-agent": [
        "Metric '{metric}' exceeded alert threshold ({value} > {threshold})",
        "Health check probe to {target} returned status {status}",
        "Log aggregation batch sent ({entries} entries, {size} bytes)",
        "Agent heartbeat missed for host '{host}' — {missed}/{window} intervals",
    ],
    "serverless-function": [
        "Function '{func}' invoked (duration: {duration}ms, memory: {mem}MB)",
        "Cold start detected for function '{func}' ({cold_start}ms init time)",
        "Function '{func}' timed out after {timeout}s",
        "Concurrent execution limit reached for function '{func}' ({running} running)",
    ],
}

# Realistic placeholder generators
USERS = ["alice", "bob", "charlie", "dave", "eve", "francis", "grace", "henry"]
IPS = [f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}" for _ in range(20)]
PATHS = ["/api/v1/users", "/api/v1/orders", "/api/v1/products", "/health", "/metrics", "/graphql"]
METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
STATUSES = [200, 201, 204, 301, 400, 401, 403, 404, 500, 502, 503]
PROVIDERS = ["google", "github", "microsoft", "okta", "auth0"]
BACKENDS = ["web-01", "web-02", "web-03", "app-01", "app-02", "db-primary", "db-replica-01"]
DOMAINS = ["example.com", "api.example.com", "cdn.example.com"]
LOCATIONS = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]


def format_message(template: str, service: str) -> str:
    """Fill a log template with realistic random values."""
    data = {
        "method": random.choice(METHODS),
        "path": random.choice(PATHS),
        "status": random.choice(STATUSES),
        "duration": random.randint(1, 5000),
        "ip": random.choice(IPS),
        "user": random.choice(USERS),
        "provider": random.choice(PROVIDERS),
        "rows": random.randint(0, 10000),
        "percent": random.randint(10, 100),
        "active": random.randint(1, 200),
        "max": 200,
        "lag": round(random.uniform(0.1, 30.0), 1),
        "table": random.choice(["users", "orders", "products", "sessions"]),
        "tx1": f"0x{random.randint(1000, 9999):04x}",
        "tx2": f"0x{random.randint(1000, 9999):04x}",
        "backend": random.choice(BACKENDS),
        "load": random.randint(0, 100),
        "domain": random.choice(DOMAINS),
        "days": random.randint(1, 90),
        "ttl": random.randint(60, 86400),
        "location": random.choice(LOCATIONS),
        "pattern": f"/{random.choice(['images', 'css', 'js', 'api'])}/*",
        "files": random.randint(1, 5000),
        "key": f"logs/{datetime.now().strftime('%Y/%m/%d')}/app-{random.randint(1, 9)}.log",
        "size": random.randint(50, 10_000_000),
        "expiry": random.randint(60, 3600),
        "count": random.randint(1, 10000),
        "bucket": random.choice(["app-logs", "user-data", "backups", "static-assets"]),
        "topic": random.choice(["orders.created", "users.updated", "events.ingested"]),
        "consumer": f"consumer-{random.randint(1, 20)}",
        "msg_id": f"msg-{random.randint(10000, 99999)}",
        "queue": random.choice(["order-processing", "email-queue", "notification-queue"]),
        "depth": random.randint(100, 50000),
        "retries": random.randint(3, 10),
        "pod": f"pod-{random.choice(['api', 'web', 'worker'])}-{random.randint(1, 10)}",
        "node": f"node-{random.randint(1, 20)}",
        "cpu": random.randint(100, 4000),
        "mem": random.randint(128, 8192),
        "container": random.choice(["nginx", "app", "sidecar-proxy", "init"]),
        "restart_count": random.randint(1, 15),
        "deployment": random.choice(["api-v2", "web-frontend", "worker-pool"]),
        "replicas": random.randint(1, 20),
        "metric": random.choice(["cpu_usage", "memory_usage", "request_latency", "error_rate"]),
        "value": round(random.uniform(50, 99), 1),
        "threshold": 85.0,
        "target": random.choice(IPS),
        "entries": random.randint(50, 5000),
        "requests": random.randint(100, 10000),
        "func": random.choice(["process-order", "send-email", "resize-image", "validate-user"]),
        "mem": random.randint(128, 2048),
        "cold_start": random.randint(100, 3000),
        "timeout": 30,
        "running": random.randint(10, 100),
        "host": f"host-{random.randint(1, 50)}",
        "missed": random.randint(1, 5),
        "window": 10,
        "intervals": 10,
        "files": random.randint(1, 1000),
        "pattern_tmp": f"/{random.choice(['images', 'css', 'js', 'api'])}/*",
    }
    return template.format(**{k: v for k, v in data.items() if f"{{{k}}}" in template})


def create_log_entry() -> Dict:
    """
    Generate a single simulated cloud log entry.

    Returns a dictionary with timestamp, level, service, message, and metadata.
    """
    # Pick a log level based on weighted probability
    levels = list(LOG_LEVELS.keys())
    weights = list(LOG_LEVELS.values())
    level = random.choices(levels, weights=weights, k=1)[0]

    # Pick a service
    service = random.choice(SERVICES)

    # Pick a message template for that service and format it
    template = random.choice(LOG_MESSAGES[service])
    message = format_message(template, service)

    # Generate ISO 8601 timestamp with timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    return {
        "timestamp": timestamp,
        "level": level,
        "severity": LEVEL_TO_SEVERITY[level],
        "service": service,
        "service_index": SERVICE_TO_INDEX[service],
        "message": message,
        "environment": random.choice(["production", "staging", "development"]),
        "region": random.choice(LOCATIONS),
    }


def format_log_line(entry: Dict) -> str:
    """Format a log entry dictionary into a human-readable log line."""
    return (
        f"[{entry['timestamp']}] "
        f"[{entry['level']:>8}] "
        f"[{entry['service']:<22}] "
        f"{entry['message']}"
    )


def load_logs_from_csv(csv_path: str = "cloud_security_logs.csv") -> List[Dict]:
    """
    Load security logs from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing security logs.
        
    Returns:
        List of log entry dictionaries.
    """
    logs = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert CSV row to our log entry format
                log_entry = {
                    "timestamp": row.get('timestamp', ''),
                    "level": "INFO",  # Default level for CSV logs
                    "severity": 1,  # INFO level
                    "service": "security-monitor",  # Generic service for security logs
                    "service_index": 0,  # Will be mapped to a service
                    "message": format_csv_log_message(row),
                    "environment": "production",  # Security logs are typically from production
                    "region": "us-east-1",  # Default region
                    "user_id": row.get('user_id', ''),
                    "user_role": row.get('user_role', ''),
                    "mfa_verified": row.get('mfa_verified', '0'),
                    "request_source": row.get('request_source', ''),
                    "ip_country": row.get('ip_country', ''),
                    "action_type": row.get('action_type', ''),
                    "resource_type": row.get('resource_type', ''),
                    "is_public_exposed": row.get('is_public_exposed', '0'),
                    "privilege_level": row.get('privilege_level', '0'),
                }
                
                # Adjust severity based on security indicators
                if row.get('is_public_exposed') == '1':
                    log_entry["severity"] = 3  # ERROR
                    log_entry["level"] = "ERROR"
                elif row.get('mfa_verified') == '0':
                    log_entry["severity"] = 2  # WARN
                    log_entry["level"] = "WARN"
                elif 'compromised' in row.get('user_role', '').lower() or \
                     'hacker' in row.get('user_id', '').lower() or \
                     'anomalous' in row.get('user_id', '').lower():
                    log_entry["severity"] = 4  # CRITICAL
                    log_entry["level"] = "CRITICAL"
                
                logs.append(log_entry)
        
        print(f"  📊 Loaded {len(logs)} security logs from {csv_path}")
        return logs
    except FileNotFoundError:
        print(f"  ⚠️  CSV file not found: {csv_path}. Using synthetic logs instead.")
        return []
    except Exception as e:
        print(f"  ⚠️  Error loading CSV: {e}. Using synthetic logs instead.")
        return []


def format_csv_log_message(row: Dict) -> str:
    """Format a CSV log row into a human-readable message."""
    user_id = row.get('user_id', 'unknown')
    action = row.get('action_type', 'unknown_action')
    resource = row.get('resource_type', 'unknown_resource')
    source = row.get('request_source', 'unknown')
    country = row.get('ip_country', 'XX')
    
    return f"Security event: {user_id} performed {action} on {resource} via {source} from {country}"


# ──────────────────────────────────────────────────────────────
# 2. Feature Extraction & ML Mock Inference
# ──────────────────────────────────────────────────────────────

# Dummy model weights: a (NUM_FEATURES x 3) matrix for 3-class prediction
# We bias the weights so that high-severity / high-error logs trigger
# the "critical" (class 2) prediction, ensuring the LLM explainer fires.
NUM_FEATURES = 8
NUM_CLASSES = 3
np.random.seed(42)
DUMMY_WEIGHTS: np.ndarray = np.random.randn(NUM_FEATURES, NUM_CLASSES).astype(np.float32)
# Boost class 2 (critical) weights for severity (feat 0) and error_flag (feat 4)
DUMMY_WEIGHTS[0, 2] += 2.0   # severity → critical
DUMMY_WEIGHTS[4, 2] += 2.5   # error_flag → critical
# Boost class 0 (normal) for service_index (feat 1) and region (feat 3)
DUMMY_WEIGHTS[1, 0] += 1.5
DUMMY_WEIGHTS[3, 0] += 1.0
DUMMY_BIAS: np.ndarray = np.random.randn(NUM_CLASSES).astype(np.float32)
DUMMY_BIAS[2] += 1.0  # bias critical class higher


def extract_features(entry: Dict) -> np.ndarray:
    """
    Convert a single log entry into a fixed-length NumPy feature vector.

    Extracted features (8 total):
      0. severity             (0-4, normalized to ~0-1)
      1. service_index        (0-9, normalized to ~0-1)
      2. env_score            (production=1.0, staging=0.5, development=0.0)
      3. region_score         (us-east-1=0.0, us-west-2=0.33, eu-west-1=0.67, ap-southeast-1=1.0)
      4. error_flag           (1 if level is ERROR or CRITICAL, else 0)
      5. hour_fraction        (hour of day / 23.0)
      6. minute_fraction      (minute of hour / 59.0)
      7. second_fraction      (second of minute / 59.0)
    """
    sev = entry["severity"] / 4.0  # normalize 0-4 → 0-1
    svc = entry["service_index"] / 9.0  # normalize 0-9 → 0-1

    # Environment score
    env_map = {"production": 1.0, "staging": 0.5, "development": 0.0}
    env_score = env_map.get(entry["environment"], 0.0)

    # Region score
    region_map = {"us-east-1": 0.0, "us-west-2": 0.33, "eu-west-1": 0.67, "ap-southeast-1": 1.0}
    region_score = region_map.get(entry["region"], 0.0)

    # Error flag
    error_flag = 1.0 if entry["level"] in ("ERROR", "CRITICAL") else 0.0

    # Time-based features from the timestamp string
    try:
        t = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
        hour_frac = t.hour / 23.0
        min_frac = t.minute / 59.0
        sec_frac = t.second / 59.0
    except (ValueError, KeyError):
        hour_frac = min_frac = sec_frac = 0.5

    return np.array(
        [sev, svc, env_score, region_score, error_flag, hour_frac, min_frac, sec_frac],
        dtype=np.float32,
    )


def simulate_predict_on_batch(features: np.ndarray) -> np.ndarray:
    """
    Simulate a model.predict_on_batch() call.

    Performs a dummy matrix multiply (features @ weights + bias) followed by
    a softmax to produce pseudo-probabilities for NUM_CLASSES.

    A small artificial compute delay is added proportional to batch size to
    simulate real inference latency.

    Args:
        features: shape (batch_size, NUM_FEATURES) NumPy array

    Returns:
        predictions: shape (batch_size, NUM_CLASSES) pseudo-probabilities
    """
    batch_size = features.shape[0]

    # ── Simulate compute latency: ~0.5ms per sample + 1ms base ──
    compute_delay = 0.001 + (batch_size * 0.0005)
    time.sleep(compute_delay)  # blocking sleep to mimic CPU-bound inference

    # ── Dense layer: logits = features @ weights + bias ──
    logits = features @ DUMMY_WEIGHTS + DUMMY_BIAS

    # ── Softmax to get pseudo-probabilities ──
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    predictions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return predictions


# ──────────────────────────────────────────────────────────────
# 3. Async Log Generator
# ──────────────────────────────────────────────────────────────

async def log_stream(
    worker_id: int,
    total_logs: int,
    log_queue: asyncio.Queue,
    min_delay: float = 0.05,
    max_delay: float = 0.5,
    burst_probability: float = 0.15,
) -> int:
    """
    Coroutine that generates a stream of log entries and enqueues them
    for batch processing.

    Simulates realistic traffic patterns with random delays and occasional bursts.

    Args:
        worker_id: Unique identifier for this worker coroutine.
        total_logs: Number of log entries this worker should produce.
        log_queue: Shared asyncio.Queue to push entries into for batch consumption.
        min_delay: Minimum delay (seconds) between log entries.
        max_delay: Maximum delay (seconds) between log entries.
        burst_probability: Probability of entering 'burst mode' with higher frequency logs.

    Returns:
        Number of log entries generated by this worker.
    """
    logs_generated = 0

    try:
        while logs_generated < total_logs:
            # ── Simulate burst traffic ──
            if random.random() < burst_probability:
                burst_size = random.randint(3, 12)
                burst_size = min(burst_size, total_logs - logs_generated)

                for _ in range(burst_size):
                    entry = create_log_entry()
                    # Print for human-readable display
                    print(format_log_line(entry))
                    # Enqueue for ML batch processing (non-blocking put)
                    await log_queue.put(entry)
                    logs_generated += 1

                    # Very short delay between burst logs (simulating high traffic)
                    await asyncio.sleep(random.uniform(0.005, 0.03))
            else:
                # Normal mode: single log with a realistic delay
                entry = create_log_entry()
                print(format_log_line(entry))
                await log_queue.put(entry)
                logs_generated += 1

                # Simulate realistic time between log entries
                delay = random.uniform(min_delay, max_delay)
                await asyncio.sleep(delay)
    except asyncio.CancelledError:
        # Worker was cancelled - return the number of logs generated so far
        pass

    return logs_generated


async def background_tasks_monitor(
    tasks: List[asyncio.Task],
    total_logs_expected: int,
    check_interval: float = 0.5,
) -> None:
    """
    Monitor the progress of all worker tasks and report completion status.

    Args:
        tasks: List of asyncio Task objects to monitor.
        total_logs_expected: Total number of logs expected across all workers.
        check_interval: How often (in seconds) to check task status.
    """
    running = len(tasks)
    start_time = time.monotonic()

    while running > 0:
        running = sum(1 for t in tasks if not t.done())
        elapsed = time.monotonic() - start_time
        logs_per_sec = (
            (total_logs_expected - sum(
                t.result() if t.done() and not t.cancelled() else 0
                for t in tasks
            ))
            / elapsed
            if elapsed > 0
            else 0
        )

        # Print a status summary
        print(
            f"\n  ⏳ [MONITOR] Workers running: {running}/{len(tasks)} | "
            f"Elapsed: {elapsed:.1f}s | "
            f"Rate: ~{logs_per_sec:.0f} logs/sec\n",
            flush=True,
        )

        await asyncio.sleep(check_interval)

    return


# ──────────────────────────────────────────────────────────────
# 4. ML Batch Consumer (ml_batch_processor)
# ──────────────────────────────────────────────────────────────

BATCH_SIZE = 32
FLUSH_INTERVAL = 0.10  # 100 milliseconds


async def ml_batch_processor(
    log_queue: asyncio.Queue,
    total_logs_expected: int,
    alert_queue: asyncio.Queue,
    batch_size: int = BATCH_SIZE,
    flush_interval: float = FLUSH_INTERVAL,
) -> Tuple[int, float, int]:
    """
    Asynchronous batch consumer that accumulates log entries into batches,
    extracts numerical features, converts them to a NumPy array, and
    simulates a model.predict_on_batch() call.

    Implements the "batch-or-flush" pattern:
        - Accumulate entries until batch_size is reached, OR
        - Flush on a timer every flush_interval seconds (whichever comes first)

    For any log predicted as 'critical' (class 2), the original entry and
    its prediction details are pushed into alert_queue for downstream
    LLM-based explanation.

    Args:
        log_queue: Shared asyncio.Queue to pull log entries from workers.
        total_logs_expected: Total number of logs expected (used to detect completion).
        alert_queue: Shared asyncio.Queue to push critical alerts for LLM explanation.
        batch_size: Maximum batch size before forcing an inference call.
        flush_interval: Maximum time (seconds) to wait before flushing a partial batch.

    Returns:
        Tuple of (total_batches_processed, total_predict_time_seconds, total_alerts_pushed).
    """
    batch_buffer: List[Dict] = []
    total_batches = 0
    total_predict_time = 0.0
    last_flush = time.monotonic()
    logs_processed = 0
    total_alerts_pushed = 0

    print(f"\n  🧠 [ML BATCH PROCESSOR] Started | "
          f"batch_size={batch_size}, flush_interval={flush_interval*1000:.0f}ms\n")

    while logs_processed < total_logs_expected:
        # ── Collect entries from the queue with a timeout ──
        timeout = flush_interval - (time.monotonic() - last_flush)
        timeout = max(timeout, 0.001)  # minimum 1ms timeout

        try:
            entry = await asyncio.wait_for(log_queue.get(), timeout=timeout)
            batch_buffer.append(entry)
            logs_processed += 1
            log_queue.task_done()

            # Drain any additional entries immediately available (up to batch_size)
            while len(batch_buffer) < batch_size and not log_queue.empty():
                try:
                    entry = log_queue.get_nowait()
                    batch_buffer.append(entry)
                    logs_processed += 1
                    log_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        except asyncio.TimeoutError:
            pass  # Timer expired — proceed to flush check

        # ── Decide whether to process the batch ──
        elapsed_since_flush = time.monotonic() - last_flush
        should_flush = (
            len(batch_buffer) >= batch_size
            or (len(batch_buffer) > 0 and elapsed_since_flush >= flush_interval)
        )

        if should_flush:
            batch = batch_buffer
            batch_buffer = []
            last_flush = time.monotonic()

            # ── Convert batch to NumPy feature array ──
            features_list = [extract_features(entry) for entry in batch]
            feature_matrix = np.stack(features_list, axis=0)  # shape: (B, NUM_FEATURES)

            # ── Simulate model.predict_on_batch() ──
            t0 = time.monotonic()
            predictions = simulate_predict_on_batch(feature_matrix)
            infer_time = time.monotonic() - t0
            total_predict_time += infer_time
            total_batches += 1

            # ── Compute aggregate predictions for display ──
            pred_classes = np.argmax(predictions, axis=1)
            confidence = float(np.mean(np.max(predictions, axis=1)))
            class_distribution = {
                "normal": int(np.sum(pred_classes == 0)),
                "anomalous": int(np.sum(pred_classes == 1)),
                "critical": int(np.sum(pred_classes == 2)),
            }

            print(
                f"  🧠 [ML BATCH #{total_batches}] "
                f"size={len(batch):>2} | "
                f"infer={infer_time*1000:>5.1f}ms | "
                f"confidence={confidence:.2f} | "
                f"preds={class_distribution}"
            )

            # ── Push critical-prediction logs to alert_queue ──
            for i, entry in enumerate(batch):
                if pred_classes[i] == 2:  # critical class
                    alert_payload = {
                        "entry": entry,
                        "prediction": predictions[i].tolist(),
                        "predicted_class": int(pred_classes[i]),
                        "confidence": float(np.max(predictions[i])),
                        "batch_id": total_batches,
                    }
                    await alert_queue.put(alert_payload)
                    total_alerts_pushed += 1

    # ── Flush any remaining entries in the buffer ──
    if batch_buffer:
        features_list = [extract_features(entry) for entry in batch_buffer]
        feature_matrix = np.stack(features_list, axis=0)
        t0 = time.monotonic()
        predictions = simulate_predict_on_batch(feature_matrix)
        infer_time = time.monotonic() - t0
        total_predict_time += infer_time
        total_batches += 1

        pred_classes = np.argmax(predictions, axis=1)
        confidence = float(np.mean(np.max(predictions, axis=1)))
        class_distribution = {
            "normal": int(np.sum(pred_classes == 0)),
            "anomalous": int(np.sum(pred_classes == 1)),
            "critical": int(np.sum(pred_classes == 2)),
        }
        print(
            f"  🧠 [ML BATCH #{total_batches}] "
            f"size={len(batch_buffer):>2} (final flush) | "
            f"infer={infer_time*1000:>5.1f}ms | "
            f"confidence={confidence:.2f} | "
            f"preds={class_distribution}"
        )

        # Push critical predictions from final flush
        for i, entry in enumerate(batch_buffer):
            if pred_classes[i] == 2:
                alert_payload = {
                    "entry": entry,
                    "prediction": predictions[i].tolist(),
                    "predicted_class": int(pred_classes[i]),
                    "confidence": float(np.max(predictions[i])),
                    "batch_id": total_batches,
                }
                await alert_queue.put(alert_payload)
                total_alerts_pushed += 1

    print(f"\n  🧠 [ML BATCH PROCESSOR] Done — {total_batches} batches | "
          f"{total_predict_time:.3f}s total inference time | "
          f"{total_alerts_pushed} critical alerts pushed")

    return total_batches, total_predict_time, total_alerts_pushed


# ──────────────────────────────────────────────────────────────
# 5. LLM Alert Explainer (llm_explain_worker)
# ──────────────────────────────────────────────────────────────

# Gemini-style prompt template for explaining critical alerts
GEMINI_PROMPT_TEMPLATE = """
You are a senior SRE (Site Reliability Engineer) analyzing a critical cloud infrastructure alert.

ALERT DETAILS:
- Timestamp: {timestamp}
- Service: {service}
- Log Level: {level}
- Message: {message}
- Environment: {environment}
- Region: {region}
- ML Prediction Confidence: {confidence:.2f}
- Predicted Class: CRITICAL (class 2)

TASK:
Provide a concise incident analysis covering:
1. ROOT CAUSE: What likely caused this alert?
2. IMPACT: Which systems/users are affected?
3. RECOMMENDED ACTION: What immediate steps should the on-call engineer take?
4. SEVERITY: P0/P1/P2 classification with justification

RESPONSE FORMAT:
```
## Incident Analysis
**Root Cause:** [1-2 sentences]
**Impact:** [1-2 sentences]
**Recommended Action:** [2-3 bullet points]
**Severity:** [P0/P1/P2] — [justification]
```
"""


async def llm_explain_worker(
    alert_queue: asyncio.Queue,
    total_alerts_expected: int,
    simulate_llm_delay: bool = True,
) -> int:
    """
    Background worker that monitors the alert_queue, pulls critical alerts
    one by one, and simulates a Gemini LLM call to generate an incident
    explanation using a structured prompt template.

    This runs as a separate asyncio task and does NOT block the primary
    log processing pipeline.

    Args:
        alert_queue: Shared asyncio.Queue containing critical alert payloads.
        total_alerts_expected: Total number of alerts to expect before stopping.
        simulate_llm_delay: If True, adds a realistic ~1-3s delay per alert
                            to simulate LLM inference latency.

    Returns:
        Total number of alerts processed.
    """
    alerts_processed = 0

    print(f"\n  🤖 [LLM EXPLAIN WORKER] Started | "
          f"expecting ~{total_alerts_expected} critical alerts\n")

    while alerts_processed < total_alerts_expected:
        try:
            # Wait for an alert with a generous timeout
            alert = await asyncio.wait_for(alert_queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            # No alert yet — loop back and check again
            continue

        entry = alert["entry"]
        confidence = alert["confidence"]

        # ── Build the Gemini prompt (for display / logging) ──
        prompt = GEMINI_PROMPT_TEMPLATE.format(
            timestamp=entry["timestamp"],
            service=entry["service"],
            level=entry["level"],
            message=entry["message"],
            environment=entry["environment"],
            region=entry["region"],
            confidence=confidence,
        )

        # ── Simulate LLM inference delay (non-blocking) ──
        if simulate_llm_delay:
            llm_latency = random.uniform(0.8, 2.5)
            await asyncio.sleep(llm_latency)
        else:
            llm_latency = 0.0

        # ── Generate a simulated LLM response ──
        simulated_response = _simulate_gemini_response(entry, confidence)

        alerts_processed += 1
        alert_queue.task_done()

        print(
            f"\n  🤖 [LLM ALERT #{alerts_processed}] "
            f"Service: {entry['service']} | "
            f"LLM latency: {llm_latency:.1f}s | "
            f"Confidence: {confidence:.2f}"
        )
        print(f"{simulated_response}")
        print(f"  {'─'*66}")

    print(f"\n  🤖 [LLM EXPLAIN WORKER] Done — {alerts_processed} alerts explained")
    return alerts_processed


def _simulate_gemini_response(entry: Dict, confidence: float) -> str:
    """
    Generate a realistic simulated Gemini LLM response for a critical alert.

    In production this would call the actual Gemini API. Here we produce
    a structured incident analysis based on the alert's service and message.

    Args:
        entry: The original log entry dictionary.
        confidence: ML model confidence score for the critical prediction.

    Returns:
        A formatted incident analysis string.
    """
    service = entry["service"]
    message = entry["message"]
    level = entry["level"]

    # Generate root cause and impact based on service type
    root_causes = {
        "api-gateway": "Upstream service degradation or connection pool exhaustion causing request queuing and timeouts.",
        "auth-service": "Authentication provider latency or token validation failures indicating potential OAuth provider issue.",
        "database-cluster": "Query performance degradation, possible missing index or replication lag impacting read/write throughput.",
        "load-balancer": "Backend health check failures causing service pool depletion and increased error rates.",
        "cdn-edge": "Cache miss storm or origin fetch failures leading to increased origin load and latency.",
        "storage-bucket": "Storage operation failures or lifecycle policy errors indicating potential IAM misconfiguration.",
        "message-queue": "Consumer lag or dead-letter queue accumulation indicating processing pipeline bottleneck.",
        "container-orchestrator": "Pod scheduling failures or resource exhaustion on cluster nodes.",
        "monitoring-agent": "Agent heartbeat loss or metric collection failures indicating potential host or network issues.",
        "serverless-function": "Cold start latency spikes or concurrent execution limit breaches causing invocation failures.",
    }

    impacts = {
        "api-gateway": "End-user API requests may experience increased latency or 503 errors. Up to 15% of traffic affected.",
        "auth-service": "User authentication and token refresh flows may be degraded. Login success rate could drop by 20%.",
        "database-cluster": "Read/write operations on affected tables may see 2-5x latency increase. Partial query failures possible.",
        "load-balancer": "Reduced backend capacity may cause uneven traffic distribution and increased error rates for affected routes.",
        "cdn-edge": "Users may experience slower content delivery and increased origin load. Cache hit ratio could drop by 30%.",
        "storage-bucket": "Object upload/retrieval operations may fail intermittently. Data durability not impacted.",
        "message-queue": "Message processing throughput may decrease. Backlog could grow by 10K+ messages per minute.",
        "container-orchestrator": "New deployments may fail to schedule. Existing workloads may experience resource contention.",
        "monitoring-agent": "Monitoring coverage gap for affected hosts. Alerting and dashboards may have stale data.",
        "serverless-function": "Function invocation latency may spike. Concurrent execution limits may cause request throttling.",
    }

    severities = ["P0", "P1", "P2"]
    severity_weights = [0.3, 0.5, 0.2]
    severity = random.choices(severities, weights=severity_weights, k=1)[0]

    severity_justifications = {
        "P0": "Critical production outage affecting core user-facing functionality. Immediate escalation required.",
        "P1": "Significant degradation of a critical service. Requires urgent investigation within 15 minutes.",
        "P2": "Non-critical degradation or potential issue. Should be investigated within 1 hour.",
    }

    root_cause = root_causes.get(service, "Unexpected system behavior detected by ML anomaly classifier.")
    impact = impacts.get(service, "Service degradation affecting a subset of users or internal systems.")
    severity_just = severity_justifications[severity]

    # Generate 2-3 recommended actions
    action_templates = [
        f"Immediately investigate {service} logs and metrics in the monitoring dashboard.",
        f"Check recent deployments or configuration changes to {service} in the last 30 minutes.",
        f"Escalate to the {service} on-call team with the alert details and ML prediction confidence ({confidence:.0%}).",
        f"Verify upstream dependencies and external service health for {service}.",
        f"Consider scaling out {service} instances or increasing resource limits.",
        f"Review recent code changes or config updates that may have triggered this regression.",
        f"Check if this is part of a broader incident by correlating with other active alerts.",
    ]
    selected_actions = random.sample(action_templates, min(3, len(action_templates)))

    actions_str = "\n".join(f"  • {a}" for a in selected_actions)

    return (
        f"  ┌─ Incident Analysis ──────────────────────────────────┐\n"
        f"  │ **Root Cause:** {root_cause}\n"
        f"  │ **Impact:** {impact}\n"
        f"  │ **Recommended Action:**\n{actions_str}\n"
        f"  │ **Severity:** {severity} — {severity_just}\n"
        f"  └──────────────────────────────────────────────────────┘"
    )


# ──────────────────────────────────────────────────────────────
# 6. Main Orchestrator
# ──────────────────────────────────────────────────────────────

async def simulate_cloud_logs(
    total_logs: int = 2000,
    num_workers: int = 10,
    min_delay: float = 0.05,
    max_delay: float = 0.5,
    simulate_time: bool = True,
    use_csv: bool = False,
    csv_path: str = "cloud_security_logs.csv",
) -> None:
    """
    Main async function that orchestrates the entire log simulation
    including ML batch processing and LLM alert explanation.

    Spawns multiple worker coroutines that generate log entries concurrently,
    pushes them through a shared async Queue into the ml_batch_processor,
    which in turn pushes critical alerts into a second queue consumed by
    the llm_explain_worker.

    Args:
        total_logs: Total number of log entries to generate.
        num_workers: Number of concurrent worker coroutines.
        min_delay: Minimum delay (seconds) between log entries.
        max_delay: Maximum delay (seconds) between log entries.
        simulate_time: If True, prints timing summary at the end.
    """
    print(f"\n{'='*70}")
    print(f"  🌩️  CLOUD LOG SIMULATOR with ML Batch Consumer & LLM Explainer")
    print(f"{'='*70}")
    print(f"  Total logs    : {total_logs:,}")
    print(f"  Workers       : {num_workers}")
    print(f"  Services      : {len(SERVICES)} ({', '.join(SERVICES[:4])}, ...)")
    print(f"  Min delay     : {min_delay}s")
    print(f"  Max delay     : {max_delay}s")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Flush interval: {FLUSH_INTERVAL*1000:.0f}ms")
    if use_csv:
        print(f"  Data source   : CSV file ({csv_path})")
    else:
        print(f"  Data source   : Synthetic logs")
    print(f"{'─'*70}\n")

    # ── Shared queues ──
    # log_queue: producers (workers) → consumer (batch processor)
    log_queue: asyncio.Queue = asyncio.Queue(maxsize=total_logs * 2)
    # alert_queue: batch processor (critical predictions) → consumer (LLM explainer)
    alert_queue: asyncio.Queue = asyncio.Queue()

    # ── Load CSV logs if requested ──
    csv_logs: List[Dict] = []
    if use_csv:
        csv_logs = load_logs_from_csv(csv_path)
        if not csv_logs:
            print("  ⚠️  Falling back to synthetic log generation")
            use_csv = False
    
    start_time = time.monotonic()
    
    # Initialize variables that will be set by either code path
    total_batches = 0
    total_predict_time = 0.0
    total_alerts_pushed = 0
    tasks = []
    monitor_task = None
    batch_processor_task = None
    llm_worker_task = None

    try:
        if use_csv and csv_logs:
            # ── Process CSV logs directly ──
            print(f"  📊 Processing {len(csv_logs)} CSV logs through ML pipeline...\n")
            
            # Create a task to feed CSV logs into the queue
            async def feed_csv_logs():
                for log_entry in csv_logs:
                    await log_queue.put(log_entry)
                    # Small delay to simulate real-time processing
                    await asyncio.sleep(0.01)
            
            # Create worker tasks for CSV mode (just monitor, don't generate)
            
            # Launch the ML batch processor concurrently
            batch_processor_task = asyncio.create_task(
                ml_batch_processor(log_queue, len(csv_logs), alert_queue),
                name="ml-batch-processor",
            )
            
            # Launch the CSV feeder
            csv_feeder_task = asyncio.create_task(
                feed_csv_logs(),
                name="csv-feeder",
            )
            
            # Wait for CSV feeder to finish
            await csv_feeder_task
            
            # Wait for batch processor to finish
            batch_result = await batch_processor_task
            
            # Extract batch processor results
            if isinstance(batch_result, tuple) and len(batch_result) == 3:
                total_batches, total_predict_time, total_alerts_pushed = batch_result
            else:
                total_batches, total_predict_time, total_alerts_pushed = 0, 0.0, 0
        else:
            # ── Original synthetic log generation ──
            # Distribute logs evenly among workers (last worker gets the remainder)
            logs_per_worker = total_logs // num_workers
            remainder = total_logs % num_workers

            # Create worker tasks — each is an independent coroutine
            for i in range(num_workers):
                worker_logs = logs_per_worker + (1 if i < remainder else 0)
                task = asyncio.create_task(
                    log_stream(
                        worker_id=i + 1,
                        total_logs=worker_logs,
                        log_queue=log_queue,
                        min_delay=min_delay,
                        max_delay=max_delay,
                    ),
                    name=f"worker-{i+1}",
                )
                tasks.append(task)

            # Launch the ML batch processor concurrently
            batch_processor_task = asyncio.create_task(
                ml_batch_processor(log_queue, total_logs, alert_queue),
                name="ml-batch-processor",
            )

            # Launch the progress monitor concurrently
            monitor_task = asyncio.create_task(
                background_tasks_monitor(tasks, total_logs)
            )

            # Wait for all workers to finish first
            await asyncio.gather(*tasks, return_exceptions=True)

            # Workers are done — wait for the batch processor to finish
            batch_result = await batch_processor_task
            
            # Cancel the monitor task
            if monitor_task and not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Extract batch processor results
            if isinstance(batch_result, tuple) and len(batch_result) == 3:
                total_batches, total_predict_time, total_alerts_pushed = batch_result
            else:
                total_batches, total_predict_time, total_alerts_pushed = 0, 0.0, 0

        # ── Now launch the LLM explain worker with the actual alert count ──
        if total_alerts_pushed > 0:
            print(f"\n  🚨 {total_alerts_pushed} critical alerts detected — launching LLM explainer...\n")

            llm_worker_task = asyncio.create_task(
                llm_explain_worker(alert_queue, total_alerts_pushed),
                name="llm-explain-worker",
            )

            # Wait for the LLM worker to finish processing all alerts
            await llm_worker_task
        else:
            print(f"\n  ✅ No critical alerts to explain.\n")

    except asyncio.CancelledError:
        # Handle cancellation gracefully - don't let it propagate
        print("\n  ⚠️  Simulation interrupted")
    
    elapsed = time.monotonic() - start_time

    # ── Summary ──
    print(f"\n{'─'*70}")
    print(f"  ✅ SIMULATION COMPLETE")
    print(f"{'─'*70}")
    print(f"  Total logs generated : {total_logs:,}")
    print(f"  Total batches        : {total_batches:,}")
    print(f"  Critical alerts      : {total_alerts_pushed}")
    print(f"  Wall-clock time      : {elapsed:.2f}s")
    if elapsed > 0:
        print(f"  Average throughput   : {total_logs / elapsed:.0f} logs/sec")
        print(f"  Total inference time : {total_predict_time:.3f}s")
        print(f"  Pct time in ML       : {total_predict_time / elapsed * 100:.1f}%")
    if simulate_time and total_logs >= 1000:
        avg_delay = (min_delay + max_delay) / 2
        simulated_time = total_logs * avg_delay / num_workers
        print(f"  Simulated time       : {simulated_time:.1f}s (across {num_workers} workers)")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────
# 7. Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    """
    Entry point for the script.

    Parses optional command-line arguments and starts the simulation.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="🌩️  Cloud Log Simulator with ML Batch Consumer & LLM Explainer — "
                    "Generate realistic cloud log entries in real-time using asyncio, "
                    "batch them, extract NumPy features, simulate predict_on_batch(), "
                    "and explain critical alerts with a simulated Gemini LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cloud_log_simulator.py
  python cloud_log_simulator.py --logs 5000 --workers 20
  python cloud_log_simulator.py --logs 100000 --workers 50 --min-delay 0.01 --max-delay 0.1
        """,
    )

    parser.add_argument(
        "--logs", "-n",
        type=int,
        default=2000,
        help="Total number of log entries to generate (default: 2000)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Number of concurrent worker coroutines (default: 10)",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.05,
        help="Minimum delay in seconds between log entries (default: 0.05)",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.5,
        help="Maximum delay in seconds between log entries (default: 0.5)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress the time simulation summary at the end",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Use CSV file (cloud_security_logs.csv) for log data instead of synthetic logs",
    )

    args = parser.parse_args()

    # Run the async simulation
    asyncio.run(
        simulate_cloud_logs(
            total_logs=args.logs,
            num_workers=args.workers,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            simulate_time=not args.no_summary,
            use_csv=args.csv,
        )
    )


if __name__ == "__main__":
    main()