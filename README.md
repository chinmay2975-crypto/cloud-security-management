# Cloud Posture Management - Log Simulator

A real-time cloud log generator with ML batch processing and LLM-powered alert explanation system. This project simulates thousands of cloud service log entries, processes them through a machine learning pipeline, and uses AI to explain critical security alerts.

## Features

- **Async Log Generation**: Simulates realistic cloud service logs from 10 different services using Python's asyncio
- **ML Batch Processing**: Batches log entries, extracts numerical features, and performs simulated inference
- **LLM Alert Explanation**: Background worker that explains critical alerts using a structured prompt template
- **Real-time Monitoring**: Progress tracking and performance metrics throughout execution
- **Configurable**: Adjustable log volume, worker count, delays, and batch sizes

## Architecture

```
┌─────────────┐
│   Workers   │ (10 concurrent coroutines generating logs)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  ML Batch Processor │ (batches logs, extracts features, runs inference)
└──────┬──────────────┘
       │
       ▼
┌──────────────────┐
│  Alert Queue     │ (critical predictions)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  LLM Explainer   │ (generates incident analysis for critical alerts)
└──────────────────┘
```

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the simulator with default settings (2000 logs, 10 workers):

```bash
python cloud_log_simulator.py
```

### Advanced Usage

Generate more logs with more workers:

```bash
python cloud_log_simulator.py --logs 5000 --workers 20
```

High-speed simulation with minimal delays:

```bash
python cloud_log_simulator.py --logs 100000 --workers 50 --min-delay 0.01 --max-delay 0.1
```

### Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--logs` | `-n` | 2000 | Total number of log entries to generate |
| `--workers` | `-w` | 10 | Number of concurrent worker coroutines |
| `--min-delay` | - | 0.05 | Minimum delay (seconds) between log entries |
| `--max-delay` | - | 0.5 | Maximum delay (seconds) between log entries |
| `--no-summary` | - | False | Suppress the time simulation summary |

## Project Structure

```
cloud-posture-management/
├── cloud_log_simulator.py    # Main simulator script
├── cloud_security_logs.csv   # Sample security log dataset (1000 entries)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## How It Works

### 1. Log Generation

The simulator creates realistic cloud service logs from 10 different services:
- API Gateway
- Auth Service
- Database Cluster
- Load Balancer
- CDN Edge
- Storage Bucket
- Message Queue
- Container Orchestrator
- Monitoring Agent
- Serverless Function

Each log entry includes:
- Timestamp (ISO 8601 format)
- Log level (DEBUG, INFO, WARN, ERROR, CRITICAL)
- Service name
- Formatted message with realistic data
- Environment (production/staging/development)
- Region

### 2. Feature Extraction

Each log entry is converted into an 8-dimensional feature vector:
- Severity (0-4, normalized)
- Service index (0-9, normalized)
- Environment score (production=1.0, staging=0.5, development=0.0)
- Region score (geographic encoding)
- Error flag (1 if ERROR/CRITICAL, else 0)
- Hour fraction (time of day)
- Minute fraction
- Second fraction

### 3. ML Inference Simulation

Features are batched and processed through a simulated neural network:
- Dense layer with pre-defined weights
- Softmax activation for 3-class prediction (normal/anomalous/critical)
- Artificial compute delay to mimic real inference latency
- Biased weights ensure high-severity logs trigger "critical" predictions

### 4. LLM Alert Explanation

Critical alerts (class 2 predictions) are sent to a background LLM worker that:
- Formats a structured Gemini-style prompt
- Simulates LLM inference latency (0.8-2.5 seconds)
- Generates incident analysis including:
  - Root cause assessment
  - Impact analysis
  - Recommended actions (2-3 bullet points)
  - Severity classification (P0/P1/P2)

## Example Output

```
  🧠 [ML BATCH #1] size=32 | infer=  2.1ms | confidence=0.67 | preds={'normal': 28, 'anomalous': 3, 'critical': 1}

  🤖 [LLM ALERT #1] Service: database-cluster | LLM latency: 1.3s | Confidence: 0.89
  ┌─ Incident Analysis ──────────────────────────────────┐
  │ **Root Cause:** Query performance degradation, possible missing index or replication lag impacting read/write throughput.
  │ **Impact:** Read/write operations on affected tables may see 2-5x latency increase. Partial query failures possible.
  │ **Recommended Action:**
  │   • Immediately investigate database-cluster logs and metrics in the monitoring dashboard.
  │   • Check recent deployments or configuration changes to database-cluster in the last 30 minutes.
  │   • Escalate to the database-cluster on-call team with the alert details and ML prediction confidence (89%).
  │ **Severity:** P1 — Significant degradation of a critical service. Requires urgent investigation within 15 minutes.
  └──────────────────────────────────────────────────────┘
```

## Performance

Typical performance metrics (2000 logs, 10 workers):
- **Throughput**: ~20-50 logs/second
- **ML Inference**: ~0.5-2% of total time
- **Critical Alerts**: ~3-5% of logs (based on severity distribution)
- **LLM Processing**: ~1-3 seconds per alert

## Data Format

The included `cloud_security_logs.csv` contains 1000 sample security log entries with the following fields:
- timestamp
- user_id
- user_role
- mfa_verified
- request_source
- ip_country
- action_type
- resource_type
- is_public_exposed
- privilege_level

## Requirements

- Python 3.7+
- numpy >= 1.20.0
- asyncio (built-in)
- Standard library modules: random, time, math, datetime, typing, argparse

## Use Cases

- **Security Training**: Demonstrate cloud security monitoring workflows
- **ML Pipeline Testing**: Test batch processing and feature extraction pipelines
- **Load Testing**: Generate realistic log volumes for testing monitoring systems
- **Demo/Prototyping**: Quick way to generate sample data for presentations
- **SRE Training**: Practice incident response with AI-generated explanations

## Future Enhancements

- Integration with real ML models (TensorFlow/PyTorch)
- Actual LLM API integration (OpenAI, Anthropic, Google Gemini)
- Log persistence to files or databases
- Web dashboard for real-time monitoring
- Alert aggregation and deduplication
- Integration with real cloud services (AWS CloudWatch, Azure Monitor, etc.)

## License

This project is provided as-is for educational and demonstration purposes.

## Contributing

Feel free to submit issues and enhancement requests.