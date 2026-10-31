# Execution & Reproduction Guide

**Project:** Proof of Concept Network Attack Onset Forecasting  
**Team:** TECH ROOKIES, PSC26153 (II-year Engineering Students)  
**Dataset:** CIC-IDS2017 Offline Replay (GeneratedLabelledFlows / TrafficLabelling)  
**Target Environment:** Python 3.10.12, Docker, Docker Compose

---

## Important Note on Model Artifacts
When `poc_harness.py` runs, it saves two model files:
1. `model/{run-id}_model.joblib` (e.g., `model/R01_model.joblib` and `model/R02_model.joblib` to preserve audit evidence across runs).
2. `model/model.joblib` (a copy of the most recently executed run's model, which is the file loaded by the API).

> **Note:** Running R02 after R01 updates `model/model.joblib` to the R02 model (window duration = 15s). If you want the containerised API to serve the R01 model (window duration = 10s), manually copy `model/R01_model.joblib` over `model/model.joblib` before starting or testing the container:
> - **Linux/macOS:** `cp model/R01_model.joblib model/model.joblib`
> - **Windows (PowerShell):** `Copy-Item model/R01_model.joblib model/model.joblib -Force`

---

## Step-by-Step Instructions

### Step 1: Create Virtual Environment and Install Dependencies

#### Linux / macOS
```bash
# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
# Create Python virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 2: Run Experiment R01 (10-second Windows)

Run the harness on the CIC-IDS2017 dataset folder using a 10-second aggregation window:

#### Linux / macOS
```bash
python poc_harness.py \
  --run-id R01 \
  --window-seconds 10 \
  --data-dir /path/to/CIC-IDS2017/GeneratedLabelledFlows \
  --out-dir ./
```

#### Windows (PowerShell)
```powershell
python poc_harness.py `
  --run-id R01 `
  --window-seconds 10 `
  --data-dir "C:\path\to\CIC-IDS2017\GeneratedLabelledFlows" `
  --out-dir ./
```

**Artifacts Generated:**
- `R01_log.csv` (eval window scoring log)
- `R01_eval_features.csv` (canonical features for latency benchmarking)
- `R01_summary.json` (audit metrics: T1, T2, T3, counts, timestamp strategy)
- `model/R01_model.joblib` and `model/model.joblib`

---

### Step 3: Run Experiment R02 (15-second Windows)

Run the harness with a 15-second aggregation window:

#### Linux / macOS
```bash
python poc_harness.py \
  --run-id R02 \
  --window-seconds 15 \
  --data-dir /path/to/CIC-IDS2017/GeneratedLabelledFlows \
  --out-dir ./
```

#### Windows (PowerShell)
```powershell
python poc_harness.py `
  --run-id R02 `
  --window-seconds 15 `
  --data-dir "C:\path\to\CIC-IDS2017\GeneratedLabelledFlows" `
  --out-dir ./
```

**Artifacts Generated:**
- `R02_log.csv`
- `R02_eval_features.csv`
- `R02_summary.json`
- `model/R02_model.joblib` and `model/model.joblib`

---

### Step 4: Build and Deploy the Containerised Serving API

Ensure `model/model.joblib` is present in the `model/` directory (mounted read-only into `/app/model`):

#### Linux / macOS / Windows (Docker Compose commands are identical)
```bash
# Build the container image
docker compose build

# Start the serving container in detached mode
docker compose up -d

# Inspect container status and healthcheck
docker compose ps
```

Verify the endpoint is healthy:

#### Linux / macOS
```bash
curl -i http://localhost:8000/health
```

#### Windows (PowerShell)
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```
Expected response:
```json
{"status": "ok"}
```

---

### Step 5: Run Latency and Throughput Benchmark

Benchmark the serving container using the canonical evaluation features from R01:

#### Linux / macOS
```bash
python latency_test.py \
  --n 1000 \
  --source R01_eval_features.csv \
  --base-url http://127.0.0.1:8000 \
  --warmup 20
```

#### Windows (PowerShell)
```powershell
python latency_test.py `
  --n 1000 `
  --source R01_eval_features.csv `
  --base-url http://127.0.0.1:8000 `
  --warmup 20
```

**Artifacts Generated:**
- `latency_log.csv` (per-request latency, HTTP status, failure flags)
- `latency_summary.json` (n_requests, failed, p50, p95, p99, system/server CPU %)

---

### Step 6: Teardown Container
When testing is complete, stop and remove the container:
```bash
docker compose down
```
