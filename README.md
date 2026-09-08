# SkyGuard AI

**Intelligent Real-Time Anomaly Detection for Automatic Weather Station Sensors**

SkyGuard AI is a research-oriented system for detecting abnormal, inconsistent, or potentially faulty observations from Automatic Weather Stations (AWS) using:

- Temperature (°C)
- Atmospheric Pressure (hPa)
- Relative Humidity (%)

The MVP focuses on **one station/location** and uses historical observations to learn normal temporal behavior. The first ML baseline is **Isolation Forest**, combined with deterministic quality-control checks and explainable anomaly classification.

---

## 1. Problem

AWS observations can be affected by sensor spikes, frozen values, gradual drift, missing data, communication errors, and other failures. A useful system should flag suspicious observations while avoiding false alarms for genuine meteorological variation.

SkyGuard therefore uses a **hybrid QC + ML architecture**:

```text
AWS / Dataset
      |
      v
Data Ingestion
      |
      v
Preprocessing + Quality Control
      |
      v
Temporal / Cross-Sensor Features
      |
      +----------------------+
      |                      |
      v                      v
Rule-Based QC          ML Anomaly Detection
                             |
                             v
                       Isolation Forest
      |                      |
      +----------+-----------+
                 v
          Decision / Fusion
                 |
                 v
       Explanation + Confidence
                 |
       +---------+---------+
       |                   |
       v                   v
  PostgreSQL          WebSocket
                           |
                           v
                    React Dashboard
```

## 2. MVP Goal

The first milestone is **not deployment**.

The first milestone is:

> Given historical T/P/RH data from one station, can SkyGuard detect realistic injected sensor faults with a useful false-positive/false-negative trade-off?

Only after the ML pipeline is validated should the team build the full real-time application.

## 3. Repository Structure

```text
skyguard-ai/
├── README.md
├── LICENSE
├── .gitignore
├── docker-compose.yml
│
├── docs/
│   ├── research/
│   │   └── skyguard_ai_research.pdf
│   ├── architecture/
│   ├── api/
│   └── decisions/
│
├── data/
│   ├── raw/                 # Original datasets; do not modify
│   ├── processed/           # Cleaned/feature-ready data
│   └── synthetic/           # Injected anomaly datasets
│
├── ml/
│   ├── src/
│   │   ├── data/            # Loading, validation, preprocessing
│   │   ├── features/        # Temporal/cross-sensor features
│   │   ├── models/          # Isolation Forest and future models
│   │   └── evaluation/      # Metrics and error analysis
│   ├── models/              # Versioned model artifacts
│   ├── notebooks/           # Exploration only
│   ├── tests/
│   └── requirements.txt
│
├── backend/
│   ├── pom.xml
│   └── src/
│       ├── main/java/com/skyguard/api/
│       │   ├── controller/
│       │   ├── service/
│       │   ├── repository/
│       │   ├── model/
│       │   └── config/
│       └── main/resources/
│
├── frontend/
│   ├── package.json
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       ├── hooks/
│       ├── types/
│       └── utils/
│
├── simulator/               # CSV/MQTT sensor simulator
├── infra/
│   ├── docker/
│   └── k8s/
└── scripts/
```

## 4. Data Contract

Canonical observation:

```json
{
  "station_id": "AWS_001",
  "timestamp": "2026-01-01T10:00:00Z",
  "temperature_c": 24.3,
  "pressure_hpa": 1008.2,
  "humidity_pct": 61.4
}
```

Keep raw observations immutable. Quality-control flags and model outputs should be stored separately rather than silently deleting observations.

## 5. Anomaly Types for the MVP

The synthetic-data generator should support at least:

1. Sudden spike
2. Sudden drop
3. Frozen/stuck sensor
4. Gradual sensor drift
5. High-frequency noise
6. Missing values / communication gaps
7. Cross-sensor inconsistency

The injected anomaly must be labeled separately from the original data so that evaluation has ground truth.

## 6. Feature Engineering

Start with:

- Raw T/P/RH
- First differences: ΔT, ΔP, ΔRH
- Rolling mean
- Rolling standard deviation
- Rate of change
- Repeated-value/frozen-value counts
- Cross-sensor consistency features
- Time-of-day / seasonal features when justified by the dataset

Avoid leakage: features must only use information available at or before the observation being classified.

## 7. ML Baseline

### Isolation Forest

Isolation Forest is the first baseline because it is appropriate for unsupervised anomaly detection and can learn from predominantly normal observations.

The experiment should not assume it is the final model.

Compare it later with:

- Autoencoder
- LSTM/sequence model
- Other appropriate anomaly detectors

Choose the final model based on measured performance, operational complexity, and false-alarm behavior.

## 8. Evaluation

Do not report accuracy alone.

Track:

- Precision
- Recall
- F1-score
- False Positive Rate
- False Negative Rate
- Detection latency
- Per-anomaly-type performance

For the MVP, evaluate separately on:

- clean normal observations
- synthetic spikes
- frozen values
- drift
- noise
- missing data
- cross-sensor inconsistencies

Use time-aware train/validation/test splits to avoid temporal leakage.

## 9. Backend

Spring Boot owns:

- station management
- telemetry ingestion
- anomaly records
- REST APIs
- authentication later
- PostgreSQL persistence
- WebSocket/STOMP updates

Suggested endpoints:

```text
POST /api/telemetry
GET  /api/telemetry/latest
GET  /api/telemetry/history
GET  /api/anomalies
GET  /api/anomalies/{id}
GET  /api/stations/{id}/health
```

The ML service should remain a separate Python service initially.

## 10. ML Service

Use:

- Python
- FastAPI
- Pandas
- NumPy
- scikit-learn
- joblib or an equivalent model-artifact mechanism

Example inference response:

```json
{
  "status": "ANOMALOUS",
  "score": 0.91,
  "sensor": "temperature",
  "type": "spike",
  "confidence": 0.94
}
```

The exact score semantics and thresholds must be defined by experiments rather than arbitrary constants.

## 11. Real-Time Flow

MVP:

```text
CSV / Sensor Simulator
        |
        v
Spring Boot
        |
        v
Feature + ML Service
        |
        +----> PostgreSQL
        |
        +----> WebSocket/STOMP
                    |
                    v
               React UI
```

Later:

```text
AWS -> MQTT -> Ingestion -> Stream Processing -> ML -> DB -> Dashboard
```

Kafka should be introduced only when there is a demonstrated need for higher-throughput streaming, buffering, or multiple consumers.

## 12. Dashboard

The dashboard should show:

- current T/P/RH
- live time-series charts
- anomaly timeline
- severity
- confidence
- affected sensor
- anomaly type
- explanation
- station health
- historical anomaly statistics

A good demo should inject a known temperature spike and show SkyGuard detecting and explaining it in real time.

## 13. Development Order

### Phase 1 — Research + Dataset

- Select dataset
- Document station/location
- Validate timestamp frequency
- Understand missingness
- Establish data dictionary
- Build preprocessing pipeline

### Phase 2 — Synthetic Anomalies

- Implement anomaly injection
- Generate labeled benchmark datasets
- Verify that injected anomalies look realistic

### Phase 3 — ML Baseline

- Feature engineering
- Isolation Forest
- Threshold tuning
- Evaluation
- Error analysis

### Phase 4 — Backend

- Spring Boot
- PostgreSQL
- ML-service integration
- REST APIs

### Phase 5 — Real-Time Dashboard

- React
- WebSocket/STOMP
- Live charts
- Anomaly explanations

### Phase 6 — Deployment

- Docker
- Environment configuration
- Observability
- Cloud deployment

### Phase 7 — Advanced Research

- Autoencoder/LSTM comparison
- Explainability
- Multi-station learning
- Kafka
- Edge inference

## 14. Engineering Rules

1. Never overwrite raw data.
2. Never delete a suspected anomaly before the detection stage.
3. Keep preprocessing reproducible.
4. Version datasets, features, and model artifacts.
5. Prevent temporal leakage.
6. Record the model version with every prediction.
7. Keep anomaly explanations tied to observable evidence.
8. Do not claim real-world fault detection performance from synthetic anomalies alone.
9. Benchmark false alarms explicitly.
10. Keep the MVP small enough to validate scientifically.

## 15. Definition of Done for MVP

The MVP is complete when:

- [ ] A documented dataset is selected
- [ ] Raw data can be reproduced from source
- [ ] Data validation/preprocessing is automated
- [ ] Synthetic anomaly generator works
- [ ] Isolation Forest baseline works
- [ ] Time-aware evaluation is implemented
- [ ] Precision/recall/F1 and false-positive behavior are reported
- [ ] Backend can ingest observations
- [ ] ML service can return anomaly decisions
- [ ] PostgreSQL stores observations and results
- [ ] React dashboard displays live/simulated data
- [ ] Anomalies are explainable at a useful level
- [ ] Entire MVP runs with Docker

## 16. Research Principle

SkyGuard should be treated as a **research + engineering project**, not just an application.

Every major design decision should answer:

> What evidence supports this choice, and how will we test whether it works?

See `docs/research/skyguard_ai_research.pdf` for the initial research brief.
