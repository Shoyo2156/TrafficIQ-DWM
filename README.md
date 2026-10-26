# TrafficIQ — Smart City Traffic Intelligence | DWM Practical

**Functional local-first demo:** React + Vite + Recharts frontend, Python FastAPI backend, SQLite **star-schema** warehouse, ETL, OLAP, fitted RandomForest congestion classifier, KMeans traffic profiles, scenario prediction, CSV import/export, model evaluation.

> **DATA HONESTY:** Included traffic observations are reproducible **synthetic examples**, not real Mumbai traffic, live feeds, GPS readings or verified accident records. Names of regions are labels for demonstration. Metrics are computed from actual fitting and holdout evaluation on the synthetic dataset. No internet API key, PostgreSQL installation or account is necessary.

## Windows setup (Python 3.10–3.12 recommended, Node.js 18+)

1. Extract ZIP to an ordinary local folder, e.g. `C:\TrafficIQ`.
2. Install Python and Node.js if missing. Check: `py --version` and `node --version`.
3. Double-click **START_ALL.bat**. It opens two command windows. First start installs Python/npm dependencies, which requires internet.
4. Open **http://127.0.0.1:5173**. Backend API documentation: **http://127.0.0.1:8000/docs**.
5. Keep both terminals open during the demo.

### Manual commands (VS Code PowerShell)

Backend terminal, project root:
```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m backend.pipeline
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Frontend terminal:
```powershell
cd frontend
npm install
npm run dev
```
Open http://127.0.0.1:5173. If 8000/5173 are occupied, stop the program on that port before starting. API requests are proxied through Vite.

## Application modules

1. **Overview:** average speed, observations, volume, high congestion share, hourly and monthly trends.
2. **OLAP Explorer:** SQL `GROUP BY` (roll-up), selected area (slice), area plus congestion class (dice); dimensions: area, road, month, hour, congestion level.
3. **Predict Traffic:** type speed, volume, weather, day and location; view Random Forest class and probability; no external inference service.
4. **ML Laboratory:** holdout accuracy, macro F1, confusion matrix, actual KMeans clusters and silhouette score.
5. **Hotspots:** area high-congestion ranking (synthetic statistical profiles only).
6. **Data Pipeline:** view flow, upload a schema-compatible CSV, retrain the models, export monthly summaries.

## Warehouse architecture

`traffic_demo.csv` → `clean()` (required columns, type validation, duplicates removed, missing-number median imputation, safe clipping, invalid classes excluded) → SQLite:

- `dim_location(location_id, area, road)`
- `dim_time(time_id, timestamp, date, month, hour, day_of_week)`
- `fact_traffic(observation_id, location_id, time_id, vehicle_count, avg_speed_kmh, rainfall_mm, temperature_c, incident_flag, congestion_level)`

Relationships: fact_traffic.location_id → dim_location.location_id; fact_traffic.time_id → dim_time.time_id. Measures: vehicle volume, speed, incidents. Dimensions: area, road, date, month, hour and class.

**ML:** RandomForestClassifier with 120 trees; categorical OneHotEncoder, numeric StandardScaler; stratified 78/22 train/test split. Classes: Low/Moderate/High. `KMeans(n_clusters=4)` on scaled volume, speed, rain and hour. Metrics written to JSON after model training. Algorithms are real scikit-learn fitting; however, labels in the demo are generated from a synthetic rule so real-world predictive utility is unproven.

## CSV format

`timestamp,area,road,hour,day_of_week,rainfall_mm,temperature_c,vehicle_count,avg_speed_kmh,incident_flag,congestion_level`

The included `data/traffic_demo.csv` is a sample. API upload is limited to 8 MB and requires 120+ valid rows and at least 2 congestion classes. Uploaded data and fitted outputs replace current local copies; back them up first. New model features should match training schema. No login/access control; **do not publicly deploy upload and retrain endpoints**.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm run build
```

## Source files

- `backend/pipeline.py`: ETL, SQLite DDL, synthetic data generation, model training.
- `backend/main.py`: HTTP API endpoints and OLAP queries.
- `frontend/src/main.jsx`: React dashboard and API requests.
- `frontend/src/style.css`: responsive custom dark theme.
- `data/`: generated model `.joblib`, warehouse `.db`, CSV and metrics.
- `docs/VIVA_GUIDE.md`: explanation and rapid-fire answers.

**Known scope limitation:** no true geospatial route maps, no real traffic feeds, no Apriori algorithm in this version, no user/admin authentication, and traffic predictions are based on the demo dataset, not road sensors.
