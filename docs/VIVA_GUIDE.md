# TrafficIQ — Viva and Presentation Guide (Hinglish + English)

## 90-second presentation script
“Good morning sir/ma'am. Our project is TrafficIQ, a smart city traffic intelligence system built for Data Warehousing and Mining. Our goal is to demonstrate how raw traffic records can be converted into insights using an ETL pipeline and a star-schema warehouse. We use Pandas to clean data, store it in SQLite with fact and dimension tables, and perform OLAP queries with SQL. A Random Forest classification model predicts low, moderate or high congestion for user-input conditions. We also use K-Means to discover patterns in traffic observations. The React dashboard displays KPIs, trends, model results, clustering and dynamic OLAP tables. For safe and reproducible demonstration, the supplied dataset is synthetic, not live Mumbai traffic.”

## Demonstration flow (3 minutes)
1. Open **Overview**: explain total observations, mean speed, mean volume and percentage high congestion. Filter area.
2. Open **OLAP Explorer**: select `road` or `month` for roll-up, area for slice, congestion class for dice. Explain `GROUP BY`.
3. Open **Predict Traffic**: use rush-hour 18, vehicle_count 220, avg_speed_kmh 15; run the classifier. Try 03:00 with count 35, speed 65 for a second scenario. Explain probabilities are synthetic-data estimates.
4. Open **ML Laboratory**: explain actual test accuracy, macro F1, confusion matrix and K-Means segmentation.
5. Open **Hotspots**: explain area rankings, acknowledge simulated geography.
6. Open **Data Pipeline**: explain CSV -> ETL -> SQLite -> train -> API -> React and show CSV export.

## 25 quick viva answers

1. **What is DWM?** Data Warehousing stores integrated analytical data; Data Mining extracts useful patterns/predictions.
2. **What is ETL?** Extract from CSV, Transform/clean, Load into database.
3. **What is a data warehouse?** An integrated, structured store optimized for analysis rather than day-to-day transactions.
4. **What is a fact table?** Central table of observations and measures with foreign keys.
5. **What is a dimension table?** Contextual attributes like time and location.
6. **What is star schema?** A central fact table linked to dimensions.
7. **Why SQLite?** Lightweight, reliable, local database with real SQL, no separate database server to start.
8. **What is OLAP?** Online Analytical Processing: multidimensional aggregation for decisions.
9. **Roll-up?** Aggregate to a less detailed level, like records grouped by month rather than timestamp.
10. **Drill-down?** Analyze finer detail, e.g. month -> hour in a defined grouping.
11. **Slice?** Filter one dimension, like Andheri.
12. **Dice?** Filter two or more dimensions, like Andheri plus High congestion.
13. **What is Random Forest?** An ensemble of decision trees, combining their votes to classify congestion.
14. **What is supervised learning?** Model learns from examples with known class labels.
15. **Why Random Forest?** Handles nonlinear feature interactions and mixed traffic predictors well.
16. **What is K-Means?** Unsupervised method grouping similar numeric observations into K clusters.
17. **Why standardize features?** Prevents high-range metrics, such as vehicle counts, from dominating distance-based clustering.
18. **Why one-hot encode?** Convert area/road text categories into machine-readable indicator columns.
19. **What is train/test split?** Held-out test observations estimate performance on unseen examples from the same dataset distribution.
20. **What is accuracy?** Correct predictions divided by total test predictions.
21. **What is macro F1?** Average of each class's harmonic mean of precision and recall, giving classes equal weight.
22. **What is confusion matrix?** Table of actual versus predicted classes to show correct and incorrect classifications.
23. **What is silhouette score?** Measures how well samples fit their own cluster relative to other clusters.
24. **React vs FastAPI?** React renders frontend UI; FastAPI runs Python analytics and exposes JSON HTTP endpoints.
25. **Can this predict real Mumbai congestion?** Not reliably: included data is synthetic; the next step is ingesting verified sensor or government traffic data and evaluating on real-world held-out data.

## Exactly how frontend communicates with backend
React calls `fetch('/api/overview')` → Vite proxies `/api` to FastAPI on port 8000 → backend queries SQLite via `sqlite3` → returns JSON → React uses Recharts to render line, bar and donut charts. For predictions React POSTs JSON to `/api/predict` → Pydantic validates it → `joblib.load()` loads fitted pipeline → model outputs class/probabilities → React displays results.

## Code locations external may ask about

- Synthetic data generation: `backend/pipeline.py` `generate_demo()`
- ETL logic: `backend/pipeline.py` `clean()` and `build_db()`
- Actual SQL warehouse tables: `backend/pipeline.py` `CREATE TABLE` statements
- Actual classification model training: `backend/pipeline.py` `train()`
- Clustering model: `backend/pipeline.py` `KMeans()` inside `train()`
- OLAP database queries: `backend/main.py` `/api/olap`
- Inference: `backend/main.py` `/api/predict`
- React charts and page components: `frontend/src/main.jsx`

## Limitations / future scope

We use reproducible synthetic records and approximate areas, not verified realtime Mumbai conditions. No authenticated admin role or genuine road geospatial coordinates. In future, integrate official/live feed with proper permissions, weather API, authenticated pipeline, temporal validation, dashboards by measured routes, and compare several ML classifiers. High scores on synthetic labels must NOT be claimed as real-city prediction accuracy.
