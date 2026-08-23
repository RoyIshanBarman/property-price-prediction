# Property Investment Advisor

A machine learning system that predicts property prices and generates buy/hold/sell-style investment recommendations for real estate listings, served through a Flask API and a browser-based dashboard.

Given a location, the app pulls every property listed there, predicts what each one *should* be worth using a stacked ML model, compares that prediction to the actual listed price, and labels each property **Strong Buy / Buy / Hold / Don't Buy** based on the predicted appreciation gap.

---

## Table of Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [API reference](#api-reference)
- [Frontend](#frontend)
- [Known limitations](#known-limitations)
- [Requirements file gaps](#requirements-file-gaps)

---

## How it works

1. `dataset.csv` holds ~5,000 property records across 20 locations, each with real features (price, size, income, crime rate, etc.).
2. Training scripts engineer additional features (see [Dataset](#dataset)) and fit two regressors — **XGBoost** and **Random Forest** — then combine them with a **Stacking Regressor** (XGBoost meta-learner) to predict `Price`.
3. The trained models are serialized with `joblib` (`xgb_model.pkl`, `rf_model.pkl`, `scaler.pkl`, `stacked_model.pkl`).
4. A Flask API (`apptest.py`) loads those artifacts, filters the dataset by location, runs predictions, and computes an appreciation rate:
   ```
   appreciation_rate = (predicted_price - actual_price) / actual_price * 100
   ```
5. Each property is labeled based on that rate:

   | Appreciation rate | Recommendation |
   |---|---|
   | > 30% | Strong Buy |
   | > 10% | Buy |
   | > 0% | Hold |
   | ≤ 0% | Don't Buy |

6. The static frontend (`index.html` + `app.js`) lets a user type a location, calls the API, and renders a sortable results table with an "average appreciation" and "top recommendation" summary.

---

## Project structure

```
Property_price_prediction_model/
├── dataset.csv                  # Source data — 5,000 properties, 20 raw columns
├── test2.py                     # ★ Main training script (RF + XGBoost stack) — produces the .pkl files actually used
├── apptest.py                   # ★ Flask API that serves predictions (the backend the frontend calls)
├── index.html                   # Dashboard UI
├── app.js                       # Frontend logic — calls the Flask API at localhost:5000
├── server.py                    # Minimal static file server for the frontend (serves a ./static folder on :5500)
├── xgb_model.pkl                # Trained XGBoost regressor
├── rf_model.pkl                 # Trained Random Forest regressor
├── stacked_model.pkl            # Trained StackingRegressor (xgb + rf → xgb meta-learner)
├── scaler.pkl                   # Fitted StandardScaler used to scale features before prediction
├── requirement.txt              # Python dependencies (incomplete — see below)
├── main.py                      # Alternate/earlier training script — XGBoost + Keras neural net stack
├── app.py                       # Alternate/earlier FastAPI backend for the main.py models
├── ds.py                        # Matplotlib/Seaborn script to plot actual vs. predicted price for a CSV of results
├── .idea/, __pycache__/         # Editor/build artifacts
└── .gitignore
```
★ = the pipeline that is actually wired together and working end-to-end (training → saved models → API → frontend).

`main.py` and `app.py` are a separate, earlier attempt at the same problem using a Keras neural network instead of Random Forest, and a FastAPI server instead of Flask. They are **not** used by `index.html`/`app.js` (which call the Flask server on port 5000) and `app.py` currently references a saved-model filename that `main.py` never actually creates — see [Known limitations](#known-limitations).

---

## Dataset

`dataset.csv` — 5,000 rows, one per property, across 20 Kolkata-area locations (Alipore, Ballygunge, Baranagar, Baguiati, Behala, Chinar Park, Dum Dum, Garia, Howrah, Jadavpur, Joka, Kasba, Lake Town, New Town, Park Street, Rajarhat, Salt Lake, Sealdah, Tollygunge, Ultadanga).

**Raw columns:**

| Column | Description |
|---|---|
| `Property_Name` | Listing identifier |
| `Location` | Neighborhood/area |
| `Price` | Target variable — current listed price |
| `Size_sqft` | Property size |
| `Bedrooms`, `Bathrooms` | Room counts |
| `Crime_Rate` | Local crime index |
| `School_Rating` | Nearby school quality score |
| `Rental_Yield` | Estimated rental yield |
| `Price_per_sqft` | Price density |
| `Median_Income` | Median household income in the area |
| `Unemployment_Rate` | Local unemployment rate |
| `Infrastructure_Score` | Existing infrastructure quality |
| `Flood_Risk` | Flood risk index |
| `Historical_Appreciation` | Past appreciation figure |
| `Growth_Projection` | Projected growth score |
| `Distance_to_Metro` | Distance to nearest metro station |
| `Property_Age` | Age of the property in years |
| `Air_Quality_Index` | Local AQI |
| `Planned_Infrastructure` | Flag/count for planned infrastructure projects |

**Engineered features** (computed at training and inference time from the raw columns above):

- `Price_to_Income_Ratio` = `Price / Median_Income`
- `Growth_vs_Infra` = `Growth_Projection * Infrastructure_Score`
- `Total_Rooms` = `Bedrooms + Bathrooms`
- `Appreciation_Rate` = `(Price - Historical_Appreciation) / Historical_Appreciation` (guarded against divide-by-zero in `test2.py`/`apptest.py`)

**Synthetic features** (randomly generated, seeded with `np.random.seed(52)` in the active pipeline): `Economic_Growth_Index`, `Crime_Trend`, `Public_Transport_Accessibility`, `Flood_Risk_Trend`, `Air_Quality_Trend`, `Historical_Price_Trends`, `Interest_Rate`, `Population_Density`, `Walk_Score`, `Seasonality`, `Future_Infrastructure_Projects`, `Government_Policies`. See [Known limitations](#known-limitations) for why these should be treated with caution.

`Location` is one-hot encoded (`pd.get_dummies(..., drop_first=True)`) before training/inference.

---

## Models

Trained in `test2.py`:

- **XGBoost Regressor** — `n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42`
- **Random Forest Regressor** — `n_estimators=100, max_depth=10, random_state=42`
- **Stacking Regressor** — combines the two above, with a second, smaller XGBoost model (`n_estimators=50, learning_rate=0.05, max_depth=3`) as the meta-learner, `cv=5`
- Features are scaled with `StandardScaler` before every model sees them.
- Train/test split: 80/20, `random_state=42`.
- All artifacts are saved with `joblib.dump(...)`: `xgb_model.pkl`, `rf_model.pkl`, `scaler.pkl`, `stacked_model.pkl`.

The alternate pipeline in `main.py` instead stacks XGBoost with a 14-layer-deep Keras `Sequential` neural network (`KerasRegressor` via `scikeras`) and saves the NN separately as `nn_model.h5`.

---

## Setup

```bash
git clone https://github.com/RoyIshanBarman/Property_price_prediction_model.git
cd Property_price_prediction_model
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirement.txt
pip install flask flask-cors    # needed for apptest.py — see "Requirements file gaps"
```

---

## Running the app

**1. (Optional) Retrain the models** — the repo already ships trained `.pkl` files, so this step is only needed if you change the dataset or model code:

```bash
python test2.py
```
This will prompt you for a location on the command line (used for a quick CLI sanity check) and re-writes `xgb_model.pkl`, `rf_model.pkl`, `scaler.pkl`, and `stacked_model.pkl`.

**2. Start the API server:**

```bash
python apptest.py
```
Runs a Flask dev server on `http://127.0.0.1:5000`.

**3. Serve the frontend:**

```bash
mkdir static && cp index.html app.js static/
python server.py
```
Serves the dashboard on `http://localhost:5500`. Open that URL in a browser, enter a location (e.g. `Salt Lake`), and view recommendations.

> `server.py` does `os.chdir('static')` before serving, so `index.html` and `app.js` must be inside a `static/` folder relative to where you run it — they currently sit at the repo root and need to be moved/copied first.

---

## API reference

Base URL: `http://127.0.0.1:5000`

### `GET /`
Health check.
```json
{ "message": "Property Investment Advisor API is running" }
```

### `GET /locations`
Returns the sorted list of all locations present in the dataset.
```json
["Alipore", "Ballygunge", "Baguiati", "..."]
```

### `POST /predict`
Body:
```json
{ "Location": "Salt Lake" }
```
Response — one object per property in that location:
```json
[
  {
    "property_name": "Horizon Towers 67",
    "actual_price": 23081788.0,
    "predicted_price": 24500000.0,
    "appreciation_rate": 6.14,
    "recommendation": "Hold",
    "bedrooms": 1,
    "bathrooms": 3,
    "size_sqft": 1594.0
  }
]
```
Errors: `400` if `Location` is missing, `404` if no properties match the given location, `500` on internal errors.

---

## Frontend

- Plain HTML + Tailwind classes + vanilla JS (`app.js`), no build step.
- On search, `app.js` POSTs to `http://127.0.0.1:5000/predict`, renders a results table (property, price, predicted price, appreciation %, recommendation, beds/baths/size), and computes summary stats (property count, average appreciation, top recommendation).
- Includes a CSV export button for the current result set.
- The API URL is hardcoded to `127.0.0.1:5000` — update it in `app.js` if you deploy the backend elsewhere.

---

## Known limitations

These are worth knowing before trusting the output of this app:

- **The "synthetic" macro features are randomly generated, not real data.** Columns like `Interest_Rate`, `Population_Density`, `Walk_Score`, `Economic_Growth_Index`, etc. are filled with `np.random` calls rather than sourced from anywhere. They're seeded (`np.random.seed(52)`) so results are at least reproducible across runs of the *same* script, but the values carry no real-world signal — model weight assigned to these columns is essentially noise-fitting. Replace them with real data sources (e.g. actual interest rate feeds, census density data) if you want the "Interest Rate" or "Population Density" influence on price to mean anything.
- **`main.py` / `app.py` are a disconnected, unused pipeline.** `app.py` loads a file (`stacked_model_final_estimator.pkl`) that `main.py` never actually saves — that FastAPI server won't start as committed. The Flask pipeline (`test2.py` + `apptest.py`) is the one that's actually wired to the frontend and works end-to-end.
- **No model evaluation is surfaced.** `test2.py` fits the stack but doesn't print or save an R²/RMSE against the held-out test set, so there's currently no visibility into how accurate the predictions actually are.
- **CORS/dev-only setup.** `apptest.py` runs Flask's built-in dev server (`debug=True`) — fine locally, not suitable for production as-is (no WSGI server, no auth on the API).
- **`rf_model.pkl` and `xgb_model.pkl` are saved but never loaded individually at inference** — `apptest.py` loads them into `app_state.models` but only `app_state.models['stacked']` is actually used in `/predict`.

---

## Requirements file gaps

`requirement.txt` only lists the FastAPI/`main.py`-era dependencies. Running the *actual* active pipeline (`test2.py` + `apptest.py`) additionally needs:

```
flask
flask-cors
```

And running the unused `main.py`/`app.py` pipeline needs, on top of `requirement.txt`:

```
tensorflow
scikeras
```

Consider splitting `requirement.txt` into the Flask pipeline's actual dependencies, or trimming the repo down to one pipeline, to avoid confusion for anyone cloning this.
