# Sales Prediction Pipeline

This minimal project demonstrates a full ML pipeline for sales prediction with:
- data loading
- preprocessing and feature engineering
- EDA and visualizations saved to `outputs/`
- train/test split, model training and evaluation
- saving model and preprocessor (`joblib`)
- simple Flask app for real-time predictions

Quick setup (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\train.py --data data\sample_sales.csv
# Start app
python src\app.py
``` 

Endpoints:
- `POST /predict` JSON body with fields: `date` (YYYY-MM-DD), `store`, `item`, `promotion` (0/1), `price` (float)

Outputs produced:
- `artifacts/model.joblib` - trained model
- `artifacts/preprocessor.joblib` - preprocessing pipeline
- `outputs/` - EDA plots

See the `src/` folder for implementation details.
# Sales Prediction Pipeline

This project demonstrates a small end-to-end ML demo for sales prediction with:
- data loading
- preprocessing and feature engineering
- EDA and visualizations saved to `outputs/`
- model training and saving (`joblib` artifacts)
- a Flask app providing a prediction endpoint and a simple dashboard

Quickstart (Windows PowerShell)
1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. (Optional) create dummy artifacts for local testing if included:

```powershell
python create_dummy_artifacts.py
```

3. Run the app (development server):

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Open the UI at `http://127.0.0.1:5000/`.

Files & folders of interest
- `app.py` — Flask application and endpoints
- `uploads/` — uploaded CSV files (created at runtime)
- `outputs/` — generated EDA images (created at runtime)
- `artifacts/` — model and preprocessor `joblib` files used by the prediction endpoint

Notes
- For local development the Flask dev server is used. For production, run behind a WSGI server.
- If the browser shows upload/network issues, open DevTools (F12) → Console + Network and inspect requests to `/upload_csv` or `/upload_base64`.
- A helper PowerShell script is available at `scripts\setup_venv.ps1` to bootstrap the `.venv` and install dependencies automatically on Windows.

Troubleshooting
- If plots are not generated, check `logs/upload.log` for upload/plotting errors and ensure `matplotlib` and `seaborn` are installed.

License & Safety
- This is demo code for local development only. Do not expose the development server publicly without proper hardening.

