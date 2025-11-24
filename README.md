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
