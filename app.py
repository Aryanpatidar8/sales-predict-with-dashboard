from flask import Flask, request, jsonify, render_template
from ml_utils import load_artifacts, predict_single
from pathlib import Path
import time
import os
import traceback
import base64
import io

app = Flask(__name__)
model, preprocessor = None, None

# Keep a short in-memory history of recent predictions for the dashboard
app.config.setdefault("PREDICTION_HISTORY", [])
app.config.setdefault("PREDICTION_HISTORY_MAX", 200)


# Lazy-load artifacts before handling the first request. Use before_request
# and a guard so this works on Flask versions that don't provide
# `before_first_request`.
@app.before_request
def ensure_models_loaded():
    global model, preprocessor
    # Attempt to reload artifacts if files changed, or load initially.
    _check_and_reload_artifacts()


def _artifact_paths(artifacts_dir: str = "artifacts") -> dict:
    base = Path(artifacts_dir)
    return {
        "model": base / "model.joblib",
        "preprocessor": base / "preprocessor.joblib",
    }


def _get_mtimes(paths: dict) -> dict:
    mt = {}
    for k, p in paths.items():
        try:
            mt[k] = p.stat().st_mtime
        except FileNotFoundError:
            mt[k] = None
    return mt


def _check_and_reload_artifacts(artifacts_dir: str = "artifacts"):
    """Reload artifacts if their mtimes changed. Sets `ARTIFACTS_LOADED` and `ARTIFACTS_MTIMES` in app.config."""
    global model, preprocessor
    paths = _artifact_paths(artifacts_dir)
    current_mtimes = _get_mtimes(paths)

    last_mtimes = app.config.get("ARTIFACTS_MTIMES")

    # If mtimes are unchanged and artifacts already loaded, nothing to do.
    if last_mtimes is not None and last_mtimes == current_mtimes and app.config.get("ARTIFACTS_LOADED"):
        return

    # If any artifact is missing, mark as not loaded but don't raise.
    if any(v is None for v in current_mtimes.values()):
        app.logger.debug("One or more artifact files missing: %s", current_mtimes)
        app.config["ARTIFACTS_LOADED"] = False
        app.config["ARTIFACTS_MTIMES"] = current_mtimes
        model, preprocessor = None, None
        return

    # Try to load artifacts (may raise other exceptions which we catch)
    try:
        model, preprocessor = load_artifacts(artifacts_dir)
        app.config["ARTIFACTS_LOADED"] = True
        app.config["ARTIFACTS_MTIMES"] = current_mtimes
        app.logger.info("Loaded model artifacts; mtimes=%s", current_mtimes)
    except Exception as ex:
        app.logger.exception("Failed to load artifacts: %s", ex)
        app.config["ARTIFACTS_LOADED"] = False
        app.config["ARTIFACTS_MTIMES"] = current_mtimes
        model, preprocessor = None, None


@app.route("/predict", methods=["POST"])
def predict_route():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    if not app.config.get('ARTIFACTS_LOADED', False):
        return (
            jsonify({"error": "Model artifacts not available on server. Place 'model.joblib' and 'preprocessor.joblib' in the 'artifacts' folder."}),
            503,
        )

    try:
        pred = predict_single(payload, model, preprocessor)

        # Record prediction in server-side history
        try:
            hist = app.config.setdefault("PREDICTION_HISTORY", [])
            entry = {
                "ts": time.time(),
                "input": payload,
                "prediction": pred,
            }
            hist.append(entry)
            # keep history bounded
            max_len = app.config.get("PREDICTION_HISTORY_MAX", 200)
            if len(hist) > max_len:
                del hist[0 : len(hist) - max_len]
        except Exception:
            app.logger.exception("Failed to record prediction to history")

        return jsonify({"prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve a simple dynamic UI for local testing
@app.route("/")
def index():
    return render_template("index.html", artifacts_loaded=app.config.get("ARTIFACTS_LOADED", False))


@app.route("/health")
def health():
    """Return basic health information including whether artifacts are loaded."""
    return jsonify({
        "status": "ok",
        "artifacts_loaded": bool(app.config.get("ARTIFACTS_LOADED", False)),
        "artifacts_mtimes": app.config.get("ARTIFACTS_MTIMES", {}),
    })


@app.route("/predictions")
def predictions():
    """Return recent predictions recorded in-memory."""
    hist = app.config.get("PREDICTION_HISTORY", [])
    # convert timestamps to readable form
    out = [
        {"ts": e["ts"], "ts_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["ts"])), "input": e["input"], "prediction": e["prediction"]}
        for e in hist
    ]
    return jsonify(out)


@app.route('/plot_csv', methods=['POST'])
def plot_csv():
    """Generate a plot from an uploaded CSV (saved under `uploads/`).
    Expects JSON: {upload_path: '<uploads/...csv>', x: '<col>', y: '<col or empty>', kind: 'line|bar|scatter|hist'}
    Returns: {'image': '<outputs/..png>'}
    """
    data = request.get_json() or {}
    upload_path = data.get('upload_path')
    x = data.get('x')
    y = data.get('y')
    kind = data.get('kind', 'line')

    if not upload_path:
        return jsonify({'error': 'upload_path is required'}), 400

    # Ensure path is under uploads
    up = Path(upload_path)
    try:
        uploads_dir = Path('uploads').resolve()
        full = up.resolve()
        if uploads_dir not in full.parents and uploads_dir != full.parent:
            return jsonify({'error': 'Invalid upload path'}), 400
    except Exception:
        return jsonify({'error': 'Invalid upload path'}), 400

    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        return jsonify({'error': f'Missing plotting dependencies: {e}'}), 500

    try:
        df = pd.read_csv(full)
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV for plotting: {e}'}), 400

    outdir = Path('outputs')
    outdir.mkdir(exist_ok=True)
    ts = int(time.time())

    try:
        img_name = f'user_plot_{ts}.png'
        fig, ax = plt.subplots(figsize=(8,4))

        if kind == 'hist':
            col = y or x
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
        elif kind == 'scatter':
            sns.scatterplot(data=df, x=x, y=y, ax=ax)
            ax.set_title(f'{y} vs {x}')
        elif kind == 'bar':
            agg = df.groupby(x)[y].sum().nlargest(50)
            agg.plot(kind='bar', ax=ax)
            ax.set_title(f'{y} by {x}')
        else:  # line
            try:
                df[x] = pd.to_datetime(df[x])
                agg = df.groupby(x)[y].sum().sort_index()
                sns.lineplot(x=agg.index, y=agg.values, ax=ax)
                ax.set_title(f'{y} over time')
            except Exception:
                # fallback to simple line of numeric series
                sns.lineplot(data=df, x=x, y=y, ax=ax)
                ax.set_title(f'{y} vs {x}')

        fig.savefig(outdir / img_name, bbox_inches='tight')
        plt.close(fig)
        return jsonify({'image': img_name})
    except Exception as e:
        return jsonify({'error': f'Failed to create plot: {e}'}), 500


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route('/outputs/<path:filename>')
def outputs_static(filename):
    """Serve files from the `outputs` folder (EDA images)."""
    outdir = Path('outputs')
    if not outdir.exists():
        return ("No outputs available", 404)
    # send file from outputs directory
    from flask import send_from_directory

    return send_from_directory(outdir.resolve(), filename)


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Accept an uploaded CSV file, run a few simple EDA plots and save images to `outputs/`.
    Returns JSON with the generated image filenames (relative to `/outputs/<name>`).
    """
    uploaded = request.files.get('file')
    if not uploaded:
        return jsonify({'error': 'No file uploaded'}), 400

    # Log basic request info to help diagnose browser upload problems
    try:
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / 'upload.log'
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(f"\n----\nTS={time.time()} Received upload request. Files: {list(request.files.keys())} Content-Length: {request.content_length}\n")
    except Exception:
        pass

    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        # log
        with open('logs/upload.log', 'a', encoding='utf-8') as lf:
            lf.write(f"Missing plotting deps: {e}\n")
        return jsonify({'error': f'Missing plotting dependencies: {e}'}), 500

    try:
        # Save uploaded CSV and also keep a DataFrame copy
        uploads_dir = Path('uploads')
        uploads_dir.mkdir(exist_ok=True)
        ts = int(time.time())
        safe_name = f"user_{ts}_{uploaded.filename}"
        upload_path = uploads_dir / safe_name
        uploaded.save(str(upload_path))
        df = pd.read_csv(upload_path)
    except Exception as e:
        # log traceback
        tb = traceback.format_exc()
        try:
            with open('logs/upload.log', 'a', encoding='utf-8') as lf:
                lf.write(f"Failed to read CSV: {e}\n{tb}\n")
        except Exception:
            pass
        return jsonify({'error': f'Failed to read CSV: {e}'}), 400

    outdir = Path('outputs')
    outdir.mkdir(exist_ok=True)
    ts = int(time.time())
    images = []

    # Helper to save figure
    def save_fig(fig, name):
        path = outdir / name
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        images.append(str(name))

    # Try timeseries if 'date' column exists
    try:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                if num_cols:
                    val_col = num_cols[0]
                    agg = df.groupby('date')[val_col].sum().sort_index()
                    fig, ax = plt.subplots(figsize=(8,3))
                    sns.lineplot(x=agg.index, y=agg.values, ax=ax)
                    ax.set_title(f'{val_col} over time')
                    fname = f'user_{ts}_over_time.png'
                    save_fig(fig, fname)
            except Exception:
                pass

        # Distribution of first numeric column
        if num_cols:
            val_col = num_cols[0]
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(df[val_col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {val_col}')
            fname = f'user_{ts}_distribution.png'
            save_fig(fig, fname)

        # Category breakdown (first non-numeric)
        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        if cat_cols:
            cat = cat_cols[0]
            if num_cols:
                val_col = num_cols[0]
                agg = df.groupby(cat)[val_col].sum().nlargest(20)
                fig, ax = plt.subplots(figsize=(6,4))
                agg.plot(kind='bar', ax=ax)
                ax.set_title(f'{val_col} by {cat}')
                fname = f'user_{ts}_by_{cat}.png'
                save_fig(fig, fname)

    except Exception as e:
        tb = traceback.format_exc()
        try:
            with open('logs/upload.log', 'a', encoding='utf-8') as lf:
                lf.write(f"Failed generating visualizations: {e}\n{tb}\n")
        except Exception:
            pass
        return jsonify({'error': f'Failed generating visualizations: {e}'}), 500

    # Prepare metadata to help client render interactive plot controls
    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
    preview = df.head(5).to_dict(orient='records')

    return jsonify({'images': images, 'upload_path': str(upload_path.as_posix()), 'columns': cols, 'numeric_columns': num_cols, 'categorical_columns': cat_cols, 'preview': preview})


@app.route('/upload_base64', methods=['POST'])
def upload_base64():
    """Accept JSON payload: {filename: 'name.csv', data: '<data:...;base64,AAA...>'}
    Saves file under uploads/ and runs same EDA logic as `/upload_csv`.
    """
    payload = request.get_json(silent=True) or {}
    filename = payload.get('filename')
    data = payload.get('data')
    if not filename or not data:
        return jsonify({'error': 'filename and data required'}), 400

    # strip data:prefix if present
    if ',' in data:
        data = data.split(',', 1)[1]

    try:
        blob = base64.b64decode(data)
    except Exception as e:
        return jsonify({'error': f'base64 decode failed: {e}'}), 400

    uploads_dir = Path('uploads')
    uploads_dir.mkdir(exist_ok=True)
    ts = int(time.time())
    safe_name = f"user_{ts}_{filename}"
    upload_path = uploads_dir / safe_name
    try:
        with open(upload_path, 'wb') as fh:
            fh.write(blob)
    except Exception as e:
        return jsonify({'error': f'Failed to save decoded file: {e}'}), 500

    # Now reuse upload processing: read CSV and generate images
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        return jsonify({'error': f'Missing plotting dependencies: {e}'}), 500

    try:
        df = pd.read_csv(upload_path)
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {e}'}), 400

    outdir = Path('outputs')
    outdir.mkdir(exist_ok=True)
    images = []

    def save_fig(fig, name):
        path = outdir / name
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        images.append(str(name))

    try:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                if num_cols:
                    val_col = num_cols[0]
                    agg = df.groupby('date')[val_col].sum().sort_index()
                    fig, ax = plt.subplots(figsize=(8,3))
                    sns.lineplot(x=agg.index, y=agg.values, ax=ax)
                    ax.set_title(f'{val_col} over time')
                    fname = f'user_{ts}_over_time.png'
                    save_fig(fig, fname)
            except Exception:
                pass

        if num_cols:
            val_col = num_cols[0]
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(df[val_col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {val_col}')
            fname = f'user_{ts}_distribution.png'
            save_fig(fig, fname)

        cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        if cat_cols:
            cat = cat_cols[0]
            if num_cols:
                val_col = num_cols[0]
                agg = df.groupby(cat)[val_col].sum().nlargest(20)
                fig, ax = plt.subplots(figsize=(6,4))
                agg.plot(kind='bar', ax=ax)
                ax.set_title(f'{val_col} by {cat}')
                fname = f'user_{ts}_by_{cat}.png'
                save_fig(fig, fname)
    except Exception as e:
        return jsonify({'error': f'Failed generating visualizations: {e}'}), 500

    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
    preview = df.head(5).to_dict(orient='records')

    return jsonify({'images': images, 'upload_path': str(upload_path.as_posix()), 'columns': cols, 'numeric_columns': num_cols, 'categorical_columns': cat_cols, 'preview': preview})


@app.route('/debug/upload_log')
def debug_upload_log():
    p = Path('logs') / 'upload.log'
    if not p.exists():
        return jsonify({'error': 'no log'}), 404
    # return last 100 lines
    try:
        text = p.read_text(encoding='utf-8')
        lines = text.splitlines()
        tail = '\n'.join(lines[-200:])
        return app.response_class(tail, mimetype='text/plain')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/debug/js_errors', methods=['POST'])
def debug_js_errors():
    payload = request.get_json(silent=True)
    p = Path('logs')
    p.mkdir(exist_ok=True)
    try:
        with open(p / 'js_errors.log', 'a', encoding='utf-8') as lf:
            lf.write(f"TS={time.time()}\n")
            if payload:
                lf.write(str(payload) + "\n")
            else:
                lf.write('no json payload\n')
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # For local testing only. Use a production server for deployment.
    app.run(host="0.0.0.0", port=5000, debug=True)
