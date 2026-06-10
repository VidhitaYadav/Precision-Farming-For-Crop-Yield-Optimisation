"""
Precision Farming for Crop Yield Optimisation — Flask API Backend
==================================================================
Serves the web dashboard and exposes endpoints for metadata query,
real-time single predictions, bulk predictions via CSV, and
model retraining on new uploaded datasets.
"""

import os
import sys
import shutil
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app)

# Paths
MODEL_PATH = "outputs/models/rf_model.pkl"
DATASET_PATH = "data/processed/clustered_master_dataset.csv"
FIGURES_DIR = "outputs/figures"
RAW_DATA_DIR = "data/raw"

# Global model artifacts
model_package = None
rf_model = None
encoders = None
crop_lookup = {}


def load_assets():
    """Load serialised model package and build the crop parameters lookup table."""
    global model_package, rf_model, encoders, crop_lookup

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[ERROR] Trained model not found at {MODEL_PATH}. Please run main.py first.")

    model_package = joblib.load(MODEL_PATH)
    rf_model = model_package['model']
    encoders = model_package['encoders']
    print(f"[INFO] Loaded trained Random Forest model successfully from {MODEL_PATH}")

    # Build crop soil/weather parameters lookup table from clustered dataset
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        grouped = df.groupby("Crop").first().reset_index()
        crop_lookup = {}
        for _, row in grouped.iterrows():
            crop_name = row["Crop"]
            crop_lookup[crop_name] = {
                "N_mean": float(row["N_mean"]),
                "P_mean": float(row["P_mean"]),
                "K_mean": float(row["K_mean"]),
                "ph_mean": float(row["ph_mean"]),
                "temperature_mean": float(row["temperature_mean"]),
                "humidity_mean": float(row["humidity_mean"]),
                "NPK_Index": float(row["NPK_Index"]),
                "Soil_Cluster": int(row["Soil_Cluster"])
            }
        print(f"[INFO] Loaded crop properties lookup table for {len(crop_lookup)} crops.")
    else:
        print(f"[WARNING] Master dataset not found at {DATASET_PATH}. Lookup table is empty.")


# Backup original datasets on startup if backup does not exist
def backup_original_datasets():
    """Create backup copies of original Kaggle CSVs so they can be restored."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    for f in ["crop_yield.csv", "crop_recommendation.csv"]:
        path = os.path.join(RAW_DATA_DIR, f)
        backup_path = os.path.join(RAW_DATA_DIR, f.replace(".csv", "_original.csv"))
        if os.path.exists(path) and not os.path.exists(backup_path):
            shutil.copy(path, backup_path)
            print(f"[INFO] Created backup of original raw dataset: {backup_path}")


# Init assets
try:
    backup_original_datasets()
    load_assets()
except Exception as e:
    print(e)


# ──────────────────────────────────────────────
#  WEB ROUTES & SERVING FILES
# ──────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the web UI page."""
    return render_template("index.html")


@app.route("/outputs/figures/<path:filename>")
def serve_figure(filename):
    """Serve figures directly from outputs/figures/ folder."""
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/outputs/<path:filename>")
def serve_output_file(filename):
    """Serve prediction or analysis files from outputs/ folder as attachments."""
    return send_from_directory("outputs", filename, as_attachment=True)


# ──────────────────────────────────────────────
#  API ENDPOINTS
# ──────────────────────────────────────────────
@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Return available states, seasons, crops, and their default soil properties."""
    if encoders is None:
        return jsonify({"error": "Model package not loaded. Run main.py first."}), 500

    crops = sorted(list(encoders["Crop"].classes_))
    seasons = sorted(list(encoders["Season"].classes_))
    states = sorted(list(encoders["State"].classes_))

    crop_profiles = {}
    for crop in crops:
        if crop in crop_lookup:
            crop_profiles[crop] = crop_lookup[crop]
        else:
            crop_profiles[crop] = {
                "N_mean": 50.0, "P_mean": 40.0, "K_mean": 30.0, "ph_mean": 6.5,
                "temperature_mean": 24.0, "humidity_mean": 65.0, "NPK_Index": 40.0,
                "Soil_Cluster": 0
            }

    return jsonify({
        "crops": crops,
        "seasons": seasons,
        "states": states,
        "crop_profiles": crop_profiles
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict crop yield (tonnes/ha) based on single record input parameters."""
    if rf_model is None or encoders is None:
        return jsonify({"error": "Model package not loaded. Run main.py first."}), 500

    try:
        data = request.json
        crop = data.get("crop")
        season = data.get("season")
        state = data.get("state")
        area = float(data.get("area"))
        rainfall = float(data.get("rainfall"))
        fertilizer = float(data.get("fertilizer"))
        pesticide = float(data.get("pesticide"))

        if crop in crop_lookup:
            props = crop_lookup[crop]
        else:
            props = {
                "N_mean": 50.0, "P_mean": 40.0, "K_mean": 30.0, "ph_mean": 6.5,
                "temperature_mean": 24.0, "humidity_mean": 65.0, "NPK_Index": 40.0,
                "Soil_Cluster": 0
            }

        try:
            crop_encoded = int(encoders["Crop"].transform([crop])[0])
            season_encoded = int(encoders["Season"].transform([season])[0])
            state_encoded = int(encoders["State"].transform([state])[0])
        except ValueError as ve:
            return jsonify({"error": f"Invalid category: {str(ve)}"}), 400

        log_area = np.log1p(area)
        npk_index = props["NPK_Index"]
        rainfall_fert_ratio = rainfall / (fertilizer + 1.0)

        # Build feature vector matching model.py FEATURE_COLS
        feature_vector = np.array([[
            props["Soil_Cluster"],
            rainfall,
            fertilizer,
            pesticide,
            props["N_mean"],
            props["P_mean"],
            props["K_mean"],
            props["ph_mean"],
            props["temperature_mean"],
            props["humidity_mean"],
            npk_index,
            rainfall_fert_ratio,
            log_area,
            season_encoded,
            crop_encoded,
            state_encoded
        ]])

        prediction = rf_model.predict(feature_vector)[0]
        estimated_production = prediction * area

        cluster_info = {
            0: "Nitrogen-rich loamy soil suitable for groundnuts and general legumes.",
            1: "Higher potassium/alkaline soil suited for pulses like Gram/Chickpea.",
            2: "Standard dry-weather crop profile cluster optimized for black grams.",
            3: "Acidic weather soil suited for pigeon peas and deep-rooted pulses.",
            4: "Highly fertile wet-clay soils with high water requirement, suited for Rice.",
            5: "High temperature, humid sandy loam soil suited for green grams.",
            6: "Rich alluvial nitrogen-dense soil optimized for cotton cultivation.",
            7: "Arid plains weather soil profile suited for maize and coarse grains.",
            8: "Mild temperature, damp soil profiles optimized for banana plantations.",
            9: "Coastal humid weather soil, high salinity tolerance, optimized for coconut."
        }
        cluster_desc = cluster_info.get(props["Soil_Cluster"], "General agricultural soil profiling.")

        return jsonify({
            "predicted_yield": round(float(prediction), 4),
            "estimated_production": round(float(estimated_production), 2),
            "soil_cluster": props["Soil_Cluster"],
            "cluster_description": cluster_desc,
            "soil_properties": props
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ──────────────────────────────────────────────
#  CSV BULK PREDICTION ENDPOINT
# ──────────────────────────────────────────────
@app.route("/api/upload/predict", methods=["POST"])
def upload_predict():
    """
    Accepts a CSV file of crops and parameters, runs Random Forest predictions for all rows,
    and returns a preview of predictions plus a download link for the processed CSV.
    """
    if rf_model is None or encoders is None:
        return jsonify({"error": "Model package not loaded. Run main.py first."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    try:
        df = pd.read_csv(file)

        # Normalise column headers (strip whitespace & make title-case matching expected schema)
        normalized_cols = {col: col.strip() for col in df.columns}
        df = df.rename(columns=normalized_cols)

        # Validate headers
        required_cols = ["Crop", "Season", "State", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return jsonify({
                "error": f"Uploaded CSV is missing required column(s): {missing_cols}. Please verify headers."
            }), 400

        # ── Vectorized bulk prediction (fast) ──────────────────────
        # Strip whitespace from categorical columns
        df["Crop"] = df["Crop"].astype(str).str.strip()
        df["Season"] = df["Season"].astype(str).str.strip()
        df["State"] = df["State"].astype(str).str.strip()

        # Build soil properties lookup vectors for every row
        default_props = {
            "N_mean": 50.0, "P_mean": 40.0, "K_mean": 30.0, "ph_mean": 6.5,
            "temperature_mean": 24.0, "humidity_mean": 65.0, "NPK_Index": 40.0,
            "Soil_Cluster": 0
        }
        soil_keys = ["N_mean", "P_mean", "K_mean", "ph_mean",
                     "temperature_mean", "humidity_mean", "NPK_Index", "Soil_Cluster"]
        soil_matrix = np.array([
            [crop_lookup.get(crop, default_props)[k] for k in soil_keys]
            for crop in df["Crop"].values
        ])  # shape: (n_rows, 8)

        n_mean = soil_matrix[:, 0]
        p_mean = soil_matrix[:, 1]
        k_mean = soil_matrix[:, 2]
        ph_mean = soil_matrix[:, 3]
        temp_mean = soil_matrix[:, 4]
        humid_mean = soil_matrix[:, 5]
        npk_index = soil_matrix[:, 6]
        soil_cluster = soil_matrix[:, 7].astype(int)

        # Encode categorical columns in batch (gracefully handle unseen labels)
        def safe_encode(encoder, values):
            """Encode known labels, assign 0 to unknown ones."""
            known = set(encoder.classes_)
            result = np.zeros(len(values), dtype=int)
            mask = np.array([v in known for v in values])
            if mask.any():
                result[mask] = encoder.transform(values[mask]).astype(int)
            return result

        crop_encoded = safe_encode(encoders["Crop"], df["Crop"].values)
        season_encoded = safe_encode(encoders["Season"], df["Season"].values)
        state_encoded = safe_encode(encoders["State"], df["State"].values)

        # Build numeric columns
        area = df["Area"].values.astype(float)
        rainfall = df["Annual_Rainfall"].values.astype(float)
        fertilizer = df["Fertilizer"].values.astype(float)
        pesticide = df["Pesticide"].values.astype(float)

        log_area = np.log1p(area)
        rainfall_fert_ratio = rainfall / (fertilizer + 1.0)

        # Assemble full feature matrix — column order must match FEATURE_COLS
        feature_matrix = np.column_stack([
            soil_cluster,
            rainfall,
            fertilizer,
            pesticide,
            n_mean,
            p_mean,
            k_mean,
            ph_mean,
            temp_mean,
            humid_mean,
            npk_index,
            rainfall_fert_ratio,
            log_area,
            season_encoded,
            crop_encoded,
            state_encoded
        ])

        # Single batch prediction — orders of magnitude faster than per-row
        predictions = np.round(rf_model.predict(feature_matrix), 4)
        productions = np.round(predictions * area, 2)

        # Append prediction columns
        df["Soil_Cluster"] = soil_cluster
        df["Predicted_Yield"] = predictions
        df["Estimated_Production"] = productions

        # Save output CSV to outputs directory
        output_filename = "bulk_predictions_output.csv"
        output_path = os.path.join("outputs", output_filename)
        os.makedirs("outputs", exist_ok=True)
        df.to_csv(output_path, index=False)

        # Send back top 5 rows as a preview
        preview_data = df.head(5).to_dict(orient="records")

        return jsonify({
            "message": "Bulk prediction completed successfully!",
            "download_url": f"/outputs/{output_filename}",
            "preview": preview_data,
            "total_rows": len(df)
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process CSV file: {str(e)}"}), 500


# ──────────────────────────────────────────────
#  MODEL DYNAMIC RETRAINING ENDPOINT
# ──────────────────────────────────────────────
@app.route("/api/upload/train", methods=["POST"])
def upload_train():
    """
    Accepts new crop_yield.csv or crop_recommendation.csv dataset, validates headers,
    runs the full main.py ML pipeline end-to-end, and hot-reloads the updated model assets.
    """
    if 'yield_file' not in request.files and 'soil_file' not in request.files:
        return jsonify({"error": "No file uploaded. Please supply a yield or soil CSV."}), 400

    yield_file = request.files.get('yield_file')
    soil_file = request.files.get('soil_file')

    try:
        # Validate and save crop yield dataset
        if yield_file and yield_file.filename != '':
            df = pd.read_csv(yield_file)
            normalized_cols = {col.strip(): col.strip() for col in df.columns}
            df = df.rename(columns=normalized_cols)
            
            # Verify required columns for crop_yield.csv
            required_cols = ["Crop", "Crop_Year", "Season", "State", "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return jsonify({"error": f"Invalid crop yield CSV schema. Missing: {missing_cols}"}), 400
            
            # Save file
            yield_path = os.path.join(RAW_DATA_DIR, "crop_yield.csv")
            df.to_csv(yield_path, index=False)
            print(f"[INFO] Replaced crop_yield.csv with uploaded data -> {df.shape}")

        # Validate and save soil recommendation dataset
        if soil_file and soil_file.filename != '':
            df = pd.read_csv(soil_file)
            normalized_cols = {col.strip(): col.strip() for col in df.columns}
            df = df.rename(columns=normalized_cols)
            
            # Verify required columns for crop_recommendation.csv
            required_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return jsonify({"error": f"Invalid soil recommendation CSV schema. Missing: {missing_cols}"}), 400
            
            # Save file
            soil_path = os.path.join(RAW_DATA_DIR, "crop_recommendation.csv")
            df.to_csv(soil_path, index=False)
            print(f"[INFO] Replaced crop_recommendation.csv with uploaded data -> {df.shape}")

        # Execute full pipeline end-to-end by importing and running main.py orchestrator
        print("[INFO] Starting pipeline retraining pipeline...")
        from main import run_pipeline
        run_pipeline()
        print("[INFO] Pipeline retraining successfully completed.")

        # Reload trained model assets dynamically in the server
        load_assets()

        # Load new evaluation metrics to display
        metrics_path = "outputs/figures/baseline_comparison.csv"
        metrics = []
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            metrics = metrics_df.to_dict(orient="records")

        return jsonify({
            "message": "Model retraining and hot-reload completed successfully!",
            "metrics": metrics
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Retraining failed: {str(e)}"}), 500


if __name__ == "__main__":
    # In production, Render/Railway injects the PORT environment variable.
    # We bind to 0.0.0.0 to allow public external routing.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
