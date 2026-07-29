from pathlib import Path

from flask import Flask

from data_store import DataStore
from metrics import compute_model_metrics
from routes import register_routes
from similarity import SimilarityIndex, title_case, trim_plate, format_organelles

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "app_data"


def create_app():
    app = Flask(__name__)
    app.jinja_env.filters["titlecase"] = title_case
    app.jinja_env.filters["trimplate"] = trim_plate
    app.jinja_env.filters["organelles"] = format_organelles
    store = DataStore(DATA_DIR)
    sim_index = SimilarityIndex(store.embeddings)
    model_metrics = compute_model_metrics(store)
    register_routes(app, store, sim_index, model_metrics)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5050)
