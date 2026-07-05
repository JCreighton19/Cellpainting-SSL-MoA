from pathlib import Path

from flask import Flask

from data_store import DataStore
from routes import register_routes
from similarity import SimilarityIndex

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "app_data"


def create_app():
    app = Flask(__name__)
    store = DataStore(DATA_DIR)
    sim_index = SimilarityIndex(store.embeddings)
    compound_sim_index = SimilarityIndex(store.compound_embeddings)  # Phase 2
    register_routes(app, store, sim_index, compound_sim_index)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5050)
