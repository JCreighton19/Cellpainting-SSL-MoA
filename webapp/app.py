from pathlib import Path

from flask import Flask

from data_store import DataStore
from routes import register_routes
from similarity import SimilarityIndex, title_case

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "app_data"


def create_app():
    app = Flask(__name__)
    app.jinja_env.filters["titlecase"] = title_case
    store = DataStore(DATA_DIR)
    sim_index = SimilarityIndex(store.embeddings)
    register_routes(app, store, sim_index)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5050)
