"""
Flask Application — Movie Recommendation System
Serves the frontend and API endpoints for the RBM recommender.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, jsonify, request
from src.recommend import get_engine

app = Flask(__name__)

# ── Eager startup: load model + data once at boot ──────────────
# This prevents first-request timeouts on Railway / Render.
print("[startup] Loading recommendation engine...")
get_engine()
print("[startup] Engine ready.")


# ── Pages ──────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


# ── API Endpoints ──────────────────────────────────────────────

@app.route("/api/search")
def search():
    """Search movies by title."""
    try:
        q = request.args.get("q", "").strip().lower()
        limit = request.args.get("limit", 10, type=int)
        if not q:
            return jsonify({"success": True, "results": []})
        engine = get_engine()
        mask = engine.movies_df["title"].str.lower().str.contains(q, na=False)
        results = engine.movies_df[mask].head(limit)
        return jsonify({
            "success": True,
            "results": results[["movie_id", "title", "genres"]].to_dict("records")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommend/<int:user_id>")
def recommend(user_id):
    """Get recommendations for a user."""
    try:
        top_n = request.args.get("n", 10, type=int)
        engine = get_engine()
        recs = engine.get_user_recommendations(user_id, top_n=top_n)
        return jsonify({"success": True, "recommendations": recs, "user_id": user_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/similar/<int:movie_id>")
def similar(movie_id):
    """Get movies similar to a given movie."""
    try:
        engine = get_engine()
        results = engine.get_similar_movies(movie_id, top_n=8)
        return jsonify({"success": True, "similar": results, "movie_id": movie_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/user/<int:user_id>/rated")
def user_rated(user_id):
    """Get movies rated by a user."""
    try:
        engine = get_engine()
        rated = engine.get_user_rated_movies(user_id)
        return jsonify({"success": True, "rated_movies": rated, "user_id": user_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/popular")
def popular():
    """Get popular movies."""
    try:
        engine = get_engine()
        movies = engine.get_popular_movies(top_n=20)
        return jsonify({"success": True, "movies": movies})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stats")
def stats():
    """Get dataset and model stats."""
    try:
        engine = get_engine()
        return jsonify({"success": True, "stats": engine.get_stats()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    import os
    debug = os.environ.get("FLASK_ENV") != "production"
    port  = int(os.environ.get("PORT", 5000))
    print("\n[*] Movie Recommendation System")
    print("    http://127.0.0.1:{}\n".format(port))
    app.run(debug=debug, host="0.0.0.0", port=port)
