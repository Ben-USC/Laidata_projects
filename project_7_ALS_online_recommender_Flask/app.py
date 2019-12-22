from flask import Blueprint
main = Blueprint('main', __name__)

import json
from engine import RecommendationEngine

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask import Flask, request

@main.route("/<int:user_id>/ratings/top/<int:count>", methods=["GET"])
def top_ratings(user_id, count):
    logger.debug(f"User {user_id}'s TOP {count} movies requested ... ")
    top_ratings = recommendation_engine.get_top_ratings(user_id, count)
    return json.dumps(top_ratings)

@main.route("/<int:user_id>/ratings/<int:movie_id>", method=["GET"])
def movie_ratings(user_id, movie_id):
    logger.debug(f"User {user_id} rating for movie {movie_id} requested ... ")
    user_movie_rating = recommendation_engine.get_ratings_for_movies(user_id, [movie_id])
    return json.dumps(user_movie_rating)

@main.route("/<int:user_id>/ratings/", method=["POST"])
def add_ratings(user_id):
    ## get ratings from the Flask POST request object
    ratings_list = request.form.keys()[0].strip().split("\n")
    ratings_list = map(lambda x: x.split(","), ratings_list)
    ## create a list with the format required by the engine: user_id, movie_id, ratings
    new_ratings = map(lambda x: (user_id, int(x[0]), float(x[1])), ratings_list)
    ## add ratings to the engine
    recommendation_engine.add_ratings(new_ratings)
    return json.dumps(new_ratings)

## create app
def create_app(spark_context, data_path):
    global recommendation_engine
    recommendation_engine = RecommendationEngine(sc, data_path)
    app = Flask(__name__)
    app.register_blueprint(main)
    return app
