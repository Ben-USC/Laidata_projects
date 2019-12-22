import time, sys, cherrypy, os
from paste.translogger import TransLogger
from app import create_app
from pyspark import SparkContext, SparkConf

def init_spark_context():
    conf = SparkConf().setAppName("MovieLens-Recommendation-Server")
    sc = SparkContext(conf = conf, pyFiles = ["engine.py", "app.py"])
    return sc

def run_server(app):
    ## enable WSGI access logging via Paste
    app_logged = TransLogger(app)
    
    ## Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, "/")
    
    ## set the configuration for the web server
    cherrypy.config.update( { "engine.autoreload.com": True,
                              "log.screen": True,
                              "server.socket_port": 7788,
                              "server.socket_host": "0.0.0.0" } )
    ## start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == "main":
    ## initialize spark context
    sc = init_spark_context()
    data_path = path.join("data", "ml-latest")
    app = create_app(sc, data_path)
    
    ## start the web server
    run_server(app)
