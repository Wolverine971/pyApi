from flask import Flask, request, jsonify
from flask_cors import CORS
from graph import createGraph, annotateDoc
import asyncio

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Hello, World!"


@app.route('/getTripples/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return "get graph"

    if request.method == 'POST':
        # data = request.get_json()
        data = request.json
        createGraph(data['index'])
        return 'OK'

@app.route('/getTripples/doc', methods=['GET', 'POST'])
def doc():
    if request.method == 'GET':
        return "get graph"

    if request.method == 'POST':
        # data = request.get_json()
        data = request.json
        return asyncio.run(annotateDoc(data))

if __name__ == "__main__":
    print('server is running on localhost!!')
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)
