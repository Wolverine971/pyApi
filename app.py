from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

# @app.route('/addToGraph')
# def index():
#     return 'Hello World'

# @app.route('/sendInfo', methods=['POST'])
# def post_user():
#     return redirect(url_for('index'))

# if __name__ == "__main__":
#     print('server is running on localhost!!')
#     app.run(debug=True)