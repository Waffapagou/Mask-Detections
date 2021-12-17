from flask import Flask
app = Flask(__name__)

@app.route('/Documents/')
def hello_world():
    return 'Hello World!'
