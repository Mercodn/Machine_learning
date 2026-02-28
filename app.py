from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def first page(name):
    return  render_template('index.html'), name= name)
    


