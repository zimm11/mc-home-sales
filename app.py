from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)


@app.route('/')
@app.route('/index')

def index():
    return render_template('index.html')

def prediction(prediction_list):
    prediction_values = np.array(prediction_list).reshape((1, -1))
    model = load('gbr.joblib')
    sale_price = model.predict(prediction_values)
    return sale_price[0]


@app.route('/prediction', methods=['POST'])
def result():
    if request.method == 'POST':
        prediction_list = request.form.to_dict()
        prediction_list = list(prediction_list.values())
        prediction_list = list(map(int, prediction_list))
        sale_price = prediction(prediction_list)

        return render_template("index.html", prediction=sale_price)

if __name__ == '__main__':
    app.run()
