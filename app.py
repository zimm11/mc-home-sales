from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import numpy as np

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    prediction_list = request.form.to_dict()
    prediction_list = list(prediction_list.values())
    predicted_sale_price = ('{:,}'.format(round(prediction(prediction_list), 2)))
    return redirect(url_for('predicted_result', predicted_sale_price=predicted_sale_price))


def prediction(prediction_list):
    prediction_values = np.array(prediction_list).reshape((1, -1))
    model = load('gbr.joblib')
    sale_price = model.predict(prediction_values)
    return sale_price[0]


@app.route('/predicted_result')
def predicted_result():
    return render_template('prediction.html', predicted_sale_price=request.args.get('predicted_sale_price'))

@app.route('/jupyter')
def jupyter():
    return render_template('jupyter-nb.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/sources')
def sources():
    return render_template('sources.html')


if __name__ == '__main__':
    app.run()
