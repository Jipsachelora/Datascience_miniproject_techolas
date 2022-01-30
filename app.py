from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('saved_model', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['Nanomaterial shape']
    data2 = request.form['Nanomaterial core']
    data3 = request.form['Nanomaterial size']
    data4 = request.form['Nanomaterial ligand1']
    data5 = request.form['Nanomaterial ligand2']
    data6 = request.form['Nanomaterial ligand3']
    data7 = request.form['Nanomaterial ligand4']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7]])
    pred = model.predict(arr)
    return render_template('index.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
