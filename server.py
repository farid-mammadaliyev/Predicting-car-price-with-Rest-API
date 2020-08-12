from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import lightgbm as lgb

model = p.load(open('finalized_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json(force = True)
    prediction = np.array2string(model.predict(data))

    return jsonify(prediction)

app.run(port = 5000, debug = False)