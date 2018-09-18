"""Filename: server.py
"""

import os
import pandas as pd
from flask import Flask, jsonify, request
# import tensorflow as tf
from SP_API.eval import eval_SP

app = Flask(__name__)

@app.route('/eval', methods=['POST'])
def apicall():
    """API Call
    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        eval_json = request.get_json()
        eval = pd.read_json(eval_json, orient='records')

    except Exception as e:
        raise e

    # clf = 'model_v1.pk'

    if eval.empty:
        return(bad_request())
    else:
        #Load the saved model
        print("Getting Model Predictions...")
        predictions = eval_SP()

        """Add the predictions as Series to a new pandas dataframe
                                OR
           Depending on the use-case, the entire test data appended with the new files
        """
        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(prediction_series)

        """We can be as creative in sending the responses.
           But we need to send the response codes as well.
        """
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)