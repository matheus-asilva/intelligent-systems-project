import os
import pickle
from flask import Flask, request
import pandas as pd


app = Flask(__name__)

# needed because of portuguese characters like "ê"
app.config['JSON_AS_ASCII'] = False

MODEL_PATH = os.getenv("MODEL_PATH")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

def pre_process(df):
    df['title'] = df['title'].str.lower()
    string_columns = df.select_dtypes("object").columns.tolist()
    df.loc[:, string_columns] = df.loc[:, string_columns].fillna("")
    return df

@app.route('/v1/categorize', methods=["POST"])
def categorize():
    try:
        products = request.json['products']
        products_df = pd.DataFrame(products)
        products_df = pre_process(products_df)
        categories_array = model.predict(products_df)
        categories_list = categories_array.tolist()
        return {"categories": categories_list}
    except (TypeError, KeyError):
        return "(Bad Request)", 400
