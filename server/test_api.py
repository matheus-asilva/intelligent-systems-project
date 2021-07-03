import pytest
from flask import json
from api import app
import os
import cloudpickle as cp
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

categorize_route = "v1/categorize"
valid_categories_set = {"Lembrancinhas", "Bebê", "Decoração", "Outros",
                        "Papel e Cia", "Bijuterias e Jóias"}

TEST_PRODUCTS_PATH = os.getenv("TEST_PRODUCTS_PATH")
TEST_PRODUCTS_CSV_PATH = os.path.join("../", "data", "test_products.csv")
with open(TEST_PRODUCTS_PATH, "r") as json_file:
    test_json = json.load(json_file)


def categorize_request(input_data):
    return app.test_client().post(categorize_route,
                                  data=json.dumps(input_data),
                                  content_type="application/json")


@pytest.mark.parametrize("input_data", [
        None,
        "",
        {},
        {"products": []},
        {"products": [{"title": ""}]},
        {"products": [{"concatenated_tags": ""}]},
        {"products": [{"other1": "", "other2": ""}]}
    ])
def test_request_with_invalid_data(input_data):
    response = categorize_request(input_data)

    assert response.status_code == 400
    assert response.data == b"(Bad Request)"


@pytest.mark.parametrize("input_data", [
    {"products": [{"title": None, "concatenated_tags": None}]},
    {"products": [{"title": "", "concatenated_tags": ""}]},
    {"products": [{"title": "", "concatenated_tags": "", "other": ""}]},
    {"products": [{"title": "a", "concatenated_tags": "a"},
                  {"title": "b", "concatenated_tags": "b"}]},
    test_json])
def test_request_with_valid_data(input_data):
    response = categorize_request(input_data)

    assert response.status_code == 200
    assert len(response.json["categories"]) == len(input_data['products'])
    assert set(response.json['categories']).issubset(valid_categories_set)

def load_model():
    with open(os.getenv("MODEL_PATH"), "rb") as file:
        return cp.load(file)

def load_data():
    data = pd.read_csv(TEST_PRODUCTS_CSV_PATH)
    string_columns = data.select_dtypes("object").columns.tolist()
    data.loc[:, string_columns] = data.loc[:, string_columns].fillna("")
    return data


def test_check_columns():   
    data = load_data()
    expected = ['title', 'query', 'concatenated_tags']

    assert np.all(pd.Series(expected).isin(data.columns))

def test_load_pipeline_model():
    model = load_model()
    expected = Pipeline
    assert expected == model.__class__

def test_column_concatenation():
    data = load_data()
    model = load_model()

    expected = data["title"] + " " + data["concatenated_tags"]
    assert expected.equals(model["preprocessor"]["text_column_concatenation"].transform(data))

def test_preprocessor_pipeline_output_class():
    data = load_data()
    model = load_model()

    expected = csr_matrix
    assert expected == model["preprocessor"].transform(data).__class__

def test_pipeline_predict():
    data = load_data()
    model = load_model()
    labels = model.classes_

    y_true = data["category"]
    y_proba = model.predict_proba(data)

    assert roc_auc_score(y_true, y_proba, multi_class="ovr") > 0.97
