import os
import pandas as pd
import json


TEST_PRODUCTS_JSON_PATH = os.path.join("../", "data", "test_products.json")
TEST_PRODUCTS_CSV_PATH = os.path.join("../", "data", "test_products.csv")

products_df = pd.read_csv(TEST_PRODUCTS_CSV_PATH)

products_df = products_df[['title', 'concatenated_tags']]
products_dict = products_df.to_dict(orient='records')
products_dict = {"products": products_dict}
products_json = json.dumps(products_dict, indent=4, ensure_ascii=False)

with open(TEST_PRODUCTS_JSON_PATH, "w") as f:
    f.write(products_json)
