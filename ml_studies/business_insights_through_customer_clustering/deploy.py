from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)


class RFMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date):
        self.reference_date = pd.to_datetime(reference_date)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        df_with_id = df[~df['CustomerID'].isna()]

        rfm = df_with_id.groupby("CustomerID").agg({
            'InvoiceDate': lambda x: (self.reference_date - x.max()).days,
            'InvoiceNo': 'count',
            'TotalPrice': 'sum'
        }).rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'TotalPrice': 'Monetary'
        })

        return rfm

@app.route("/predict", methods=["POST"])
def predict():
    content = request.get_json()

    if not content or "data" not in content:
        return jsonify({"error": "Missing transaction data"}), 400

    n_clusters = int(content.get("n_clusters", 4))
    reference_date = content.get("reference_date")
    if reference_date is None:
        return jsonify({"error": "Missing reference_date"}), 400

    try:
        reference_date = pd.to_datetime(reference_date)
    except Exception as e:
        return jsonify({"error": f"Invalid reference_date format: {e}"}), 400

    transactions = content["data"]
    df = pd.DataFrame(transactions)

    try:
        transformer = RFMTransformer(reference_date=reference_date)
        rfm_data = transformer.transform(df)
    except Exception as e:
        return jsonify({"error": f"Error during RFM transformation: {e}"}), 500

    try:
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(rfm_data)
    except Exception as e:
        return jsonify({"error": f"Clustering failed: {e}"}), 500

    return jsonify({
        "n_clusters": n_clusters,
        "reference_date": str(reference_date),
        "customer_ids": rfm_data.index.tolist(),
        "labels": labels.tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
