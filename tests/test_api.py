from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def upload_sales():
    csv = (
        b"order_id,order_date,region,product,customer,revenue,quantity,profit\n"
        b"1,2025-01-05,North,Laptop,Asha,1200,2,180\n"
        b"2,2025-01-09,South,Phone,Rahul,850,3,120\n"
        b"3,2025-02-11,North,Tablet,Meera,430,2,65\n"
        b"4,2025-02-20,South,Laptop,Asha,99999,1,100\n"
    )
    response = client.post(
        "/datasets/upload",
        files=[("files", ("sales.csv", csv, "text/csv"))],
    )
    assert response.status_code == 200
    return response.json()["dataset_id"]


def ask(dataset_id, question):
    return client.post(
        "/analyze",
        json={"dataset_id": dataset_id, "question": question, "history": []},
    )


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_quality_and_summary():
    dataset_id = upload_sales()
    quality = client.get(f"/datasets/{dataset_id}/quality")
    assert quality.status_code == 200
    response = ask(dataset_id, "Give me a summary")
    assert response.status_code == 200
    assert "rows" in response.json()["answer"].lower()
    assert response.json()["plan"]["operation"] == "summary"


def test_grouped_revenue_returns_chart_sql_and_pandas():
    dataset_id = upload_sales()
    response = ask(dataset_id, "Which region generated the highest total revenue?")
    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["operation"] == "groupby"
    assert body["chart"]["type"] == "bar"
    assert body["sql"] and "GROUP BY" in body["sql"]
    assert body["pandas_code"]
    assert body["table"][0]["region"] == "South"


def test_monthly_trend_returns_line_chart():
    dataset_id = upload_sales()
    response = ask(dataset_id, "Show monthly sales trends")
    assert response.status_code == 200
    body = response.json()
    assert body["chart"]["type"] == "line"
    assert body["plan"]["time_granularity"] == "month"
    assert all("2025-" in row["period"] for row in body["table"])


def test_anomaly_detection_explains_outliers():
    dataset_id = upload_sales()
    response = ask(dataset_id, "Detect anomalies in the dataset")
    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["operation"] == "anomalies"
    assert body["anomalies"]
    assert all(item["reason"] for item in body["anomalies"])


def test_schema_and_data_quality_questions():
    dataset_id = upload_sales()
    schema = ask(dataset_id, "What columns are available?")
    quality = ask(dataset_id, "Show data quality and missing values")
    assert schema.status_code == 200
    assert schema.json()["plan"]["operation"] == "schema"
    assert quality.status_code == 200
    assert quality.json()["plan"]["operation"] == "quality"
