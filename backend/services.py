
import json
import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# In-memory dataset storage for the assignment demo.
# Data is cleared when the backend restarts.
_DATASETS = {}


# --------------------------------------------------
# Dataset registration and retrieval
# --------------------------------------------------

def register_dataset(dataset_id, files):
    _DATASETS[dataset_id] = [
        {"name": name, "df": df.copy()}
        for name, df in files
    ]

    return [
        {
            "name": item["name"],
            "rows": int(len(item["df"])),
            "columns": int(len(item["df"].columns)),
            "column_names": list(item["df"].columns),
        }
        for item in _DATASETS[dataset_id]
    ]


def _entries(dataset_id):
    if dataset_id not in _DATASETS:
        raise KeyError(dataset_id)
    return _DATASETS[dataset_id]


def _records(df, limit=100):
    safe_df = df.head(limit).copy()

    # Convert NumPy scalar types for JSON serialization.
    safe_df = safe_df.astype(object).where(
        pd.notna(safe_df), None
    )

    return json.loads(
        safe_df.to_json(
            orient="records",
            date_format="iso",
        )
    )


def dataset_preview(dataset_id):
    return [
        {
            "name": item["name"],
            "preview": _records(item["df"], 10),
        }
        for item in _entries(dataset_id)
    ]


def dataset_quality(dataset_id):
    results = []

    for item in _entries(dataset_id):
        df = item["df"]

        results.append({
            "name": item["name"],
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": {
                str(col): int(count)
                for col, count in df.isna().sum().items()
            },
            "dtypes": {
                str(col): str(dtype)
                for col, dtype in df.dtypes.items()
            },
        })

    return results


# --------------------------------------------------
# Anomaly detection
# --------------------------------------------------

def _detect_df_anomalies(df):
    findings = []
    numeric = df.select_dtypes(include="number")

    for column in numeric.columns:
        series = numeric[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            mask = (series < q1) | (series > q3)
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)

        for index, value in series[mask].items():
            findings.append({
                "row_index": int(index),
                "column": str(column),
                "value": float(value),
                "reason": (
                    f"Value is outside the 1.5×IQR range "
                    f"for column {column}."
                ),
            })

    return findings[:500]


def dataset_anomalies(dataset_id):
    return [
        {
            "name": item["name"],
            "anomalies": _detect_df_anomalies(item["df"]),
        }
        for item in _entries(dataset_id)
    ]


# --------------------------------------------------
# Question and column matching
# --------------------------------------------------

def _normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _column_mentioned(question, column):
    """
    Check whether a real dataset column is explicitly mentioned.
    Handles names such as order_date and Order Date.
    """
    q = _normalize(question)
    name = _normalize(column)

    if not name:
        return False

    return bool(
        re.search(
            r"(?<![a-z0-9])"
            + re.escape(name)
            + r"(?![a-z0-9])",
            q,
        )
    )


def _explicit_column(df, question, candidates=None):
    """
    Prefer columns explicitly named by the user.
    """
    columns = list(df.columns)

    if candidates:
        columns = [
            col for col in columns
            if col in candidates
        ]

    matches = [
        col for col in columns
        if _column_mentioned(question, col)
    ]

    if not matches:
        return None

    # Prefer the longest explicit column name if names overlap.
    return max(matches, key=lambda col: len(_normalize(col)))


def _find_metric(df, question):
    """
    Find a numeric metric mentioned by the user.
    """
    numeric = list(df.select_dtypes(include="number").columns)

    explicit = _explicit_column(
        df,
        question,
        candidates=numeric,
    )

    if explicit is not None:
        return explicit

    q = _normalize(question)

    common_terms = {
        "revenue": ["revenue", "sales amount", "total sales"],
        "profit": ["profit", "earnings"],
        "quantity": ["quantity", "units"],
        "price": ["price", "unit price"],
        "cost": ["cost", "expense"],
        "amount": ["amount", "value"],
    }

    for column in numeric:
        normalized_column = _normalize(column)

        if normalized_column and normalized_column in q:
            return column

    for column in numeric:
        col_name = _normalize(column)

        for key, aliases in common_terms.items():
            if key in col_name and any(alias in q for alias in aliases):
                return column

    for preferred in (
        "revenue", "sales", "profit",
        "amount", "quantity", "price", "cost",
    ):
        for column in numeric:
            if preferred in _normalize(column):
                return column

    return numeric[0] if numeric else None


def _explicit_group_column(df, question):
    """Find the column named after a grouping phrase such as 'by region'."""
    q = _normalize(question)
    normalized_columns = [(column, _normalize(column)) for column in df.columns]

    # Prefer the words immediately following a grouping cue. This prevents a
    # metric mentioned earlier in the question (e.g. revenue) being selected
    # as the group column instead of region.
    cues = (
        "grouped by ", "group by ", "breakdown by ",
        "broken down by ", "by ", "per ", "for each ",
    )
    for cue in cues:
        start = q.find(cue)
        if start < 0:
            continue
        phrase = q[start + len(cue):]
        matches = [
            (column, name) for column, name in normalized_columns
            if name and re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", phrase)
        ]
        if matches:
            return max(matches, key=lambda pair: len(pair[1]))[0]

    # If the user used a semantic grouping word, prefer the matching column.
    hints = (
        ("region", ("region", "area", "territory")),
        ("product", ("product", "item")),
        ("category", ("category", "type")),
        ("customer", ("customer", "client")),
        ("segment", ("segment",)),
        ("country", ("country",)),
        ("city", ("city",)),
        ("date", ("date", "daily", "day")),
        ("month", ("month", "monthly")),
        ("year", ("year", "yearly", "annual")),
    )
    for preferred, aliases in hints:
        if any(alias in q for alias in aliases):
            for column, name in normalized_columns:
                if preferred in name:
                    return column

    # Fall back only to a non-numeric/categorical column.
    categorical = list(df.select_dtypes(include=["object", "category", "string"]).columns)
    return categorical[0] if categorical else None


def _find_group_column(df, question):
    """Choose a grouping column, prioritizing the explicit 'by X' phrase."""
    q = _normalize(question)
    explicit_group = _explicit_group_column(df, question)
    if explicit_group is not None:
        return explicit_group

    # A general explicit column mention is a last resort, not the first choice:
    # metric columns are often mentioned before the requested group column.
    explicit = _explicit_column(df, question)
    if explicit is not None and not pd.api.types.is_numeric_dtype(df[explicit].dtype):
        return explicit

    time_question = any(
        word in q for word in (
            "trend", "over time", "monthly", "daily", "weekly",
            "by date", "by month", "by year",
        )
    )
    if time_question:
        for column in df.columns:
            if any(word in _normalize(column) for word in ("date", "month", "year", "time")):
                return column

    categorical = list(df.select_dtypes(include=["object", "category", "string"]).columns)
    return categorical[0] if categorical else None


def _pick_file(entries, question):
    q = _normalize(question)

    for item in entries:
        filename = _normalize(
            item["name"].rsplit(".", 1)[0]
        )

        if filename and filename in q:
            return item

    return entries[0]


# --------------------------------------------------
# Gemini analysis planner
# --------------------------------------------------

def _llm_plan(df, question, history):
    """
    Ask Gemini for a restricted JSON analysis plan.
    The plan is validated and corrected against real columns.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=20000),
        )

        schema = {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        }

        prompt = f"""
You are a data analysis planning assistant.

Return a single JSON object. Do not generate Python or SQL code.

Allowed operations:
summary, schema, quality, anomalies, aggregate, groupby, top_rows

Allowed keys:
operation, metric, group, aggregation, sort, limit, chart

Allowed aggregation values:
sum, mean, min, max, count

Allowed sort values:
asc, desc

Allowed chart values:
bar, line, pie, scatter

Important rules:
- Use exact column names from the supplied schema.
- If the question says "by region", group by the region column.
- If the question says "by product", group by the product column.
- Do not select a date column unless the question asks for date,
  monthly, daily, yearly, or time-based analysis.
- If the user asks to group, rank, compare, or chart categories,
  choose groupby.
- If the user asks for one overall total or average, choose aggregate.
- Do not invent columns.
- limit must be an integer between 1 and 100.

Dataset schema:
{json.dumps(schema)}

User question:
{question}

Recent conversation:
{json.dumps(history[-6:], default=str)}
"""

        response = client.models.generate_content(
            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash",
            ),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        plan = json.loads(response.text or "{}")

        allowed_operations = {
            "summary",
            "schema",
            "quality",
            "anomalies",
            "aggregate",
            "groupby",
            "top_rows",
        }

        if plan.get("operation") not in allowed_operations:
            return None

        if plan.get("aggregation") not in (
            None, "sum", "mean", "min", "max", "count"
        ):
            return None

        if plan.get("chart") not in (
            None, "bar", "line", "pie", "scatter"
        ):
            return None

        # Correct hallucinated or invalid column names.
        metric = plan.get("metric")
        group = plan.get("group")
        if not isinstance(metric, str) or metric not in df.columns:
            plan["metric"] = None

        if not isinstance(group, str) or group not in df.columns:
            plan["group"] = None

        return plan

    except Exception as exc:
        print(f"Gemini planning error: {exc}")
        return None


# --------------------------------------------------
# Rule-based fallback planner
# --------------------------------------------------

def _sort_direction(question):
    """Resolve ranking direction without misreading 'highest to lowest'."""
    q = _normalize(question)
    high_words = r"(?:highest|largest|most|best|maximum|max)"
    low_words = r"(?:lowest|smallest|least|worst|minimum|min)"

    if re.search(high_words + r".*\bto\b.*" + low_words, q):
        return "desc"
    if re.search(low_words + r".*\bto\b.*" + high_words, q):
        return "asc"

    if any(word in q for word in ("ascending", "lowest", "smallest", "least", "worst")):
        if not any(word in q for word in ("highest to lowest", "largest to smallest", "most to least")):
            return "asc"
    return "desc"


def _heuristic_plan(df, question):
    q = _normalize(question)

    if any(
        word in q
        for word in ("anomaly", "anomalies", "outlier", "outliers", "unusual")
    ):
        return {"operation": "anomalies"}

    if any(
        word in q
        for word in (
            "missing", "data quality", "duplicate",
            "duplicates", "null values",
        )
    ):
        return {"operation": "quality"}

    if any(
        word in q
        for word in (
            "schema", "columns", "column names",
            "what fields", "available fields",
        )
    ):
        return {"operation": "schema"}

    metric = _find_metric(df, question)
    group = _find_group_column(df, question)

    aggregation = "sum"

    if any(word in q for word in ("average", "mean", "avg")):
        aggregation = "mean"
    elif "minimum" in q or "smallest" in q:
        aggregation = "min"
    elif "maximum" in q or "largest" in q:
        aggregation = "max"
    elif "count" in q or "number of" in q:
        aggregation = "count"

    # Explicit grouping intent takes priority over a generic summary.
    grouping_intent = (
        " by " in f" {q} "
        or "group by" in q
        or "grouped by" in q
        or "per " in q
        or "for each " in q
        or "breakdown" in q
        or "compare" in q
        or "ranking" in q
        or "rank " in q
        or "highest" in q
        or "lowest" in q
        or "top " in q
        or "most " in q
        or "best " in q
        or "chart" in q
        or "plot" in q
        or "graph" in q
        or "visual" in q
        or "trend" in q
        or "over time" in q
        or "monthly" in q
        or "daily" in q
        or "weekly" in q
        or "yearly" in q
    )

    if grouping_intent and metric and group:
        sort_direction = _sort_direction(question)

        chart = "bar"

        if any(
            word in q
            for word in ("trend", "over time", "monthly", "daily")
        ):
            chart = "line"

        if "pie" in q:
            chart = "pie"
        elif "scatter" in q:
            chart = "scatter"

        return {
            "operation": "groupby",
            "metric": metric,
            "group": group,
            "aggregation": aggregation,
            "sort": sort_direction,
            "limit": 100 if chart == "line" else 20,
            "chart": chart,
            **({"time_granularity": "month"} if "monthly" in q else {}),
            **({"time_granularity": "day"} if "daily" in q else {}),
            **({"time_granularity": "year"} if "yearly" in q or "annual" in q else {}),
        }

    # Match aggregation words as whole words. A substring check for "sum"
    # incorrectly treats "summary" as a request to sum the first metric.
    if metric and any(
        re.search(r"\b" + re.escape(word) + r"\b", q)
        for word in (
            "sum", "total", "average", "mean",
            "avg", "minimum", "maximum",
        )
    ):
        return {
            "operation": "aggregate",
            "metric": metric,
            "aggregation": aggregation,
        }

    return {"operation": "summary"}


# --------------------------------------------------
# Execute the validated analysis plan
# --------------------------------------------------

def _run_plan(item, plan):
    df = item["df"].copy()
    operation = plan.get("operation", "summary")

    result = {
        "answer": "",
        "table": None,
        "chart": None,
        "sql": None,
        "pandas_code": None,
        "anomalies": None,
    }

    if operation == "schema":
        result["table"] = [
            {
                "column": str(column),
                "dtype": str(dtype),
            }
            for column, dtype in df.dtypes.items()
        ]

        result["answer"] = (
            f"Dataset contains {len(df)} rows and "
            f"{len(df.columns)} columns."
        )
        return result

    if operation == "quality":
        result["table"] = [
            {
                "column": str(column),
                "missing_values": int(count),
            }
            for column, count in df.isna().sum().items()
        ]

        result["answer"] = (
            f"Dataset contains {len(df)} rows, "
            f"{len(df.columns)} columns and "
            f"{int(df.duplicated().sum())} duplicate rows. "
            "Missing values are listed in the table."
        )
        return result

    if operation == "anomalies":
        rows = _detect_df_anomalies(df)
        result["anomalies"] = rows
        result["table"] = rows[:100]

        result["answer"] = (
            f"Found {len(rows)} potential numeric outlier(s) "
            "using the 1.5×IQR rule. Review them before "
            "treating them as data errors."
        )
        return result

    if operation == "summary":
        result["table"] = [
            {
                "column": str(column),
                "dtype": str(dtype),
                "non_null": int(df[column].notna().sum()),
            }
            for column, dtype in df.dtypes.items()
        ]

        result["answer"] = (
            f"**{item['name']}** contains {len(df):,} rows "
            f"and {len(df.columns)} columns. Ask for a total, "
            "average, ranking, chart, data-quality report, "
            "or anomaly check."
        )
        return result

    metric = plan.get("metric")

    if operation in ("aggregate", "groupby", "top_rows"):
        if metric not in df.columns:
            metric = _find_metric(df, "")

        if metric not in df.columns:
            result["answer"] = (
                "No suitable metric column was found. "
                "Please specify a numeric column."
            )
            return result

    aggregation = plan.get("aggregation", "sum")

    if aggregation not in ("sum", "mean", "min", "max", "count"):
        aggregation = "sum"

    sort = "asc" if plan.get("sort") == "asc" else "desc"

    try:
        limit = int(plan.get("limit", 10))
    except (TypeError, ValueError):
        limit = 10

    limit = max(1, min(limit, 100))

    table_name = '"' + str(
        item["name"].rsplit(".", 1)[0]
    ).replace('"', '""') + '"'

    def quote_identifier(value):
        return '"' + str(value).replace('"', '""') + '"'

    if operation == "aggregate":
        series = pd.to_numeric(df[metric], errors="coerce")
        value = getattr(series, aggregation)()

        if pd.isna(value):
            result["answer"] = (
                f"No valid numeric values found in `{metric}`."
            )
            result["table"] = []
            return result

        result["answer"] = (
            f"**{aggregation.title()} of `{metric}`:** "
            f"{value:,.4f}"
        )

        result["table"] = [{
            "metric": metric,
            "aggregation": aggregation,
            "value": float(value),
        }]

        result["sql"] = (
            f"SELECT {aggregation.upper()}("
            f"{quote_identifier(metric)}) AS result "
            f"FROM {table_name};"
        )

        result["pandas_code"] = (
            f"pd.to_numeric(df[{metric!r}], errors='coerce')"
            f".{aggregation}()"
        )

        return result

    if operation == "top_rows":
        temp = df.copy()
        temp["_metric_sort"] = pd.to_numeric(
            temp[metric],
            errors="coerce",
        )

        selected = (
            temp.sort_values(
                "_metric_sort",
                ascending=(sort == "asc"),
            )
            .head(limit)
            .drop(columns="_metric_sort")
        )

        result["answer"] = (
            f"Showing up to {limit} rows sorted by "
            f"`{metric}` ({sort})."
        )

        result["table"] = _records(selected, limit)
        result["sql"] = (
            f"SELECT * FROM {table_name} "
            f"ORDER BY {quote_identifier(metric)} "
            f"{sort.upper()} LIMIT {limit};"
        )

        result["pandas_code"] = (
            f"df.sort_values({metric!r}, "
            f"ascending={sort == 'asc'}).head({limit})"
        )

        return result

    if operation == "groupby":
        group = plan.get("group")

        if not isinstance(group, str) or group not in df.columns:
            group = _find_group_column(df, "")

        # A group column and metric must be different columns. Otherwise
        # df[[group, metric]] creates duplicate labels and work[metric]
        # becomes a DataFrame, which makes pd.to_numeric raise TypeError.
        if group == metric:
            alternatives = [
                column for column in df.select_dtypes(include="number").columns
                if column != group
            ]
            if alternatives:
                metric = alternatives[0]
            else:
                result["answer"] = (
                    "The selected grouping and metric columns are the same. "
                    "Please specify a numeric metric column and a different grouping column."
                )
                return result

        if group not in df.columns:
            result["answer"] = (
                "No suitable grouping column was found. "
                "Please specify a column such as region or product."
            )
            return result

        work = df[[group, metric]].copy()
        time_granularity = plan.get("time_granularity")
        if time_granularity in ("month", "day", "year"):
            parsed_dates = pd.to_datetime(work[group], errors="coerce")
            if time_granularity == "month":
                work[group] = parsed_dates.dt.to_period("M").astype(str)
            elif time_granularity == "year":
                work[group] = parsed_dates.dt.year.astype("Int64").astype(str)
            else:
                work[group] = parsed_dates.dt.strftime("%Y-%m-%d")
            work = work[work[group].notna() & (work[group] != "NaT")]
            work = work.rename(columns={group: "period"})
            group = "period"
        work[metric] = pd.to_numeric(
            work[metric],
            errors="coerce",
        )

        value_column = f"{aggregation}_{metric}"

        grouped = (
            work.groupby(group, dropna=False)[metric]
            .agg(aggregation)
            .reset_index(name=value_column)
        )

        grouped = (
            grouped.sort_values(
                value_column,
                ascending=(sort == "asc"),
            )
            .head(limit)
        )

        rows = _records(grouped, limit)

        result["answer"] = (
            f"Grouped `{metric}` by `{group}` using "
            f"{aggregation}. Showing {len(rows)} group(s)."
        )

        result["table"] = rows

        chart_type = plan.get("chart", "bar")

        if chart_type not in ("bar", "line", "pie", "scatter"):
            chart_type = "bar"

        result["chart"] = {
            "type": chart_type,
            "title": (
                f"{aggregation.title()} of {metric} by {group}"
            ),
            "x": group,
            "y": value_column,
            "data": rows,
        }

        result["sql"] = (
            f"SELECT {quote_identifier(group)}, "
            f"{aggregation.upper()}({quote_identifier(metric)}) "
            f"AS {quote_identifier(value_column)} "
            f"FROM {table_name} "
            f"GROUP BY {quote_identifier(group)} "
            f"ORDER BY {quote_identifier(value_column)} "
            f"{sort.upper()} LIMIT {limit};"
        )

        result["pandas_code"] = (
            f"df.groupby({group!r}, dropna=False)"
            f"[{metric!r}].{aggregation}()."
            f"reset_index(name={value_column!r})"
        )

        return result

    result["answer"] = (
        "I could not map that question to a supported analysis. "
        "Try asking for a summary, total, average, ranking, "
        "chart, data-quality report, or anomaly check."
    )

    return result


# --------------------------------------------------
# Public analysis function
# --------------------------------------------------

def analyze_question(dataset_id, question, history):
    entries = _entries(dataset_id)
    item = _pick_file(entries, question)
    df = item["df"]

    # Ask Gemini first; use the rule-based planner if unavailable.
    plan = _llm_plan(df, question, history)

    if not plan:
        plan = _heuristic_plan(df, question)

    q = _normalize(question)

    # Explicit grouping columns in the question must override
    # an incorrect grouping column selected by the LLM.
    explicit_group = _explicit_group_column(df, question)

    grouping_intent = (
        " by " in f" {q} "
        or "group by" in q
        or "grouped by" in q
        or "per " in q
        or "for each " in q
        or "breakdown" in q
        or "compare" in q
        or "ranking" in q
        or "rank " in q
        or "highest" in q
        or "lowest" in q
        or "top " in q
        or "most " in q
        or "best " in q
        or "chart" in q
        or "plot" in q
        or "graph" in q
        or "visual" in q
        or "trend" in q
        or "over time" in q
        or "monthly" in q
        or "daily" in q
        or "weekly" in q
        or "yearly" in q
    )

    time_granularity = None
    if "monthly" in q or "by month" in q:
        time_granularity = "month"
    elif "daily" in q or "by day" in q:
        time_granularity = "day"
    elif "yearly" in q or "annual" in q or "by year" in q:
        time_granularity = "year"

    is_time_trend = any(word in q for word in ("trend", "over time", "monthly", "daily", "weekly", "yearly", "by month", "by year"))
    if is_time_trend and time_granularity is None:
        time_granularity = "month" if "weekly" not in q else "day"

    if is_time_trend:
        date_column = next((col for col in df.columns if any(token in _normalize(col) for token in ("date", "time"))), None)
        metric = _find_metric(df, question)
        if date_column is not None and metric is not None:
            plan.update({"operation": "groupby", "group": date_column, "metric": metric, "aggregation": "sum", "sort": "asc", "limit": 100, "chart": "line", "time_granularity": time_granularity or "month"})

    if explicit_group and grouping_intent and not is_time_trend:
        metric = _find_metric(df, question)

        if metric:
            plan["operation"] = "groupby"
            plan["group"] = explicit_group
            plan["metric"] = metric

            if plan.get("aggregation") not in (
                "sum", "mean", "min", "max", "count"
            ):
                plan["aggregation"] = (
                    "mean"
                    if any(word in q for word in ("average", "mean", "avg"))
                    else "sum"
                )

            plan["sort"] = _sort_direction(question)

            if plan.get("chart") not in (
                "bar", "line", "pie", "scatter"
            ):
                plan["chart"] = (
                    "line"
                    if any(
                        word in q
                        for word in (
                            "trend", "over time", "monthly", "daily"
                        )
                    )
                    else "bar"
                )

            plan.setdefault("limit", 100)

    result = _run_plan(item, plan)

    result["dataset"] = item["name"]
    result["plan"] = plan
    result["method_note"] = (
        "Calculations are performed using Pandas. "
        "Potential outliers require human review."
    )

    result["answer"] += (
        "\n\nMethod: Results are calculated from the uploaded "
        "dataset. Statistical outliers are candidates for review."
    )

    return result
