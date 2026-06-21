"""
utils.py — Tool implementations for the Data Analyst Agent v2
New: auto_profile(), export helpers
"""

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = [
    "#63b3ed", "#68d391", "#f6ad55", "#fc8181",
    "#b794f4", "#76e4f7", "#fbd38d", "#9ae6b4"
]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1a1f2e",
    font=dict(family="Space Grotesk, sans-serif", color="#cbd5e1"),
    title_font=dict(size=16, color="#e2e8f0"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a3441"),
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=PALETTE,
    xaxis=dict(gridcolor="#2a3441", linecolor="#2a3441"),
    yaxis=dict(gridcolor="#2a3441", linecolor="#2a3441"),
)


def generate_chart(df, chart_type, x_column, y_column=None, title="Chart",
                   color_column=None, agg_func="sum"):
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found. Available: {list(df.columns)}")
    if y_column and y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found. Available: {list(df.columns)}")

    agg_map = {"sum": "sum", "mean": "mean", "count": "count", "max": "max", "min": "min"}
    func = agg_map.get(agg_func, "sum")

    if y_column and chart_type in ["bar", "line"] and x_column in df.select_dtypes(["object", "category"]).columns:
        df_plot = df.groupby(x_column)[y_column].agg(func).reset_index()
    else:
        df_plot = df.copy()

    kwargs = dict(data_frame=df_plot, title=title, color_discrete_sequence=PALETTE)

    if chart_type == "bar":
        fig = px.bar(x=x_column, y=y_column or df_plot.columns[1], **kwargs)
        fig.update_traces(marker_line_width=0, opacity=0.9)
    elif chart_type == "line":
        fig = px.line(x=x_column, y=y_column or df_plot.columns[1],
                      color=color_column, markers=True, **kwargs)
    elif chart_type == "scatter":
        fig = px.scatter(x=x_column, y=y_column, color=color_column, opacity=0.7, **kwargs)
    elif chart_type == "histogram":
        fig = px.histogram(x=x_column, **kwargs)
        fig.update_traces(marker_line_width=0)
    elif chart_type == "pie":
        fig = px.pie(names=x_column, values=y_column, **kwargs)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker=dict(colors=PALETTE))
    elif chart_type == "box":
        fig = px.box(x=color_column, y=y_column or x_column, **kwargs)
    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="Blues", text=corr.round(2).values,
            texttemplate="%{text}", showscale=True
        ))
        fig.update_layout(title=title)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def run_pandas_query(df, query_type, column=None, agg_column=None,
                     agg_func="sum", n=10, filter_expr=None):
    agg_map = {"sum": "sum", "mean": "mean", "count": "count", "max": "max", "min": "min"}
    func = agg_map.get(agg_func, "sum")

    if query_type == "groupby":
        if not column:
            raise ValueError("'column' required for groupby.")
        agg_col = agg_column or df.select_dtypes(include="number").columns[0]
        result = df.groupby(column)[agg_col].agg(func).reset_index()
        result.columns = [column, f"{func}_{agg_col}"]
        return result.sort_values(f"{func}_{agg_col}", ascending=False)
    elif query_type == "filter":
        if not filter_expr:
            raise ValueError("'filter_expr' required for filter.")
        return df.query(filter_expr)
    elif query_type == "top_n":
        agg_col = agg_column or column
        if not agg_col:
            raise ValueError("'column' or 'agg_column' required for top_n.")
        group_col = column if column != agg_col else df.select_dtypes(include="object").columns[0]
        result = df.groupby(group_col)[agg_col].agg(func).reset_index()
        return result.nlargest(n or 10, agg_col)
    elif query_type == "value_counts":
        if not column:
            raise ValueError("'column' required for value_counts.")
        vc = df[column].value_counts().reset_index()
        vc.columns = [column, "count"]
        return vc
    elif query_type == "correlation":
        numeric = df.select_dtypes(include="number")
        if column and column in numeric.columns:
            return numeric.corr()[column].sort_values(ascending=False).reset_index()
        return numeric.corr().round(3)
    elif query_type == "sort":
        agg_col = agg_column or column
        if not agg_col:
            return df.head(n or 10)
        return df.sort_values(agg_col, ascending=False).head(n or 10)
    else:
        return df.describe()


def compute_statistics(df, stat_type, columns=None):
    subset = df[columns] if columns else df
    if stat_type == "describe":
        return subset.select_dtypes(include="number").describe().round(3)
    elif stat_type == "missing":
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        result = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        return result[result["missing_count"] > 0].sort_values("missing_count", ascending=False)
    elif stat_type == "dtypes":
        return pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.values.astype(str),
            "unique_values": [df[c].nunique() for c in df.columns],
            "null_count": df.isnull().sum().values
        })
    elif stat_type == "correlation":
        return subset.select_dtypes(include="number").corr().round(3)
    elif stat_type == "unique_counts":
        cat_cols = subset.select_dtypes(include=["object", "category"]).columns
        result = {col: df[col].nunique() for col in cat_cols}
        return pd.DataFrame(list(result.items()), columns=["column", "unique_values"])
    else:
        return df.describe()


def auto_profile(df: pd.DataFrame) -> dict:
    """
    Auto-profile a DataFrame on upload.
    Returns a dict with summary stats, missing info, top categoricals, and key numerics.
    """
    profile = {}

    # Basic shape
    profile["shape"] = {"rows": len(df), "cols": len(df.columns)}

    # Missing values
    missing = df.isnull().sum()
    profile["missing_total"] = int(missing.sum())
    profile["missing_cols"] = missing[missing > 0].to_dict()

    # Numeric summary
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().round(2)
        profile["numeric_summary"] = desc.to_dict()
        # Find column with highest variance (most interesting)
        profile["most_variable_col"] = df[num_cols].std().idxmax()

    # Categorical summary
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    profile["categorical_cols"] = cat_cols
    profile["top_categoricals"] = {}
    for col in cat_cols[:4]:  # limit to 4
        vc = df[col].value_counts()
        profile["top_categoricals"][col] = {
            "unique": int(df[col].nunique()),
            "top": str(vc.index[0]) if len(vc) > 0 else "N/A",
            "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0
        }

    # Auto-detect potential KPI columns (revenue, sales, profit, amount, price)
    kpi_keywords = ["revenue", "sales", "profit", "amount", "price", "cost", "total", "value"]
    profile["kpi_cols"] = [
        c for c in num_cols
        if any(k in c.lower() for k in kpi_keywords)
    ]

    # Duplicate rows
    profile["duplicate_rows"] = int(df.duplicated().sum())

    return profile


def build_dashboard(df: pd.DataFrame):
    """Build a 4-panel auto-dashboard from any DataFrame."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Distribution: " + num_cols[0],
            "Top Categories" if cat_cols else "Numeric Spread",
            "Correlation Heatmap" if len(num_cols) > 2 else "Second Metric",
            "Summary: " + (num_cols[1] if len(num_cols) > 1 else num_cols[0])
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Panel 1: Histogram of first numeric col
    fig.add_trace(
        go.Histogram(x=df[num_cols[0]], marker_color=PALETTE[0],
                     name=num_cols[0], showlegend=False),
        row=1, col=1
    )

    # Panel 2: Top categories bar OR box plot
    if cat_cols:
        vc = df[cat_cols[0]].value_counts().head(8)
        fig.add_trace(
            go.Bar(x=vc.index.tolist(), y=vc.values.tolist(),
                   marker_color=PALETTE[1], name=cat_cols[0], showlegend=False),
            row=1, col=2
        )
    else:
        fig.add_trace(
            go.Box(y=df[num_cols[0]], marker_color=PALETTE[1],
                   name=num_cols[0], showlegend=False),
            row=1, col=2
        )

    # Panel 3: Heatmap if enough numerics, else scatter
    if len(num_cols) >= 3:
        corr = df[num_cols[:6]].corr().round(2)
        fig.add_trace(
            go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                       colorscale="Blues", showscale=False,
                       text=corr.values, texttemplate="%{text}"),
            row=2, col=1
        )
    elif len(num_cols) >= 2:
        fig.add_trace(
            go.Scatter(x=df[num_cols[0]], y=df[num_cols[1]],
                       mode="markers", marker=dict(color=PALETTE[2], opacity=0.6),
                       name="scatter", showlegend=False),
            row=2, col=1
        )

    # Panel 4: Second numeric distribution or category x numeric
    col2 = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    if cat_cols:
        agg = df.groupby(cat_cols[0])[col2].sum().nlargest(8)
        fig.add_trace(
            go.Bar(x=agg.index.tolist(), y=agg.values.tolist(),
                   marker_color=PALETTE[3], name=col2, showlegend=False),
            row=2, col=2
        )
    else:
        fig.add_trace(
            go.Histogram(x=df[col2], marker_color=PALETTE[3],
                         name=col2, showlegend=False),
            row=2, col=2
        )

    fig.update_layout(
        title_text="Auto Dashboard",
        **LAYOUT_DEFAULTS,
        height=600,
    )
    fig.update_annotations(font=dict(color="#94a3b8", size=12))
    return fig


# ── Export helpers ─────────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def fig_to_png_bytes(fig) -> bytes:
    return fig.to_image(format="png", scale=2)


def load_google_sheet(url: str) -> pd.DataFrame:
    """
    Load a public Google Sheet as a DataFrame.
    Accepts both the share URL and the /edit URL formats.
    """
    if "/edit" in url:
        base = url.split("/edit")[0]
    elif "/pub" in url:
        base = url.split("/pub")[0]
    else:
        base = url.rstrip("/")

    csv_url = base + "/export?format=csv"
    df = pd.read_csv(csv_url)
    return df


def detect_anomalies(df: pd.DataFrame, column: str = None) -> dict:
    """Detect outliers using IQR method across all numeric columns."""
    import numpy as np
    num_cols = [column] if column else df.select_dtypes(include="number").columns.tolist()
    results = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            results[col] = {
                "count": len(outliers),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outlier_rows": outliers
            }
    return results


def build_anomaly_chart(df: pd.DataFrame, column: str):
    """Box plot highlighting outliers for a given column."""
    fig = px.box(df, y=column, title=f"Outlier Detection — {column}",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def predict_trend(df: pd.DataFrame, target_col: str, periods: int = 5):
    """
    Fit a simple linear regression on a numeric column (using row index as X)
    and forecast the next N values.
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np

    series = df[target_col].dropna()
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    model = LinearRegression()
    model.fit(X, y)

    # Forecast next N periods
    future_X = np.arange(len(series), len(series) + periods).reshape(-1, 1)
    forecast = model.predict(future_X)
    fitted = model.predict(X)

    r2 = model.score(X, y)

    # Build chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(series))), y=y,
        mode="lines+markers", name="Actual",
        line=dict(color=PALETTE[0])
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(series))), y=fitted,
        mode="lines", name="Trend Line",
        line=dict(color=PALETTE[1], dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(series), len(series) + periods)), y=forecast,
        mode="lines+markers", name=f"Forecast (+{periods})",
        line=dict(color=PALETTE[2], dash="dot"),
        marker=dict(symbol="star", size=8)
    ))
    fig.update_layout(title=f"Trend & Forecast — {target_col}", **LAYOUT_DEFAULTS)

    return {
        "chart": fig,
        "forecast_values": forecast.round(2).tolist(),
        "r2_score": round(r2, 3),
        "slope": round(float(model.coef_[0]), 4),
        "direction": "upward 📈" if model.coef_[0] > 0 else "downward 📉"
    }


def generate_pdf_report(df: pd.DataFrame, messages: list, profile: dict) -> bytes:
    """
    Generate a PDF report summarising the dataset and all chat insights.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    import io, datetime

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                 fontSize=20, textColor=colors.HexColor("#1e3a5f"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
                              fontSize=13, textColor=colors.HexColor("#2b6cb0"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                fontSize=10, leading=14)
    insight_style = ParagraphStyle("Insight", parent=styles["Normal"],
                                   fontSize=9, leading=13,
                                   leftIndent=10, textColor=colors.HexColor("#374151"))

    story = []

    # Title
    story.append(Paragraph("📊 AI Data Analyst Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}", body_style))
    story.append(Spacer(1, 0.5*cm))

    # Dataset Overview
    story.append(Paragraph("Dataset Overview", h2_style))
    overview_data = [
        ["Rows", str(profile["shape"]["rows"])],
        ["Columns", str(profile["shape"]["cols"])],
        ["Missing Values", str(profile["missing_total"])],
        ["Duplicate Rows", str(profile["duplicate_rows"])],
        ["KPI Columns", ", ".join(profile.get("kpi_cols", [])) or "None detected"],
    ]
    t = Table(overview_data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dbeafe")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # Column Info
    story.append(Paragraph("Column Summary", h2_style))
    col_data = [["Column", "Type", "Missing"]]
    for col in df.columns:
        col_data.append([col, str(df[col].dtype), str(df[col].isnull().sum())])
    ct = Table(col_data, colWidths=[6*cm, 4*cm, 4*cm])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(ct)
    story.append(Spacer(1, 0.4*cm))

    # Chat Insights
    insights = [(m["content"], m["role"]) for m in messages if m["role"] == "assistant"]
    if insights:
        story.append(Paragraph("Analysis Insights", h2_style))
        for i, (text, _) in enumerate(insights, 1):
            clean = text.replace("**", "").replace("*", "").replace("#", "")
            clean = clean[:800] + "..." if len(clean) > 800 else clean
            story.append(Paragraph(f"<b>Insight {i}:</b>", body_style))
            story.append(Paragraph(clean, insight_style))
            story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


