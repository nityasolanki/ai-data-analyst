# 📊 AI Data Analyst Agent — v2

> Upload any CSV or connect a Google Sheet → Ask questions in plain English → Get instant charts, insights & analysis

Built with **LLaMA 3.3 70B (Groq)** · **Custom tool-calling agent** · **Pandas** · **Plotly** · **Streamlit**

---

## 🚀 Quick Start

```bash
# 1. Enter the project folder
cd ai_data_analyst_v2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY=your_key_here      # Mac/Linux
set GROQ_API_KEY=your_key_here         # Windows

# 4. Run
streamlit run app.py
```

Open http://localhost:8501, upload `sample_data.csv`, and start asking!

> **Tip:** You can also create a `.env` file in the project root with `GROQ_API_KEY=your_key_here` — it will be loaded automatically.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Chat with your data** | Ask questions in plain English — agent picks the right tool automatically |
| 🧠 **Conversation memory** | Follow-up questions work: "now filter that by North" |
| 📊 **Auto Dashboard** | 4-panel chart dashboard generated the moment a file loads |
| 🔍 **Auto Profile** | Instant data quality report — missing values, KPIs, categoricals, duplicates |
| 🚨 **Anomaly Detection** | IQR-based outlier detection across all numeric columns |
| 🔮 **Trend Forecasting** | Linear regression forecast for any numeric column |
| 📄 **PDF Report** | Export a full analysis report as a formatted PDF |
| 📤 **Export results** | Download any chart as PNG or any table as CSV |
| 🌐 **Google Sheets** | Paste a public Google Sheet URL instead of uploading a file |
| 📁 **CSV + Excel** | Supports both .csv and .xlsx uploads |

---

## 💡 Example Questions

| Question | What the agent does |
|---|---|
| "Show a bar chart of revenue by region" | `generate_chart` → bar chart |
| "What are the top 5 products by profit?" | `run_pandas_query` (top_n) → ranked table |
| "Now filter that to just Electronics" | Uses memory → filtered result |
| "Is there a correlation between price and units sold?" | `compute_statistics` (correlation) → heatmap |
| "Give me a statistical summary" | `compute_statistics` (describe) → stats table |
| "Which sales rep performed best?" | `run_pandas_query` (groupby) → ranked table |
| "Are there any missing values?" | `compute_statistics` (missing) → missing report |

---

## 🏗️ Architecture

```
ai_data_analyst_v2/
├── app.py            ← Streamlit UI (6 tabs: Chat, Dashboard, Profile, Data, Anomalies, Predict)
├── agent.py          ← DataAnalystAgent with memory + tool-calling loop
├── utils.py          ← Tool implementations + dashboard + export helpers
├── requirements.txt  ← Dependencies
├── sample_data.csv   ← Demo dataset (30-row sales data)
└── README.md         ← This file
```

### Agent Flow

```
User Question
     │
     ▼
conversation history injected
     │
     ▼
LLaMA 3.3 70B (Groq) ──► picks tool ──► generate_chart
     │                                ├── run_pandas_query
     │                                └── compute_statistics
     ▼
Tool executes on real DataFrame (no hallucination)
     │
     ▼
LLaMA 3.3 interprets results → writes insight
     │
     ▼
Text + optional Chart/Table → UI
     │
     ▼
Exchange saved to memory (capped at 20 turns)
```

### Tools

| Tool | Operations |
|---|---|
| `generate_chart` | bar, line, scatter, histogram, pie, box, heatmap |
| `run_pandas_query` | groupby, filter, top_n, value_counts, correlation, sort |
| `compute_statistics` | describe, missing, dtypes, correlation matrix, unique_counts |

---

## 🌐 Deploy to Streamlit Cloud (free)

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "Initial commit: AI Data Analyst Agent v2"
git remote add origin https://github.com/YOUR_USERNAME/ai-data-analyst-v2.git
git push -u origin main

# 2. Go to share.streamlit.io
# 3. Connect repo → Advanced Settings → add secret:
#    GROQ_API_KEY = your_key_here
# 4. Click Deploy — live URL in ~2 minutes
```

---

## 📦 Dependencies

```
streamlit>=1.35.0     # UI framework
groq>=0.9.0           # LLaMA 3.3 inference (fast + free)
pandas>=2.0.0         # Data operations
plotly>=5.18.0        # Interactive charts
kaleido>=0.2.1        # PNG chart export
openpyxl>=3.1.0       # Excel (.xlsx) support
requests>=2.31.0      # Google Sheets fetch
python-dotenv>=1.0.0  # .env file support
scikit-learn>=1.4.0   # Linear regression (Predict tab)
reportlab>=4.0.0      # PDF report generation
```
