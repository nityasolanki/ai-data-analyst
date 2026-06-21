"""
AI Data Analyst Agent — v2
──────────────────────────
New in v2:
  • Conversation memory — follow-up questions work naturally
  • Auto data profiling — instant insights on file upload
  • Google Sheets support — load data from a public sheet URL
"""

import json
import pandas as pd
from groq import Groq
from utils import generate_chart, run_pandas_query, compute_statistics, auto_profile


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": (
                "Generate a Plotly chart from the dataframe. Use when the user asks "
                "for a bar chart, line chart, scatter plot, histogram, pie chart, or "
                "any other visualisation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "histogram", "pie", "box", "heatmap"],
                        "description": "The type of chart to generate."
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column name for the X axis (or labels for pie)."
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column name for the Y axis (or values for pie). Leave empty for histogram."
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title."
                    },
                    "color_column": {
                        "type": "string",
                        "description": "Optional column to use for colour grouping."
                    },
                    "agg_func": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "max", "min"],
                        "description": "Aggregation function when grouping by x_column."
                    }
                },
                "required": ["chart_type", "x_column", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_pandas_query",
            "description": (
                "Run a pandas operation on the dataframe to answer analytical questions. "
                "Use for filtering, groupby, top-N, value counts, correlations, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["groupby", "filter", "top_n", "value_counts", "correlation", "sort", "custom"],
                        "description": "Type of pandas operation."
                    },
                    "column": {"type": "string", "description": "Primary column to operate on."},
                    "agg_column": {"type": "string", "description": "Column to aggregate (for groupby)."},
                    "agg_func": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "max", "min"],
                        "description": "Aggregation function."
                    },
                    "n": {"type": "integer", "description": "Number of rows for top_n or head operations."},
                    "filter_expr": {"type": "string", "description": "Pandas query string."}
                },
                "required": ["query_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_statistics",
            "description": (
                "Compute descriptive statistics, missing value analysis, data types, "
                "distributions and correlations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stat_type": {
                        "type": "string",
                        "enum": ["describe", "missing", "dtypes", "correlation", "unique_counts"],
                        "description": "Type of statistical analysis."
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific columns to analyse. Leave empty for all."
                    }
                },
                "required": ["stat_type"]
            }
        }
    }
]


class DataAnalystAgent:
    def __init__(self, df: pd.DataFrame, api_key: str):
        self.df = df
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        # Conversation memory
        self.history: list = []

    def _build_system_prompt(self) -> str:
        cols = ", ".join(self.df.columns.tolist())
        dtype_str = ", ".join(f"{c}: {t}" for c, t in self.df.dtypes.to_dict().items())
        sample = self.df.head(3).to_dict(orient="records")
        return f"""You are an expert AI Data Analyst with memory of the full conversation.

Dataset:
  Columns  : {cols}
  Dtypes   : {dtype_str}
  Shape    : {self.df.shape[0]} rows x {self.df.shape[1]} columns
  Sample   : {json.dumps(sample, default=str)}

Rules:
1. Use conversation history to answer follow-up questions in context.
2. Always call a tool unless the question is purely conversational.
3. Pick the most appropriate chart type for the data.
4. After tool results, write clear, insightful analysis with bullet points.
5. Reference previous answers when relevant (e.g. "As we saw earlier...").
"""

    def run(self, user_question: str) -> dict:
        messages = (
            [{"role": "system", "content": self._build_system_prompt()}]
            + self.history
            + [{"role": "user", "content": user_question}]
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.3,
        )

        message = response.choices[0].message
        chart_result = None
        df_result = None
        tool_output_text = ""

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            try:
                if fn_name == "generate_chart":
                    chart_result = generate_chart(self.df, **fn_args)
                    tool_output_text = f"Chart generated: {fn_args.get('title', 'Chart')}"
                elif fn_name == "run_pandas_query":
                    result = run_pandas_query(self.df, **fn_args)
                    if isinstance(result, pd.DataFrame):
                        df_result = result
                        tool_output_text = result.to_string(max_rows=20)
                    else:
                        tool_output_text = str(result)
                elif fn_name == "compute_statistics":
                    result = compute_statistics(self.df, **fn_args)
                    if isinstance(result, pd.DataFrame):
                        df_result = result
                        tool_output_text = result.to_string()
                    else:
                        tool_output_text = str(result)
            except Exception as e:
                tool_output_text = f"Tool error: {str(e)}"

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": message.tool_calls
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output_text
            })

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.4,
            )
            final_text = final_response.choices[0].message.content
        else:
            final_text = message.content or "I couldn't generate a response."

        # Save to memory (bounded to last 20 turns)
        self.history.append({"role": "user", "content": user_question})
        self.history.append({"role": "assistant", "content": final_text})
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return {
            "text": final_text,
            "chart": chart_result,
            "dataframe": df_result,
        }

    def get_auto_profile(self) -> dict:
        """Run automatic data profiling when a file is first uploaded."""
        return auto_profile(self.df)
