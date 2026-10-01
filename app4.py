"""
U.S. Provisional Natality Exploration Dashboard (2025)
Integrated Single-File Streamlit Application
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:  # The app must still load (and show a warning) if openai is not installed.
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

# -----------------------------------------------------------------------------
# 1. CONSTANTS & LOOKUPS
# -----------------------------------------------------------------------------

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

STATE_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

SEX_COLORS = {
    "Female": "#2b5c8f",
    "Male": "#d95f02",
}

# AI chatbot settings ----------------------------------------------------------
# Provider is detected from the API-key prefix.
LLM_PROVIDERS = {
    "gsk_": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
        # Tried in order if the chosen model is retired / unknown / unavailable.
        "fallback_models": [
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
        ],
    },
    "xai-": {
        "name": "xAI (Grok)",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-3-mini",
        "fallback_models": [],
    },
}

# Any of these Streamlit secret names may hold the API key.
API_KEY_SECRET_NAMES = ["GROQ_API_KEY", "GROK_API_KEY", "XAI_API_KEY", "LLM_API_KEY"]
MODEL_OVERRIDE_SECRET_NAME = "LLM_MODEL"

# Phrases (lower-case) that mean "this model can't be used" -> trigger fallback.
MODEL_UNAVAILABLE_PHRASES = [
    "not found",
    "decommissioned",
    "deprecated",
    "does not exist",
    "not available",
    "model_not_found",
    "model_decommissioned",
]

SUGGESTED_QUESTIONS = [
    "Which 3 states had the most births?",
    "Which month had the fewest births, and why might that be?",
    "What is the male-to-female ratio in the current selection?",
]

MAX_HISTORY_MESSAGES = 6      # only the last N chat messages are sent to the model
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2000         # gpt-oss is a reasoning model and spends tokens "thinking"

SYSTEM_PROMPT = """You are a friendly data assistant for CDC/NCHS provisional 2025 U.S. natality (birth) data.

Rules:
- Answer ONLY from the DATA SUMMARY below. It reflects the user's current sidebar filters.
- Values are raw birth COUNTS, not rates. When comparing states, remind the user that population size drives the counts.
- The data is provisional and may be revised.
- If a question cannot be answered from the data (race, mother's age, other years, etc.), say so and suggest what data would be needed.
- Use thousands separators, double-check your arithmetic, and be concise.
"""

# -----------------------------------------------------------------------------
# 2. PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="U.S. Provisional Natality Dashboard (2025)",
    page_icon="📊",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 3. DATA LOADING & VALIDATION
# -----------------------------------------------------------------------------

def get_data_path() -> Path:
    """Resolve file path across root and data/ directories."""
    current_dir = Path(__file__).resolve().parent
    candidate_paths = [
        current_dir / "Provisional_Natality_2025_CDC.csv",
        current_dir / "data" / "Provisional_Natality_2025_CDC.csv",
        Path("Provisional_Natality_2025_CDC.csv"),
        Path("data/Provisional_Natality_2025_CDC.csv"),
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Provisional_Natality_2025_CDC.csv not found in current folder or data/ folder."
    )


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate dataframe structure and data integrity."""
    required_cols = {
        "state_of_residence", "month", "month_code",
        "year_code", "sex_of_infant", "births"
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if not pd.api.types.is_numeric_dtype(df["births"]):
        raise TypeError("Column 'births' must be numeric.")

    if (df["births"] < 0).any():
        raise ValueError("Column 'births' contains negative values.")


@st.cache_data(show_spinner="Loading CDC Natality Data...")
def load_and_preprocess_data() -> pd.DataFrame:
    """Load, clean, order categorical variables, and map state abbreviations."""
    file_path = get_data_path()
    df = pd.read_csv(file_path)
    validate_raw_data(df)

    # State postal code mapping
    df["state_abbr"] = df["state_of_residence"].map(STATE_TO_ABBR)

    # Clean and order chronological months
    df["month"] = df["month"].astype(str).str.strip()
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)

    # Clean strings and enforce integer counts
    df["sex_of_infant"] = df["sex_of_infant"].astype(str).str.strip()
    df["births"] = df["births"].astype(int)

    return df

# -----------------------------------------------------------------------------
# 4. KPI METRIC COMPUTATIONS
# -----------------------------------------------------------------------------

def compute_kpis(filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary figures from the active filtered slice."""
    if filtered_df.empty:
        return {
            "total_births": 0,
            "selected_geographies": 0,
            "avg_monthly_births": 0.0,
            "top_geography_name": "N/A",
            "top_geography_count": 0,
            "peak_month_name": "N/A",
            "peak_month_count": 0,
        }

    total_births = int(filtered_df["births"].sum())
    num_geos = int(filtered_df["state_of_residence"].nunique())
    num_months = max(1, int(filtered_df["month"].nunique()))
    avg_monthly_births = total_births / num_months

    # Top state by count
    geo_totals = (
        filtered_df.groupby("state_of_residence")["births"]
        .sum()
        .sort_values(ascending=False)
    )
    top_geo = geo_totals.index[0]
    top_geo_val = int(geo_totals.iloc[0])

    # Top month by count (respects categorical ordering)
    month_totals = (
        filtered_df.groupby("month", observed=False)["births"]
        .sum()
        .sort_values(ascending=False)
    )
    peak_month = str(month_totals.index[0])
    peak_month_val = int(month_totals.iloc[0])

    return {
        "total_births": total_births,
        "selected_geographies": num_geos,
        "avg_monthly_births": avg_monthly_births,
        "top_geography_name": top_geo,
        "top_geography_count": top_geo_val,
        "peak_month_name": peak_month,
        "peak_month_count": peak_month_val,
    }

# -----------------------------------------------------------------------------
# 5. VISUALIZATION GENERATORS
# -----------------------------------------------------------------------------

def plot_top_bottom_geographies(filtered_df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Horizontal bar chart comparing highest and lowest volume states."""
    geo_agg = (
        filtered_df.groupby("state_of_residence")["births"]
        .sum()
        .reset_index()
        .sort_values("births", ascending=True)
    )

    if len(geo_agg) <= top_n * 2:
        chart_data = geo_agg.copy()
        chart_data["Group"] = "Selected Entities"
    else:
        bottoms = geo_agg.head(top_n).copy()
        bottoms["Group"] = f"Bottom {top_n}"
        tops = geo_agg.tail(top_n).copy()
        tops["Group"] = f"Top {top_n}"
        chart_data = pd.concat([bottoms, tops])

    fig = px.bar(
        chart_data,
        x="births",
        y="state_of_residence",
        color="Group",
        orientation="h",
        labels={"births": "Total Births", "state_of_residence": "State / Geography"},
        title=f"Highest and Lowest Birth Volumes (Top & Bottom {top_n})",
        color_discrete_map={
            f"Top {top_n}": "#2b5c8f",
            f"Bottom {top_n}": "#d95f02",
            "Selected Entities": "#2b5c8f",
        },
    )
    fig.update_layout(
        xaxis=dict(rangemode="tozero", tickformat=","),
        yaxis=dict(categoryorder="total ascending"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
        legend_title_text="",
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Births: %{x:,.0f}<extra></extra>")
    return fig


def plot_macro_trendline(filtered_df: pd.DataFrame) -> go.Figure:
    """Aggregate monthly time-series line chart."""
    trend = (
        filtered_df.groupby("month", observed=False)["births"]
        .sum()
        .reset_index()
    )
    fig = px.line(
        trend,
        x="month",
        y="births",
        markers=True,
        labels={"month": "Month", "births": "Total Births"},
        title="Aggregate Monthly Birth Trend",
    )
    fig.update_traces(
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=8),
        hovertemplate="Month: <b>%{x}</b><br>Births: %{y:,.0f}<extra></extra>",
    )
    fig.update_layout(
        yaxis=dict(rangemode="tozero", tickformat=","),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def plot_choropleth_map(filtered_df: pd.DataFrame) -> go.Figure:
    """Interactive US Choropleth map with state abbreviations."""
    state_totals = (
        filtered_df.dropna(subset=["state_abbr"])
        .groupby(["state_of_residence", "state_abbr"])["births"]
        .sum()
        .reset_index()
    )
    fig = px.choropleth(
        state_totals,
        locations="state_abbr",
        locationmode="USA-states",
        color="births",
        scope="usa",
        color_continuous_scale="Blues",
        labels={"births": "Total Births"},
        hover_name="state_of_residence",
        title="Geographic Distribution of Provisional Births",
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b> (%{location})<br>Births: %{z:,.0f}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title="Births", tickformat=","),
    )
    return fig


def plot_state_rankings(filtered_df: pd.DataFrame) -> go.Figure:
    """Full ranked horizontal bar chart of selected states."""
    geo_totals = (
        filtered_df.groupby("state_of_residence")["births"]
        .sum()
        .reset_index()
        .sort_values("births", ascending=True)
    )
    fig = px.bar(
        geo_totals,
        x="births",
        y="state_of_residence",
        orientation="h",
        labels={"births": "Total Births", "state_of_residence": "State / Geography"},
        title="Total Births by State (Ranked)",
    )
    fig.update_traces(
        marker_color="#2b5c8f",
        hovertemplate="<b>%{y}</b><br>Births: %{x:,.0f}<extra></extra>",
    )
    height = max(450, len(geo_totals) * 18)
    fig.update_layout(
        height=height,
        xaxis=dict(rangemode="tozero", tickformat=","),
        yaxis=dict(categoryorder="total ascending"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def plot_monthly_sex_comparison(filtered_df: pd.DataFrame) -> go.Figure:
    """Side-by-side grouped bar chart comparing monthly births by infant sex."""
    trend_sex = (
        filtered_df.groupby(["month", "sex_of_infant"], observed=False)["births"]
        .sum()
        .reset_index()
    )
    fig = px.bar(
        trend_sex,
        x="month",
        y="births",
        color="sex_of_infant",
        barmode="group",
        labels={"month": "Month", "births": "Births", "sex_of_infant": "Infant Sex"},
        color_discrete_map=SEX_COLORS,
        title="Monthly Birth Comparison by Infant Sex",
    )
    fig.update_traces(
        hovertemplate="Month: <b>%{x}</b><br>Sex: %{fullData.name}<br>Births: %{y:,.0f}<extra></extra>"
    )
    fig.update_layout(
        yaxis=dict(rangemode="tozero", tickformat=","),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_state_month_heatmap(filtered_df: pd.DataFrame) -> go.Figure:
    """Seasonality cross-tabulation heatmap (State vs. Month)."""
    pivot = filtered_df.pivot_table(
        index="state_of_residence",
        columns="month",
        values="births",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="State / Geography", color="Births"),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Seasonality Heatmap: State vs. Month",
    )
    fig.update_traces(
        hovertemplate="State: <b>%{y}</b><br>Month: <b>%{x}</b><br>Births: %{z:,.0f}<extra></extra>"
    )
    height = max(500, len(pivot) * 16)
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=30),
        coloraxis_colorbar=dict(title="Births", tickformat=","),
    )
    return fig

# -----------------------------------------------------------------------------
# 6. AI DATA ASSISTANT (CHATBOT)
# -----------------------------------------------------------------------------

class AllModelsUnavailableError(Exception):
    """Raised when every configured model is retired / unknown / unavailable."""


def get_api_key() -> Optional[str]:
    """Read the LLM API key from Streamlit secrets (never hard-coded)."""
    for name in API_KEY_SECRET_NAMES:
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
        if value and str(value).strip():
            return str(value).strip()
    return None


def get_model_override() -> Optional[str]:
    """Optional LLM_MODEL secret that overrides the provider's default model."""
    try:
        value = st.secrets.get(MODEL_OVERRIDE_SECRET_NAME)
    except Exception:
        value = None
    return str(value).strip() if value and str(value).strip() else None


def detect_provider(api_key: str) -> Optional[Dict[str, Any]]:
    """Pick the provider config from the key prefix (gsk_ -> Groq, xai- -> xAI)."""
    for prefix, cfg in LLM_PROVIDERS.items():
        if api_key.startswith(prefix):
            return cfg
    return None


def is_model_unavailable_error(err: Exception) -> bool:
    """True only if the error says the model is retired/unknown/unavailable."""
    # Auth and rate-limit problems are never a model problem.
    if getattr(err, "status_code", None) in (401, 429):
        return False
    text = str(err).lower()
    return any(phrase in text for phrase in MODEL_UNAVAILABLE_PHRASES)


def get_candidate_models(provider: Dict[str, Any]) -> List[str]:
    """Primary model first, then fallbacks; the last model that worked goes first."""
    primary = get_model_override() or provider["default_model"]
    models = [primary] + [m for m in provider["fallback_models"] if m != primary]
    active = st.session_state.get("active_model")
    if active in models:
        models.remove(active)
        models.insert(0, active)
    return models


def build_data_summary(filtered_df: pd.DataFrame) -> str:
    """Compact text summary of the FILTERED data (the raw CSV is never sent)."""
    sex_filter = st.session_state.get("selected_sex", "All")
    n_geos = int(filtered_df["state_of_residence"].nunique())
    n_months = int(filtered_df["month"].nunique())
    total = int(filtered_df["births"].sum())

    lines = [
        "ACTIVE FILTERS",
        f"- Sex: {sex_filter}",
        f"- Geographies: {n_geos}",
        f"- Months: {n_months}",
        "",
        f"TOTAL BIRTHS: {total:,}",
        "",
        "BIRTHS BY SEX",
    ]

    by_sex = filtered_df.groupby("sex_of_infant")["births"].sum()
    for sex, val in by_sex.items():
        lines.append(f"- {sex}: {int(val):,}")

    lines += ["", "BIRTHS BY MONTH"]
    by_month = filtered_df.groupby("month", observed=True)["births"].sum()
    for month, val in by_month.items():
        lines.append(f"- {month}: {int(val):,}")

    lines += ["", "BIRTHS BY STATE (ranked high to low) - State | Total | Female | Male"]
    by_state = (
        filtered_df.pivot_table(
            index="state_of_residence",
            columns="sex_of_infant",
            values="births",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=["Female", "Male"], fill_value=0)
    )
    by_state["Total"] = by_state["Female"] + by_state["Male"]
    by_state = by_state.sort_values("Total", ascending=False)
    for state, row in by_state.iterrows():
        lines.append(
            f"- {state} | {int(row['Total']):,} | {int(row['Female']):,} | {int(row['Male']):,}"
        )

    return "\n".join(lines)


def create_chat_stream(client: Any, provider: Dict[str, Any], api_messages: List[Dict[str, str]]) -> Any:
    """Open a streaming completion, falling back ONLY on 'model unavailable' errors."""
    last_err: Optional[Exception] = None
    for model in get_candidate_models(provider):
        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=api_messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
        )
        if "gpt-oss" in model:
            # Sent via extra_body so it works with any openai>=1.40 SDK version.
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        try:
            stream = client.chat.completions.create(**kwargs)
        except Exception as err:
            if is_model_unavailable_error(err):
                last_err = err
                continue  # try the next model
            raise  # any other error is raised normally
        st.session_state["active_model"] = model
        return stream
    raise AllModelsUnavailableError(str(last_err)) from last_err


def iter_stream_text(stream: Any) -> Iterator[str]:
    """Yield only delta.content text, skipping chunks that have no choices."""
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece


def friendly_error_message(err: Exception) -> str:
    """Translate an exception into a plain-English message for the chat."""
    if isinstance(err, AllModelsUnavailableError):
        return (
            "None of the configured AI models are available. Add a LLM_MODEL secret "
            "in Streamlit Cloud (⋮ → Settings → Secrets) with a model name that is "
            "currently available for your account."
        )
    status = getattr(err, "status_code", None)
    text = str(err).lower()
    if status == 401 or "invalid api key" in text or "invalid_api_key" in text:
        return "The API key was rejected. Please check the key saved in your Streamlit Secrets."
    if status == 429 or "rate limit" in text or "rate_limit" in text:
        return "The free-tier rate limit was reached. Please wait a minute and try again."
    return f"Sorry, something went wrong contacting the AI service: {err}"


def clear_chat() -> None:
    st.session_state.messages = []


def render_chatbot(filtered_df: pd.DataFrame) -> None:
    """Chat tab: ask questions about the data matching the sidebar filters."""
    st.subheader("Ask the Data Assistant")

    api_key = get_api_key()
    if not api_key:
        st.warning(
            "No AI API key was found. To enable the assistant, add your key as "
            "`GROQ_API_KEY` in Streamlit Cloud: **⋮ → Settings → Secrets**, for example:\n\n"
            '`GROQ_API_KEY = "gsk_your_key_here"`'
        )
        return
    if OpenAI is None:
        st.warning("The `openai` package is not installed. Add `openai>=1.40.0` to requirements.txt.")
        return
    provider = detect_provider(api_key)
    if provider is None:
        st.warning(
            "The saved API key has an unrecognized prefix. Groq keys start with `gsk_` "
            "and xAI keys start with `xai-`."
        )
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    caption_slot = st.empty()

    def show_caption() -> None:
        model = st.session_state.get("active_model") or get_candidate_models(provider)[0]
        caption_slot.caption(
            f"Powered by {provider['name']} · model {model}. "
            "Answers are based on the data matching your current sidebar filters. "
            "AI can make mistakes — verify key numbers with the charts."
        )

    show_caption()

    # Suggested questions + clear button
    clicked_question: Optional[str] = None
    cols = st.columns(len(SUGGESTED_QUESTIONS) + 1)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        if cols[i].button(question, key=f"suggested_q_{i}", width="stretch"):
            clicked_question = question
    cols[-1].button("🗑️ Clear chat", key="clear_chat", on_click=clear_chat, width="stretch")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed_question = st.chat_input("Ask a question about the births data…")
    question = typed_question or clicked_question
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            system_content = SYSTEM_PROMPT + "\nDATA SUMMARY:\n" + build_data_summary(filtered_df)
            api_messages = [{"role": "system", "content": system_content}] + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-MAX_HISTORY_MESSAGES:]
            ]
            client = OpenAI(api_key=api_key, base_url=provider["base_url"], timeout=60.0, max_retries=1)
            stream = create_chat_stream(client, provider, api_messages)
            reply = st.write_stream(iter_stream_text(stream))
            reply = reply if isinstance(reply, str) else "".join(str(r) for r in reply)
            if not reply.strip():
                reply = "The model returned an empty answer. Please try rephrasing your question."
                st.markdown(reply)
        except Exception as err:
            reply = friendly_error_message(err)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    show_caption()

# -----------------------------------------------------------------------------
# 7. MAIN APPLICATION EXECUTION
# -----------------------------------------------------------------------------

def main():
    try:
        df_raw = load_and_preprocess_data()
    except Exception as exc:
        st.error(f"Error loading dataset: {exc}")
        st.stop()

    all_states = sorted(df_raw["state_of_residence"].unique().tolist())
    all_months = MONTH_ORDER
    sex_options = ["All", "Female", "Male"]

    # Filter State Callbacks
    if "selected_states" not in st.session_state:
        st.session_state.selected_states = all_states
    if "selected_months" not in st.session_state:
        st.session_state.selected_months = all_months
    if "selected_sex" not in st.session_state:
        st.session_state.selected_sex = "All"

    def reset_filters():
        st.session_state.selected_states = all_states
        st.session_state.selected_months = all_months
        st.session_state.selected_sex = "All"

    def select_all_states():
        st.session_state.selected_states = all_states

    def select_all_months():
        st.session_state.selected_months = all_months

    # Sidebar
    st.sidebar.header("Filter Controls")

    st.sidebar.selectbox("Infant Sex", options=sex_options, key="selected_sex")

    col_s_btn, _ = st.sidebar.columns([1, 1])
    with col_s_btn:
        st.button("Select All States", on_click=select_all_states, width="stretch")

    st.sidebar.multiselect(
        "State / Geography",
        options=all_states,
        key="selected_states",
        help="Select one or multiple geographies.",
    )

    col_m_btn, _ = st.sidebar.columns([1, 1])
    with col_m_btn:
        st.button("Select All Months", on_click=select_all_months, width="stretch")

    st.sidebar.multiselect(
        "Month (Chronological)",
        options=all_months,
        key="selected_months",
        help="Select calendar months.",
    )

    st.sidebar.markdown("---")
    st.sidebar.button("Reset All Filters", on_click=reset_filters, width="stretch")

    st.sidebar.markdown("### Active Filters Summary")
    st.sidebar.caption(f"• **Sex:** {st.session_state.selected_sex}")
    st.sidebar.caption(f"• **Geographies:** {len(st.session_state.selected_states)} of {len(all_states)} selected")
    st.sidebar.caption(f"• **Months:** {len(st.session_state.selected_months)} of {len(all_months)} selected")

    # Header & Context
    st.title("U.S. Provisional Natality Exploration Dashboard (2025)")
    st.markdown(
        "Designed for exploratory data analysis of geographic, monthly, and infant-sex patterns "
        "using CDC vital statistics."
    )

    st.info(
        "**Source & Methodology Notice:**\n\n"
        "- **Data Source:** Centers for Disease Control and Prevention (CDC) National Center for Health Statistics (NCHS).\n"
        "- **Provisional Status:** All counts shown are provisional and subject to reporting revisions and registration delays.\n"
        "- **Metric Definition:** Values represent raw **birth counts**, not birth or fertility rates. "
        "High volumes reflect both birth propensity and underlying state population size."
    )

    # Filter Application
    filtered_df = df_raw.copy()
    if st.session_state.selected_sex != "All":
        filtered_df = filtered_df[filtered_df["sex_of_infant"] == st.session_state.selected_sex]

    filtered_df = filtered_df[
        (filtered_df["state_of_residence"].isin(st.session_state.selected_states)) &
        (filtered_df["month"].isin(st.session_state.selected_months))
    ]

    if filtered_df.empty:
        st.warning("⚠️ No observations match your current filter selections. Please expand your filter criteria in the sidebar.")
        st.stop()

    # Dynamic KPI Cards
    kpis = compute_kpis(filtered_df)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    kpi_col1.metric("Total Births", f"{kpis['total_births']:,}")
    kpi_col2.metric("Selected Geographies", f"{kpis['selected_geographies']}")
    kpi_col3.metric("Avg Births / Month", f"{kpis['avg_monthly_births']:,.0f}")
    kpi_col4.metric("Top Geography", kpis["top_geography_name"], f"{kpis['top_geography_count']:,} births", delta_color="off")
    kpi_col5.metric("Peak Month", kpis["peak_month_name"], f"{kpis['peak_month_count']:,} births", delta_color="off")

    st.markdown("---")

    # Tabs
    tab_overview, tab_geo, tab_monthly_sex, tab_chat, tab_table, tab_about = st.tabs([
        "Overview",
        "Geographic Analysis",
        "Monthly & Sex Analysis",
        "🤖 Ask the Data (AI)",
        "Data Table & Download",
        "About the Data",
    ])

    with tab_overview:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(plot_top_bottom_geographies(filtered_df, top_n=5), width="stretch")
        with c2:
            st.plotly_chart(plot_macro_trendline(filtered_df), width="stretch")

    with tab_geo:
        st.subheader("Geographic Distribution")
        st.plotly_chart(plot_choropleth_map(filtered_df), width="stretch")
        st.markdown("#### State Volume Rankings")
        st.plotly_chart(plot_state_rankings(filtered_df), width="stretch")

    with tab_monthly_sex:
        st.subheader("Monthly Seasonality & Sex Breakdown")
        st.plotly_chart(plot_monthly_sex_comparison(filtered_df), width="stretch")
        st.markdown("#### Geographic Seasonality Matrix")
        st.plotly_chart(plot_state_month_heatmap(filtered_df), width="stretch")

    with tab_chat:
        render_chatbot(filtered_df)

    with tab_table:
        st.subheader("Searchable Filtered Records")
        display_df = filtered_df[[
            "state_of_residence", "month", "sex_of_infant", "births"
        ]].rename(columns={
            "state_of_residence": "State",
            "month": "Month",
            "sex_of_infant": "Infant Sex",
            "births": "Birth Count",
        })
        st.dataframe(
            display_df.style.format({"Birth Count": "{:,}"}),
            width="stretch",
            hide_index=True,
        )
        csv_buffer = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_buffer,
            file_name="filtered_provisional_natality_2025.csv",
            mime="text/csv",
        )

    with tab_about:
        st.subheader("Data Documentation & Analytics Guidance")
        st.markdown(
            """
            ### Background and Provenance
            This dataset originates from the **Centers for Disease Control and Prevention (CDC)** National Vital Statistics System (NVSS).
            The records document provisional monthly live birth counts categorized by maternal state of residence and infant sex for the year 2025.

            ### Critical Analytical Notes for Students
            1. **Counts vs. Rates:**
               * The figures presented are raw birth counts ($N$).
               * Larger values in states such as California, Texas, and Florida primarily reflect base population rather than higher birth rates.
               * To calculate standardized birth rates in deeper analytics exercises, join these counts with U.S. Census Bureau population estimates:
                 $$\\text{Crude Birth Rate} = \\frac{\\text{Total Births}}{\\text{Total Population}} \\times 1{,}000$$
            2. **Provisional Data Considerations:**
               * Provisional data files reflect ongoing vital record reporting.
               * Counts for the most recent reporting months are subject to upward revisions as late certificates are processed.
            3. **Sex Ratio at Birth:**
               * Across large demographic samples, the natural human secondary sex ratio at birth typically hovers around 105 male births per 100 female births (~51.2% male).
               * Students can test for statistical deviations from this ratio across states using chi-squared goodness-of-fit tests.
            """
        )

if __name__ == "__main__":
    main()
