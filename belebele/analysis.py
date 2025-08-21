import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st


def load_results():
    """Load all JSON result files from the results directory."""
    results_dir = "results"
    data = []

    # Define the model mapping from the evaluation script
    MODELS = {
        "mistralai/Mistral-7B-v0.1": {},
        "mistralai/Mistral-7B-Instruct-v0.3": {},
        "utter-project/EuroLLM-9B": {},
        "utter-project/EuroLLM-1.7B": {},
        "google/gemma-3-1b-pt": {},
        "google/gemma-3-1b-it": {},
        "google/gemma-3-4b-pt": {},
        "google/gemma-3-4b-it": {},
        "google/gemma-3-12b-pt": {},
        "google/gemma-3-12b-it": {},
        "hplt-monolingual": {
            "deu_Latn": "HPLT/hplt2c_deu_checkpoints",
            "fra_Latn": "HPLT/hplt2c_fra_checkpoints",
            "spa_Latn": "HPLT/hplt2c_spa_checkpoints",
            "ita_Latn": "HPLT/hplt2c_ita_checkpoints",
            "pol_Latn": "HPLT/hplt2c_pol_checkpoints",
            "por_Latn": "HPLT/hplt2c_por_checkpoints",
            "eng_Latn": "HPLT/hplt2c_eng_checkpoints",
            "est_Latn": "HPLT/hplt2c_est_checkpoints",
        },
        "allenai/OLMo-2-1124-13B-Instruct": {},
        "allenai/OLMo-2-1124-13B": {},
        "allenai/OLMo-2-1124-7B-Instruct": {},
        "allenai/OLMo-2-1124-7B": {},
        "HuggingFaceTB/SmolLM3-3B-Base": {},
        "HuggingFaceTB/SmolLM3-3B": {},
    }

    LANGUAGES = [
        "eng_Latn",
        "deu_Latn",
        "fra_Latn",
        "spa_Latn",
        "ita_Latn",
        "pol_Latn",
        "por_Latn",
        "est_Latn",
    ]

    if not os.path.exists(results_dir):
        st.error(f"Results directory '{results_dir}' not found!")
        return pd.DataFrame()

    # Create a mapping from filename patterns to model keys
    filename_to_model = {}

    for model_key, language_variants in MODELS.items():
        for language in LANGUAGES:
            if language_variants and language in language_variants:
                actual_model_name = language_variants[language]
            elif not language_variants:
                actual_model_name = model_key
            else:
                continue

            # Create expected filename pattern
            filename_pattern = (
                f"belebe-{actual_model_name.split('/')[-1]}_{language}.json"
            )
            filename_to_model[filename_pattern] = model_key

    # Load all JSON files
    for filename in os.listdir(results_dir):
        if filename.endswith(".json") and filename.startswith("belebe-"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    result = json.load(f)

                # Find the corresponding model key
                model_key = filename_to_model.get(filename)
                if model_key:
                    # Extract language from filename - should be the last two parts joined
                    # Format: belebe-{model}_{lang_code}_Latn.json
                    parts = (
                        filename.replace("belebe-", "").replace(".json", "").split("_")
                    )
                    if len(parts) >= 2:
                        # Take the last two parts to get full language code like "spa_Latn"
                        language = "_".join(parts[-2:])
                    else:
                        language = "unknown"

                    data.append(
                        {
                            "model": model_key,
                            "language": language,
                            "correct_percent": result.get("correct_percent", 0),
                            "total_questions": result.get("total", 0),
                            "correct_answers": result.get("correct", 0),
                        }
                    )
                else:
                    st.warning(f"Could not map {filename} to a model key")

            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")

    return pd.DataFrame(data)


def create_box_plot(df):
    """Create a box plot showing correctness distribution for each LLM across languages."""
    if df.empty:
        st.error("No data available for visualization.")
        return None

    # Sort models by median performance (lowest first)
    model_medians = df.groupby("model")["correct_percent"].median().sort_values()
    model_order = model_medians.index.tolist()

    # Create box plot using plotly - one box per model
    fig = px.box(
        df,
        x="model",
        y="correct_percent",
        title="LLM Performance Distribution Across Languages",
        labels={"correct_percent": "Correct Percentage (%)", "model": "Language Model"},
        points=False,  # Remove individual points for cleaner look like the image
        category_orders={"model": model_order},  # Sort by median performance
    )

    fig.update_layout(
        height=600, xaxis={"tickangle": 45}, plot_bgcolor="white", paper_bgcolor="white"
    )

    # Style the boxes to match the reference image
    fig.update_traces(
        marker=dict(color="lightblue", line=dict(color="black", width=1)),
        line=dict(color="black"),
    )

    return fig


def main():
    st.set_page_config(page_title="LLM Evaluation Analysis", layout="wide")

    st.title("LLM Evaluation Analysis")
    st.markdown(
        "Analysis of language model performance on the Belebele dataset across different languages."
    )

    # Load data
    df = load_results()

    if df.empty:
        st.error(
            "No evaluation results found. Make sure the results directory contains JSON files."
        )
        return

    # Display summary statistics
    st.header("Summary Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Models", len(df["model"].unique()))
    with col2:
        st.metric("Languages Tested", len(df["language"].unique()))
    with col3:
        st.metric("Total Evaluations", len(df))

    # Box plot visualization
    st.header("Performance Distribution by Model")
    box_fig = create_box_plot(df)
    if box_fig:
        st.plotly_chart(box_fig, use_container_width=True)

    # Data table
    st.header("Detailed Results")

    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect(
            "Filter by Model:",
            options=df["model"].unique(),
            default=df["model"].unique(),
        )
    with col2:
        selected_languages = st.multiselect(
            "Filter by Language:",
            options=df["language"].unique(),
            default=df["language"].unique(),
        )

    # Filter data
    filtered_df = df[
        (df["model"].isin(selected_models)) & (df["language"].isin(selected_languages))
    ]

    # Display filtered data
    st.dataframe(
        filtered_df.sort_values("correct_percent", ascending=False),
        use_container_width=True,
    )

    # Model performance summary
    st.header("Model Performance Summary")
    model_summary = (
        df.groupby("model")["correct_percent"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(2)
    )
    model_summary.columns = ["Mean %", "Std Dev", "Min %", "Max %", "Languages"]
    model_summary_sorted = model_summary.sort_values("Mean %", ascending=False)
    
    st.dataframe(model_summary_sorted)
    
    # Add download button for HTML table
    html_table = model_summary_sorted.to_html(classes='model-summary-table', table_id='model-summary')
    
    st.download_button(
        label="Download Table as HTML",
        data=html_table,
        file_name="model_performance_summary.html",
        mime="text/html"
    )


main()
