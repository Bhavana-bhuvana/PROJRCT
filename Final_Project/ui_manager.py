import streamlit as st
import pandas as pd
from preprocessing.data_cleaner import DataCleaner
from preprocessing.data_cleaner import DataCleaner  # Assuming your cleaner class is here
import streamlit as st
import pandas as pd
def display_upload_ui():
    st.title(" Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])
    if uploaded_file:
        try:
            file_type = uploaded_file.name.split(".")[-1]
            if file_type == "csv":
                data = pd.read_csv(uploaded_file)
            elif file_type in ["xlsx", "xls"]:
                data = pd.read_excel(uploaded_file)
            elif file_type == "json":
                data = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format!")
                return

            st.session_state.data = data
            st.success(f"{uploaded_file.name} uploaded successfully!")

            # Run cleaner immediately
            cleaner = DataCleaner(data)
            cleaned_data, histograms, heatmap, report = cleaner.run_pipeline()

            st.session_state.cleaned_data = cleaned_data
            st.session_state.histograms = histograms
            st.session_state.correlation_heatmap = heatmap
            st.session_state.cleaning_report = report
            st.session_state.page = "Data Cleaning"

        except Exception as e:
            st.error(f"Error reading file: {e}")

def display_cleaning_ui():
    st.title(" Data Cleaning & Exploration")

    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("Please upload a dataset first from the Upload section.")
        return

    if "cleaned_data" not in st.session_state:
        cleaner = DataCleaner(st.session_state.data)
        cleaned_data, histograms, heatmap, report = cleaner.run_pipeline()
        st.session_state.cleaned_data = cleaned_data
        st.session_state.histograms = histograms
        st.session_state.correlation_heatmap = heatmap
        st.session_state.cleaning_report = report

    st.subheader("Cleaned Dataset")
    st.dataframe(st.session_state.cleaned_data.head(), use_container_width=True)
    csv = st.session_state.cleaned_data.to_csv(index=False).encode("utf-8")
    st.download_button(" Download Cleaned Dataset", csv, "cleaned_dataset.csv", "text/csv")

    if st.session_state.histograms:
        st.subheader("Histograms")
        for col, hist_buf in st.session_state.histograms.items():
            st.image(hist_buf, caption=f"Histogram of {col}")

    if st.session_state.correlation_heatmap:
        st.subheader(" Correlation Heatmap")
        st.image(st.session_state.correlation_heatmap, caption="Correlation Heatmap")

    if st.session_state.cleaning_report:
        st.subheader(" Cleaning Report Summary")
        st.code(st.session_state.cleaning_report)
        st.download_button(" Download Report", st.session_state.cleaning_report, file_name="cleaning_report.txt")

    if st.button(" Re-run Cleaning"):
        cleaner = DataCleaner(st.session_state.data)
        cleaned_data, histograms, heatmap, report = cleaner.run_pipeline()
        st.session_state.cleaned_data = cleaned_data
        st.session_state.histograms = histograms
        st.session_state.correlation_heatmap = heatmap
        st.session_state.cleaning_report = report
        st.success("Cleaning re-run successfully!")
        st.rerun()

def display_model_selection_ui():
    st.subheader(" Select Models to Apply")
    models = ["Regression", "Classification", "Clustering"]
    selected = st.multiselect("Choose models to run:", models)

    if st.button(" Confirm Selection") and selected:
        st.session_state.selected_models = selected
        st.session_state.page = selected[0]  # Go to first selected model
        st.rerun()
