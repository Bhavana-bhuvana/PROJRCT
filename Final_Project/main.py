# import streamlit as st
# from utils import load_data, show_sidebar, reset_session
# from model_suggestions import suggest_models
# from model_manager import run_selected_model

# st.set_page_config(page_title="ML Playground", layout="wide")
# st.title("Machine Learning Playground")

# # Session init
# if "uploaded_data" not in st.session_state:
#     st.session_state.uploaded_data = None
# if "active_models" not in st.session_state:
#     st.session_state.active_models = []
# if "model_results" not in st.session_state:
#     st.session_state.model_results = {}

# # Sidebar
# show_sidebar()

# # Upload data
# uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])
# if uploaded_file:
#     st.session_state.uploaded_data = load_data(uploaded_file)
#     st.success("File uploaded successfully!")
#     st.write(st.session_state.uploaded_data.head())

#     # Suggest and select models
#     suggestions = suggest_models(st.session_state.uploaded_data)
#     st.info("Suggested Models: " + ", ".join(suggestions))

#     st.session_state.active_models = st.multiselect(
#         "Select models to apply:",
#         ["Regression", "Classification", "Clustering"],
#         default=suggestions
#     )

#     # Run models
#     for model_name in st.session_state.active_models:
#         run_selected_model(model_name, st.session_state.uploaded_data)
# else:
#     st.stop()

# # Reset
# if st.button("Upload New File"):
#************************************************************************************
# import streamlit as st
# from model_suggestions import suggest_models
# from model_manager import run_selected_model
# from ui_manager import display_upload_ui, display_model_selection_ui
# # Streamlit Page Configuration
# st.set_page_config(page_title="ML Playground", layout="wide")
# st.title("Machine Learning Playground")
# st.caption("Upload a dataset and apply Regression, Classification, or Clustering models.")
# st.markdown("### Instructions:")
# st.markdown("- **Step 1:** Upload a dataset (CSV, Excel, or JSON).")
# st.markdown("- **Step 2:** Choose a model type (Regression, Classification, or Clustering).")
# st.markdown("- **Step 3:** Modify model parameters and observe results interactively.")

# # Initialize session state
# if "page" not in st.session_state:
#     st.session_state.page = "upload"
# if "selected_models" not in st.session_state:
#     st.session_state.selected_models = []

# if "model_results" not in st.session_state:
#     st.session_state.model_results = {}

# # Sidebar Navigation
# with st.sidebar:
#     st.title("Model Selection")

#     if st.session_state.page == "upload":
#         st.info("Upload dataset to begin")
#     elif st.session_state.page == "model_selection":
#         st.info("Choose a model to apply or explore dataset")
#     else:
#         for model in st.session_state.selected_models:
#             if st.button(f"🔍 {model}"):
#                 st.session_state.page = model
#                 st.rerun()
#         if st.button(" Upload New Data"):
#             st.session_state.page = "upload"
#             st.session_state.selected_models = []
#             st.session_state.model_results = {}
#             st.rerun()

# # Main Page Routing
# if st.session_state.page == "upload":
#     display_upload_ui()
# elif st.session_state.page == "select_model":
#     st.header(" Model Configuration")
#     data = st.session_state.data
#     st.dataframe(data, use_container_width=True)
#     st.markdown("Review your dataset above before selecting models.")

#     if st.button("Continue to Model Selection"):
#         st.session_state.page = "model_selection"
#         st.rerun()
# elif st.session_state.page == "model_selection":
#     display_model_selection_ui()

# elif st.session_state.page in st.session_state.selected_models:
#     run_selected_model(st.session_state.page, st.session_state.data)
# else:
#     st.error("Invalid page state.")
#*****************************************************************************************************
import streamlit as st
from model_suggestions import suggest_models
from model_manager import run_selected_model
from ui_manager import display_upload_ui, display_model_selection_ui

# Page Config
st.set_page_config(page_title="ML Playground", layout="wide")
st.title("Machine Learning Playground")
st.caption("Upload a dataset and apply Regression, Classification, or Clustering models.")

# st.markdown("### Instructions:")
# st.markdown("- **Step 1:** Upload a dataset (CSV, Excel, or JSON).")
# st.markdown("- **Step 2:** Review the dataset.")
# st.markdown("- **Step 3:** Choose models and explore results interactively.")
if "data" not in st.session_state:
    st.markdown("**Step 1:** Upload a dataset to get started.")
elif "selected_model" not in st.session_state:
    st.markdown("**Step 2:** Select a model from the sidebar.")
elif st.session_state["selected_model"] == "Regression":
    st.markdown("**Regression Instructions:** Select a target and feature columns...")
elif st.session_state["selected_model"] == "Clustering":
    st.markdown("**Clustering Instructions:** Select at least two feature columns...")

# Session State Init
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []
if "model_results" not in st.session_state:
    st.session_state.model_results = {}

# Sidebar Navigation
# with st.sidebar:
#     st.title("Model Navigation")
#     if st.session_state.page == "upload":
#         st.info(" Upload dataset to begin.")
#     elif st.session_state.page == "select_model":
#         st.info(" Preview the dataset.")
#     elif st.session_state.page == "model_selection":
#         st.info(" Select models to apply.")
#     else:
#         for model in st.session_state.selected_models:
#             if st.button(f" {model}"):
#                 st.session_state.page = model
#                 st.rerun()
#         if st.button(" Upload New Data"):
#             st.session_state.page = "upload"
#             st.session_state.selected_models = []
#             st.session_state.model_results = {}
#             st.rerun()
with st.sidebar:
    st.title("Model Navigation")
    # Persistent Dataset Preview 
    if "data" in st.session_state:
        with st.expander("📄 Preview Dataset"):
            st.dataframe(st.session_state.data.head(), use_container_width=True)
    # Show selected models with remove option
    if st.session_state.selected_models:
        st.subheader("Selected Models:")
        for model in st.session_state.selected_models:
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f" {model}", key=f"goto_{model}"):
                    st.session_state.page = model
                    st.rerun()
            with col2:
                if st.button("❌", key=f"remove_{model}"):
                    st.session_state.selected_models.remove(model)
                    if st.session_state.page == model:
                        st.session_state.page = "model_selection"
                    st.rerun()

    # Add new model selection dropdown
    if "data" in st.session_state:
        available_models = ["Regression", "Classification", "Clustering"]
        remaining_models = list(set(available_models) - set(st.session_state.selected_models))
        if remaining_models:
            #new_model = st.selectbox(" Add  models", [""] + remaining_models, key="add_model_select")
            new_model=st.selectbox("Add another model", options=remaining_models, index=0)
            if new_model and st.button(" Add Model"):
                st.session_state.selected_models.append(new_model)
                st.session_state.page = new_model
                st.rerun()
        else:
            st.info("All models added.")

    st.markdown("---")
    if st.button(" Upload New Data"):
        st.session_state.page = "upload"
        st.session_state.selected_models = []
        st.session_state.model_results = {}
        st.rerun()


# Page Routing
if st.session_state.page == "upload":
    display_upload_ui()
elif st.session_state.page == "model_selection":
    display_model_selection_ui()
elif st.session_state.page in st.session_state.selected_models:
    run_selected_model(st.session_state.page, st.session_state.data)
else:
    st.error("Invalid page state.")
