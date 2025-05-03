# # ui_manager.py
# import streamlit as st
# import pandas as pd

# def display_upload_ui():
#     st.title(" Upload Your Dataset")
#     uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])
#     if uploaded_file is not None:
#         file_type = uploaded_file.name.split(".")[-1]
#         try:
#             if file_type == "csv":
#                 data = pd.read_csv(uploaded_file)
#             elif file_type in ["xlsx", "xls"]:
#                 data = pd.read_excel(uploaded_file)
#             elif file_type == "json":
#                 data = pd.read_json(uploaded_file)
#             else:
#                 st.error("Unsupported file format!")
#                 st.stop()
#         except Exception as e:
#             st.error(f"Error reading file: {e}")
#             st.stop()
#         st.session_state.uploaded_data = data
#         st.success(" Dataset uploaded successfully!{uploaded_file.name}")
#         st.write(data.head())
#         st.session_state.data = data
#         st.session_state.page = "select_model"
#         st.rerun()
    
#     else:
#         st.warning("Please upload file to proceed.")
    
# #Model Selection
# def display_model_selection_ui():
#     st.subheader(" Select Models to Apply")
#     models = ["Regression", "Classification", "Clustering"]
#     selected = st.multiselect("Choose models to run:", models)
#     if st.button(" Confirm Selection") and selected:
#         st.session_state.selected_models = selected
#         st.session_state.page = selected[0]  # Navigate to first selected model
#         st.rerun()
# ui_manager.py
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
            st.success(f" {uploaded_file.name} uploaded successfully!")

            # Immediately route to model selection page
            st.session_state.page = "model_selection"
            st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")


# def display_model_preview_ui():
#     st.header(" Dataset Preview")
#     data = st.session_state.get("data")
#     if data is not None:
#         st.dataframe(data, use_container_width=True)
#         st.markdown("Review your dataset above before proceeding.")
#         if st.button(" Continue to Model Selection"):
#             st.session_state.page = "model_selection"
#             st.rerun()
    # else:
    #     st.warning("No dataset found. Please upload a dataset first.")
    #     st.session_state.page = "upload"
    #     st.rerun()

def display_model_selection_ui():
    st.subheader(" Select Models to Apply")
    models = ["Regression", "Classification", "Clustering"]
    selected = st.multiselect("Choose models to run:", models)

    if st.button(" Confirm Selection") and selected:
        st.session_state.selected_models = selected
        st.session_state.page = selected[0]  # Go to first selected model
        st.rerun()
