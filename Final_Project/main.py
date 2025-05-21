import streamlit as st
from model_manager import run_selected_model
from ui_manager import display_upload_ui, display_cleaning_ui
from db import log_user_action,users_col,logs_col # MongoDB logger
from login_signup import login_signup_ui
# from history import show_history_page
from datetime import datetime, timedelta, timezone
from db import logs_col  # Assuming logs_col is your MongoDB logs collection
from PIL import Image
import base64
from io import BytesIO

# Ensure user is logged in before using the app
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login_signup_ui()
    st.stop()

# Page Config
st.set_page_config(page_title="ML Playground", layout="wide")

# Header with controls
col1, col2, col3, col4, col5 = st.columns([5, 1, 1, 1, 1])
with col2:
    if st.button("History"):
        st.session_state["page"] = "history"
with col3:
    if st.button("My Profile"):
        st.session_state["page"] = "profile"
with col4:
    if st.button("Reset Session"):
        preserved = {
            "logged_in": st.session_state.logged_in,
            "user_email": st.session_state.get("user_email"),
            "user_name": st.session_state.get("user_name"),
        }
        st.session_state.clear()
        st.session_state.update(preserved)
        st.success("Session reset successfully.")
        st.rerun()
with col5:
    if st.button(" Logout"):
        st.session_state.clear()
        st.rerun()
# Session State Init
if "page" not in st.session_state:
    st.session_state.page = "upload"
if st.session_state.page=="profile":
    st.markdown("---")
    st.subheader("👤 My Profile")

    # Get logged in user's email or username
    user_email = st.session_state.get("user_email", None)

    if not user_email:
        st.error("User email not found in session.")
        st.stop()

    # Fetch user info from MongoDB
    user_data = users_col.find_one({"email": user_email})

    if not user_data:
        st.warning("User profile not found.")
    else:
        st.markdown(f"*Full Name:* {user_data.get('full_name', 'N/A')}")
        st.markdown(f"*Username:* {user_data.get('username', 'N/A')}")
        st.markdown(f"*Email:* {user_data.get('email', 'N/A')}")

    st.markdown("---")
    st.button("🔙 Back to Home", on_click=lambda: st.session_state.update({"page": "upload"}))
    st.stop()
#Function to decode and display base64 image
def display_base64_image(base64_str, caption="Image"):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    st.image(image, caption=caption)


if st.session_state.page == "history":
    button=st.button("🔙 Back to Home", on_click=lambda: st.session_state.update({"page": "upload"}))
    if(button):
        st.stop()
    st.markdown("---")
    st.subheader("📜 My Activity History")
    user_email = st.session_state.get("user_email", None)
    if not user_email:
        st.error("User email not found in session.")
        st.stop()

    # Step 1: Fetch the user's log document
    user_log = logs_col.find_one({"email": user_email})

    if not user_log or "actions" not in user_log:
        st.info("No activity history found.")
    else:
        # Step 2: Extract and sort actions by timestamp descending
        sorted_actions = sorted(user_log["actions"], key=lambda x: x.get("timestamp", ""), reverse=True)
        IST = timezone(timedelta(hours=5, minutes=30))
        # Step 3: Display each action
        for entry in sorted_actions:
            action = entry.get("action", "Unknown Action")
            timestamp_utc = entry.get("timestamp", "No Time Recorded")
            details = entry.get("details", {})

            # Convert timestamp to IST format if valid
            try:
                if isinstance(timestamp_utc, str):
                # Convert string to datetime, assuming no timezone info (i.e., naive UTC)
                    utc_dt = datetime.strptime(timestamp_utc, "%Y-%m-%d %H:%M:%S.%f")
                    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
                elif isinstance(timestamp_utc, datetime):
                # Already a datetime object — just set UTC if naive
                    utc_dt = timestamp_utc if timestamp_utc.tzinfo else timestamp_utc.replace(tzinfo=timezone.utc)
                else:
                    raise ValueError("Unrecognized timestamp format")

                # Convert to IST and format
                ist_dt = utc_dt.astimezone(IST)
                formatted_time = ist_dt.strftime("%d/%m/%Y %H:%M:%S")
            except Exception as e:
                formatted_time = f"Invalid timestamp: {timestamp_utc} ({e})"

            with st.container():
                st.markdown(f"**🕒 {formatted_time}**")
                st.markdown(f"**Action:** {action}")
                if details:
                    st.markdown("**Details:**")
                    one = "regression_plot"
                    two = "correlation_heatmap"

                    #Fetch the actual base64 values using the keys stored in one and two
                    one_value = details.get(one, "")
                    two_value = details.get(two, "")
                    display_base64_image(one_value, caption="Regression Plot")
                    display_base64_image(two_value, caption="Correlation Heatmap")
                    for key, val in details.items():
                        # if key=="regression_plot" or key=="correlation_heatmap":
                        #     continue
                        st.markdown(f"- {key.capitalize()}: {val}")
                st.markdown("---")
   

st.caption("Upload a dataset and apply Regression, Classification, or Clustering models.")

# Session State Init
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []
if "model_results" not in st.session_state:
    st.session_state.model_results = {}

# Sidebar Navigation
page = st.session_state.page

st.sidebar.title("Navigation")

if st.sidebar.button("Upload Data"):
    st.session_state.page = "upload"
    st.rerun()

if st.sidebar.button("Data Cleaning"):
    st.session_state.page = "Data Cleaning"
    st.rerun()

if "raw_data" in st.session_state:
    with st.sidebar.expander("Original Dataset"):
        st.dataframe(st.session_state.raw_data.head(), use_container_width=True)

if "cleaned_data" in st.session_state:
    with st.sidebar.expander("Cleaned Dataset"):
        st.dataframe(st.session_state.cleaned_data.head(), use_container_width=True)

if st.session_state.selected_models:
    st.sidebar.subheader("Selected Models:")
    for model in st.session_state.selected_models:
        col1, col2 = st.sidebar.columns([5, 1])
        with col1:
            if st.button(f"{model}", key=f"goto_{model}"):
                log_user_action(st.session_state.get("user_email", "anonymous"), f"Switched to model: {model}")
                st.session_state.page = model
                st.rerun()
        with col2:
            if st.button("❌", key=f"remove_{model}"):
                st.session_state.selected_models.remove(model)
                log_user_action(st.session_state.get("user_email", "anonymous"), f"Removed model: {model}")
                if st.session_state.page == model:
                    st.session_state.page = "model_selection"
                st.rerun()

# Only show model options after data is uploaded
submodels = {
    "Regression": [
        "Linear Regression", "Polynomial Regression", "Multiple Linear Regression",
        "Decision Tree Regression", "Random Forest Regression", "Support Vector Regression"
    ],
    "Classification": [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "SVM", "KNN", "Naive Bayes"
    ],
    "Clustering": [
        "K-Means", "DBSCAN", "Gaussian Mixture Model", "Hierarchical"
    ]
}

st.sidebar.subheader("Add Models")
main_models = ["Regression", "Classification", "Clustering"]
selected_main_model = st.sidebar.selectbox("Choose Model Type", options=main_models)

# Filter already added submodels
added_submodels = {m.split(": ")[1] for m in st.session_state.selected_models if m.startswith(selected_main_model)}
available_submodels = [s for s in submodels[selected_main_model] if s not in added_submodels]

if available_submodels:
    selected_submodels = st.sidebar.multiselect(
        f"Select {selected_main_model} algorithms",
        options=available_submodels,
        key=f"select_{selected_main_model}"
    )

    if selected_submodels and st.sidebar.button("Add Model(s)"):
        for sub in selected_submodels:
            model_key = f"{selected_main_model}: {sub}"
            if model_key not in st.session_state.selected_models:
                st.session_state.selected_models.append(model_key)
                log_user_action(st.session_state.get("user_email", "anonymous"), f"Added model: {model_key}")
        st.session_state.page = f"{selected_main_model}: {selected_submodels[0]}"
        st.rerun()
else:
    st.sidebar.info("All models added.")

st.sidebar.markdown("---")


# Main Area Routing
if st.session_state.page == "upload":
    display_upload_ui()
elif st.session_state.page == "Data Cleaning":
    display_cleaning_ui()
elif st.session_state.page in st.session_state.selected_models:
    run_selected_model(st.session_state.page, st.session_state.get("cleaned_data", st.session_state.get("raw_data")))
elif st.session_state.page == "Results":
    if st.session_state.model_results:
        st.success(" Model Results Displayed Below:")
        for model_name, results in st.session_state.model_results.items():
            st.subheader(f"Model: {model_name}")
            st.write(results)
    else:
        st.info("No model results available yet.")
