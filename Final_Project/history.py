# import streamlit as st
# from datetime import datetime, timedelta, timezone
# from db import logs_col  # Assuming logs_col is your MongoDB logs collection
# from PIL import Image
# import base64
# from io import BytesIO

# def display_base64_image(base64_str, caption="Image"):
#     image_data = base64.b64decode(base64_str)
#     image = Image.open(BytesIO(image_data))
#     st.image(image, caption=caption)
# def show_history_page():
#     st.markdown("---")
#     st.subheader("My Activity History")

#     user_email = st.session_state.get("user_email", None)
#     if not user_email:
#         st.error("User email not found in session.")
#         st.stop()

#     user_log = logs_col.find_one({"email": user_email})
#     st.write("DEBUG: user_log", user_log)

#     if not user_log or "actions" not in user_log:
#         st.info("No activity history found.")
#     else:
#         sorted_actions = sorted(user_log["actions"], key=lambda x: x.get("timestamp", ""), reverse=True)
#         IST = timezone(timedelta(hours=5, minutes=30))

#         for entry in sorted_actions:
#             action = entry.get("action", "Unknown Action")
#             timestamp_utc = entry.get("timestamp", "No Time Recorded")
#             details = entry.get("details", {})

#             try:
#                 if isinstance(timestamp_utc, str):
#                     utc_dt = datetime.strptime(timestamp_utc, "%Y-%m-%d %H:%M:%S.%f")
#                     utc_dt = utc_dt.replace(tzinfo=timezone.utc)
#                 elif isinstance(timestamp_utc, datetime):
#                     utc_dt = timestamp_utc if timestamp_utc.tzinfo else timestamp_utc.replace(tzinfo=timezone.utc)
#                 else:
#                     raise ValueError("Unrecognized timestamp format")

#                 ist_dt = utc_dt.astimezone(IST)
#                 formatted_time = ist_dt.strftime("%d/%m/%Y %H:%M:%S")
#             except Exception as e:
#                 formatted_time = f"Invalid timestamp: {timestamp_utc} ({e})"

#             with st.container():
#                 st.markdown(f"**{formatted_time}**")
#                 st.markdown(f"**Action:** {action}")
#                 if details:
#                     st.markdown("**Details:**")
#                     for key, val in details.items():
#                         if key in ["regression_plot", "correlation_heatmap"]:
#                             try:
#                                 display_base64_image(val, caption=key.replace("_", " ").title())
#                             except Exception as e:
#                                 st.warning(f"Image display failed for {key}: {e}")
#                         else:
#                             st.markdown(f"- {key.capitalize()}: {val}")

#     st.markdown("---")

#     if st.button("🔙 Back to Home", key="back_to_home_button"):
#         st.session_state.page = "upload"
#         st.rerun()

# # import streamlit as st
# # from PIL import Image
# # import base64
# # from io import BytesIO

# # # Display base64-encoded image helper
# # def display_base64_image(base64_string, caption=None):
# #     try:
# #         img_data = base64.b64decode(base64_string)
# #         image = Image.open(BytesIO(img_data))
# #         st.image(image, caption=caption)
# #     except Exception as e:
# #         st.error(f"Failed to load image: {e}")

# # # Simulated MongoDB document (replace this with actual DB call)

# # # Display
# # st.title("📂 Session History Viewer")

# # # Loop through sessions
# # for i, session in enumerate(data.get("sessions", []), 1):
# #     with st.expander(f"Session {i} - {session.get('model_name')}"):
# #         st.markdown(f"**Model Type:** {session.get('model_type')}")
# #         st.markdown(f"**Model Name:** {session.get('model_name')}")

# #         session_data = session.get("session_data", {})
# #         st.markdown(f"**Timestamp:** `{session_data.get('timestamp')}`")
# #         st.markdown(f"**Target:** `{session_data.get('target')}`")
# #         st.markdown(f"**Features:** `{session_data.get('features')}`")
# #         st.markdown(f"**Train-Test Split:** `{session_data.get('train_test_split')}`")

# #         st.subheader("📈 Prediction History")
# #         history_list = session_data.get("prediction_history", [])
# #         for j, history in enumerate(history_list, 1):
# #             with st.expander(f"Prediction {j}"):
# #                 for key, val in history.items():
# #                     # Auto-detect and display images
# #                     if "base64" in key:
# #                         display_base64_image(val, caption=key.replace("_", " ").title())
# #                     else:
# #                         st.markdown(f"- **{key.replace('_', ' ').title()}**: `{val}`")
import streamlit as st
from datetime import datetime, timedelta, timezone
from db import logs_col  # Your MongoDB collection
from PIL import Image
import base64
from io import BytesIO

# ✅ Updated display function
def display_base64_image(base64_str, caption=None):
    try:
        # Clean and fix padding
        base64_str = base64_str.replace("\n", "").strip()
        missing_padding = len(base64_str) % 4
        if missing_padding:
            base64_str += "=" * (4 - missing_padding)

        # Decode and show image
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        st.image(image, caption=caption, use_column_width=True)
    except Exception as e:
        st.warning(f"Could not display image: {e}")

# ✅ Main history display function
def show_history_page():
    st.markdown("---")
    st.subheader(" My Activity History")

    # Get user session email
    user_email = st.session_state.get("user_email", None)
    if not user_email:
        st.error(" User email not found in session.")
        st.stop()

    # Fetch logs from MongoDB
    user_log = logs_col.find_one({"email": user_email})
    st.write("🔍 DEBUG: user_log", user_log)

    if not user_log or "actions" not in user_log:
        st.info("ℹ No activity history found.")
    else:
        sorted_actions = sorted(user_log["actions"], key=lambda x: x.get("timestamp", ""), reverse=True)
        IST = timezone(timedelta(hours=5, minutes=30))

        for entry in sorted_actions:
            action = entry.get("action", "Unknown Action")
            timestamp_utc = entry.get("timestamp", "No Time Recorded")
            details = entry.get("details", {})

            # Format timestamp to IST
            try:
                if isinstance(timestamp_utc, str):
                    utc_dt = datetime.strptime(timestamp_utc, "%Y-%m-%d %H:%M:%S.%f")
                    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
                elif isinstance(timestamp_utc, datetime):
                    utc_dt = timestamp_utc if timestamp_utc.tzinfo else timestamp_utc.replace(tzinfo=timezone.utc)
                else:
                    raise ValueError("Unrecognized timestamp format")
                ist_dt = utc_dt.astimezone(IST)
                formatted_time = ist_dt.strftime("%d/%m/%Y %H:%M:%S")
            except Exception as e:
                formatted_time = f"Invalid timestamp: {timestamp_utc} ({e})"

            # 🖼️ Display section
            with st.container():
                st.markdown(f"** Time:** {formatted_time}")
                st.markdown(f"** Action:** {action}")
                if details:
                    st.markdown("** Details:**")
                    for key, val in details.items():
                        if key in ["regression_plot", "correlation_heatmap"]:
                            st.markdown(f"- {key.replace('_', ' ').title()}:")
                            if isinstance(val, str):
                                st.code(val[:100] + "...", language="text")  # Debug snippet
                                display_base64_image(val, caption=key.replace("_", " ").title())
                            else:
                                st.warning(f"{key} is not a base64 string (type: {type(val)})")
                        else:
                            st.markdown(f"- {key.capitalize()}: {val}")

    st.markdown("---")

    if st.button("🔙 Back to Home", key="back_to_home_button"):
        st.session_state.page = "upload"
        st.rerun()

