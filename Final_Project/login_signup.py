import streamlit as st
import re
from auth import login_user, signup_user, send_otp, verify_otp, reset_password,mark_user_verified
from utils import send_otp_email
from db import pending_users_collection
def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def validate_password_strength(password):
    return (
        len(password) >= 8 and
        any(c.islower() for c in password) and
        any(c.isupper() for c in password) and
        any(c.isdigit() for c in password) and
        any(c in "!@#$%^&*()_+-=" for c in password)
    )

def login_signup_ui():
    st.set_page_config(page_title="Login | ML Playground")
    st.title("🔐 Login or Signup")

    auth_mode = st.radio("Choose Action", ["Login", "Signup"], horizontal=True)

    if auth_mode == "Signup":
        st.subheader("Create an Account")

        # Initialize session flags
        if "otp_sent" not in st.session_state:
            st.session_state.otp_sent = False
        if "otp_verified" not in st.session_state:
            st.session_state.otp_verified = False

        full_name = st.text_input("Full Name")
        username = st.text_input("Username")
        email = st.text_input("Email")

        if email and not st.session_state.otp_sent:
            if st.button("Send OTP"):
                otp = send_otp(email)
                send_otp_email(email, otp)
                st.success(f"OTP sent to {email}.")
                st.session_state.otp_sent = True
                st.session_state.pending_email = email

        if st.session_state.otp_sent and not st.session_state.otp_verified:
            entered_otp = st.text_input("Enter OTP")
            if st.button("Verify OTP"):
                if not entered_otp:
                    st.error("Please enter the OTP.")
                elif verify_otp(st.session_state.pending_email, entered_otp):
                    mark_user_verified(st.session_state.pending_email)
                    st.success("OTP verified. You can now set your password.")
                    st.session_state.otp_verified = True
                else:
                    st.error("Invalid OTP. Please try again.")

        if st.session_state.otp_verified:
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")

            if st.button("Sign Up"):
                if not all([full_name, username, email, password, confirm]):
                    st.error("All fields are required.")
                elif not validate_email(email):
                    st.error("Invalid email format.")
                elif not validate_password_strength(password):
                    st.warning("Password must be at least 8 characters and include uppercase, lowercase, number, and symbol.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    success, msg = signup_user(full_name, username, email, password)
                    if success:
                        st.success(msg)
                        # Reset flow
                        st.session_state.otp_sent = False
                        st.session_state.otp_verified = False
                        del st.session_state["pending_email"]
                    else:
                        st.error(msg)

    else:  # Login
        identifier = st.text_input("Email or Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if not identifier or not password:
                st.error("Both fields are required.")
            elif not validate_email(identifier) and "@" in identifier:
                st.error("Invalid email format.")
            else:
                success, msg = login_user(identifier, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_email = identifier
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("---")
        with st.expander("Forgot Password?"):
            if "otp_verified" not in st.session_state:
                st.session_state.otp_verified = False
            reset_email = st.text_input("Enter your registered email")
            if st.button("Send OTP", key="send_otp"):
                if not validate_email(reset_email):
                    st.error("Enter a valid email.")
                else:
                    otp = send_otp(reset_email,purpose="reset")
                    send_otp_email(reset_email, otp)  # ✅ Added email sending
                    st.session_state.reset_email = reset_email
                    st.session_state.otp_sent = True
                    st.success(f"OTP sent to email)")
                    

            if st.session_state.get("otp_sent"):
                otp_input = st.text_input("Enter OTP", key="otp_input")
                if st.button("Verify OTP"):
                    if not otp_input:
                        st.error("Please enter the OTP.")
                    elif verify_otp(reset_email, otp_input,purpose="reset"):
                        st.success("OTP verified. You can now set your password.")
                        st.session_state.otp_verified = True
                    else:
                        st.error("Invalid OTP. Please try again.")


                if st.session_state.otp_verified:
                    new_pass = st.text_input("New Password", type="password", key="new_pass")
                    confirm_new = st.text_input("Confirm New Password", type="password", key="confirm_new")
                    if st.button("Reset Password", key="reset_password"):
                        if not new_pass or not confirm_new:
                            st.error("All fields are required.")
                        elif new_pass != confirm_new:
                            st.error("Passwords do not match.")
                        elif not validate_password_strength(new_pass):
                            st.warning("Password must be at least 8 characters with uppercase, lowercase, number, and symbol.")
                        else:
                            reset_password(reset_email, new_pass)
                            mark_user_verified(reset_email, purpose="reset")
                            pending_users_collection.delete_one({"email": email, "purpose": "reset"})
                            st.success("Password reset successful! Refresh the page and login again.")

