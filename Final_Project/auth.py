
import bcrypt
from pymongo import MongoClient
import random
import streamlit as st
from utils import send_otp_email
import login_signup
MONGO_URI = "mongodb+srv://komal0mallaram:Qwerty%401234@cluster0.lhfjf0c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["New_DB"]
users_collection = db["Users"]
pending_users_collection = db["pending_users"]

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def send_otp(email, full_name=None, username=None, purpose="signup"):
    import random

    otp = str(random.randint(100000, 999999))

    if purpose == "signup":
        # check if already signed up
        if users_collection.find_one({"email": email}):
            return None  # signup not allowed

        data = {
            "email": email,
            "full_name": full_name,
            "username": username,
            "otp": otp,
            "otp_verified": False,
            "purpose": "signup"
        }

    elif purpose == "reset":
        # don't add name or username
        if not users_collection.find_one({"email": email}):
            return None  # can't reset non-existing user

        data = {
            "email": email,
            "otp": otp,
            "otp_verified": False,
            "purpose": "reset"
        }

    pending_users_collection.update_one(
        {"email": email, "purpose": purpose},
        {"$set": data},
        upsert=True
    )

    return otp


def signup_user(full_name, username, email, password):
    user = pending_users_collection.find_one({"email": email})
    if not user or not user.get("otp_verified"):
        return False, "OTP not verified or email not found."

    # Now move to main users collection
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        "full_name": user["full_name"],
        "username": user["username"],
        "email": user["email"],
        "password": hashed_pw
    })
    # Clean up pending user
    pending_users_collection.delete_one({"email": email})
    return True, "Signup successful!"

def login_user(identifier, password):
    user = users_collection.find_one({
        "$or": [{"email": identifier}, {"username": identifier}]
    })
    if not user:
        return False, "User not found."
    if not check_password(password, user["password"]):
        return False, "Incorrect password." 
    return True, "Login successful."
def verify_otp(email, otp, purpose="signup"):

    record = pending_users_collection.find_one({"email": email, "purpose": purpose})
    if record and record["otp"] == otp:
        pending_users_collection.update_one(
            {"email": email, "purpose": purpose},
            {"$set": {"otp_verified": True}}
        )
        return True
    return False

def reset_password(email, new_password):
    hashed = hash_password(new_password)
    users_collection.update_one(
        {"email": email},
        {"$set": {"password": hashed}}
    )
    return True

def mark_user_verified(email,purpose="signup"):
    pending_users_collection.delete_one({"email": email, "purpose": purpose})



def get_user_by_email_or_username(identifier):
    return users_collection.find_one({
        "$or": [{"email": identifier}, {"username": identifier}]
    })
