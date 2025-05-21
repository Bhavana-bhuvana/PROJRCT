import pymongo
from datetime import datetime
from bson import ObjectId

# MongoDB setup
MONGO_URI = "mongodb+srv://komal0mallaram:Qwerty%401234@cluster0.lhfjf0c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["New_DB"]

# Collections
users_col = db["Users"]
logs_col = db["Logs"]
pending_users_collection = db["pending_users"]

# ========== USERS ==========

def create_user(fullname, username, email, hashed_password):
    # Generate user ID from email (before '@gmail.com')
    user_id = email.split("@")[0] if "@gmail.com" in email else email

    result = users_col.insert_one({
        "user_id": user_id,  # <-- Add user_id here
        "fullname": fullname,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.now()
    })
    return result.inserted_id

# ========== LOGGING ==========

def log_user_action(user_email, action, details=None):
    logs_col.update_one(
        {"email": user_email},
        {
            "$push": {
                "actions": {
                    "action": action,
                    "details": details or {},
                    "timestamp": datetime.now()
                }
            }
        },
        upsert=True
    )

def save_log(user_email, model_type, model_name, session_data):
    logs_col.update_one(
        {"email": user_email},
        {
            "$push": {
                "sessions": {
                    "model_type": model_type,
                    "model_name": model_name,
                    "session_data": session_data,
                    "timestamp": datetime.now()
                }
            }
        },
        upsert=True
    )

def get_logs(user_email):
    """Returns the full log object (actions + sessions) for a user."""
    log_doc = logs_col.find_one({"email": user_email})
    if log_doc:
        return {
            "email": log_doc["email"],
            "actions": log_doc.get("actions", []),
            "sessions": log_doc.get("sessions", [])
        }
    return None
