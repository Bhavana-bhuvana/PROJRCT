import pymongo
from datetime import datetime
from bson import ObjectId

MONGO_URI = "mongodb+srv://komal0mallaram:Qwerty%401234@cluster0.lhfjf0c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["New_DB"]

# Collections
users_col = db["Users"]     # Basic user info
logs_col = db["Logs"]       # User action logs
pending_users_collection = db["pending_users"]
# Add a user (ensure it returns user ID for linking)
def create_user(fullname, username, email, hashed_password):
    result = users_col.insert_one({
        "fullname": fullname,
        "username": username,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.now()
    })
    return result.inserted_id  # This is the user's unique ID

def log_user_action(user_email, action, details=None):
    users_col.update_one(
        {"email": user_email},
        {
            "$push": {
                "actions": {
                    "action": action,
                    "details": details or {},
                    "timestamp": datetime.now()
                }
            }
        }
    )
