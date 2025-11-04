from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
client = MongoClient(MONGO_URI)
db = client['student_ai']

students_col = db['students']
predictions_col = db['predictions']

def save_prediction(record: dict):
    return predictions_col.insert_one(record)

def save_student(student: dict):
    return students_col.insert_one(student)

def get_recent_predictions(limit=20):
    return list(predictions_col.find().sort('_id', -1).limit(limit))
