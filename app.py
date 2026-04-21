from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware

# Step 1: Create app FIRST
app = FastAPI()

# Step 2: Add middleware AFTER app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 3: Safe file loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

# Step 4: Schema
class Email(BaseModel):
    text: str

# Step 5: Routes
@app.get("/")
def home():
    return {"message": "Spam Classifier API Running 🚀"}

@app.post("/predict")
def predict(email: Email):
    vec = vectorizer.transform([email.text])
    pred = model.predict(vec)[0]

    return {
        "prediction": "Spam" if pred == 1 else "Not Spam"
    }