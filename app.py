from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class Email(BaseModel):
    text: str

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