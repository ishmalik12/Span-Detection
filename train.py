import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 🔹 Step 1: Download dataset
path = "path.csv"

df = pd.read_csv(path)

# print("\nFirst rows:\n", df.head())
# print("\nColumns:\n", df.columns)

# 🔹 Step 4: Use correct columns
df = df[['text', 'spam']]
df.columns = ['text', 'spam']

# 🔹 Step 5: Ensure labels are numeric
df['spam'] = df['spam'].astype(int)

# 🔹 Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['spam'], test_size=0.2, random_state=42
)

# 🔹 Step 7: Vectorization
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) 

# 🔹 Step 8: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 🔹 Step 9: Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# 🔹 Step 10: Custom test
test_email = ["you're selected for the role of AI Engineer"]

test_vec = vectorizer.transform(test_email)
prediction = model.predict(test_vec)

print("\nPrediction:", "Spam" if prediction[0] == 1 else "Not Spam")
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))