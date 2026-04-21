import kagglehub
import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🔹 Step 1: Download dataset
path = kagglehub.dataset_download("jackksoncsie/spam-email-dataset")

print("Path to dataset files:", path)

# 🔹 Step 2: Get file
files = os.listdir(path)
print("Files in dataset:", files)

file_path = os.path.join(path, files[0])

# 🔹 Step 3: Load data
df = pd.read_csv(file_path)

print("\nFirst rows:\n", df.head())
print("\nColumns:\n", df.columns)

# 🔹 Step 4: Use correct columns
df = df[['text', 'spam']]
df.columns = ['text', 'label']

# 🔹 Step 5: Ensure labels are numeric
df['label'] = df['label'].astype(int)

# 🔹 Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 🔹 Step 7: Vectorization
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)   # ✅ FIXED (was wrong before)

# 🔹 Step 8: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 🔹 Step 9: Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# 🔹 Step 10: Custom test
test_email = ["please contact and collect the win amount"]

test_vec = vectorizer.transform(test_email)
prediction = model.predict(test_vec)

print("\nPrediction:", "Spam" if prediction[0] == 1 else "Not Spam")