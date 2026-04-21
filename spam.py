
# from sklearn.feature_extraction.text import CountVectorizer

# email = [
#     "win money now",
#     "meeting at 5PM",
#     "win a free lottery",
#     "Project discussion tomorrow"
# ]

# vertorizer = CountVectorizer()

# x = vertorizer.fit_transform(email)

# print("Vocabulary: ", vertorizer.get_feature_names_out())
# print("Matrix: \n", x.toarray())

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Data
emails = [
    "win money now",
    "meeting at 5 pm",
    "free lottery win",
    "project discussion tomorrow"
]

labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Step 2: Convert text → numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Step 3: Train model
model = MultinomialNB()
model.fit(X, labels)

# Step 4: Test on new email
test_email = ["meeting at 5 pm"]
test_vector = vectorizer.transform(test_email)

prediction = model.predict(test_vector)

print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")
