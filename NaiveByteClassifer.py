from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "win money now",
    "meeting at 5 pm",
    "free lottery win",
    "project discussion tomorrow"
]
Lables = [1,0,1,0]
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(x, Lables)



test_email = ["please contact and collect the win amount"]
test_vectorize = vectorizer.transform(test_email)

prediction = model.predict(test_vectorize)

print("prediction: ", "Spam" if prediction[0] == 1 else "Not Spam")