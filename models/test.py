import pandas as pd
from sklearn.calibration import LabelEncoder
from centering_model import TransitionAnalyzer, prepare_training_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


text = (
    "Sarah went to the bakery to buy some bread. She has been going to this bakery for years. "
    "Sarah was excited to try the new sourdough bread. The bakery had recently started making it. "
    "It was a delightful surprise to find that the bakery was also offering pastries now."
)

analyzer = TransitionAnalyzer(text)
results = analyzer.analyze()


# Example dataset (replace with actual data)
data = [
    {'current_entities': 'Sarah', 'next_entities': 'She', 'current_words': 'Sarah went to the bakery to buy some bread', 'next_words': 'She has been going to this bakery for years', 'transition': 'Retaining Transition'},
    {'current_entities': 'The bakery', 'next_entities': 'It', 'current_words': 'Sarah was excited to try the new sourdough bread', 'next_words': 'The bakery had recently started making it', 'transition': 'Retaining Transition'},
    {'current_entities': 'The bakery', 'next_entities': 'The bakery', 'current_words': 'It was a delightful surprise to find that the bakery was also offering pastries now', 'next_words': 'The bakery had recently started making it', 'transition': 'Continuation Transition'},
    # Add more data
]

df = pd.DataFrame(data)
X = df[['current_entities', 'next_entities', 'current_words', 'next_words']]
y = df['transition']

# Flatten the features
X_flattened = X.apply(lambda row: ' '.join(row), axis=1)

# Vectorize the features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_flattened)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

def predict_transition(model, vectorizer, label_encoder, current_entities, next_entities, current_words, next_words):
    features = {
        'current_entities': ' '.join(current_entities),
        'next_entities': ' '.join(next_entities),
        'current_words': ' '.join(current_words),
        'next_words': ' '.join(next_words)
    }
    feature_vectorized = vectorizer.transform([' '.join(features.values())])
    prediction_encoded = model.predict(feature_vectorized)[0]
    return label_encoder.inverse_transform([prediction_encoded])[0]

# Example prediction
current_entities = ['Sarah']
next_entities = ['She']
current_words = ['Sarah', 'went', 'to', 'the', 'bakery', 'to', 'buy', 'some', 'bread']
next_words = ['She', 'has', 'been', 'going', 'to', 'this', 'bakery', 'for', 'years']

predicted_transition = predict_transition(model, vectorizer, label_encoder, current_entities, next_entities, current_words, next_words)
print(f"Predicted Transition: {predicted_transition}")
