import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from analyze_transitions import analyze_transitions  # Adjust the import path as necessary

# Example text for transition analysis
text = (
    "Alice was excited about her upcoming vacation. She had been planning it for months. "
    "Her friend Bob decided to join her. They both agreed that visiting Paris would be the highlight of the trip."
)

# Analyze transitions and get the DataFrame
transition_df = analyze_transitions(text)

# Prepare features and labels
transition_df['features'] = transition_df['current_sentence'] + " " + transition_df['next_sentence']
transition_df['label'] = transition_df['transition_type'].astype('category').cat.codes

# Prepare text data
sentences = transition_df['features'].tolist()  # Extract features from the DataFrame
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)  # Use features for fitting
sequences = tokenizer.texts_to_sequences(sentences)  # Convert features to sequences
max_length = max(len(s) for s in sequences)
X_padded = pad_sequences(sequences, maxlen=max_length, padding='post')

# Prepare labels
y = transition_df['label'].values  # Extract labels directly from DataFrame
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # You can skip this if labels are already numeric

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(transition_df['transition_type'].unique()), activation='softmax')  # Correct number of classes
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# Example prediction
test_sentences = ["Alice was excited about her vacation.", "She had been planning it for months."]
test_seq = tokenizer.texts_to_sequences([" ".join(test_sentences)])  # Combine for prediction
test_padded = pad_sequences(test_seq, maxlen=max_length, padding='post')

predictions = model.predict(test_padded)

# Get predicted transition type
predicted_class = predictions.argmax()  # Get the index of the highest probability
predicted_transition = label_encoder.inverse_transform([predicted_class])[0]  # Decode back to label

print(predicted_transition)

# Model kaydetme konumunu belirleyin
model_save_dir = 'models'  # Klasör adı
os.makedirs(model_save_dir, exist_ok=True)  # Klasörü oluştur (varsa hata vermez)

# Modelinizi burada eğittikten sonra kaydedin
model.save(os.path.join(model_save_dir, 'transition_model.h5'))  # Modeli kaydet
print("Model başarıyla kaydedildi!")

