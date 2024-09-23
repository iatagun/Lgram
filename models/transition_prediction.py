from centering_model import CenteringModel
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from analyze_transitions import analyze_transitions  # Adjust the import path as necessary

# Metin dosyasını yükleme
def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Örnek metin dosyası
file_path = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_data.txt'  # Dosya yolunu buraya girin
text = load_text_data(file_path)

# Geçiş analizini yap ve DataFrame'i al
transition_df = analyze_transitions(text)

# Özellikleri ve etiketleri hazırlama
transition_df['features'] = transition_df['current_sentence'] + " " + transition_df['next_sentence']
transition_df['label'] = transition_df['transition_type'].astype('category').cat.codes

# Metin verilerini hazırlama
sentences = transition_df['features'].tolist()  # DataFrame'den özellikleri çıkar
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)  # Özellikleri kullanarak fit et
sequences = tokenizer.texts_to_sequences(sentences)  # Özellikleri dizelere dönüştür
max_length = max(len(s) for s in sequences)
X_padded = pad_sequences(sequences, maxlen=max_length, padding='post')

# Etiketleri hazırlama
y = transition_df['label'].values  # DataFrame'den etiketleri çıkar

# Veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Modeli inşa etme
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(transition_df['transition_type'].unique()), activation='softmax')  # Doğru sınıf sayısı
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, batch_size=8, validation_data=(X_test, y_test))



# Model kaydetme konumunu belirleme
model_save_dir = 'models'  # Klasör adı
os.makedirs(model_save_dir, exist_ok=True)  # Klasörü oluştur (varsa hata vermez)

# Modelinizi burada eğittikten sonra kaydedin
model.save(os.path.join(model_save_dir, 'transition_model.h5'))  # Modeli kaydet
print("Model başarıyla kaydedildi!")

