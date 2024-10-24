from centering_model import CenteringModel
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from analyze_transitions import analyze_transitions

# Metin dosyasını belirli boyutlarda yükleme
def load_text_data_in_chunks(file_path, chunk_size=19000):
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Örnek metin dosyası
file_path = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_data.txt'

# Geçiş analizini yapacak DataFrame'leri birleştirme
all_transition_dfs = []

for chunk in load_text_data_in_chunks(file_path, chunk_size=19000):
    transition_df = analyze_transitions(chunk)
    transition_df['features'] = transition_df['current_sentence'] + " " + transition_df['next_sentence']
    transition_df['label'] = transition_df['transition_type'].astype('category').cat.codes
    all_transition_dfs.append(transition_df)

full_transition_df = pd.concat(all_transition_dfs, ignore_index=True)

# Metin verilerini hazırlama
sentences = full_transition_df['features'].tolist()
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
max_length = max(len(s) for s in sequences)
X_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

# Etiketleri hazırlama
y = full_transition_df['label'].values

# Veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Sınıf ağırlıklarını hesapla
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Model kaydetme dizinini tanımlama
model_save_dir = 'models'
os.makedirs(model_save_dir, exist_ok=True)

# Modeli inşa etme
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),  # Bidirectional LSTM
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),  # Ek bir Bidirectional LSTM katmanı
    tf.keras.layers.Dropout(0.4),  # Dropout oranını artır
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # L2 düzenlemesi
    tf.keras.layers.Dense(len(full_transition_df['transition_type'].unique()), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Early stopping ve model checkpoint ayarları
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_save_dir, 'best_transition_model.keras'), save_best_only=True)

# Modeli eğitme
model.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test),
          class_weight=class_weights_dict, callbacks=[early_stopping, model_checkpoint])

# Test kaybı ve doğruluğunu hesapla
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Kaybı: {test_loss:.4f}, Test Doğruluğu: {test_accuracy:.4f}')

# Eğitim ve test verilerinin şekillerini yazdır
print(f'\nEğitim Verisi Şekli: {X_train.shape}, Etiket Şekli: {y_train.shape}')
print(f'Test Verisi Şekli: {X_test.shape}, Etiket Şekli: {y_test.shape}')
