from centering_model import CenteringModel
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
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

# Veri inceleme: Geçiş DataFrame'inin başını ve sonunu yazdır
print("Geçiş Analizi Sonucu:")
print(transition_df.head())  # İlk 5 satırı göster
print("\nGeçiş Analizi Sonucu (Son 5 Satır):")
print(transition_df.tail())  # Son 5 satırı göster

# Metin verilerini hazırlama
sentences = transition_df['features'].tolist()  # DataFrame'den özellikleri çıkar
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)  # Özellikleri kullanarak fit et
sequences = tokenizer.texts_to_sequences(sentences)  # Özellikleri dizelere dönüştür
max_length = max(len(s) for s in sequences)
X_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

# Etiketleri hazırlama
y = transition_df['label'].values  # DataFrame'den etiketleri çıkar

# Veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Sınıf ağırlıklarını hesapla
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Model kaydetme dizinini tanımlama
model_save_dir = 'models'  # Klasör adı
os.makedirs(model_save_dir, exist_ok=True)  # Klasörü oluştur (varsa hata vermez)

# Modeli inşa etme
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
    tf.keras.layers.LSTM(128, return_sequences=True),  # LSTM hücre sayısını artır
    tf.keras.layers.LSTM(64, return_sequences=False),  # Ek bir LSTM katmanı ekle
    tf.keras.layers.Dropout(0.5),  # Aşırı öğrenmeyi önlemek için Dropout ekle
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(transition_df['transition_type'].unique()), activation='softmax')  # Doğru sınıf sayısı
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping ve model checkpoint ayarları
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_save_dir, 'best_transition_model.keras'), save_best_only=True)

# Modeli eğitme
model.fit(X_train, y_train, epochs=300, batch_size=8, validation_data=(X_test, y_test),
          class_weight=class_weights_dict, callbacks=[early_stopping, model_checkpoint])

# Test kaybı ve doğruluğunu hesapla
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Kaybı: {test_loss:.4f}, Test Doğruluğu: {test_accuracy:.4f}')

# Eğitim ve test verilerinin şekillerini yazdır
print(f'\nEğitim Verisi Şekli: {X_train.shape}, Etiket Şekli: {y_train.shape}')
print(f'Test Verisi Şekli: {X_test.shape}, Etiket Şekli: {y_test.shape}')
