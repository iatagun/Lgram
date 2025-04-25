import pickle
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import os
import plotly.io as pio

# === 1. Modeli Yükle ===
with open('C:\\Users\\ilker\\Documents\\GitHub\\Lgram\\ngrams\\bigram_model.pkl', 'rb') as f:
    model = pickle.load(f)

# === 2. Bigramlardan Tek Kelimeleri Ayır
words = set()
for ngram in model.keys():
    if isinstance(ngram, tuple):
        words.update(ngram)
    else:
        words.add(ngram)

words = list(words)

# === 3. En sık geçen ilk 200 kelimeyi al
words = sorted(words, key=lambda w: model.get(w, 0), reverse=True)[:500]

# === 4. Word2Vec Modelini Yükle
print("Önceden eğitilmiş Word2Vec modeli yükleniyor...")
w2v_model = api.load('glove-wiki-gigaword-100')

# === 5. Küçük harfe çevirerek Word2Vec eşleşmesi
filtered_words = []
vectors = []

for word in words:
    lw = word.lower()
    if lw in w2v_model:
        filtered_words.append(word)
        vectors.append(w2v_model[lw])

print(f"{len(filtered_words)} kelime eşleşti.")

if not filtered_words:
    raise ValueError("Hiç kelime eşleşmedi!")

# === 6. PCA ile 3 Boyuta İndir
vectors = np.array(vectors)
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(vectors)

# === 7. Anlamsal Benzerlik ile Çizgileri Belirle
similarities = cosine_similarity(vectors)

threshold = 0.5  # Benzerlik eşik değeri
edges = []

for i in range(len(filtered_words)):
    for j in range(i + 1, len(filtered_words)):
        if similarities[i, j] >= threshold:
            edges.append((i, j))

# === 8. Çizgi Koordinatlarını Hazırla
edge_x = []
edge_y = []
edge_z = []

for i, j in edges:
    x0, y0, z0 = reduced_vectors[i]
    x1, y1, z1 = reduced_vectors[j]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode='lines',
    line=dict(color='lightblue', width=1),
    hoverinfo='none'
)

# === 9. Kelime Noktalarını Hazırla
node_trace = go.Scatter3d(
    x=reduced_vectors[:,0],
    y=reduced_vectors[:,1],
    z=reduced_vectors[:,2],
    mode='markers+text',
    marker=dict(
        size=6,
        color='gold',
        opacity=0.9,
        line=dict(width=0.5, color='black')
    ),
    text=filtered_words,
    textposition="top center",
    hoverinfo='text'
)

# === 10. Slayt Gösterisi (Kamera Dönüşü)
frames = []
angles = np.linspace(0, 360, 300)

for angle in angles:
    camera = dict(
        eye=dict(
            x=2*np.cos(np.radians(angle)),
            y=2*np.sin(np.radians(angle)),
            z=0.5 + 0.3*np.sin(np.radians(2*angle))
        )
    )
    frames.append(go.Frame(layout=dict(scene_camera=camera)))

# === 11. Layout
layout = go.Layout(
    title="🌌 3D Anlamsal Kelime Haritası - Çizgili + Slaytlı 🌌",
    margin=dict(l=0, r=0, b=0, t=50),
    paper_bgcolor='black',
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False),
        bgcolor='black'
    ),
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        buttons=[dict(
            label='Slaytı Başlat 🚀',
            method='animate',
            args=[None, {
                "frame": {"duration":20, "redraw":True},
                "fromcurrent":True,
                "mode":"immediate"
            }]
        )]
    )]
)

# === 12. Figürü Oluştur (frames DOĞRU yerde!)
fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=layout,
    frames=frames
)

# === 13. HTML'e Kaydet ve Aç
html_file = "semantic_word_network_slideshow.html"
pio.write_html(fig, file=html_file, auto_open=False)
os.startfile(os.path.abspath(html_file))
