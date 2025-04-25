import pickle
import random
import numpy as np
import plotly.graph_objects as go
import networkx as nx

# === 1. Modeli Yükle ===
with open('language_model.pkl', 'rb') as f:
    model = pickle.load(f)

# === 2. Graph Kur ===
G = nx.Graph()

for key in model.keys():
    if isinstance(key, tuple):
        words = key
    else:
        words = (key,)
    
    for word in words:
        G.add_node(word)
    
    for i in range(len(words) - 1):
        weight = model.get(key, 1)
        G.add_edge(words[i], words[i + 1], weight=weight)

# === 3. 3D Pozisyonlar ===
pos = nx.spring_layout(G, dim=3, seed=42)

x_nodes = [pos[k][0] for k in G.nodes()]
y_nodes = [pos[k][1] for k in G.nodes()]
z_nodes = [pos[k][2] for k in G.nodes()]

# === 4. Düğüm Frekansları (ısı için) ===
node_weights = {}
for node in G.nodes():
    node_weights[node] = 0

for edge in G.edges(data=True):
    node_weights[edge[0]] += edge[2]['weight']
    node_weights[edge[1]] += edge[2]['weight']

# Normalize et
max_weight = max(node_weights.values()) if node_weights else 1
node_colors = [node_weights[node] / max_weight for node in G.nodes()]
node_sizes = [5 + 15 * c for c in node_colors]  # frekansa göre boyut da değişiyor

# === 5. Kenarları Ayarla ===
edge_x = []
edge_y = []
edge_z = []

for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

# === 6. Ana Düğümler ve Kenarlar ===
node_trace = go.Scatter3d(
    x=x_nodes, y=y_nodes, z=z_nodes,
    mode='markers+text',
    marker=dict(
        size=node_sizes,
        color=node_colors,
        colorscale='Hot',  # İşte burası ısı haritası
        cmin=0,
        cmax=1,
        colorbar=dict(title='Frekans Sıcaklığı'),
        line=dict(width=0.5, color='black'),
        opacity=0.9
    ),
    text=[str(node) for node in G.nodes()],
    textposition="top center",
    hoverinfo='text'
)

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(
        width=0.5,
        color='lightblue',
    ),
    hoverinfo='none'
)

# === 7. Kamera Animasyonu ===
frames = []
angles = np.linspace(0, 360, 120)

for angle in angles:
    camera = dict(
        eye=dict(x=2*np.cos(np.radians(angle)),
                 y=2*np.sin(np.radians(angle)),
                 z=0.5 + 0.3*np.sin(np.radians(2*angle)))
    )
    frames.append(go.Frame(layout=dict(scene_camera=camera)))

layout = go.Layout(
    title="🔥 Kelime Sıcaklığı Galaksisi 🔥",
    showlegend=False,
    margin=dict(l=0, r=0, b=0, t=50),
    paper_bgcolor='black',
    scene=dict(
        xaxis=dict(showbackground=False, showticklabels=False, title=''),
        yaxis=dict(showbackground=False, showticklabels=False, title=''),
        zaxis=dict(showbackground=False, showticklabels=False, title=''),
        bgcolor='black'
    ),
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        buttons=[dict(label='Başlat 🔥', method='animate', args=[None, {
            "frame": {"duration":50, "redraw":True},
            "fromcurrent":True,
            "mode":"immediate"
        }])]
    )]
)

fig = go.Figure(data=[edge_trace, node_trace], layout=layout, frames=frames)

fig.show()
