import os
import json
import networkx as nx
from itertools import combinations
import pandas as pd
import random
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
import numpy as np
from networkx.algorithms.community import modularity
def construir_red_global_usuarios(input_car="usuarios_por_canal"):
    """
    Crea un grafo global con videos de todos los canales
    """
    G = nx.Graph()
    todos_los_datos = {}
    
    # Cargar datos de todos los canales
    for archivo in os.listdir(input_car):
        nombre_canal = archivo.split(".")[0]
        ruta = os.path.join(input_car, archivo)
        
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Agregar nodos (videos) con identificador único por canal
        for video_id, usuarios_video in data.items():
            video_node = f"{nombre_canal}_{video_id}"
            todos_los_datos[video_node] = {
                'usuarios': set(usuarios_video),
                'canal': nombre_canal,
                'video_id': video_id
            }
            
            G.add_node(video_node, 
                      canal=nombre_canal, 
                      video_id=video_id,
                      num_usuarios=str(len(set(usuarios_video))))
    
    # Crear aristas entre todos los videos (inter e intra canal)
    videos = list(todos_los_datos.keys())
    for v1, v2 in combinations(videos, 2):
        usuarios_v1 = todos_los_datos[v1]['usuarios']
        usuarios_v2 = todos_los_datos[v2]['usuarios']
        interseccion = usuarios_v1 & usuarios_v2
        
        peso = len(interseccion)
        if peso > 0:  # solo agregar aristas si hay usuarios en común
            canal1 = todos_los_datos[v1]['canal']
            canal2 = todos_los_datos[v2]['canal']
            tipo_conexion = 'intra_canal' if canal1 == canal2 else 'inter_canal'
            
            G.add_edge(v1, v2, 
                      weight=peso,
                      tipo_conexion=tipo_conexion,
                      usuarios_compartidos=str(list(interseccion)))
            
    # nx.write_gexf(G, "grafo_para_gephi.gexf")
    return G

def construir_red_usuarios_canales(input_car="usuarios_por_canal"):
    """
    Construye un grafo donde cada nodo es un canal y las aristas representan
    la cantidad de usuarios compartidos entre canales.

    Además detecta comunidades mediante Louvain, agrega el atributo 
    'comunidad' a cada nodo y devuelve la modularidad.
    """

    G = nx.Graph()
    datos_canales = {}

    # =====================================================
    # 1. Cargar todos los usuarios por canal
    # =====================================================
    for archivo in os.listdir(input_car):
        if not archivo.endswith(".json"):
            continue

        nombre_canal = archivo.split(".")[0]
        ruta = os.path.join(input_car, archivo)

        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Unión de usuarios del canal (todos los videos del canal)
        usuarios_del_canal = set()
        for video_id, lista_usuarios in data.items():
            usuarios_del_canal.update(lista_usuarios)

        datos_canales[nombre_canal] = usuarios_del_canal

        # Agregar nodo al grafo
        G.add_node(
            nombre_canal,
            num_videos=len(data),
            num_usuarios=len(usuarios_del_canal)
        )

    # =====================================================
    # 2. Crear aristas canal ↔ canal
    # =====================================================
    canales = list(datos_canales.keys())

    for c1, c2 in combinations(canales, 2):
        interseccion = datos_canales[c1] & datos_canales[c2]
        peso = len(interseccion)

        if peso > 0:
            # ⚠️ No guardamos la lista de usuarios, solo el peso (GEXF-friendly)
            G.add_edge(c1, c2, weight=peso)

    # =====================================================
    # 3. Detectar comunidades (Louvain)
    # =====================================================
    comunidades = nx.algorithms.community.louvain_communities(G,weight='weight')

    # Mapear nodo → comunidad_id
    comunidad_id = {}
    for i, com in enumerate(comunidades):
        for nodo in com:
            comunidad_id[nodo] = i

    # Asignar comunidad a cada nodo como atributo
    nx.set_node_attributes(G, comunidad_id, "comunidad")

    # =====================================================
    # 4. Calcular modularidad
    # =====================================================
    Q = modularity(G, comunidades, weight="weight")

    # =====================================================
    # 5. Guardar para Gephi
    # =====================================================
    nx.write_gexf(G, "grafo_canales_usuarios.gexf", encoding="utf-8")

    # =====================================================
    # 6. Devolver todo
    # =====================================================
    return G, comunidades, Q

def construir_red_global_sentimientos_canales(
    input_car="analisis_por_canal",
    umbral_similitud=0.8,
    output_gexf="red_global_sentimientos.gexf"
):
    """
    Construye una red de videos basada en la similitud de sentimientos promedio
    y la exporta a formato Gephi (.gexf).

    Parámetros
    ----------
    input_car : str
        Carpeta que contiene los archivos JSON por canal.
    umbral_similitud : float
        Solo se conectan videos con similitud coseno > umbral.
    output_gexf : str
        Nombre del archivo .gexf de salida.

    Retorna
    -------
    G : networkx.Graph
        Grafo con nodos (videos) y aristas (similitudes).
    todos_los_datos : dict
        Información agregada por video.
    """
    G = nx.Graph()
    todos_los_datos = {}
    nombres, vectores, canales = [], [], []
    # === 1. Cargar datos y calcular promedios ===
    for archivo in os.listdir(input_car):
        nombre_canal = archivo.split("_")[1]
        ruta = os.path.join(input_car, archivo)

        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        sent_videos=[]
        for video_id, comentarios in data.items():
            if not comentarios:
                continue

            sent_vals = []
            for c in comentarios.keys():
                if c == '_metrics':
                    continue
                sent_proba = comentarios[c].get('sentimiento_probas', None)
                if sent_proba is None:
                    continue

                vals = [sent_proba.get("POS"), sent_proba.get("NEG"), sent_proba.get("NEU")]
                # Solo agregar si todos los valores existen y no son None
                if None not in vals:
                    sent_vals.append(vals)

            sent_vals = np.array(sent_vals)

            # --- Salteá si no hay datos válidos ---
            if sent_vals.size == 0:
                continue

            prom = np.nanmean(sent_vals, axis=0)  # promedio ignorando NaN

        # --- Validar que prom sea un vector 3D sin NaN ---
            if not isinstance(prom, np.ndarray) or prom.shape != (3,) or np.isnan(prom).any():
                continue

            sent_videos.append(prom)

        prom=np.mean(sent_videos,axis=0)
            
        canal_node = f"{nombre_canal}"
        nombres.append(canal_node)
        vectores.append(prom)
        canales.append(nombre_canal)
        todos_los_datos[canal_node] = {
                "prom_sent": prom.tolist(),
                "canal": nombre_canal,
        }
        
        G.add_node(
                canal_node,
                POS=prom[0],
                NEG=prom[1],
                NEU=prom[2]
        )
    vectores = np.array(vectores)
    vectores = (vectores - vectores.mean(axis=0))
    n = len(vectores)
    # === 2. Calcular matriz de distancias ===
    matriz_sim = cosine_similarity(vectores)
    i_idx, j_idx = np.where(np.triu(matriz_sim >= umbral_similitud, k=1))
    edges = [
    (
        nombres[i],
        nombres[j],
        {
            'weight': float(matriz_sim[i, j]),           # peso = similitud
            'similitud': float(matriz_sim[i, j]),
        }
    )
    for i, j in zip(i_idx, j_idx)
    ]
    G.add_edges_from(edges)

    # === 4. Exportar a Gephi (.gexf) ===
    nx.write_gexf(G, output_gexf, encoding="utf-8")
    nodos_df = pd.DataFrame([
    {"id": n, **G.nodes[n]} for n in G.nodes()
    ])
    nodos_df.to_csv("nodos_canales_sent.csv", index=False)

    # 5. Exportar aristas a CSV
    aristas_df = pd.DataFrame([
    {"source": u, "target": v, "similitud": d["similitud"]}
    for u, v, d in G.edges(data=True)
    ])
    aristas_df.to_csv("aristas_canales_sent.csv", index=False)

    return G

def construir_red_global_sentimientos(
    input_car="analisis_por_canal",
    umbral_similitud=0.8,
    output_gexf="red_global_sentimientos.gexf",
    n_videos_por_canal=100,  
    seed=42,
    cantidad_aristas=80,
    tipo='ambos'                  
):
    """
    Construye una red de videos basada en la similitud de sentimientos promedio
    y la exporta a formato Gephi (.gexf).

    Parámetros
    ----------
    input_car : str
        Carpeta que contiene los archivos JSON por canal.
    umbral_similitud : float
        Solo se conectan videos con similitud coseno > umbral.
    output_gexf : str
        Nombre del archivo .gexf de salida.
    n_videos_por_canal : int, opcional
        Número máximo de videos a procesar por canal (muestra aleatoria).
    seed : int, opcional
        Semilla aleatoria para reproducibilidad.
    """

    random.seed(seed)
    np.random.seed(seed)

    G = nx.Graph()
    todos_los_datos = {}
    nombres, vectores, canales = [], [], []

    # === 1. Cargar datos y calcular promedios ===
    for archivo in os.listdir(input_car):
        nombre_canal = archivo.split("_")[1]
        ruta = os.path.join(input_car, archivo)

        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_ids = list(data.keys())
        if "_metrics" in video_ids:
            video_ids.remove("_metrics")

        # --- Elegir muestra aleatoria de videos ---
        if n_videos_por_canal is not None and len(video_ids) > n_videos_por_canal:
            video_ids = random.sample(video_ids, n_videos_por_canal)

        for video_id in video_ids:
            comentarios = data[video_id]
            if not comentarios:
                continue

            sent_vals = []
            for c in comentarios.keys():
                if c == '_metrics':
                    continue
                sent_proba = comentarios[c].get('sentimiento_probas', None)
                if sent_proba is None:
                    continue

                vals = [sent_proba.get("POS"), sent_proba.get("NEG"), sent_proba.get("NEU")]
                if None not in vals:
                    sent_vals.append(vals)

            sent_vals = np.array(sent_vals)

            if sent_vals.size == 0:
                continue

            prom = np.nanmean(sent_vals, axis=0)
            if not isinstance(prom, np.ndarray) or prom.shape != (3,) or np.isnan(prom).any():
                continue

            video_node = f"{nombre_canal}_{video_id}"

            G.add_node(
                video_node,
                canal=nombre_canal,
                video_id=video_id,
                POS=prom[0],
                NEG=prom[1],
                NEU=prom[2]
            )

            nombres.append(video_node)
            vectores.append(prom)
            canales.append(nombre_canal)

            todos_los_datos[video_node] = {
                "prom_sent": prom.tolist(),
                "canal": nombre_canal,
                "video_id": video_id
            }

    # === 2. Calcular similitudes ===
    vectores = np.array(vectores)
    vectores = vectores - vectores.mean(axis=0)

    matriz_sim = cosine_similarity(vectores)
    i_idx, j_idx = np.where(np.triu(matriz_sim >= umbral_similitud, k=1))

    edges = [
        (
            nombres[i],
            nombres[j],
            {
                'weight': float(matriz_sim[i, j]),
                'similitud': float(matriz_sim[i, j]),
                'tipo_conexion': 'intra_canal' if canales[i] == canales[j] else 'inter_canal'
            }
        )
        for i, j in zip(i_idx, j_idx)
    ]
    # --- Seleccionar el k% de las aristas con mayor peso ---
    # k = cantidad_aristas  # porcentaje deseado, por ejemplo 30%
    # num_edges = int(len(edges) * k / 100)

    # # Ordenar las aristas por 'weight' en orden descendente
    # edges_ordenadas = sorted(edges, key=lambda e: e[2]['weight'], reverse=True)

    # # Seleccionar las k% de mayor peso
    # edges_muestreadas = edges_ordenadas[:num_edges]
    #--- Seleccionar un n% de las aristas de manera aleatoria ---
    # --- Filtrado según la variable 'tipo' ---
    if tipo == 'negativa':
        edges = [e for e in edges if e[2]['weight'] < 0]

    elif tipo == 'positiva':
        edges = [e for e in edges if e[2]['weight'] >= 0]

    elif tipo == 'ambos':
        pass  # no filtramos nada
    n = cantidad_aristas  # porcentaje deseado, por ejemplo 30%
    num_edges = int(len(edges) * n / 100)
    edges_muestreadas = random.sample(edges, num_edges)

    # --- Agregar solo esas aristas al grafo ---
    G.add_edges_from(edges_muestreadas) 

    # === 3. Exportar ===
    nx.write_gexf(G, output_gexf, encoding="utf-8")

    pd.DataFrame([{"id": n, **G.nodes[n]} for n in G.nodes()]).to_csv("nodos.csv", index=False)
    pd.DataFrame([
        {"source": u, "target": v, "similitud": d["similitud"]}
        for u, v, d in G.edges(data=True)
    ]).to_csv("aristas.csv", index=False)

    return G

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta

def evolucion_red_comentarios(carpeta='comentarios_por_canal', output="evolucion.mp4", fps=2):
    """
    Construye la evolución temporal de una red de comentarios y genera una animación.

    Parámetros:
    - carpeta: str, ruta a la carpeta con los JSON de comentarios.
    - output: str, nombre del archivo de salida (gif o mp4).
    - fps: int, cuadros por segundo de la animación.

    Retorna:
    - evolucion: lista de (fecha, grafo) con el estado de la red día por día.
    """

    G = nx.Graph()
    comentarios = []

    # --- 1. Cargar todos los comentarios ---
    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)

        for video_id, contenido in data.items():
            if contenido=={}:
                continue
            for comment_id, comment_data in contenido.items():
                if comment_id == "_metrics":
                    continue
                fecha_str = comment_data.get("fecha")
                autor=comment_data.get("fecha")
                if not fecha_str or not autor:
                    continue
                fecha = datetime.fromisoformat(fecha_str.replace("Z", "+00:00"))
                comentarios.append((fecha.date(), autor, video_id))

    # --- 2. Ordenar por fecha ---
    comentarios.sort(key=lambda x: x[0])

    # --- 3. Construir red día a día ---
    fecha_inicio = comentarios[0][0]
    fecha_fin = comentarios[-1][0]
    delta = timedelta(days=1)

    evolucion = []
    fecha_actual = fecha_inicio

    while fecha_actual <= fecha_fin:
        # Filtrar comentarios del día actual
        del_dia = [c for c in comentarios if c[0] == fecha_actual]

        for _, autor, video_id in del_dia:
            G.add_node(autor)
            # conectar con autores previos en el mismo video
            autores_previos = [a for f, a, v in comentarios 
                               if v == video_id and f <= fecha_actual and a != autor]
            for otro in autores_previos:
                G.add_edge(autor, otro, video=video_id, fecha=str(fecha_actual))

        evolucion.append((fecha_actual, G.copy()))
        fecha_actual += delta

    # --- 4. Crear animación ---
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(i):
        ax.clear()
        fecha, g_dia = evolucion[i]
        pos = nx.spring_layout(g_dia, seed=42)  # posiciones fijas para estabilidad visual
        nx.draw(
            g_dia, pos, with_labels=True,
            node_size=500, font_size=8,
            node_color="skyblue", edge_color="gray", ax=ax
        )
        ax.set_title(f"Red de comentarios al {fecha}")

    ani = animation.FuncAnimation(fig, update, frames=len(evolucion), interval=1000//fps, repeat=False)

    # Guardar animación
    if output.endswith(".gif"):
        ani.save(output, writer="pillow", fps=fps)
    else:
        ani.save(output, writer="ffmpeg", fps=fps)

    plt.close(fig)

    return evolucion


            