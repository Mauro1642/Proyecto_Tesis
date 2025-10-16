import os
import json
import networkx as nx
from itertools import combinations
from scipy.spatial.distance import cdist
import numpy as np
def construir_red_global_andres(input_car="usuarios_por_canal"):
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
            
    nx.write_gexf(G, "grafo_para_gephi.gexf")
    return G

def construir_red_global_pablo(input_car="analisis_por_canal"):
    G=nx.Graph()
    nodos = []
    vectores = []
    canales = []
    videos_id = []
    todos_los_datos={}
    for archivo in os.listdir(input_car):
        nombre_canal=archivo.split("_")[1]
        ruta=os.path.join(input_car, archivo)
        
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        videos_id=list(data.keys())
        for video_id in videos_id:
            if(data[video_id]=={}):
                continue
            else:
                comentarios=data[video_id]
                id_comentarios=list(comentarios.keys())
                prom_pos=0
                prom_neg=0
                prom_neu=0
                for id_comentario in id_comentarios:
                    sent_probas=comentarios[id_comentario]["sentimiento_probas"]
                    prom_pos+=sent_probas["POS"]
                    prom_neg+=sent_probas["NEG"]
                    prom_neu+=sent_probas["NEU"]
                prom_pos=prom_pos/len(id_comentarios)
                prom_neg=prom_neg/len(id_comentarios)
                prom_neu=prom_neu/len(id_comentarios)
            prom=[prom_pos,prom_neg,prom_neu]
            video_node = f"{nombre_canal}_{video_id}"
            todos_los_datos[video_node] = {
                'prom_sent':prom,
                'canal': nombre_canal,
                'video_id': video_id
            }
            G.add_node(video_node, 
                      canal=nombre_canal, 
                      video_id=video_id)
            nodos.append(video_node)
            vectores.append(prom)
            canales.append(nombre_canal)
            videos_id.append(video_id)
    vectores = np.array(vectores)
    dist_matrix = cdist(vectores, vectores, metric="cosine")
    n = len(nodos)
    for i in range(n):
        for j in range(i+1, n):
            tipo_conexion = 'intra_canal' if canales[i] == canales[j] else 'inter_canal'
            G.add_edge(nodos[i], nodos[j],
                       weight=dist_matrix[i, j],
                       tipo_conexion=tipo_conexion)

    nx.write_gexf(G, "grafo_para_gephi_pablo.gexf")
    return G
    # for v1, v2 in combinations(videos, 2):
    #     promedio_v1 = todos_los_datos[v1]['prom_sent']
    #     promedio_v2 = todos_los_datos[v2]['prom_sent']
    #     promedio_v1=np.array(promedio_v1)
    #     promedio_v2=np.array(promedio_v2)
    #     distancia_coseno=cosine(promedio_v1.ravel(),promedio_v2.ravel())

    #     canal1 = todos_los_datos[v1]['canal']
    #     canal2 = todos_los_datos[v2]['canal']
    #     tipo_conexion = 'intra_canal' if canal1 == canal2 else 'inter_canal'
            
    #     G.add_edge(v1, v2, 
    #                   weight=distancia_coseno,
    #                   tipo_conexion=tipo_conexion)

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


            