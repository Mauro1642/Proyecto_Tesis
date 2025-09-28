import os
import json
import networkx as nx
from itertools import combinations
from scipy.spatial.distance import cosine
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
                      num_usuarios=str(len(usuarios_video)))
    
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
    videos = list(todos_los_datos.keys())
    for v1, v2 in combinations(videos, 2):
        promedio_v1 = todos_los_datos[v1]['prom_sent']
        promedio_v2 = todos_los_datos[v2]['prom_sent']
        promedio_v1=np.array(promedio_v1)
        promedio_v2=np.array(promedio_v2)
        distancia_coseno=cosine(promedio_v1.ravel(),promedio_v2.ravel())

        canal1 = todos_los_datos[v1]['canal']
        canal2 = todos_los_datos[v2]['canal']
        tipo_conexion = 'intra_canal' if canal1 == canal2 else 'inter_canal'
            
        G.add_edge(v1, v2, 
                      weight=distancia_coseno,
                      tipo_conexion=tipo_conexion)
    nx.write_gexf(G, "grafo_para_gephi_pablo.gexf")
    return G


            