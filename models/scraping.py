import pandas as pd
import re
import json
import os
from datetime import datetime, timedelta
import pytz
from yt_dlp import YoutubeDL
ap_key="AIzaSyCnx9KNELdutZ4XJPZgWfFspQTnPmPj08M"
canales= [
         "UCj6PcyLvpnIRT_2W_mwa9Aw", "UCFgk2Q2mVO1BklRQhSv6p0w",
         "UCba3hpU7EFBSk817y9qZkiA", "UCrpMfcQNog595v5gAS-oUsQ",
         "UCR9120YBAqMfntqgRTKmkjQ","UCvsU0EGXN7Su7MfNqcTGNHg","UChxGASjdNEYHhVKpl667Huw",
         "UCnejI42HK6cKapzJRWZXCbQ","UCiaePeoCqpU8hBHiNrgkzrA","UCXgsCoIhEUIwWvGK_JDY21w",
         "UCYvINPByAdCcpA0sWrF3I_w","UC_49ElhhVd1BO7MsdBPm77Q","UC5wAqJ9NF0fpGH9dVf3h6HA","UCz489cQmrgH57sShDiatwfw"
     ]
#Funcion principal para extraer datos
import time
import random
def extraer_comentarios(channel_ids=canales,
    processed_file="processed_videos.json",
    comments_dir="comentarios_por_canal",
    actualizar_videos=True,
    max_comen=50):
    # --- Cargar historial de videos procesados ---
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_videos = json.load(f)
            processed_videos = {ch: set(ids) for ch, ids in processed_videos.items()}
    else:
        processed_videos = {}
    #Extraigo nombre del canal
    def get_channel_name(channel_id):
        try:
            url=f"https://www.youtube.com/channel/{channel_id}/about"
            channel_id=channel_id.strip('"\'')
            ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "no_warnings": True,
            "extract_flat": True,
            "ignoreerrors":True

            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                name=info["channel"]
                handle=info["uploader_id"]
                return name,handle
        except Exception as e:
            print(f"❌ Error al obtener info del canal {channel_id}: {e}")
            return "desconocido", None

        
    for channel_id in channel_ids:
        channel_name,handle = get_channel_name(channel_id)
        url=f"https://www.youtube.com/{handle}/videos"


        # Archivo de comentarios por canal
        canal_comments_file = os.path.join(comments_dir, f"comentarios_{channel_name}.json")
        if os.path.exists(canal_comments_file):
            with open(canal_comments_file, "r", encoding="utf-8") as f:
                canal_comments = json.load(f)
        else:
            canal_comments = {}
        
        updated_video_ids = set()
        videos_a_actualizar=set(processed_videos[channel_name])
        #Actualizo los videos que estan en processed_videos
        if (actualizar_videos==True):
            for video_id in videos_a_actualizar:
                # Inicializar stats con valores por defecto
                stats = {"views": 0, "likes": 0}

                for attempt in range(3):
                    try:
                        stats = get_video_metadata(video_id)
                        if stats["views"] is not None or stats["likes"] is not None:
                            break  # Éxito si obtuvo al menos una métrica
                    except Exception as e:
                        print(f"❌ Error obteniendo métricas de {video_id}, intento {attempt+1}/3: {e}")
                        if attempt < 2:  # No esperar en el último intento
                            time.sleep(2 ** attempt)  # Backoff exponencial
                else:
                    print(f"⚠️ No se pudieron obtener métricas de {video_id}, usando valores por defecto.")
                # # --- Obtener métricas ---
                # try:
                #     for attempt in range(3):
                #         try:
                #             stats = get_video_metadata(video_id)
                #             break
                #         except Exception as e:
                #             print(f"❌ Error obteniendo métricas de {video_id}, intento {attempt+1}/3: {e}")
                #             time.sleep(2)
                # except:
                #     print(f"⚠️ No se pudieron obtener métricas de {video_id}, se salta.")
                #     continue

                metrics = {
                    "_metrics": {
                        "views": stats["views"],
                        "likes": stats["likes"],
                    }
                }

                # --- Obtener comentarios ---
                for attempt in range(3):
                    try:
                        nuevos_comentarios = get_comments(video_id=video_id)
                        break
                    except Exception as e:
                        print(f"❌ Error obteniendo comentarios de {video_id}, intento {attempt+1}/3: {e}")
                        time.sleep(2)
                else:
                    print(f"⚠️ No se pudieron obtener comentarios de {video_id}, se salta.")
                    continue

                # Recuperar histórico de comentarios del JSON
                historico_video = {
                    k: v for k, v in canal_comments.get(video_id, {}).items() if k != "_metrics"
                }

                # Fusionar: agregar solo comentarios nuevos
                for cid, data in nuevos_comentarios.items():
                    if cid not in historico_video:
                        historico_video[cid] = data

                # Guardar métrica actualizada + comentarios fusionados
                canal_comments[video_id] = {**metrics, **historico_video}

                # Control de antigüedad
                fecha = fecha_video(video_id=video_id)
                diferencia, pasaron48 = pasaron_48_horas(fecha)
                if not pasaron48:
                    updated_video_ids.add(video_id)
                else:
                    print(f"📌 Última actualización y retiro de {video_id}, pasaron {diferencia}h desde publicación")

        videos=id_ultimos_videos(url)
        print(videos)
        for video_id in videos:
            # Control de antigüedad
            pausa = random.uniform(1.5, 4.0) # Pausa entre 1.5 y 4.0 segundos
            time.sleep(pausa)
            ydl_opts = {
                'quiet': True,
                'skip_download': True,
                'no_warnings': True,
                'extract_flat': False,
                'getcomments': True,
                'comment_limit': max_comen,
                'ignoreerrors': True # Recomendado para que no falle
            }
            video_id = video_id.strip('"\'')
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
            fecha=info.get('upload_date')
            diferencia,pasaron48=pasaron_48_horas(fecha)
            if (pasaron48==False and video_id not in videos_a_actualizar):
                updated_video_ids.add(video_id)
                stats = {
                'views': info.get('view_count'),
                'likes': info.get('like_count'),
                }
                metrics = {
                    "_metrics": {
                        "views": stats["views"],
                        "likes": stats["likes"],
                    }
                }
                nuevos_comentarios=get_comments(info=info)        
                historico_video = {k: v for k, v in canal_comments.get(video_id, {}).items() if k != "_metrics"}
                for cid, data in nuevos_comentarios.items():
                    if cid not in historico_video:
                        historico_video[cid] = data
                canal_comments[video_id] = {**metrics, **historico_video}
            else:
                continue
        # Guardar comentarios actualizados
        with open(canal_comments_file, "w", encoding="utf-8") as f:
            json.dump(canal_comments, f, ensure_ascii=False, indent=2)
        # Actualizar processed_videos para este canal
        processed_videos[channel_name] = updated_video_ids
    # Guardar processed_videos global
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump({ch: list(ids) for ch, ids in processed_videos.items()}, f, ensure_ascii=False, indent=2)

#Extraigo fecha de video
from yt_dlp import YoutubeDL, DownloadError

def fecha_video(info):
        return info.get('upload_date')

# def fecha_video(video_id):
    
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     ydl_opts = {
#         'quiet': True,
#         'skip_download': True,
#         'no_warnings':True
#     }

#     with YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=False)
#         return info.get('upload_date')  # formato: 'YYYYMMDD'
    
#me fijo si pasaron 48hs
def pasaron_48_horas(upload_date_str):
    # Convertir la fecha 'YYYYMMDD' a datetime
    fecha_publicacion = datetime.strptime(upload_date_str, "%Y%m%d")

    # Asignar zona horaria Argentina
    argentina_tz = pytz.timezone("America/Argentina/Buenos_Aires")
    fecha_publicacion = argentina_tz.localize(fecha_publicacion)

    # Obtener fecha actual en Argentina
    ahora = datetime.now(argentina_tz)

    # Calcular diferencia
    diferencia = ahora - fecha_publicacion
    return [diferencia,diferencia.total_seconds() > 24 * 3600]

#Cargo el JSON
def cargar_json(path):
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
#Guardo data en el JSON
def guardar_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

#Extrae los ultimos id de videos
def id_ultimos_videos(channel_url, nv=10):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,  # No descarga, solo lista los videos
        'skip_download': True,
        'force_json': False,
        'playlistend': nv,
        'no_warnings': True,
        'ignoreerrors':True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        id_videos=[]
        for entry in info["entries"]:
            if (entry["live_status"]==None and entry["availability"]==None):
                id_videos.append(entry["id"])
        return id_videos
    
#Extrae views y likes de un video
def get_video_metadata(info):
    return {
                'views': info.get('view_count'),
                'likes': info.get('like_count'),
            }
    
#Convierto a horario argentino
def convertir_a_argentina(published_at_str):
    utc_dt = datetime.strptime(published_at_str, "%Y-%m-%dT%H:%M:%SZ")
    utc_dt = utc_dt.replace(tzinfo=pytz.utc)
    argentina_tz = pytz.timezone("America/Argentina/Buenos_Aires")
    local_dt = utc_dt.astimezone(argentina_tz)
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")

#Extrae comentarios
def get_comments(info):
    comentarios={}
        # 🟢 VERIFICACIÓN CLAVE: Comprueba si info existe y contiene la clave 'comments'
    if info and "comments" in info:
        for comentario in info["comments"]:
            cid = comentario["id"]
            fecha=comentario["_time_text"]
            fecha=time_text_to_iso_argentina(fecha)
            comentarios[cid] = {
                    "texto": comentario["text"],
                    "fecha": fecha,
                    "likes": comentario["like_count"],
                    "autor": comentario["author"]
                }
            return comentarios
# def get_comments(video_id,comment_lim=50):
#     ydl_opts = {
#     'quiet': True,
#     'skip_download': True,
#     'no_warnings': True,
#     'extract_flat': True,
#     'getcomments': True, # Necesitamos info completa
#     'comment_limit' : comment_lim,
#     'ignoreerrors':True
#     }
#     video_id=video_id.strip('"\'')
#     video_url = f"https://www.youtube.com/watch?v={video_id}"
#     comentarios={}
#     with YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(video_url, download=False)
#         for comentario in info["comments"]:
#             cid=comentario["id"]
#             fecha=comentario["_time_text"]
#             fecha=time_text_to_iso_argentina(fecha)
#             comentarios[cid]={
#             "texto": comentario["text"],
#             "fecha": fecha,
#             "likes": comentario["like_count"],
#             "autor": comentario["author"]
#             }
#     return comentarios
            
#Funcion para cambiar el formato de fecha
def time_text_to_iso_argentina(_time_text):
    # Hora actual en Argentina
    tz = pytz.timezone("America/Argentina/Buenos_Aires")
    now = datetime.now(tz)

    # Patrón para extraer número y unidad
    match = re.match(r"(\d+)\s(\w+)\sago", _time_text)
    if not match:
        return None  # formato desconocido

    value, unit = match.groups()
    value = int(value)

    # Convertir a timedelta
    if unit.startswith("second"):
        delta = timedelta(seconds=value)
    elif unit.startswith("minute"):
        delta = timedelta(minutes=value)
    elif unit.startswith("hour"):
        delta = timedelta(hours=value)
    elif unit.startswith("day"):
        delta = timedelta(days=value)
    else:
        return None  # unidad desconocida

    comment_time = now - delta
    return comment_time.strftime("%Y-%m-%dT%H:%M:%SZ")

#Extrae usuarios por canal
def usuarios_canal(input_dir="comentarios_por_canal", output_dir="usuarios_por_canal"):
    os.makedirs(output_dir, exist_ok=True)

    for archivo in os.listdir(input_dir):
        if archivo.endswith(".json"):
            ruta = os.path.join(input_dir, archivo)

            with open(ruta, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Inferimos canal a partir del nombre del archivo (ej: canal1_video123.json → canal1)
            canal = archivo.split("_")[1]
            salida = os.path.join(output_dir, f"{canal}")

            # Si ya existe el JSON del canal, lo cargamos
            if os.path.exists(salida):
                with open(salida, "r", encoding="utf-8") as f:
                    canal_data = json.load(f)
            else:
                canal_data = {}
            id_videos = data.keys()

            for video in id_videos:
                usuarios = set(canal_data.get(video, []))  # usuarios acumulados del video

                comentarios = data[video].keys()
                for comentario in comentarios:
                    if comentario != "_metrics":  # ignoramos métricas
                        autor = data[video][comentario]["autor"]
                        usuarios.add(autor)

                canal_data[video] = list(usuarios)
            # Guardamos el archivo actualizado
            with open(salida, "w", encoding="utf-8") as f:
                json.dump(canal_data, f, ensure_ascii=False, indent=2)

