import pandas as pd
import re
import json
import os
from datetime import datetime, timedelta
import pytz
import time
import random
import yt_dlp

ap_key = "AIzaSyCnx9KNELdutZ4XJPZgWfFspQTnPmPj08M" # Esta clave no se usa con yt_dlp, pero se mantiene.
canales = [
    "UCj6PcyLvpnIRT_2W_mwa9Aw", "UCFgk2Q2mVO1BklRQhSv6p0w",
    "UCba3hpU7EFBSk817y9qZkiA", "UCrpMfcQNog595v5gAS-oUsQ",
    "UCR9120YBAqMfntqgRTKmkjQ", "UCvsU0EGXN7Su7MfNqcTGNHg", "UChxGASjdNEYHhVKpl667Huw",
    "UCnejI42HK6cKapzJRWZXCbQ", "UCiaePeoCqpU8hBHiNrgkzrA", "UCXgsCoIhEUIwWvGK_JDY21w",
    "UCYvINPByAdCcpA0sWrF3I_w", "UC_49ElhhVd1BO7MsdBPm77Q", "UC5wAqJ9NF0fpGH9dVf3h6HA", "UCz489cQmrgH57sShDiatwfw"
]

# --- Funciones Auxiliares (Modificadas para recibir 'info' dict) ---

def get_channel_name(channel_id):
    try:
        url = f"https://www.youtube.com/channel/{channel_id}/about"
        channel_id = channel_id.strip('"\'')
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "no_warnings": True,
            "extract_flat": True, # Para obtener info del canal, no videos
            "ignoreerrors": True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info and "channel" in info and "uploader_id" in info:
                name = info["channel"]
                handle = info["uploader_id"]
                return name, handle
            else:
                print(f"❌ No se pudo obtener la información del canal {channel_id}.")
                return "desconocido", None
    except Exception as e:
        print(f"❌ Error al obtener info del canal {channel_id}: {e}")
        return "desconocido", None

def fecha_video(info):
    """Extrae la fecha de publicación de un diccionario de información de video."""
    return info.get('upload_date') # formato: 'YYYYMMDD'

def get_video_metadata(info):
    """Extrae vistas y likes de un diccionario de información de video."""
    return {
        'views': info.get('view_count'),
        'likes': info.get('like_count'),
    }

def get_comments(info):
    """Extrae comentarios de un diccionario de información de video."""
    comentarios = {}
    if info and "comments" in info:
        for comentario in info["comments"]:
            cid = comentario["id"]
            fecha = comentario["_time_text"]
            fecha = time_text_to_iso_argentina(fecha)
            comentarios[cid] = {
                "texto": comentario["text"],
                "fecha": fecha,
                "likes": comentario["like_count"],
                "autor": comentario["author"]
            }
    return comentarios

# --- Funciones sin cambios significativos ---

def pasaron_48_horas(upload_date_str):
    if upload_date_str is None: # Manejar el caso donde no se pudo obtener la fecha
        return timedelta(hours=999), True # Considerarlo como "pasaron 48h"
    fecha_publicacion = datetime.strptime(upload_date_str, "%Y%m%d")
    argentina_tz = pytz.timezone("America/Argentina/Buenos_Aires")
    fecha_publicacion = argentina_tz.localize(fecha_publicacion)
    ahora = datetime.now(argentina_tz)
    diferencia = ahora - fecha_publicacion
    return [diferencia, diferencia.total_seconds() > 24 * 3600] # Cambiado a 48h

def cargar_json(path):
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}

def guardar_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def id_ultimos_videos(channel_url, nv=10):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
        'force_json': False,
        'playlistend': 20,
        'no_warnings': True,
        'ignoreerrors': True,
        'daterange': 'today-1day'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        id_videos = []
        if info and "entries" in info:
            for entry in info["entries"]:
                # Comprobación más robusta para evitar errores si las claves no existen
                if entry.get("live_status") is None and entry.get("availability") is None and "id" in entry:
                    id_videos.append(entry["id"])
        id_videos=list(reversed(id_videos))
        return id_videos[-nv:]

def convertir_a_argentina(published_at_str):
    utc_dt = datetime.strptime(published_at_str, "%Y-%m-%dT%H:%M:%SZ")
    utc_dt = utc_dt.replace(tzinfo=pytz.utc)
    argentina_tz = pytz.timezone("America/Argentina/Buenos_Aires")
    local_dt = utc_dt.astimezone(argentina_tz)
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")

def time_text_to_iso_argentina(_time_text):
    tz = pytz.timezone("America/Argentina/Buenos_Aires")
    now = datetime.now(tz)
    match = re.match(r"(\d+)\s(\w+)\sago", _time_text)
    if not match:
        return None
    value, unit = match.groups()
    value = int(value)
    if unit.startswith("second"):
        delta = timedelta(seconds=value)
    elif unit.startswith("minute"):
        delta = timedelta(minutes=value)
    elif unit.startswith("hour"):
        delta = timedelta(hours=value)
    elif unit.startswith("day"):
        delta = timedelta(days=value)
    else:
        return None
    comment_time = now - delta
    return comment_time.strftime("%Y-%m-%dT%H:%M:%SZ")

# --- Función Principal (extraer_comentarios) Optimización Clave ---

def extraer_comentarios(channel_ids=canales,
                        processed_file="processed_videos.json",
                        comments_dir="comentarios_por_canal",
                        actualizar_videos=True,
                        max_comen=100):

    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_videos = json.load(f)
            processed_videos = {ch: set(ids) for ch, ids in processed_videos.items()}
    else:
        processed_videos = {}

    for channel_id in channel_ids:
        channel_name, handle = get_channel_name(channel_id)
        if not handle: # Si no pudimos obtener el handle, saltamos el canal
            print(f"Skipping channel {channel_id} due to missing handle.")
            continue
        url = f"https://www.youtube.com/{handle}/videos"

        canal_comments_file = os.path.join(comments_dir, f"comentarios_{channel_name}.json")
        if os.path.exists(canal_comments_file):
            with open(canal_comments_file, "r", encoding="utf-8") as f:
                canal_comments = json.load(f)
        else:
            canal_comments = {}
        
        updated_video_ids = set()
        # Asegurarse de que el canal_name esté en processed_videos
        if channel_name not in processed_videos:
            processed_videos[channel_name] = set()

        videos_a_procesar_ahora = set()

        # Añadir videos existentes que necesitan actualización
        if actualizar_videos:
            videos_a_procesar_ahora.update(processed_videos[channel_name])
            print(f"🔄 Actualizando {len(processed_videos[channel_name])} videos existentes para {channel_name}...")

        # Obtener los últimos videos del canal
        latest_video_ids = id_ultimos_videos(url)
        print(latest_video_ids)
        print(f"Found {len(latest_video_ids)} latest videos for {channel_name}.")
        videos_a_procesar_ahora.update(latest_video_ids)

        for video_id in videos_a_procesar_ahora:
            if(not(video_id in canal_comments)):
            # 🟢 PAUSA ALEATORIA antes de CADA video para evitar rate limiting
                pausa = random.uniform(1.5, 4.0)
                time.sleep(pausa)
                print(f"Pausando {pausa:.2f} segundos antes de procesar video: {video_id}...")

                # --- ÚNICA LLAMADA a yt_dlp.extract_info por video ---
                ydl_opts_video = {
                    'quiet': True,
                    'skip_download': True,
                    'no_warnings': True,
                    'extract_flat': False, # Necesitamos detalles completos
                    'getcomments': True,
                    'comment_limit': max_comen,
                    'ignoreerrors': True,
                    'force_json':False,
                }
                video_id=video_id.strip('"\'')
                video_url = f"https://www.youtube.com/watch?v={video_id}"
            
                info_dict = None
                try:
                    with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
                        info_dict = ydl.extract_info(video_url, download=False)
                except yt_dlp.DownloadError as e:
                    print(f"❌ Error al extraer info de {video_id} (ignorado por 'ignoreerrors'): {e}")
                    # info_dict ya es None, la lógica subsiguiente lo manejará.

                # 🟢 Comprobación clave: si info_dict es None, saltamos este video
                if not info_dict:
                    print(f"⚠️ No se pudo obtener la información completa del video {video_id}. Se salta.")
                    continue


            # Solo procesar si no han pasado 48h o si estamos actualizando y el video ya estaba
            # Y si el video no ha sido procesado ya en esta corrida (para evitar duplicados en updated_video_ids)
                updated_video_ids.add(video_id)

                stats = get_video_metadata(info_dict)
                metrics = {
                        "_metrics": {
                            "views": stats.get("views"), # Usar .get() para seguridad
                            "likes": stats.get("likes"),
                        }
                    }
                nuevos_comentarios = get_comments(info_dict) # Pasa info_dict directamente

                historico_video = {
                        k: v for k, v in canal_comments.get(video_id, {}).items() if k != "_metrics"
                    }

                for cid, data in nuevos_comentarios.items():
                        if cid not in historico_video:
                            historico_video[cid] = data

                canal_comments[video_id] = {**metrics, **historico_video}
            # else:
            #     # --- Extracción de datos del UNICO info_dict ---
            #     fecha = fecha_video(info_dict)
            #     if fecha is None: # Si la fecha tampoco se pudo obtener, salta el video
            #         print(f"⚠️ No se pudo obtener la fecha de publicación del video {video_id}. Se salta.")
            #         continue    
            #     diferencia, pasaron48 = pasaron_48_horas(fecha)
            #     print(f"El video {video_id} tiene {diferencia} de tiempo publicado.")
        # Guardar comentarios actualizados por canal
        os.makedirs(comments_dir, exist_ok=True)
        with open(canal_comments_file, "w", encoding="utf-8") as f:
            json.dump(canal_comments, f, ensure_ascii=False, indent=2)
        
        # Actualizar processed_videos para este canal con los IDs que sí se procesaron
        processed_videos[channel_name] = list(updated_video_ids) # Convertir a lista para JSON

    # Guardar processed_videos global
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump({ch: list(ids) for ch, ids in processed_videos.items()}, f, ensure_ascii=False, indent=2)

# --- Extrae usuarios por canal (Sin cambios) ---
def usuarios_canal(input_dir="comentarios_por_canal", output_dir="usuarios_por_canal"):
    os.makedirs(output_dir, exist_ok=True)

    for archivo in os.listdir(input_dir):
        if archivo.endswith(".json"):
            ruta = os.path.join(input_dir, archivo)

            with open(ruta, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Inferimos canal a partir del nombre del archivo (ej: comentarios_NombreCanal.json)
            # Ajustar para que el nombre del canal sea correcto si tiene espacios, etc.
            canal_nombre_archivo = os.path.splitext(archivo)[0].replace("comentarios_", "")
            salida = os.path.join(output_dir, f"{canal_nombre_archivo}.json")

            if os.path.exists(salida):
                with open(salida, "r", encoding="utf-8") as f:
                    canal_data = json.load(f)
            else:
                canal_data = {}
            
            # Obtener solo las claves que son IDs de video (excluyendo _metrics)
            id_videos = [k for k in data.keys() if k != "_metrics"]

            for video in id_videos:
                usuarios = set(canal_data.get(video, []))

                # Obtener las claves que son IDs de comentario (excluyendo _metrics)
                comentarios_del_video = data[video].keys()
                for comentario_key in comentarios_del_video:
                    if comentario_key != "_metrics":
                        autor = data[video][comentario_key]["autor"]
                        usuarios.add(autor)

                canal_data[video] = list(usuarios)
            
            with open(salida, "w", encoding="utf-8") as f:
                json.dump(canal_data, f, ensure_ascii=False, indent=2)

