from googleapiclient.discovery import build
import pandas as pd
import re
import json
import os
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from datetime import datetime, timezone
import pytz
from collections import OrderedDict
ap_key="AIzaSyCnx9KNELdutZ4XJPZgWfFspQTnPmPj08M"
canales= [
         "UCj6PcyLvpnIRT_2W_mwa9Aw", "UCFgk2Q2mVO1BklRQhSv6p0w",
         "UCba3hpU7EFBSk817y9qZkiA", "UCrpMfcQNog595v5gAS-oUsQ",
         "UCR9120YBAqMfntqgRTKmkjQ","UCvsU0EGXN7Su7MfNqcTGNHg","UChxGASjdNEYHhVKpl667Huw",
         "UCnejI42HK6cKapzJRWZXCbQ","UCiaePeoCqpU8hBHiNrgkzrA","UCXgsCoIhEUIwWvGK_JDY21w",
         "UCYvINPByAdCcpA0sWrF3I_w","UC_49ElhhVd1BO7MsdBPm77Q","UC5wAqJ9NF0fpGH9dVf3h6HA","UCz489cQmrgH57sShDiatwfw"
     ]

def extract_youtube_comments_and_metrics_always(
    channel_ids=canales,
    api_key=ap_key,
    max_videos=20,
    max_comments=100,
    processed_file="processed_videos.json",
    comments_dir="comentarios_por_canal",
    tz_name="America/Argentina/Buenos_Aires",
    max_age_hours=48
):
    os.makedirs(comments_dir, exist_ok=True)

    # --- Cargar historial de videos procesados ---
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_videos = json.load(f)
            processed_videos = {ch: set(ids) for ch, ids in processed_videos.items()}
    else:
        processed_videos = {}

    tz = pytz.timezone(tz_name)
    youtube = build('youtube', 'v3', developerKey=api_key)
    all_dataframes = {}

    def get_channel_name(channel_id):
        resp = youtube.channels().list(
            part="snippet",
            id=channel_id
        ).execute()
        items = resp.get("items", [])
        return items[0]["snippet"]["title"] if items else None

    def get_video_stats(video_id):
        resp = youtube.videos().list(
            part="statistics",
            id=video_id
        ).execute()
        items = resp.get("items", [])
        if not items:
            return None
        stats = items[0]["statistics"]
        return {
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            # dislikeCount ya no está disponible públicamente
            "dislikes": int(stats.get("dislikeCount", 0)) if "dislikeCount" in stats else None
        }

    for channel_id in channel_ids:
        channel_data = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()

        channel_name = get_channel_name(channel_id)
        uploads_playlist_id = channel_data['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        playlist_items = youtube.playlistItems().list(
            part='snippet',
            playlistId=uploads_playlist_id,
            maxResults=max_videos
        ).execute()

        # Archivo de comentarios por canal
        canal_comments_file = os.path.join(comments_dir, f"comentarios_{channel_name}.json")
        if os.path.exists(canal_comments_file):
            with open(canal_comments_file, "r", encoding="utf-8") as f:
                canal_comments = json.load(f)
        else:
            canal_comments = {}

        rows = []
        updated_video_ids = set()

        for item in playlist_items['items']:
            video_id = item['snippet']['resourceId']['videoId']
            title = item['snippet']['title']
            published_at = datetime.fromisoformat(
                item['snippet']['publishedAt'].replace("Z", "+00:00")
            ).astimezone(tz)

            diff=(datetime.now(tz) - published_at.astimezone(tz)).total_seconds()
            age_hours=round(diff/3600)
            # --- Siempre obtener métricas actuales ---
            stats = get_video_stats(video_id) or {"views": None, "likes": None, "dislikes": None}
            metrics = {
                "_metrics": {
                    "views": stats["views"],
                    "likes": stats["likes"],
                    "dislikes": stats["dislikes"]
                }
            }

            # --- Descargar comentarios ---
            nuevos_comentarios = {}
            try:
                next_page_token = None
                while True:
                    comment_data = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=max_comments,
                        textFormat='plainText',
                        pageToken=next_page_token
                    ).execute()

                    for c in comment_data['items']:
                        snippet = c['snippet']['topLevelComment']['snippet']
                        comment_id = c['id']
                        nuevos_comentarios[comment_id] = {
                            "texto": snippet['textDisplay'],
                            "fecha": snippet['publishedAt'],
                            "likes": snippet['likeCount'],
                            "autor": snippet['authorDisplayName']
                        }

                    next_page_token = comment_data.get('nextPageToken')
                    if not next_page_token:
                        break

            except Exception as e:
                nuevos_comentarios[f"error_{datetime.now().timestamp()}"] = {
                    "texto": f"Error: {str(e)}",
                    "fecha": None,
                    "likes": None,
                    "autor": None
                }

            # Fusionar con histórico y métricas
            historico_video = {k: v for k, v in canal_comments.get(video_id, {}).items() if k != "_metrics"}
            for cid, data in nuevos_comentarios.items():
                if cid not in historico_video:
                    historico_video[cid] = data

            canal_comments[video_id] = {**metrics, **historico_video}

            # Control de antigüedad
            if age_hours <= max_age_hours:
                updated_video_ids.add(video_id)
            else:
                print(f"📌 Última actualización y retiro de {video_id} ({channel_name}) — {age_hours:.1f}h desde publicación")
                print(f"   📊 Métricas finales: {metrics['_metrics']}")

            rows.append({
                'Titulo': title,
                'Video_id': video_id,
                'Fecha_publicacion': published_at.strftime("%Y-%m-%d %H:%M:%S"),
                'Vistas': metrics["_metrics"]["views"],
                'Likes': metrics["_metrics"]["likes"],
                'Dislikes': metrics["_metrics"]["dislikes"],
                'Comentarios': historico_video
            })

        if rows:
            all_dataframes[channel_name] = pd.DataFrame(rows)
        canal_comments = OrderedDict(
        sorted(canal_comments.items(), key=lambda x: x[1].get("_metrics", {}).get("publishedAt", ""))
        )
        # Guardar comentarios actualizados
        with open(canal_comments_file, "w", encoding="utf-8") as f:
            json.dump(canal_comments, f, ensure_ascii=False, indent=2)

        # Actualizar processed_videos para este canal
        processed_videos[channel_name] = updated_video_ids

    # Guardar processed_videos global
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump({ch: list(ids) for ch, ids in processed_videos.items()}, f, ensure_ascii=False, indent=2)

    return all_dataframes


def usuarios_canal(input_dir="comentarios_por_canal", output_dir="usuarios_por_canal"):
    os.makedirs(output_dir, exist_ok=True)

    for archivo in os.listdir(input_dir):
        if archivo.endswith(".json"):
            ruta = os.path.join(input_dir, archivo)

            with open(ruta, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Inferimos canal a partir del nombre del archivo (ej: canal1_video123.json → canal1)
            canal = archivo.split("_")[1]
            salida = os.path.join(output_dir, f"{canal}.json")

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

