import os
import json
from pysentimiento import create_analyzer

# Inicializamos los analizadores
sentiment_analyzer = create_analyzer(task="sentiment", lang="es")

def analisis_sentimientos(input_dir="comentarios_por_canal", output_dir="analisis_por_canal"):
    os.makedirs(output_dir, exist_ok=True)

    for archivo in os.listdir(input_dir):
        if archivo.endswith(".json"):
            canal_id = archivo.replace(".json", "")
            ruta_entrada = os.path.join(input_dir, archivo)
            ruta_salida = os.path.join(output_dir, f"{canal_id}_analisis.json")

            # Cargar comentarios originales
            with open(ruta_entrada, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Cargar análisis previo si existe
            if os.path.exists(ruta_salida):
                with open(ruta_salida, "r", encoding="utf-8") as f:
                    resultados_por_video = json.load(f)
            else:
                resultados_por_video = {}

            # Procesar cada video
            for video_id, contenido_video in data.items():
                if video_id.startswith("_"):
                    continue

                if video_id not in resultados_por_video:
                    resultados_por_video[video_id] = {}

                for comentario_id, comentario_data in contenido_video.items():
                    if comentario_id == "_metrics":
                        continue

                    # Saltar si ya fue analizado
                    if comentario_id in resultados_por_video[video_id]:
                        continue

                    texto = comentario_data.get("texto", "")
                    if not texto:
                        continue

                    sentimiento = sentiment_analyzer.predict(texto)

                    resultados_por_video[video_id][comentario_id] = {
                        "sentimiento": sentimiento.output,
                        "sentimiento_probas": dict(sentimiento.probas),
                    }

            # Guardar resultados actualizados
            with open(ruta_salida, "w", encoding="utf-8") as f:
                json.dump(resultados_por_video, f, ensure_ascii=False, indent=2)
