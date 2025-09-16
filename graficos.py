import os
import re
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

def contar_comentarios_por_canal(carpeta="comentarios_por_canal"):
    conteo_por_canal = {}

    for archivo in os.listdir(carpeta):
        if archivo.endswith(".json"):
            canal = archivo.replace("comentarios_", "").replace(".json", "")
            ruta = os.path.join(carpeta, archivo)

            with open(ruta, "r", encoding="utf-8") as f:
                data = json.load(f)

            total_comentarios = 0
            for video_id, contenido in data.items():
                for comentario in contenido:
                    if comentario!="_metrics":
                        total_comentarios+=1

            conteo_por_canal[canal] = total_comentarios

    return conteo_por_canal

def contar_videos_por_canal(carpeta="comentarios_por_canal"):
    conteo_por_canal = {}

    for archivo in os.listdir(carpeta):
        if archivo.endswith(".json"):
            canal = archivo.replace("comentarios_", "").replace(".json", "")
            ruta = os.path.join(carpeta, archivo)

            with open(ruta, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Cada clave es un video_id
            total_videos = sum(1 for video_id in data if not video_id.startswith("_"))
            conteo_por_canal[canal] = total_videos

    return conteo_por_canal

def graficar_conteos(conteo_comentarios, conteo_videos):
    canales1 = list(conteo_comentarios.keys())
    cantidades1 = list(conteo_comentarios.values())

    canales2 = list(conteo_videos.keys())
    cantidades2 = list(conteo_videos.values())

    # Crear figura con 2 subplots (1 fila, 2 columnas)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: comentarios
    axs[0].bar(canales1, cantidades1, color="skyblue")
    axs[0].set_title("Cantidad de comentarios por canal")
    axs[0].set_xlabel("Canal")
    axs[0].set_ylabel("Comentarios")
    axs[0].tick_params(axis="x", rotation=45)

    # Gráfico 2: videos
    axs[1].bar(canales2, cantidades2, color="salmon")
    axs[1].set_title("Cantidad de videos por canal")
    axs[1].set_xlabel("Canal")
    axs[1].set_ylabel("Videos")
    axs[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

def calcular_promedio_sentimiento(carpeta="analisis_por_canal",sent="POS"):
    promedios = {}

    for archivo in os.listdir(carpeta):
        if archivo.endswith(".json"):
            canal = archivo.replace(".json", "")
            canal=canal.replace("comentarios_","")
            canal=canal.replace("_analisis","")
            ruta = os.path.join(carpeta, archivo)

            with open(ruta, "r", encoding="utf-8") as f:
                data = json.load(f)

            valores = []
            for video_id, comentarios in data.items():
                for comentario_id, contenido in comentarios.items():
                    probas = contenido.get("sentimiento_probas", {})
                    sen = probas.get(sent)
                    if isinstance(sen, (int, float)):
                        valores.append(sen)

            if valores:
                promedio = sum(valores) / len(valores)
                promedios[canal] = promedio

    return promedios
def graficar_promedios(promedios, titulo="Positividad", ax=None):
    
    if ax is None:
        fig, ax = plt.subplots()

    ax.bar(promedios.keys(), promedios.values())
    ax.set_title(titulo)
    ax.set_ylabel(f"Promedio de {titulo}")
    ax.set_xticklabels(promedios.keys(), rotation=45)

def generar_nube_palabras(carpeta="comentarios_por_canal", output_dir="nubes"):
    os.makedirs(output_dir, exist_ok=True)

    stop_words = set(stopwords.words("spanish"))
    stop_words.add("q")
    stop_words.add("si")

    for archivo in os.listdir(carpeta):
        if archivo.endswith(".json"):
            canal_id = archivo.replace(".json", "")

            # Leer comentarios del archivo
            with open(os.path.join(carpeta, archivo), "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extraer solo los textos de los comentarios, ignorando _metrics
            textos = []
            for video_data in data.values():
                for id_comentario, comentario_data in video_data.items():
                    if id_comentario != "_metrics":
                        texto_comentario=comentario_data["texto"]
                        if isinstance(texto_comentario,str):
                            textos.append(texto_comentario)

            # Unir todos los comentarios en un solo texto
            texto = " ".join(textos)

            # Normalizar: minúsculas y quitar caracteres especiales
            texto = texto.lower()
            texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)

            # Generar nube
            nube = WordCloud(
                width=800,
                height=400,
                background_color="white",
                include_numbers=False,
                stopwords=stop_words,
                collocations=False
            ).generate(texto)

            # Guardar imagen
            plt.figure(figsize=(10, 6))
            plt.imshow(nube, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Nube de palabras - {canal_id}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"nube_{canal_id}.png"))
            plt.close()
