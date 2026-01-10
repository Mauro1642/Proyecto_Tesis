import os
import re
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from collections import defaultdict

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
    # Crear carpeta para guardar imágenes (si no existe)
    output_dir = "graficos_generales"
    os.makedirs(output_dir, exist_ok=True)

    # Ordenar por valor descendente
    conteo_comentarios = dict(sorted(conteo_comentarios.items(), key=lambda x: x[1], reverse=True))
    conteo_videos = dict(sorted(conteo_videos.items(), key=lambda x: x[1], reverse=True))

    canales1 = list(conteo_comentarios.keys())
    cantidades1 = list(conteo_comentarios.values())

    canales2 = list(conteo_videos.keys())
    cantidades2 = list(conteo_videos.values())
    filename_comentarios='conteo_comentarios.pdf'
    filename_videos='conteo_videos.pdf'  
    # --- Gráfico 1: Comentarios ---
    plt.style.use("seaborn-v0_8-white")

    plt.figure(figsize=(14, 8))
    plt.bar(
        canales1,
        cantidades1,
        color="#6baed6",          # azul académico suave
        edgecolor="black",
        linewidth=1.2
    )

    plt.xlabel("Canal", fontsize=16)
    plt.ylabel("Cantidad de Comentarios", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename_comentarios),
                format='pdf', bbox_inches="tight")
    plt.show()


    # --- Gráfico 2: Videos ---
    plt.figure(figsize=(14, 8))
    plt.bar(
        canales2,
        cantidades2,
        color="#9ecae1",          # azul más claro para diferenciar
        edgecolor="black",
        linewidth=1.2
    )

    plt.xlabel("Canal", fontsize=16)
    plt.ylabel("Cantidad de Videos", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename_videos),
                format='pdf', bbox_inches="tight")
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

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import traceback

def generar_nube_tfidf(carpeta="comentarios_por_canal", output_dir="nubes_tfidf_global"):
    os.makedirs(output_dir, exist_ok=True)

    stop_words = set(stopwords.words("spanish")) | set(stopwords.words("english"))
    stop_words.update([
        'q','si','gente','va','ser','gobierno','país','ma','vamo','hace',
        'año','nunca','mismo','solo','dio','van','ver','gracias','ahora',
        'bien','mas','años','the','by','argentina','presidente','dios','así','youtube','sos',
        'hola','mauro','bueno','excelente','vamos','bull','dani','tan','pueblo','parameter','cordobeses','infobae',
        'bayly','suscribite','tengo','ajajjaj','dream','laila','scammer','htlr','lucasfornaro','cristiangomez'
    ])

    # --- Leer y CONCATENAR todos los comentarios de cada canal ---
    documentos_por_canal = {}  # cada canal = 1 documento
    nombres_canales = []
    
    for archivo in os.listdir(carpeta):
        if not archivo.endswith(".json"):
            continue
        print("Leyendo:", archivo)
        canal_id = archivo.replace(".json", "")
        ruta = os.path.join(carpeta, archivo)
        
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                data = json.load(f)
 
        
        except Exception as e:
            print(f"\n❌ Error leyendo {archivo}:")
            traceback.print_exc()
            continue

        textos = []
        for video_data in data.values():
            for id_comentario, comentario_data in video_data.items():
                if id_comentario != "_metrics":
                    texto = comentario_data.get("texto", "")
                    if isinstance(texto, str):
                        texto = texto.lower()
                        texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)
                        texto = re.sub(r"\s+", " ", texto).strip()
                        textos.append(texto)
        
        if textos:
            # ✅ CLAVE: Un solo string con TODO el canal
            documentos_por_canal[canal_id] = " ".join(textos)
            nombres_canales.append(canal_id)

    if not documentos_por_canal:
        print("No se encontraron datos")
        return

    # --- Calcular TF-IDF sobre los CANALES (no comentarios individuales) ---
    corpus = [documentos_por_canal[canal] for canal in nombres_canales]
    
    vectorizer = TfidfVectorizer(
    stop_words=list(stop_words),
    max_df=0.7,          # penaliza palabras comunes
    min_df=2,            # ignora palabras que aparecen solo una vez en todo el corpus
    max_features=None,    # más vocabulario para análisis
    sublinear_tf=True,   # TF logarítmico           # no normaliza por longitud
)
    
    # ✅ Esto calcula TF-IDF donde cada FILA = un canal
    tfidf_matrix = vectorizer.fit_transform(corpus)
    palabras = vectorizer.get_feature_names_out()

    print(f"\n📊 Calculado TF-IDF sobre {len(nombres_canales)} canales")
    print(f"📝 Vocabulario: {len(palabras)} palabras\n")

    # --- Generar nubes por canal ---
    for idx, canal in enumerate(nombres_canales):
        # Extraer scores TF-IDF para este canal
        tfidf_scores = tfidf_matrix[idx].toarray().ravel()
        
        # Crear diccionario palabra:score
        tfidf_dict = {
            palabra: score 
            for palabra, score in zip(palabras, tfidf_scores) 
            if score > 0
        }
        nombre_canal=canal.split("_")[1]
        if not tfidf_dict:
            print(f"⚠️ {canal}: Sin palabras significativas")
            continue
        
        # Top 10 para debug
        top_10 = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"🔝 {canal}:")
        for palabra, score in top_10:
            print(f"   {palabra}: {score:.4f}")
        print()
        
        # Generar nube
        nube = WordCloud(
            width=1200, 
            height=600, 
            background_color="white",
            include_numbers=False, 
            collocations=False, 
            colormap="plasma",
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(tfidf_dict)

        plt.figure(figsize=(10, 5))
        plt.imshow(nube, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{nombre_canal}", fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"nube_tfidf_{canal}.pdf"), 
            format="pdf", 
            bbox_inches="tight",
            dpi=300
        )
        plt.close()
        
    print(f"✅ Nubes guardadas en {output_dir}/")
