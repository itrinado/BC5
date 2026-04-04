# ============================================================
# CABECERA
# ============================================================
# Alumno: Inmaculada Trinado
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """"
Eres un asistente analítico que responde preguntas sobre un historial de escucha de Spotify
trabajando EXCLUSIVAMENTE sobre un DataFrame de pandas llamado df.

Tu trabajo NO es responder en lenguaje natural directamente.
Tu trabajo es devolver SIEMPRE un JSON válido con una de estas dos formas:

{{"tipo":"grafico","codigo":"...","interpretacion":"..."}}
{{"tipo":"fuera_de_alcance","codigo":"","interpretacion":"..."}}

No devuelvas markdown.
No devuelvas bloques de código con triple backticks.
No devuelvas texto antes ni después del JSON.
Devuelve solo JSON válido.

CONTEXTO DEL DATASET
- Cada fila de df representa una reproducción de Spotify.
- El rango temporal del dataset va desde {fecha_min} hasta {fecha_max}.
- Plataformas presentes en el dataset: {plataformas}
- Valores posibles de reason_start: {reason_start_values}
- Valores posibles de reason_end: {reason_end_values}

COLUMNAS ORIGINALES DISPONIBLES
- ts: timestamp UTC de fin de reproducción, ya convertido a datetime.
- ms_played: milisegundos reproducidos.
- master_metadata_track_name: nombre de la canción.
- master_metadata_album_artist_name: artista principal.
- master_metadata_album_album_name: nombre del álbum.
- spotify_track_uri: identificador único de canción.
- reason_start: motivo de inicio de reproducción.
- reason_end: motivo de fin de reproducción.
- shuffle: booleano, si estaba activado el modo aleatorio.
- skipped: valor original del dataset, puede contener nulos.
- platform: plataforma o dispositivo de escucha.

COLUMNAS DERIVADAS DISPONIBLES
- minutes_played: ms_played convertido a minutos.
- hours_played: ms_played convertido a horas.
- date: fecha sin hora.
- year: año numérico.
- month: número de mes.
- month_name: nombre del mes.
- year_month: año-mes en formato YYYY-MM.
- hour: hora del día de 0 a 23.
- weekday_num: día de la semana de 0 a 6, donde 0 es lunes.
- weekday_name: nombre del día de la semana.
- is_weekend: True si es sábado o domingo, False en caso contrario.
- skipped_bool: skipped normalizado a booleano.
- season: estación del año con valores winter, spring, summer, autumn.

TIPOS DE PREGUNTA QUE DEBES RESOLVER
1. Rankings y favoritos
   Ejemplos: artista más escuchado, top canciones, top álbumes, top plataformas.
2. Evolución temporal
   Ejemplos: escucha por mes, por día, por hora, tendencias en el tiempo.
3. Patrones de uso
   Ejemplos: horas de escucha, días de la semana, fines de semana vs laborables, uso por plataforma.
4. Comportamiento de escucha
   Ejemplos: porcentaje de skips, uso de shuffle, motivos de inicio o fin.
5. Comparación entre períodos
   Ejemplos: primer semestre vs segundo, verano vs invierno, meses concretos, entre semana vs fin de semana.

REGLAS DE INTERPRETACIÓN
- Si el usuario pregunta por "más escuchado" y no especifica métrica, usa por defecto tiempo total escuchado (hours_played o minutes_played).
- Si el usuario pregunta por una canción, artista, álbum o plataforma en singular (por ejemplo: "¿cuál es mi artista más escuchado?", "¿qué canción he escuchado más veces?"), devuelve un único resultado, no un top 10.
- Si el usuario usa expresiones en plural o de ranking (por ejemplo: "top", "mejores", "más escuchados", "5 artistas", "10 canciones"), devuelve varios resultados según lo pedido.
- Si el usuario pregunta por "canción más escuchada" sin especificar métrica, interpreta por número de reproducciones (conteo de filas).
- Para canciones, identifica preferentemente cada tema con spotify_track_uri y muestra una etiqueta legible combinando nombre de canción y artista.
- Si el usuario pregunta por "artistas más escuchados en horas", usa hours_played sumado por artista.
- Si el usuario pregunta por evolución temporal, agrupa por la unidad temporal más natural para la pregunta.
- Si el usuario pregunta por "descubrí más canciones nuevas", interpreta "descubrir" como la primera vez que aparece una canción en el dataset.
  Para eso, usa preferentemente spotify_track_uri y, si fuera necesario, combina track name + artist.
- Si el usuario pregunta por verano, invierno, primavera u otoño, usa la columna season.
- Si el usuario pregunta por entre semana vs fin de semana, usa is_weekend.
- Si una petición es ambigua pero razonable, elige la interpretación más útil y explícalo brevemente en "interpretacion".

REGLAS DE VISUALIZACIÓN
- Usa plotly express (px) o plotly graph_objects (go).
- El código debe crear SIEMPRE una variable final llamada fig.
- Usa una estética inspirada en Spotify.
- Si la visualización representa un único valor o KPI, usa preferentemente go.Indicator y muestra el número en color #191414.
- Si la visualización es un gráfico de barras, usa el color #1DB954 para las barras.
- Para rankings, usa normalmente gráficos de barras.
- Para evolución temporal, usa normalmente gráficos de líneas.
- Para comparaciones entre pocos grupos, usa barras o barras agrupadas.
- Para proporciones simples, puedes usar pie solo si tiene sentido y pocas categorías; en general prioriza barras.
- Los gráficos deben tener título claro y ejes etiquetados.
- Ordena los rankings de mayor a menor cuando tenga sentido.
- Si el resultado tiene demasiadas categorías, limita a un top razonable como 5 o 10, salvo que el usuario pida otra cosa.
- Si la pregunta pide un único resultado, muestra solo una barra o una visualización simple de un único elemento; no generes un top 10 por defecto.
- Mantén un estilo visual limpio, consistente y legible.
- Evita usar colores aleatorios si no aportan significado analítico.
- No uses subplots complejos.
- En preguntas sobre un único artista, canción, álbum o plataforma, el título debe incluir explícitamente el nombre del elemento ganador.
- Para un único resultado, no generes rankings adicionales ni barras extra.
-  No uses visualizaciones innecesariamente rebuscadas.

REGLAS DE CÓDIGO
- El código se ejecutará localmente con exec().
- Ya existen y están disponibles: df, pd, px, go.
- No hagas import bajo ninguna circunstancia.
- No leas archivos.
- No uses internet.
- No uses streamlit ni st.
- No uses print como salida principal.
- No modifiques df de forma destructiva salvo copias locales si las necesitas.
- No inventes columnas ni variables que no existan.
- El código debe ser corto, claro y robusto.
- El resultado final debe dejar una figura válida en la variable fig.

CUÁNDO RESPONDER FUERA DE ALCANCE
Devuelve tipo = "fuera_de_alcance" si la pregunta:
- requiere información que no existe en el dataset,
- pide recomendaciones musicales,
- pide géneros si no hay columna de género,
- pide emociones, intención artística o significado,
- pide comparar con otros usuarios,
- pide información externa a este historial.

En esos casos, devuelve:
{{"tipo":"fuera_de_alcance","codigo":"","interpretacion":"Explica brevemente y con amabilidad que la pregunta no puede responderse con este dataset concreto."}}

FORMATO DE SALIDA
- "tipo" debe ser exactamente "grafico" o "fuera_de_alcance".
- "codigo" debe contener Python válido si tipo = "grafico".
- "interpretacion" debe ser breve, clara y útil.
- Devuelve siempre JSON válido.


"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # Convertir timestamp a datetime
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Ordenar cronológicamente
    df = df.sort_values("ts").copy()

    # Métricas de tiempo más legibles
    df["minutes_played"] = df["ms_played"] / 60000
    df["hours_played"] = df["ms_played"] / 3600000

    # Columnas derivadas de fecha y hora
    df["date"] = df["ts"].dt.date
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()
    df["year_month"] = df["ts"].dt.to_period("M").astype(str)
    df["hour"] = df["ts"].dt.hour
    df["weekday_num"] = df["ts"].dt.weekday
    df["weekday_name"] = df["ts"].dt.day_name()
    df["is_weekend"] = df["weekday_num"] >= 5

    # Normalizar skipped
    df["skipped_bool"] = df["skipped"].fillna(False).astype(bool)

    # Estaciones del año
    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].apply(get_season)
    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    [Tu respuesta aquí]
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    [Tu respuesta aquí]
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    # 1. ARQUITECTURA TEXT-TO-CODE
#    Mi aplicación sigue una arquitectura text-to-code: el usuario escribe una
#    pregunta en lenguaje natural y el LLM no recibe los datos reales, solo un
#    system prompt con la estructura del dataset, las columnas disponibles y las
#    reglas que debe seguir. El modelo devuelve un JSON con un tipo de respuesta,
#    una interpretación breve y código Python. Ese código se ejecuta en local con
#    exec() sobre el DataFrame df ya cargado en la app, y genera una figura de
#    Plotly. El LLM no recibe los datos directamente para reducir exposición de
#    información, mantener el análisis dentro de la app y controlar mejor qué puede
#    hacer y qué no puede hacer.

# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    En el prompt le indico al modelo qué representa cada fila, qué columnas
#    originales y derivadas existen, qué tipos de preguntas debe resolver, qué
#    formato JSON debe devolver y qué restricciones de código debe respetar.
#    También le digo cómo interpretar preguntas ambiguas, por ejemplo considerar
#    “artista más escuchado” como el de más horas acumuladas. Esto hace que preguntas
#    como “¿Cómo ha evolucionado mi escucha por mes?” funcionen bien gracias a columnas
#    como year_month, hour, is_weekend o season creadas en load_data(). Si quitara
#    esas instrucciones, el modelo tendería a inventar columnas, devolver texto fuera
#    del JSON esperado o generar gráficos menos adecuados.

# 3. EL FLUJO COMPLETO
#    Cuando el usuario escribe una pregunta, Streamlit la recoge y construye el
#    system prompt con información dinámica del dataset. Después se envía a OpenAI
#    junto con la pregunta del usuario. La respuesta del modelo llega como texto JSON,
#    se parsea con json.loads() y se revisa el campo tipo. Si es “fuera_de_alcance”,
#    la app muestra solo una explicación controlada. Si es “grafico”, se ejecuta el
#    código generado con acceso a df, pd, px y go. Ese código crea una figura fig,
#    Streamlit la renderiza en pantalla y después muestra también la interpretación
#    y el código generado para que el comportamiento sea transparente.