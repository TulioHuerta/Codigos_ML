from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelos"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"


st.set_page_config(
    page_title="Clasificacion de Senales Eco-Acusticas",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #0b1424;
        --bg-soft: #111c2e;
        --panel: #1b2a3d;
        --panel-2: #213247;
        --line: #31445e;
        --ink: #f8fafc;
        --muted: #9fb3ca;
        --brand: #3b82f6;
        --brand-2: #60a5fa;
        --green: #10b981;
        --yellow: #f59e0b;
        --red: #ef4444;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--ink);
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(11, 20, 36, 0.82);
    }
    [data-testid="stSidebar"] {
        background: #1f2430;
        border-right: 1px solid #2e3c52;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1360px;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
        font-weight: 800;
    }
    h1 {
        color: var(--brand-2);
        font-size: 3rem !important;
        line-height: 1.08;
    }
    h2, h3 {
        color: #f8fafc;
    }
    p, li, span, label, div {
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
        font-size: 1rem;
    }
    div[data-testid="stTabs"] [role="tablist"] {
        background: var(--panel);
        border: 1px solid #263a54;
        border-radius: 14px;
        padding: 0.45rem;
        gap: 0.3rem;
    }
    div[data-testid="stTabs"] [role="tab"] {
        color: #8fa6c1;
        border-radius: 10px;
        padding: 0.65rem 1.35rem;
        font-weight: 700;
    }
    div[data-testid="stTabs"] [aria-selected="true"] {
        background: var(--brand);
        color: #ffffff !important;
    }
    div[data-testid="stTabs"] [aria-selected="true"] p {
        color: #ffffff !important;
    }
    div[data-testid="stTabs"] [role="tab"] p {
        font-size: 1rem;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        color: #f8fafc;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #f8fafc !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700;
        color: var(--brand-2) !important;
    }
    .status-box {
        border-radius: 12px;
        padding: 1rem 1.1rem;
        border: 1px solid #334155;
        margin: 0.5rem 0 1rem 0;
        background: var(--panel);
        color: #f8fafc;
    }
    .status-green {
        background: #062d23;
        border-color: #10b981;
        color: #d1fae5;
    }
    .status-yellow {
        background: #332306;
        border-color: #f59e0b;
        color: #fef3c7;
    }
    .status-red {
        background: #3b1111;
        border-color: #ef4444;
        color: #fee2e2;
    }
    .small-muted {
        color: var(--muted);
        font-size: 0.92rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    div[role="radiogroup"] label {
        background: #111c2e;
        border: 1px solid #31445e;
        border-radius: 999px;
        padding: 0.3rem 0.7rem;
        margin-right: 0.35rem;
    }
    div[role="radiogroup"] label:hover {
        border-color: var(--brand);
        color: var(--brand-2);
    }
    img {
        border-radius: 12px;
    }
    hr {
        border-color: #263a54;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(resolve_data_path(name))


@st.cache_resource(show_spinner=False)
def load_pickle(name: str):
    with open(MODEL_DIR / name, "rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    model = load_pickle("modelo_final.pkl")
    scaler = load_pickle("scaler.pkl")
    metadata = load_pickle("metadata.pkl")
    return model, scaler, metadata


def read_optional_csv(name: str) -> pd.DataFrame | None:
    path = resolve_data_path(name, required=False)
    if path.exists():
        return pd.read_csv(path)
    return None


def resolve_data_path(name: str, required: bool = True) -> Path:
    candidates = [
        DATA_DIR / name,
        BASE_DIR / name,
        BASE_DIR.parent / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    if required:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No se encontro {name}. Rutas revisadas: {searched}")
    return candidates[0]


def image_path(*parts: str) -> Path:
    candidates = [
        ASSETS_DIR.joinpath(*parts),
        BASE_DIR.joinpath(*parts),
        BASE_DIR.parent.joinpath(*parts),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def show_image(path: Path, caption: str | None = None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"No se encontro la figura: {path.name}")


def feature_columns(metadata: dict, df: pd.DataFrame) -> list[str]:
    features = metadata.get("features")
    if features:
        return list(features)
    return [col for col in df.columns if col.startswith("mel_")]


def species_label(class_id: int | str, metadata: dict) -> str:
    try:
        numeric = int(class_id)
    except (TypeError, ValueError):
        return str(class_id)

    species_names = metadata.get("species_names", {})
    name = species_names.get(numeric, species_names.get(str(numeric)))
    if name:
        return f"{numeric} - {name}"
    return str(numeric)


def predict_row(row: pd.Series, model, scaler, metadata: dict) -> dict:
    features = feature_columns(metadata, pd.DataFrame([row]))
    x_raw = row[features].to_numpy(dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x_raw)

    start = time.perf_counter()
    probabilities = model.predict_proba(x_scaled)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    model_classes = getattr(model, "classes_", np.arange(len(probabilities)))
    best_position = int(np.argmax(probabilities))
    predicted_internal = model_classes[best_position]

    inverse_mapping = metadata.get("inverse_mapping", {})
    predicted_species = inverse_mapping.get(predicted_internal)
    if predicted_species is None:
        predicted_species = inverse_mapping.get(str(predicted_internal), predicted_internal)

    probability_table = pd.DataFrame(
        {
            "Clase interna": model_classes,
            "species_id": [
                inverse_mapping.get(cls, inverse_mapping.get(str(cls), cls))
                for cls in model_classes
            ],
            "Probabilidad": probabilities,
        }
    )
    probability_table["Etiqueta"] = probability_table["species_id"].apply(
        lambda value: species_label(value, metadata)
    )

    return {
        "predicted_species": predicted_species,
        "predicted_label": species_label(predicted_species, metadata),
        "max_probability": float(probabilities[best_position]),
        "probability_table": probability_table.sort_values(
            "Probabilidad", ascending=False
        ),
        "elapsed_ms": elapsed_ms,
    }


@st.cache_data(show_spinner=False)
def compute_scenarios(test_df: pd.DataFrame, metadata: dict) -> dict[str, int]:
    model, scaler, _ = load_model_bundle()
    features = feature_columns(metadata, test_df)
    x_scaled = scaler.transform(test_df[features].to_numpy(dtype=float))
    probabilities = model.predict_proba(x_scaled)
    max_probs = probabilities.max(axis=1)

    scenarios = {
        "Alta confianza": int(np.argmax(max_probs)),
        "Incertidumbre": int(np.argmin(np.abs(max_probs - 0.60))),
        "Baja confianza": int(np.argmin(max_probs)),
    }
    return scenarios


def policy_for_probability(probability: float, metadata: dict) -> tuple[str, str, str]:
    thresholds = metadata.get("thresholds", {"confianza": 0.85, "incertidumbre": 0.40})
    confidence = float(thresholds.get("confianza", 0.85))
    uncertainty = float(thresholds.get("incertidumbre", 0.40))

    if probability >= confidence:
        return (
            "Zona de confianza",
            "Clasificacion automatica con alta fiabilidad operativa.",
            "green",
        )
    if probability >= uncertainty:
        return (
            "Zona de incertidumbre",
            "Clasificacion asistida; el registro debe pasar a auditoria humana.",
            "yellow",
        )
    return (
        "Zona de rechazo",
        "Descarte automatico para mitigar ruido ambiental o evidencia debil.",
        "red",
    )


def render_probability_chart(probability_table: pd.DataFrame):
    fig = px.bar(
        probability_table.sort_values("Probabilidad"),
        x="Probabilidad",
        y="Etiqueta",
        orientation="h",
        text=probability_table.sort_values("Probabilidad")["Probabilidad"].map(
            lambda value: f"{value:.1%}"
        ),
        color="Probabilidad",
        color_continuous_scale=["#D64545", "#F2B84B", "#2F855A"],
        range_x=[0, 1],
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Probabilidad predictiva",
        yaxis_title="",
        coloraxis_showscale=False,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)


def render_architecture():
    dot = """
    digraph {
        graph [
            rankdir=LR,
            bgcolor="transparent",
            pad=0.25,
            nodesep=0.45,
            ranksep=0.7
        ]
        node [
            shape=box,
            style="rounded,filled",
            color="#93a4b5",
            penwidth=1.8,
            fillcolor="#f1f5f9",
            fontname="Arial",
            fontsize=18,
            margin="0.28,0.18"
        ]
        edge [
            color="#94a3b8",
            penwidth=2.2,
            arrowsize=1.0,
            fontname="Arial",
            fontsize=13
        ]

        data [label="Datos\\neco-acusticos\\nX en R64", fillcolor="#dbeafe"]
        prep [label="Preparacion\\ny escalamiento", fillcolor="#e0f2fe"]
        analysis [label="Analisis\\nPCA, UMAP\\ny clustering", fillcolor="#dcfce7"]
        model [label="Modelado\\nMLP vs ensamble", fillcolor="#f5f3ff"]
        app [label="Aplicacion\\nStreamlit", fillcolor="#ede9fe"]
        decision [label="Decision\\naceptar, auditar\\no rechazar", fillcolor="#ffedd5"]

        data -> prep
        prep -> analysis
        analysis -> model
        model -> app
        app -> decision
    }
    """
    st.graphviz_chart(dot, use_container_width=True)


try:
    model, scaler, metadata = load_model_bundle()
    train_df = load_csv("eco_acoustic_train.csv")
    test_df = load_csv("eco_acoustic_test.csv")
    app_ready = True
except Exception as exc:
    model, scaler, metadata = None, None, {}
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    app_ready = False
    st.error(
        "No se pudieron cargar los artefactos principales. "
        f"Detalle tecnico: {exc}"
    )


st.title("Clasificacion de Senales Eco-Acusticas")
st.caption("Grupo 10 | Pipeline de Machine Learning para deteccion automatizada de especies a partir de vectores Mel en R64.")

with st.sidebar:
    st.header("Artefactos")
    st.write("Modelo:", "MLP")
    st.write("F1 macro:", "0.4704")
    st.write("Train:", f"{len(train_df):,}" if app_ready else "No disponible")
    st.write("Test:", f"{len(test_df):,}" if app_ready else "No disponible")

    st.divider()
    st.header("Umbrales")
    thresholds = metadata.get("thresholds", {"confianza": 0.85, "incertidumbre": 0.40})
    st.write("Confianza:", f"{float(thresholds.get('confianza', 0.85)):.0%}")
    st.write("Incertidumbre:", f"{float(thresholds.get('incertidumbre', 0.40)):.0%}")


tabs = st.tabs(
    [
        "Dataset",
        "Reduccion dimensional",
        "Clustering",
        "Clasificacion",
        "Inferencia",
        "Datos",
    ]
)


with tabs[0]:
    st.subheader("Descripcion del dataset")

    col_a, col_b, col_c, col_d = st.columns(4)
    features = feature_columns(metadata, train_df) if app_ready else []
    col_a.metric("Observaciones train", f"{len(train_df):,}" if app_ready else "-")
    col_b.metric("Observaciones test", f"{len(test_df):,}" if app_ready else "-")
    col_c.metric("Dimensiones Mel", len(features) if features else "-")
    col_d.metric("Clases", len(metadata.get("original_classes", [])) or "-")

    st.markdown(
        """
        <div class="status-box" style="background:#111c2a;border-color:#334155;color:#f8fafc;">
            <strong>Contexto de los datos</strong><br>
            El dataset esta compuesto por registros eco-acusticos tabulares obtenidos a partir de audios biologicos.
            Cada observacion representa una muestra acustica descrita por 64 coeficientes Mel, los cuales resumen
            propiedades de frecuencia y timbre de la senal. La variable objetivo es <strong>species_id</strong>,
            que identifica la especie detectada en el registro.
        </div>
        """,
        unsafe_allow_html=True,
    )

    var_rows = pd.DataFrame(
        [
            {
                "Grupo": "Identificacion",
                "Columnas": "recording_id",
                "Descripcion": "Identificador unico del registro de audio original.",
            },
            {
                "Grupo": "Metadatos",
                "Columnas": "songtype_id, is_tp",
                "Descripcion": "Tipo de canto y marca tecnica de validacion de la senal.",
            },
            {
                "Grupo": "Variable objetivo",
                "Columnas": "species_id",
                "Descripcion": "Etiqueta numerica de la especie faunistica detectada.",
            },
            {
                "Grupo": "Predictores",
                "Columnas": "mel_0 a mel_63",
                "Descripcion": "64 caracteristicas numericas continuas que forman el espacio X en R64.",
            },
        ]
    )
    st.write("Estructura de variables")
    st.dataframe(var_rows, use_container_width=True, hide_index=True)

    taxonomy = pd.DataFrame(
        [
            {"species_id": 10, "Nombre cientifico": "Leptodactylus discodactylus", "Tipo de fauna": "Anfibio"},
            {"species_id": 12, "Nombre cientifico": "Osteocephalus taurinus", "Tipo de fauna": "Anfibio"},
            {"species_id": 17, "Nombre cientifico": "Chiroxiphia lineata", "Tipo de fauna": "Ave"},
            {"species_id": 18, "Nombre cientifico": "Saltator grossus", "Tipo de fauna": "Ave"},
            {"species_id": 23, "Nombre cientifico": "Pheucticus chrysopeplus", "Tipo de fauna": "Ave"},
        ]
    )
    st.write("Catalogo taxonomico de clases objetivo")
    st.dataframe(taxonomy, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="status-box" style="background:#111c2a;border-color:#334155;color:#f8fafc;">
            <strong>Uso analitico</strong><br>
            Las variables Mel se emplean como entrada para reduccion dimensional, clustering y clasificacion.
            La particion de entrenamiento permite ajustar los modelos, mientras que la particion de prueba se usa
            para evaluar generalizacion y simular inferencia dentro de la aplicacion.
        </div>
        """
        ,
        unsafe_allow_html=True,
    )

    if app_ready:
        class_counts = (
            train_df["species_id"]
            .value_counts()
            .rename_axis("species_id")
            .reset_index(name="Conteo")
            .sort_values("species_id")
        )
        class_counts["Etiqueta"] = class_counts["species_id"].apply(
            lambda value: species_label(value, metadata)
        )
        fig = px.bar(
            class_counts,
            x="Etiqueta",
            y="Conteo",
            color="Etiqueta",
            title="Distribucion de clases en entrenamiento",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Registros")
        st.plotly_chart(fig, use_container_width=True)


with tabs[1]:
    st.subheader("Exploracion geometrica y reduccion de dimensionalidad")

    reduction_results = read_optional_csv("resultados_reduccion_dimensionalidad.csv")
    if reduction_results is not None:
        st.dataframe(reduction_results, use_container_width=True, hide_index=True)

    view = st.radio(
        "Figura",
        [
            "PCA",
            "t-SNE",
            "UMAP",
            "Comparativa 2D",
            "Comparativa 3D",
        ],
        index=0,
        horizontal=True,
        key="dimensionality_figure_selector",
    )
    image_map = {
        "PCA": image_path("img", "pca_eco_acoustic.png"),
        "t-SNE": image_path("img", "tsne_variando_perplexity.png"),
        "UMAP": image_path("img", "umap_variando_n_neighbors.png"),
        "Comparativa 2D": image_path("img", "comparativa_final_2d.png"),
        "Comparativa 3D": image_path("img", "comparativa_final_3d.png"),
    }
    show_image(image_map[view])

    explanations = {
        "PCA": (
            "Interpretacion de PCA",
            "Esta figura proyecta las 64 caracteristicas Mel en espacios de 2 y 3 componentes principales. "
            "La curva de varianza acumulada indica cuanta informacion global retiene cada numero de componentes. "
            "Se observa solapamiento entre especies, por lo que PCA resume bien la estructura global, pero no separa completamente las clases.",
        ),
        "t-SNE": (
            "Interpretacion de t-SNE",
            "Esta figura compara distintas perplexities en 2D y 3D. t-SNE enfatiza vecindades locales y permite observar regiones cercanas que PCA puede mezclar. "
            "Como la forma global cambia con el hiperparametro, se interpreta como herramienta visual y no como prueba unica de separacion.",
        ),
        "UMAP": (
            "Interpretacion de UMAP",
            "Esta figura evalua diferentes valores de n_neighbors. Valores pequenos priorizan estructura local; valores grandes conservan una vision mas global del espacio. "
            "La comparacion permite justificar el hiperparametro usado para visualizar las especies en baja dimension.",
        ),
        "Comparativa 2D": (
            "Interpretacion de la comparativa 2D",
            "Esta figura contrasta PCA, t-SNE y UMAP con salida bidimensional. Permite comparar separacion visual, solapamiento entre especies y diferencias entre metodos lineales y no lineales.",
        ),
        "Comparativa 3D": (
            "Interpretacion de la comparativa 3D",
            "Esta figura agrega una tercera dimension para evaluar si la separacion mejora respecto al plano 2D. Sirve para discutir interpretabilidad, costo computacional y preservacion de estructura geometrica.",
        ),
    }
    explanation_title, explanation_body = explanations[view]
    st.markdown(
        f"""
        <div class="status-box" style="background:#111c2a;border-color:#334155;color:#f8fafc;">
            <strong>{explanation_title}</strong><br>
            {explanation_body}
        </div>
        """,
        unsafe_allow_html=True,
    )


with tabs[2]:
    st.subheader("Mineria de patrones y estructuras de clustering")

    clustering_file = MODEL_DIR / "clustering_artifacts.pkl"
    cols = st.columns(4)
    cols[0].metric("Artefacto", "Disponible" if clustering_file.exists() else "No")
    cols[1].metric("Paradigma 1", "DBSCAN")
    cols[2].metric("Paradigma 2", "GMM")
    cols[3].metric("Validacion", "Silhouette / BIC")

    clustering_results = read_optional_csv("resultados_dbscan_sweep_eps.csv")
    if clustering_results is not None:
        with st.expander("Barrido de eps para DBSCAN", expanded=False):
            st.dataframe(clustering_results, use_container_width=True, hide_index=True)

    c_view = st.radio(
        "Analisis",
        ["DBSCAN eps", "GMM k", "Comparativa", "Tablas cruzadas"],
        index=2,
        horizontal=True,
        key="clustering_analysis_selector",
    )
    c_images = {
        "DBSCAN eps": image_path("img2", "dbscan_sweep_eps.png"),
        "GMM k": image_path("img2", "gmm_seleccion_k.png"),
        "Comparativa": image_path("img2", "comparativa_clustering.png"),
        "Tablas cruzadas": image_path("img2", "tablas_cruzadas_clustering.png"),
    }
    if c_view == "DBSCAN eps":
        st.write("Mapa de Silhouette para seleccion de hiperparametros")
        show_image(image_path("img2", "dbs_2.png"))
        st.write("Sensibilidad de DBSCAN frente al parametro eps")
        show_image(c_images[c_view])
    elif c_view == "Comparativa":
        st.write("Comparativa sobre proyeccion UMAP 2D")
        show_image(c_images[c_view])
        st.write("Comparativa complementaria sobre proyeccion PCA 2D")
        show_image(image_path("img2", "comparativa_2.png"))
    else:
        show_image(c_images[c_view])

    clustering_explanations = {
        "DBSCAN eps": (
            "Interpretacion de DBSCAN eps",
            "El mapa de Silhouette permite comparar combinaciones de eps y MinPts para identificar zonas con mejor cohesion y separacion de clusters. "
            "La grafica de sensibilidad muestra como cambia el numero de clusters y el porcentaje de puntos marcados como ruido al variar eps. "
            "En conjunto, ambas figuras justifican la seleccion de hiperparametros y evidencian que DBSCAN es sensible a la densidad del espacio.",
        ),
        "GMM k": (
            "Interpretacion de GMM k",
            "Esta figura evalua distintos numeros de componentes para Gaussian Mixture Models. BIC y AIC miden el ajuste penalizando la complejidad, Silhouette evalua cohesion y separacion, y Davies-Bouldin se interpreta mejor cuando es menor. "
            "La linea de k=5 sirve como referencia biologica porque existen cinco especies reales en el dataset.",
        ),
        "Comparativa": (
            "Interpretacion de la comparativa de clustering",
            "Esta figura compara DBSCAN, GMM y las etiquetas reales sobre una proyeccion UMAP 2D. DBSCAN permite identificar ruido y regiones densas, mientras que GMM reparte los datos en componentes probabilisticos. "
            "La comparacion complementaria en PCA 2D permite observar el comportamiento de ambos paradigmas en un espacio lineal de referencia. "
            "En conjunto, ambas vistas muestran que los clusters descubiertos no coinciden perfectamente con las especies, lo cual evidencia solapamiento acustico entre clases.",
        ),
        "Tablas cruzadas": (
            "Interpretacion de las tablas cruzadas",
            "Estas matrices cruzan los clusters descubiertos con los species_id reales. Las concentraciones altas en una celda indican que un cluster agrupa muchos registros de una especie; valores distribuidos en varias columnas indican mezcla entre especies. "
            "Sirven para interpretar los clusters despues del entrenamiento, aunque la seleccion principal debe basarse en metricas internas.",
        ),
    }
    clustering_title, clustering_body = clustering_explanations[c_view]
    st.markdown(
        f"""
        <div class="status-box" style="background:#111c2a;border-color:#334155;color:#f8fafc;">
            <strong>{clustering_title}</strong><br>
            {clustering_body}
        </div>
        """,
        unsafe_allow_html=True,
    )


with tabs[3]:
    st.subheader("Arquitectura de clasificacion: MLP vs modelos de ensamble")

    model_summary = pd.DataFrame(
        [
            {
                "Modelo": "Random Forest",
                "F1_macro": 0.4335,
                "F1_weighted": 0.4466,
                "Tiempo_entrenamiento_s": 1.1000,
                "Tiempo_inferencia_test_s": 0.0136,
            },
            {
                "Modelo": "Gradient Boosting",
                "F1_macro": 0.4400,
                "F1_weighted": 0.4614,
                "Tiempo_entrenamiento_s": 2.3500,
                "Tiempo_inferencia_test_s": 0.0200,
            },
            {
                "Modelo": "MLP",
                "F1_macro": 0.4704,
                "F1_weighted": 0.4560,
                "Tiempo_entrenamiento_s": 4.5200,
                "Tiempo_inferencia_test_s": 0.0089,
            },
        ]
    )
    st.dataframe(model_summary, use_container_width=True, hide_index=True)

    show_image(
        image_path("img3", "mlp_loss_curves.png"),
        "Curvas de aprendizaje del MLP",
    )

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        show_image(image_path("img3", "cm_random_forest.png"), "Random Forest")
    with col_b:
        show_image(image_path("img3", "cm_gradient_boosting.png"), "Gradient Boosting")
    with col_c:
        show_image(image_path("img3", "cm_mlp.png"), "MLP")

    st.markdown(
        """
        <div class="status-box" style="background:#111c2a;border-color:#334155;color:#f8fafc;">
            <strong>Interpretacion de la clasificacion supervisada</strong><br>
            Las curvas de aprendizaje permiten evaluar la estabilidad del MLP durante el entrenamiento y observar si la perdida disminuye de forma consistente. 
            Las matrices de confusion comparan las predicciones de Random Forest, Gradient Boosting y MLP contra las especies reales. 
            La diagonal principal representa aciertos; los valores fuera de la diagonal muestran confusiones entre especies acusticamente similares. 
            La metrica F1 macro se usa como criterio principal porque resume el rendimiento por clase sin favorecer solamente a las especies con mas registros.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tabs[4]:
    st.subheader("Inferencia simulada y mitigacion de riesgos")

    if not app_ready:
        st.stop()

    scenarios = compute_scenarios(test_df, metadata)
    mode = st.radio(
        "Modo de seleccion",
        ["Escenario precargado", "Registro manual del test"],
        horizontal=True,
    )

    if mode == "Escenario precargado":
        scenario_name = st.selectbox("Escenario", list(scenarios.keys()))
        selected_index = scenarios[scenario_name]
    else:
        selected_index = st.number_input(
            "Indice del registro en test",
            min_value=0,
            max_value=len(test_df) - 1,
            value=0,
            step=1,
        )

    row = test_df.iloc[int(selected_index)]
    result = predict_row(row, model, scaler, metadata)
    zone, decision, color = policy_for_probability(result["max_probability"], metadata)

    kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
    kpi_a.metric("Registro", int(selected_index))
    kpi_b.metric("Prediccion", result["predicted_label"])
    kpi_c.metric("Probabilidad maxima", f"{result['max_probability']:.1%}")
    kpi_d.metric("Inferencia", f"{result['elapsed_ms']:.3f} ms")

    st.markdown(
        f"""
        <div class="status-box status-{color}">
            <strong>{zone}</strong><br>
            {decision}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_probs, col_record = st.columns([1.25, 1])
    with col_probs:
        render_probability_chart(result["probability_table"])
    with col_record:
        st.write("Registro evaluado")
        preview_cols = ["recording_id", "species_id", "songtype_id", "is_tp"]
        available = [col for col in preview_cols if col in row.index]
        st.dataframe(row[available].to_frame("valor"), use_container_width=True)
        st.write("Caracteristicas Mel")
        st.line_chart(row[feature_columns(metadata, test_df)].astype(float).reset_index(drop=True))

    st.divider()
    st.subheader("Distribucion de registros por zona de decision")
    st.caption(
        "La grafica resume cuántos registros caen en rechazo, auditoria o clasificacion automatica segun la probabilidad maxima del modelo."
    )
    show_image(image_path("img3", "confidence_zones.png"), "Zonas probabilisticas")


with tabs[5]:
    st.subheader("Datos y artefactos")

    if app_ready:
        dataset_name = st.selectbox("Dataset", ["Train", "Test"])
        df = train_df if dataset_name == "Train" else test_df
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)

        st.write("Resumen numerico de las caracteristicas Mel")
        st.dataframe(
            df[feature_columns(metadata, df)].describe().T,
            use_container_width=True,
        )
