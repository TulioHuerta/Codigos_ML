"""
================================================================================
 MINERÍA DE PATRONES Y ESTRUCTURAS DE CLUSTERING — Dataset Eco-Acústico
 Métodos: Gaussian Mixture Models (GMM) y DBSCAN
 Adaptado para: eco_acoustic_train.csv (X in R^64)
================================================================================
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════════
# 0. ESTILO VISUAL ACADÉMICO (tesis / artículo)
#    Mismo bloque de configuración usado en el resto de los scripts del
#    proyecto: serif tipo Times, math en STIX, alto DPI de exportación, y
#    TODO tamaño de fuente en ejes/leyendas >= 14 (regla del curso, penaliza
#    -3.0 pts si algún eje o leyenda queda por debajo de ese umbral).
# ══════════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.color': '#D9D9D9',
    'grid.linewidth': 0.6,
    'grid.linestyle': ':',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.linewidth': 0.9,
    'lines.antialiased': True,
    'patch.antialiased': True,
    'text.antialiased': True,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 150,       # calidad en pantalla
    'savefig.dpi': 600,      # calidad al exportar (300-600 recomendado para tesis)
})

RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN DE DATOS (una sola vez, reutilizada en las 3 etapas)
# ══════════════════════════════════════════════════════════════════════════
print("1. Cargando y preprocesando datos...")
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'eco_acoustic_train.csv'))

X = df.loc[:, 'mel_0':'mel_63']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Shape: {X.shape}")


# ══════════════════════════════════════════════════════════════════════════
# 2. OPTIMIZACIÓN DEL MODELO PROBABILÍSTICO (GMM)
#    Objetivo: determinar el número óptimo de componentes Gaussianos (K) que
#    mejor representen la distribución del hiperespacio acústico R^64,
#    usando el Coeficiente de Silhouette como métrica de validación interna.
# ══════════════════════════════════════════════════════════════════════════
print("\n2. Optimizando GMM (Coeficiente de Silhouette vs. K)...")

k_values = range(2, 10)
silhouette_scores_gmm = []

for k in k_values:
    gmm_k = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
    labels_k = gmm_k.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_k)
    silhouette_scores_gmm.append(score)
    print(f"  K={k} -> Silhouette: {score:.3f}")

mejor_k = k_values[int(np.argmax(silhouette_scores_gmm))]
mejor_score_gmm = max(silhouette_scores_gmm)
print(f"El número óptimo de clústeres es K = {mejor_k} con un Silhouette de {mejor_score_gmm:.3f}")

fig_gmm, ax_gmm = plt.subplots(figsize=(10, 6))
ax_gmm.plot(k_values, silhouette_scores_gmm, marker='o', linestyle='-',
            color='#1f77b4', linewidth=2.5, markersize=8)
ax_gmm.axvline(mejor_k, color='#E24B4A', ls='--', lw=1.2, alpha=0.8,
               label=f'K óptimo = {mejor_k}')
ax_gmm.set_title('GMM: Coeficiente de Silhouette vs. Número de Clústeres', pad=15)
ax_gmm.set_xlabel('Número de componentes Gaussianos (K)')
ax_gmm.set_ylabel('Puntaje de Silhouette')
ax_gmm.set_xticks(list(k_values))
ax_gmm.legend()

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'silhouette_gmm.png'), bbox_inches='tight', facecolor='white')
plt.close(fig_gmm)


# ══════════════════════════════════════════════════════════════════════════
# 3. OPTIMIZACIÓN BASADA EN DENSIDAD (DBSCAN)
#    Objetivo: explorar mediante grid search los hiperparámetros de vecindad
#    (epsilon) y densidad mínima (MinPts) para identificar clústeres
#    orgánicos y aislar el ruido espacial. Se visualiza la convergencia
#    mediante un mapa de calor del Coeficiente de Silhouette.
# ══════════════════════════════════════════════════════════════════════════
print("\n3. Optimizando DBSCAN (Grid Search: epsilon x MinPts)...")

eps_values = np.arange(3.0, 9.0, 1.0)
min_samples_values = list(range(3, 12, 2))

results_dbscan = np.zeros((len(eps_values), len(min_samples_values)))

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels_db = db.fit_predict(X_scaled)

        n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)

        if n_clusters > 1:
            score = silhouette_score(X_scaled, labels_db)
            results_dbscan[i, j] = score
        else:
            results_dbscan[i, j] = 0.0  # 0 si no logra formar grupos válidos

# Mejor combinación (eps, min_samples) según el Silhouette máximo
idx_best = np.unravel_index(np.argmax(results_dbscan), results_dbscan.shape)
mejor_eps = eps_values[idx_best[0]]
mejor_min_samples = min_samples_values[idx_best[1]]
mejor_score_dbscan = results_dbscan[idx_best]
print(f"Mejor combinación DBSCAN: eps={mejor_eps:.1f}, min_samples={mejor_min_samples} "
      f"-> Silhouette: {mejor_score_dbscan:.3f}")

fig_dbscan, ax_dbscan = plt.subplots(figsize=(10, 6))
sns.heatmap(results_dbscan, annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=min_samples_values, yticklabels=[f'{e:.1f}' for e in eps_values],
            annot_kws={"size": 14}, ax=ax_dbscan, cbar_kws={'label': 'Silhouette'})

ax_dbscan.set_title('DBSCAN: Silhouette Score en función de Epsilon y MinPts', pad=15)
ax_dbscan.set_xlabel('Número mínimo de puntos (MinPts)')
ax_dbscan.set_ylabel('Radio de vecindad (Epsilon)')
ax_dbscan.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'silhouette_dbscan.png'), bbox_inches='tight', facecolor='white')
plt.close(fig_dbscan)


# ══════════════════════════════════════════════════════════════════════════
# 4. COMPARATIVA FINAL: GMM vs. DBSCAN (proyección PCA 2D)
#    Se entrenan ambos modelos con sus hiperparámetros óptimos (obtenidos
#    en las secciones 2 y 3) y se comparan visualmente sobre la misma
#    proyección PCA 2D.
# ══════════════════════════════════════════════════════════════════════════
print("\n4. Generando comparativa final GMM vs. DBSCAN...")

gmm = GaussianMixture(n_components=mejor_k, random_state=RANDOM_STATE)
gmm_labels = gmm.fit_predict(X_scaled)

dbscan = DBSCAN(eps=mejor_eps, min_samples=mejor_min_samples)
dbscan_labels = dbscan.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

# Paletas de colores (tab10, consistente con el resto del proyecto)
palette_gmm = sns.color_palette("tab10", n_colors=mejor_k)
n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
palette_dbscan = sns.color_palette("tab10", n_colors=n_dbscan_clusters)
color_dict_dbscan = {i: palette_dbscan[i] for i in range(n_dbscan_clusters)}
color_dict_dbscan[-1] = (0.5, 0.5, 0.5, 0.5)  # Gris semitransparente para el ruido

# Gráfica 1: GMM
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=gmm_labels, palette=palette_gmm,
                ax=axes[0], s=60, alpha=0.8, legend='full')
axes[0].set_title(f'Asignación de Clústeres: GMM (K={mejor_k})', pad=15)
axes[0].set_xlabel('Componente Principal 1')
axes[0].set_ylabel('Componente Principal 2')
axes[0].legend(title="Clúster GMM", title_fontsize=14)

# Gráfica 2: DBSCAN
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette=color_dict_dbscan,
                ax=axes[1], s=60, alpha=0.8, legend='full')
axes[1].set_title(f'Asignación de Clústeres: DBSCAN (eps={mejor_eps:.1f})', pad=15)
axes[1].set_xlabel('Componente Principal 1')
axes[1].legend(title="Clúster DBSCAN\n(-1 = Ruido)", title_fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'comparativa_clusters.png'), bbox_inches='tight', facecolor='white')
plt.close(fig)

print("\n✓ Proceso completo. Figuras guardadas: silhouette_gmm.png, "
      "silhouette_dbscan.png, comparativa_clusters.png")