"""
================================================================================
 ANÁLISIS DE REDUCCIÓN DE DIMENSIONALIDAD — Dataset Eco-Acústico
 Métodos: PCA, t-SNE y UMAP (2D y 3D)
 Adaptado para: eco_acoustic_train.csv (X in R^64, target = species_id)
================================================================================
"""

import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import BoundaryNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════════
# 0. ESTILO VISUAL ACADÉMICO (tesis / artículo)
#    Tipografía serif tipo Times, math en STIX, mayor DPI de exportación,
#    y antialiasing consistente para figuras nítidas en LaTeX.
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
    'axes.titlesize': 15,
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

IMG_DIR = 'Proyecto 2/img'
os.makedirs(IMG_DIR, exist_ok=True)

RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════════
df = pd.read_csv('Proyecto 2/eco_acoustic_train.csv')
print(f"Shape: {df.shape}")

mel_cols = [f'mel_{i}' for i in range(64)]
X = df[mel_cols].values
y = df['species_id'].values

X_sc = StandardScaler().fit_transform(X)

clases = np.unique(y)
n_clases = len(clases)

colores = plt.cm.tab10(np.linspace(0, 1, 10))[:n_clases]
cmap = dict(zip(clases, colores))

# Nombres científicos para leyendas (Cuadro 1 de la documentación del dataset)
NOMBRES_ESPECIES = {
    10: 'L. discodactylus',
    12: 'O. taurinus',
    17: 'C. lineata',
    18: 'S. grossus',
    23: 'P. chrysopeplus',
}

# Tabla de resultados cuantitativos (tiempo, varianza, trustworthiness)
resultados = []


def registrar(metodo, params, t_seg, var_ret=None, trust_5=None, trust_10=None):
    resultados.append({
        'Metodo': metodo,
        'Parametros': params,
        'Tiempo_s': round(t_seg, 2),
        'Varianza_retenida_%': round(var_ret, 1) if var_ret is not None else '-',
        'Trustworthiness_k5': round(trust_5, 3) if trust_5 is not None else '-',
        'Trustworthiness_k10': round(trust_10, 3) if trust_10 is not None else '-',
    })


# ══════════════════════════════════════════════════════════════════════════
# 2. PCA (2D y 3D)
# ══════════════════════════════════════════════════════════════════════════
print("\n--- Ejecutando PCA ---")

t0 = time.time()
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
Z_pca_2d = pca_2d.fit_transform(X_sc)
t_pca_2d = time.time() - t0
var_2d = pca_2d.explained_variance_ratio_
var_ret_2d = sum(var_2d) * 100
trust_pca_2d_5 = trustworthiness(X_sc, Z_pca_2d, n_neighbors=5)
trust_pca_2d_10 = trustworthiness(X_sc, Z_pca_2d, n_neighbors=10)
print(f"PCA 2D — Varianza retenida: {var_ret_2d:.1f}% | Tiempo: {t_pca_2d:.2f}s | "
      f"Trust(k=5): {trust_pca_2d_5:.3f}")
registrar('PCA', '2D', t_pca_2d, var_ret_2d, trust_pca_2d_5, trust_pca_2d_10)

t0 = time.time()
pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
Z_pca_3d = pca_3d.fit_transform(X_sc)
t_pca_3d = time.time() - t0
var_3d = pca_3d.explained_variance_ratio_
var_ret_3d = sum(var_3d) * 100
trust_pca_3d_5 = trustworthiness(X_sc, Z_pca_3d, n_neighbors=5)
trust_pca_3d_10 = trustworthiness(X_sc, Z_pca_3d, n_neighbors=10)
print(f"PCA 3D — Varianza retenida: {var_ret_3d:.1f}% | Tiempo: {t_pca_3d:.2f}s")
registrar('PCA', '3D', t_pca_3d, var_ret_3d, trust_pca_3d_5, trust_pca_3d_10)


# ══════════════════════════════════════════════════════════════════════════
# 3. t-SNE (2D y 3D) — se varía perplexity
# ══════════════════════════════════════════════════════════════════════════
print("\n--- Ejecutando t-SNE ---")

perplexities = [5, 10, 30, 50]
PERP_RECOMENDADO = 30

best_Z_tsne_2d = None
best_Z_tsne_3d = None

fig_tsne = plt.figure(figsize=(18, 9), num="t-SNE 2D vs 3D")
plt.suptitle('t-SNE: Variando Perplexity — Dataset Eco-Acústico', fontsize=16, fontweight='bold')

for i, p in enumerate(perplexities):
    t0 = time.time()
    tsne_2d = TSNE(n_components=2, perplexity=p, max_iter=1000, init='pca', random_state=RANDOM_STATE)
    Z_2d = tsne_2d.fit_transform(X_sc)
    t_2d = time.time() - t0

    t0 = time.time()
    tsne_3d = TSNE(n_components=3, perplexity=p, max_iter=1000, init='pca', random_state=RANDOM_STATE)
    Z_3d = tsne_3d.fit_transform(X_sc)
    t_3d = time.time() - t0
    print(f"t-SNE (perp={p}) — 2D: {t_2d:.2f}s | 3D: {t_3d:.2f}s")

    trust5 = trustworthiness(X_sc, Z_2d, n_neighbors=5)
    trust10 = trustworthiness(X_sc, Z_2d, n_neighbors=10)
    registrar('t-SNE', f'2D, perplexity={p}', t_2d, None, trust5, trust10)
    registrar('t-SNE', f'3D, perplexity={p}', t_3d, None,
              trustworthiness(X_sc, Z_3d, n_neighbors=5),
              trustworthiness(X_sc, Z_3d, n_neighbors=10))

    if p == PERP_RECOMENDADO:
        best_Z_tsne_2d = Z_2d
        best_Z_tsne_3d = Z_3d

    ax_2d = fig_tsne.add_subplot(2, 4, i + 1)
    ax_2d.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap='tab10', s=15, alpha=0.8)
    ax_2d.set_title(f't-SNE 2D (perp={p})', fontsize=14)
    ax_2d.set_xticks([])
    ax_2d.set_yticks([])

    ax_3d = fig_tsne.add_subplot(2, 4, i + 5, projection='3d')
    ax_3d.scatter(Z_3d[:, 0], Z_3d[:, 1], Z_3d[:, 2], c=y, cmap='tab10', s=15, alpha=0.8)
    ax_3d.set_title(f't-SNE 3D (perp={p})', fontsize=14)
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/tsne_variando_perplexity.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_tsne)


# ══════════════════════════════════════════════════════════════════════════
# 4. UMAP (2D y 3D) — se varía n_neighbors
# ══════════════════════════════════════════════════════════════════════════
print("\n--- Ejecutando UMAP ---")

n_neighbors_list = [5, 15, 50]
dist_min = 0.1
N_NEIGHBORS_RECOMENDADO = 15

best_Z_umap_2d = None
best_Z_umap_3d = None

fig_umap = plt.figure(figsize=(16, 9), num="UMAP 2D vs 3D")
plt.suptitle(f'UMAP: Variando n_neighbors (min_dist={dist_min}) — Dataset Eco-Acústico',
             fontsize=16, fontweight='bold')

for i, vecinos in enumerate(n_neighbors_list):
    t0 = time.time()
    umap_2d = umap.UMAP(n_components=2, n_neighbors=vecinos, min_dist=dist_min, random_state=RANDOM_STATE)
    Z_u2d = umap_2d.fit_transform(X_sc)
    t_2d = time.time() - t0

    t0 = time.time()
    umap_3d = umap.UMAP(n_components=3, n_neighbors=vecinos, min_dist=dist_min, random_state=RANDOM_STATE)
    Z_u3d = umap_3d.fit_transform(X_sc)
    t_3d = time.time() - t0
    print(f"UMAP (n_neighbors={vecinos}) — 2D: {t_2d:.2f}s | 3D: {t_3d:.2f}s")

    registrar('UMAP', f'2D, n_neighbors={vecinos}', t_2d, None,
              trustworthiness(X_sc, Z_u2d, n_neighbors=5),
              trustworthiness(X_sc, Z_u2d, n_neighbors=10))
    registrar('UMAP', f'3D, n_neighbors={vecinos}', t_3d, None,
              trustworthiness(X_sc, Z_u3d, n_neighbors=5),
              trustworthiness(X_sc, Z_u3d, n_neighbors=10))

    if vecinos == N_NEIGHBORS_RECOMENDADO:
        best_Z_umap_2d = Z_u2d
        best_Z_umap_3d = Z_u3d

    ax_2d = fig_umap.add_subplot(2, 3, i + 1)
    ax_2d.scatter(Z_u2d[:, 0], Z_u2d[:, 1], c=y, cmap='tab10', s=15, alpha=0.8)
    ax_2d.set_title(f'UMAP 2D (n={vecinos})', fontsize=14)
    ax_2d.set_xticks([])
    ax_2d.set_yticks([])

    ax_3d = fig_umap.add_subplot(2, 3, i + 4, projection='3d')
    ax_3d.scatter(Z_u3d[:, 0], Z_u3d[:, 1], Z_u3d[:, 2], c=y, cmap='tab10', s=15, alpha=0.8)
    ax_3d.set_title(f'UMAP 3D (n={vecinos})', fontsize=14)
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/umap_variando_n_neighbors.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_umap)


# ══════════════════════════════════════════════════════════════════════════
# 4.1 DEMOSTRACIÓN DE transform() EN UMAP SOBRE UN REGISTRO NUEVO DEL TEST SET
#     A diferencia de t-SNE (que solo tiene fit_transform), UMAP permite
#     proyectar datos NUEVOS sin reentrenar el modelo. Relevante para el
#     pipeline de inferencia en Streamlit: un audio nuevo se proyecta con
#     el reducer ya entrenado.
# ══════════════════════════════════════════════════════════════════════════
print("\n--- Demostrando UMAP.transform() sobre un registro del set de prueba ---")

df_test = pd.read_csv('Proyecto 2/eco_acoustic_test.csv')
X_test = df_test[mel_cols].values

scaler = StandardScaler().fit(X)
reducer_umap = umap.UMAP(n_components=2, n_neighbors=N_NEIGHBORS_RECOMENDADO,
                          min_dist=dist_min, random_state=RANDOM_STATE)
reducer_umap.fit(X_sc)

x_nuevo = X_test[[0]]
x_nuevo_sc = scaler.transform(x_nuevo)
z_nuevo = reducer_umap.transform(x_nuevo_sc)

print(f"Registro: {df_test['recording_id'].iloc[0]}")
print(f"Proyección UMAP 2D (sin reentrenar): {np.round(z_nuevo[0], 3)}")
print("NOTA: t-SNE no ofrece esta capacidad; solo tiene fit_transform().")


# ══════════════════════════════════════════════════════════════════════════
# 5. COMPARATIVA FINAL 2D: PCA vs t-SNE vs UMAP
#    Layout: 3 filas x 1 columna. Leyenda = colorbar compartido (barras).
#    Sin título general (suptitle); se conservan los títulos por subplot.
# ══════════════════════════════════════════════════════════════════════════
fig_comp_2d = plt.figure(figsize=(7.5, 17), num="Comparativa Final 2D", constrained_layout=True)

# 3 filas para los métodos + 1 columna angosta para el colorbar compartido,
# que abarca las tres filas (gs[:, 1]).
gs = fig_comp_2d.add_gridspec(3, 2, width_ratios=[1, 0.05])
ax1 = fig_comp_2d.add_subplot(gs[0, 0])
ax2 = fig_comp_2d.add_subplot(gs[1, 0])
ax3 = fig_comp_2d.add_subplot(gs[2, 0])
cax = fig_comp_2d.add_subplot(gs[:, 1])

# species_id NO son consecutivos (10,12,17,18,23) -> se mapean por índice
# de posición en `clases`, no por su valor numérico, para que el colorbar
# discreto tenga exactamente n_clases bins.
idx_map = {cls: i for i, cls in enumerate(clases)}
y_idx = np.array([idx_map[v] for v in y])

cmap_discreta = mpl.colors.ListedColormap(colores)
bounds = np.arange(-0.5, n_clases + 0.5, 1)
norm = BoundaryNorm(bounds, cmap_discreta.N)

sc1 = ax1.scatter(Z_pca_2d[:, 0], Z_pca_2d[:, 1], c=y_idx, cmap=cmap_discreta, norm=norm,
                   s=20, alpha=0.85, edgecolors='white', linewidths=0.2)
ax1.set_title(f'PCA 2D (Var: {var_ret_2d:.0f}%)')

ax2.scatter(best_Z_tsne_2d[:, 0], best_Z_tsne_2d[:, 1], c=y_idx, cmap=cmap_discreta, norm=norm,
            s=20, alpha=0.85, edgecolors='white', linewidths=0.2)
ax2.set_title(f't-SNE 2D (perp={PERP_RECOMENDADO})')

ax3.scatter(best_Z_umap_2d[:, 0], best_Z_umap_2d[:, 1], c=y_idx, cmap=cmap_discreta, norm=norm,
            s=20, alpha=0.85, edgecolors='white', linewidths=0.2)
ax3.set_title(f'UMAP 2D (n_neighbors={N_NEIGHBORS_RECOMENDADO})')

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')

cbar = fig_comp_2d.colorbar(sc1, cax=cax, ticks=np.arange(n_clases))
cbar.ax.set_yticklabels([str(c) for c in clases])
cbar.set_label('species_id')

fig_comp_2d.savefig(f'{IMG_DIR}/comparativa_final_2d.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig_comp_2d)


# ══════════════════════════════════════════════════════════════════════════
# 6. COMPARATIVA FINAL 3D: PCA vs t-SNE vs UMAP
#    Layout: 3 filas x 1 columna. Leyenda = colorbar compartido (barras),
#    igual estilo que la comparativa 2D. Sin título general (suptitle);
#    se conservan los títulos por subplot.
# ══════════════════════════════════════════════════════════════════════════
fig_comp_3d = plt.figure(figsize=(8.5, 20), num="Comparativa Final 3D")

gs3d = fig_comp_3d.add_gridspec(3, 2, width_ratios=[1, 0.05])
ax1_3d = fig_comp_3d.add_subplot(gs3d[0, 0], projection='3d')
ax2_3d = fig_comp_3d.add_subplot(gs3d[1, 0], projection='3d')
ax3_3d = fig_comp_3d.add_subplot(gs3d[2, 0], projection='3d')
cax_3d = fig_comp_3d.add_subplot(gs3d[:, 1])

sc1_3d = ax1_3d.scatter(Z_pca_3d[:, 0], Z_pca_3d[:, 1], Z_pca_3d[:, 2],
                         c=y_idx, cmap=cmap_discreta, norm=norm,
                         s=20, alpha=0.85, edgecolors='white', linewidths=0.2)
ax1_3d.set_title(f'PCA 3D (Var: {var_ret_3d:.0f}%)')

ax2_3d.scatter(best_Z_tsne_3d[:, 0], best_Z_tsne_3d[:, 1], best_Z_tsne_3d[:, 2],
               c=y_idx, cmap=cmap_discreta, norm=norm,
               s=20, alpha=0.85, edgecolors='white', linewidths=0.2)
ax2_3d.set_title(f't-SNE 3D (perp={PERP_RECOMENDADO})')

ax3_3d.scatter(best_Z_umap_3d[:, 0], best_Z_umap_3d[:, 1], best_Z_umap_3d[:, 2],
               c=y_idx, cmap=cmap_discreta, norm=norm,
               s=20, alpha=0.85, edgecolors='white', linewidths=0.2)
ax3_3d.set_title(f'UMAP 3D (n_neighbors={N_NEIGHBORS_RECOMENDADO})')

for ax in [ax1_3d, ax2_3d, ax3_3d]:
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.view_init(elev=25, azim=45)

cbar_3d = fig_comp_3d.colorbar(sc1_3d, cax=cax_3d, ticks=np.arange(n_clases))
cbar_3d.ax.set_yticklabels([str(c) for c in clases])
cbar_3d.set_label('species_id')

plt.savefig(f'{IMG_DIR}/comparativa_final_3d.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig_comp_3d)


# ══════════════════════════════════════════════════════════════════════════
# 7. PCA — VARIANZA EXPLICADA ACUMULADA (todas las 64 dimensiones)
# ══════════════════════════════════════════════════════════════════════════
pca_full = PCA(random_state=RANDOM_STATE).fit(X_sc)
cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
n_comp = np.arange(1, len(cum_var) + 1)

n_80 = int(np.argmax(cum_var >= 80) + 1)
n_90 = int(np.argmax(cum_var >= 90) + 1)
n_95 = int(np.argmax(cum_var >= 95) + 1)
print(f"\nComponentes necesarios: 80%->{n_80} | 90%->{n_90} | 95%->{n_95} (de 64 originales)")


# ══════════════════════════════════════════════════════════════════════════
# 8. FIGURA RESUMEN: PCA 2D + PCA 3D + Varianza acumulada
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(19, 5.5))
fig.suptitle('PCA — Dataset Eco-Acústico (reducción de 64D a 2D y 3D)', fontsize=15, fontweight='bold')

ax1 = fig.add_subplot(1, 3, 1)
for cls in clases:
    mask = y == cls
    ax1.scatter(Z_pca_2d[mask, 0], Z_pca_2d[mask, 1], c=[cmap[cls]], s=18, alpha=0.8,
                label=f'{cls}', edgecolors='none')
ax1.set_title(f'PCA 2D\nPC1={var_2d[0]:.1%}  PC2={var_2d[1]:.1%}  Total={var_ret_2d:.1f}%')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.legend(fontsize=14, markerscale=1.4, title='species_id')

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
for cls in clases:
    mask = y == cls
    ax2.scatter(Z_pca_3d[mask, 0], Z_pca_3d[mask, 1], Z_pca_3d[mask, 2],
                c=[cmap[cls]], s=15, alpha=0.8, label=f'{cls}', depthshade=True)
ax2.set_title(
    f'PCA 3D\nPC1={var_3d[0]:.1%}  PC2={var_3d[1]:.1%}  PC3={var_3d[2]:.1%}\nTotal={var_ret_3d:.1f}%')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.legend(fontsize=14, markerscale=1.4)
ax2.view_init(elev=25, azim=45)

ax3 = fig.add_subplot(1, 3, 3)
ax3.bar(n_comp, pca_full.explained_variance_ratio_ * 100, color='#378ADD', alpha=0.7, label='Var. individual')
ax3.plot(n_comp, cum_var, color='#E24B4A', lw=2, marker='o', ms=3, label='Var. acumulada')

for pct, ls in zip([80, 90, 95], [':', '--', '-.']):
    ax3.axhline(pct, color='gray', ls=ls, lw=1, alpha=0.6)
    ax3.text(n_comp[-1] * 0.75, pct + 1.5, f'{pct}%', color='gray', fontsize=14)

ax3.axvline(2, color='#1D9E75', ls='--', lw=1.2, alpha=0.8, label='2D seleccionado')
ax3.axvline(3, color='#EF9F27', ls='--', lw=1.2, alpha=0.8, label='3D seleccionado')

ax3.set_title('Varianza explicada acumulada\n(64 componentes)')
ax3.set_xlabel('Número de componentes')
ax3.set_ylabel('Varianza explicada (%)')
ax3.set_xticks(np.arange(0, 65, 8))
ax3.set_ylim(0, 105)
ax3.legend(fontsize=14)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/pca_eco_acoustic.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\n✓ Figura guardada: pca_eco_acoustic.png")


# ══════════════════════════════════════════════════════════════════════════
# 9. TABLA COMPARATIVA CUANTITATIVA (para insertar directo en LaTeX)
# ══════════════════════════════════════════════════════════════════════════
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('Proyecto 2/resultados_reduccion_dimensionalidad.csv', index=False)
print("\n=== Tabla comparativa (guardada en resultados_reduccion_dimensionalidad.csv) ===")
print(df_resultados.to_string(index=False))

print("\n✓ Proceso completo. Figuras en ./img/")