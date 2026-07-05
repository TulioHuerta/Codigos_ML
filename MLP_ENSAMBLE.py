import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ══════════════════════════════════════════════════════════════════════════
# ESTILO VISUAL ACADÉMICO (tesis / artículo)
#    Tipografía serif tipo Times, math en STIX, mayor DPI de exportación,
#    y antialiasing consistente para figuras nítidas en LaTeX.
#    (Mismo bloque de configuración usado en el script de reducción de
#    dimensionalidad, para mantener consistencia visual entre figuras.)
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

# Fiar semilla para reproducibilidad
np.random.seed(42)
torch.manual_seed(42)

# Directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. CARGA Y PREPROCESAMIENTO DE DATOS ---
print("1. Cargando y preprocesando datos...")
train_df = pd.read_csv(os.path.join(SCRIPT_DIR, "eco_acoustic_train.csv"))
test_df = pd.read_csv(os.path.join(SCRIPT_DIR, "eco_acoustic_test.csv"))

# Mapeo de la variable objetivo (Asimétrica)
original_classes = [10, 12, 17, 18, 23]
class_mapping = {val: idx for idx, val in enumerate(original_classes)}
inverse_mapping = {idx: val for idx, val in enumerate(original_classes)}

train_df['target'] = train_df['species_id'].map(class_mapping)
test_df['target'] = test_df['species_id'].map(class_mapping)

features = [f'mel_{i}' for i in range(64)]
X_train_raw = train_df[features].values
y_train_raw = train_df['target'].values
X_test_raw = test_df[features].values
y_test_raw = test_df['target'].values

# Estandarización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Cálculo de Class Weights para Entropía Cruzada Ponderada (Corrección del desbalance)
class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_raw), y=y_train_raw)
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

print(f"Clases originales: {original_classes}")
print(f"Pesos de clases (Inversamente proporcionales): {class_weights_np}")

# Preparar tensores para PyTorch
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_raw, dtype=torch.long)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test_raw, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=128, shuffle=False)

# --- 2. ENTRENAMIENTO DE MODELOS DE ENSAMBLE ---
print("\n2. Entrenando modelos de ensamble...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_raw)
rf_preds = rf_model.predict(X_test_scaled)
rf_f1 = f1_score(y_test_raw, rf_preds, average='macro')
print(f"Random Forest - F1-Score (Macro): {rf_f1:.4f}")

gb_model = HistGradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train_raw)
gb_preds = gb_model.predict(X_test_scaled)
gb_f1 = f1_score(y_test_raw, gb_preds, average='macro')
print(f"Gradient Boosting - F1-Score (Macro): {gb_f1:.4f}")


# --- 3. CONSTRUCCIÓN DE LA RED NEURONAL ---
print("\n3. Entrenando MLP en PyTorch (Experimentación de Topología)...")

# Escenario A: BatchNorm ANTES de la activación ReLU
class MLP_ScenarioA(nn.Module):
    def __init__(self):
        super(MLP_ScenarioA, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 5)
        )
    def forward(self, x):
        return self.net(x)

# Escenario B: BatchNorm DESPUÉS de la activación ReLU
class MLP_ScenarioB(nn.Module):
    def __init__(self):
        super(MLP_ScenarioB, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),

            nn.Linear(64, 5)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(model, epochs=50):
    # Uso de la Entropía Cruzada Categórica Ponderada
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

    # Obtener predicciones finales
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())

    mlp_f1 = f1_score(y_test_raw, all_preds, average='macro')
    return train_losses, val_losses, all_preds, mlp_f1

model_A = MLP_ScenarioA()
model_B = MLP_ScenarioB()

train_loss_A, val_loss_A, preds_A, f1_A = train_mlp(model_A, epochs=60)
train_loss_B, val_loss_B, preds_B, f1_B = train_mlp(model_B, epochs=60)

print(f"MLP (Escenario A) - F1-Score (Macro): {f1_A:.4f}")
print(f"MLP (Escenario B) - F1-Score (Macro): {f1_B:.4f}")

# --- 4. VISUALIZACIONES ---
print("\n4. Generando Gráficas...")

# ── Paleta consistente con el resto del proyecto (tab10) ───────────────────
color_A = plt.cm.tab10(0)  # azul
color_B = plt.cm.tab10(1)  # naranja

# 4.1 Gráfico de Curvas de Aprendizaje (Loss vs Épocas)
fig_loss, ax_loss = plt.subplots(figsize=(9, 5.5))
ax_loss.plot(train_loss_A, label='Train Loss (Escenario A: BN $\\rightarrow$ ReLU)',
             color=color_A, linestyle='--', lw=1.5)
ax_loss.plot(val_loss_A, label='Val Loss (Escenario A: BN $\\rightarrow$ ReLU)',
             color=color_A, lw=2)
ax_loss.plot(train_loss_B, label='Train Loss (Escenario B: ReLU $\\rightarrow$ BN)',
             color=color_B, linestyle='--', lw=1.5)
ax_loss.plot(val_loss_B, label='Val Loss (Escenario B: ReLU $\\rightarrow$ BN)',
             color=color_B, lw=2)
ax_loss.set_title('Estabilidad del Aprendizaje: Impacto Relativo de BatchNorm y Dropout')
ax_loss.set_xlabel('Épocas')
ax_loss.set_ylabel('Entropía Cruzada Ponderada (Loss)')
ax_loss.legend()
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'mlp_loss_curves.png'), bbox_inches='tight', facecolor='white')
plt.close(fig_loss)


# Determinar el mejor modelo para la matriz de confusión
best_mlp_preds = preds_A if f1_A > f1_B else preds_B
best_mlp_name = "Escenario A" if f1_A > f1_B else "Escenario B"

# 4.2 Función para graficar matrices de confusión
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=original_classes, yticklabels=original_classes,
                annot_kws={"size": 14}, ax=ax_cm, cbar_kws={'label': 'N.º de muestras'})
    ax_cm.set_title(title)
    ax_cm.set_xlabel('Predicción')
    ax_cm.set_ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, filename), bbox_inches='tight', facecolor='white')
    plt.close(fig_cm)

plot_confusion_matrix(y_test_raw, rf_preds, 'Matriz de Confusión - Random Forest', 'cm_random_forest.png')

plot_confusion_matrix(y_test_raw, gb_preds, 'Matriz de Confusión - Gradient Boosting', 'cm_gradient_boosting.png')

plot_confusion_matrix(y_test_raw, best_mlp_preds, f'Matriz de Confusión - MLP ({best_mlp_name})', 'cm_mlp.png')


print("\n=== RESUMEN FINAL ===")
print("Modelo\t\t\t\tF1-Score (Macro)")
print("-------------------------------------------------")
print(f"Random Forest\t\t\t{rf_f1:.4f}")
print(f"Gradient Boosting (XGB)\t\t{gb_f1:.4f}")
print(f"MLP (Escenario A)\t\t{f1_A:.4f}")
print(f"MLP (Escenario B)\t\t{f1_B:.4f}")