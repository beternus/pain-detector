import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Facial landmarks (IDs fictícios para teste numérico)
LANDMARKS = {
    'topo_testa': 0,
    'base_queixo': 1,
    'lateral_esq': 2,
    'lateral_dir': 3,
    'sobrancelha_esq': 4,
    'sobrancelha_dir': 5,
    'testa_sob_esq': 6,
    'testa_sob_direita': 7,
    'olho_topo_esq': 8,
    'olho_base_esq': 9,
    'olho_topo_dir': 10,
    'olho_base_dir': 11,
    'bochecha_esq': 12,
    'bochecha_dir': 13,
    'labio_sup': 14,
    'labio_inf': 15,
    'pico_boca': 16,
    'base_boca': 17,
    'pupila_dir': 478,
    'pupila_esq': 479
}

def calcular_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# -----------------------------
# Parameters
# -----------------------------
FPS = 30
DURATION = 15  # seconds
N_FRAMES = FPS * DURATION

# Pain periods (start, end) in frames
DOR_PERIODS = [
    (3 * FPS, 5 * FPS),
    (8 * FPS, 10 * FPS),
    (11 * FPS, 13 * FPS)
]

# Check if index is within any pain period
def is_pain_frame(idx):
    return any(start <= idx <= end for start, end in DOR_PERIODS)

# Simulated video frames (black)
frames_simulados = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(N_FRAMES)]

# Simulate landmarks
def gerar_landmarks(frame_idx):
    base = 50
    desvio = 2
    if is_pain_frame(frame_idx):
        desvio = 50
    pontos = [(base + np.random.randn() * desvio, base + np.random.randn() * desvio) for _ in range(478)]
    pontos.append((100 + np.random.randn() * desvio, 100 + np.random.randn() * desvio))  # pupila_dir
    pontos.append((120 + np.random.randn() * desvio, 100 + np.random.randn() * desvio))  # pupila_esq

    if is_pain_frame(frame_idx):
        pontos[LANDMARKS['olho_topo_esq']] = (pontos[LANDMARKS['olho_topo_esq']][0], pontos[LANDMARKS['olho_topo_esq']][1] - 5)
        pontos[LANDMARKS['olho_topo_dir']] = (pontos[LANDMARKS['olho_topo_dir']][0], pontos[LANDMARKS['olho_topo_dir']][1] - 5)
        pontos[LANDMARKS['labio_sup']] = (pontos[LANDMARKS['labio_sup']][0], pontos[LANDMARKS['labio_sup']][1] - 3)
        pontos[LANDMARKS['labio_inf']] = (pontos[LANDMARKS['labio_inf']][0], pontos[LANDMARKS['labio_inf']][1] + 3)
    return pontos

# -----------------------------
# Variables
# -----------------------------
TP = FP = TN = FN = 0
detec_dor = []
janela_deteccao = []
y_true = []
y_pred = []

# Calibration
valores_sobr_esq = []
valores_sobr_dir = []
valores_olho_esq = []
valores_olho_dir = []
valores_bochecha_esq = []
valores_bochecha_dir = []
valores_labios = []
valores_boca = []

calibrando = True
media_sobr_esq = media_sobr_dir = 0
media_olho_esq = media_olho_dir = 0
media_bochecha_esq = media_bochecha_dir = 0
media_labios = media_boca = 0

# -----------------------------
# Main loop
# -----------------------------
for idx, frame in enumerate(frames_simulados):
    tempo_passado = idx / FPS
    pontos = gerar_landmarks(idx)

    altura_face = max(calcular_dist(pontos[LANDMARKS['topo_testa']], pontos[LANDMARKS['base_queixo']]), 1e-5)
    largura_face = max(calcular_dist(pontos[LANDMARKS['lateral_esq']], pontos[LANDMARKS['lateral_dir']]), 1e-5)
    interpupilar = max(calcular_dist(pontos[LANDMARKS['pupila_esq']], pontos[LANDMARKS['pupila_dir']]), 1e-5)

    dist_sobr_esq = calcular_dist(pontos[LANDMARKS['sobrancelha_esq']], pontos[LANDMARKS['testa_sob_esq']]) / interpupilar
    dist_sobr_dir = calcular_dist(pontos[LANDMARKS['sobrancelha_dir']], pontos[LANDMARKS['testa_sob_direita']]) / interpupilar
    dist_olho_esq = calcular_dist(pontos[LANDMARKS['olho_topo_esq']], pontos[LANDMARKS['olho_base_esq']]) / interpupilar
    dist_olho_dir = calcular_dist(pontos[LANDMARKS['olho_topo_dir']], pontos[LANDMARKS['olho_base_dir']]) / interpupilar
    dist_bochecha_esq = calcular_dist(pontos[LANDMARKS['bochecha_esq']], pontos[LANDMARKS['lateral_esq']]) / largura_face
    dist_bochecha_dir = calcular_dist(pontos[LANDMARKS['bochecha_dir']], pontos[LANDMARKS['lateral_dir']]) / largura_face
    dist_labios = calcular_dist(pontos[LANDMARKS['labio_sup']], pontos[LANDMARKS['labio_inf']]) / altura_face
    dist_boca = (calcular_dist(pontos[LANDMARKS['pico_boca']], pontos[LANDMARKS['base_boca']]) / altura_face) / 100

    # Calibration
    if calibrando and tempo_passado < 3:
        valores_sobr_esq.append(dist_sobr_esq)
        valores_sobr_dir.append(dist_sobr_dir)
        valores_olho_esq.append(dist_olho_esq)
        valores_olho_dir.append(dist_olho_dir)
        valores_bochecha_esq.append(dist_bochecha_esq)
        valores_bochecha_dir.append(dist_bochecha_dir)
        valores_labios.append(dist_labios)
        valores_boca.append(dist_boca)
        detec_dor.append(0)
        continue
    elif calibrando:
        calibrando = False
        media_sobr_esq = np.mean(valores_sobr_esq)
        media_sobr_dir = np.mean(valores_sobr_dir)
        media_olho_esq = np.mean(valores_olho_esq)
        media_olho_dir = np.mean(valores_olho_dir)
        media_bochecha_esq = np.mean(valores_bochecha_esq)
        media_bochecha_dir = np.mean(valores_bochecha_dir)
        media_labios = np.mean(valores_labios)
        media_boca = np.mean(valores_boca)
        print("Calibrated! Reference averages computed.")

    # Variation
    var_sobr_esq = (dist_sobr_esq - media_sobr_esq) * 100
    var_sobr_dir = (dist_sobr_dir - media_sobr_dir) * 100
    var_olho_esq = (dist_olho_esq - media_olho_esq) * 100
    var_olho_dir = (dist_olho_dir - media_olho_dir) * 100
    var_bochecha_esq = (dist_bochecha_esq - media_bochecha_esq) * 100
    var_bochecha_dir = (dist_bochecha_dir - media_bochecha_dir) * 100
    var_labios = (dist_labios - media_labios) * 100
    var_boca = (dist_boca - media_boca) * 100

    sobrancelhas = var_sobr_esq >= 8 and var_sobr_dir >= 8
    olhos = var_olho_esq <= -15 and var_olho_dir <= -15
    bochechas = var_bochecha_esq <= -7 and var_bochecha_dir <= -7
    labios = var_labios < -20
    boca = var_boca > 2000

    dor_instante = sum([sobrancelhas, olhos, bochechas, labios, boca]) >= 3
    janela_deteccao.append(dor_instante)

    # Smoothing over 3 frames
    if len(janela_deteccao) >= 3:
        suavizado = sum(janela_deteccao[-3:]) >= 2
    else:
        suavizado = dor_instante

    detec_dor.append(suavizado)

    dor_real = is_pain_frame(idx)
    y_true.append(int(dor_real))
    y_pred.append(int(suavizado))

    if suavizado and dor_real:
        TP += 1
    elif suavizado and not dor_real:
        FP += 1
    elif not suavizado and not dor_real:
        TN += 1
    elif not suavizado and dor_real:
        FN += 1

# -----------------------------
# Metrics
# -----------------------------
precisao = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
acuracia = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0

# Tabela de métricas
tabela_metricas = pd.DataFrame({
    'Metric': ['True Positives', 'False Positives', 'True Negatives', 'False Negatives',
               'Precision', 'Recall', 'F1-score', 'Accuracy'],
    'Result': [TP, FP, TN, FN, precisao, recall, f1, acuracia]
})

print("\n=== Evaluation Metrics ===")
print(tabela_metricas.to_string(index=False))

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
conf_mat = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Pain", "Pain"],
            yticklabels=["No Pain", "Pain"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Pain Detection Over Time
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(detec_dor, label="Detected pain", color='blue')

for start, end in DOR_PERIODS:
    plt.axvspan(start, end, color='red', alpha=0.2, label="Actual pain period" if start == DOR_PERIODS[0][0] else "")

plt.xlabel("Frames")
plt.ylabel("Detection")
plt.title("Pain detection over time")
plt.legend()
plt.tight_layout()
plt.show()
