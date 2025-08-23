from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Thread
from flask import Flask, Response
from flask import send_file
import io

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Cria a instância do app Flask
app = Flask(__name__)
frame_to_stream = None  # Armazenar o frame anotado para o streaming

# Função de streaming para o navegador ou app Android
@app.route('/video_feed')
def video_feed():
    def generate():
        global frame_to_stream
        while True:
            if frame_to_stream is not None:
                ret, buffer = cv2.imencode('.jpg', frame_to_stream)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # Cerca de 30 fps
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Thread para rodar o servidor Flask em paralelo
def start_flask():
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)

flask_thread = Thread(target=start_flask)
flask_thread.daemon = True
flask_thread.start()

last_frame = None  # Para armazenar o último frame capturado
def gerar_frame():
    global last_frame
    if last_frame is None:
        return None
    _, buffer = cv2.imencode('.jpg', last_frame)
    return io.BytesIO(buffer.tobytes())

@app.route('/frame')
def frame():
    frame_io = gerar_frame()
    if frame_io is None:
        return "Nenhum frame disponível", 503
    return send_file(frame_io, mimetype='image/jpeg')

LANDMARKS = {
    'sobrancelha_esq': 105,
    'olho_topo_esq': 159,
    'olho_base_esq': 145,
    'sobrancelha_dir': 334,
    'testa_sob_direita': 333,
    'testa_sob_esq': 104,
    'olho_topo_dir': 386,
    'olho_base_dir': 374,
    'canto_boca_esq': 61,
    'canto_boca_dir': 291,
    'labio_sup': 0,
    'labio_inf': 17,
    'bochecha_dir': 425,
    'bochecha_esq': 205,
    'topo_testa': 10,
    'base_queixo': 152,
    'lateral_esq': 234,
    'lateral_dir': 454,
    'pico_boca': 13,
    'base_boca': 14,
    'pupila_esq': 473,
    'pupila_dir': 468
}

def calcular_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def variacao_percentual(valor_atual, valor_calibrado):
    if valor_calibrado < 1e-5:
        return 0
    return ((valor_atual - valor_calibrado) / valor_calibrado) * 100

def iniciar_calibracao():
    global calibrando, tempo_inicio
    global valores_sobr_esq, valores_sobr_dir, valores_bochecha_esq, valores_bochecha_dir
    global valores_olho_esq, valores_olho_dir, valores_labios
    valores_sobr_esq = []
    valores_sobr_dir = []
    valores_bochecha_esq = []
    valores_bochecha_dir = []
    valores_olho_esq = []
    valores_olho_dir = []
    valores_labios = []
    valores_boca = []
    calibrando = True
    tempo_inicio = time.time()
    print("Reiniciando calibração...")

#cap = cv2.VideoCapture("http://192.168.100.137:8080") #wifi de casa
cap = cv2.VideoCapture("http://192.168.156.184:8080") #wifi do s24+

# Variáveis de controle
calibrando = True
tempo_inicio = time.time()
mostrar_indices = False  
pausado = False

# Coletas para calibração
valores_sobr_esq = []
valores_sobr_dir = []
valores_bochecha_esq = []
valores_bochecha_dir = []
valores_olho_esq = []
valores_olho_dir = []
valores_labios = []
valores_boca = []

media_sobr_esq = 0
media_sobr_dir = 0
media_olho_esq = 0
media_olho_dir = 0
media_bochecha_esq = 0
media_bochecha_dir = 0
media_labios = 0
media_boca = 0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        if not pausado:
            success, frame = cap.read()
            

            if not success:
                break

            tempo_atual = time.time()
            tempo_passado = tempo_atual - tempo_inicio

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            frame_h, frame_w = frame.shape[:2]
            annotated_frame = frame.copy()
            
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pontos = [(int(landmark.x * frame_w), int(landmark.y * frame_h)) for landmark in face_landmarks.landmark]

                    altura_face = calcular_dist(pontos[LANDMARKS['topo_testa']], pontos[LANDMARKS['base_queixo']])
                    largura_face = calcular_dist(pontos[LANDMARKS['lateral_esq']], pontos[LANDMARKS['lateral_dir']])
                    interpupilar = calcular_dist(pontos[LANDMARKS['pupila_esq']], pontos[LANDMARKS['pupila_dir']])

                    dist_sobrancelha_esq = calcular_dist(pontos[LANDMARKS['sobrancelha_esq']], pontos[LANDMARKS['testa_sob_esq']]) / interpupilar
                    dist_sobrancelha_dir = calcular_dist(pontos[LANDMARKS['sobrancelha_dir']], pontos[LANDMARKS['testa_sob_direita']]) / interpupilar
                    dist_olho_esq = calcular_dist(pontos[LANDMARKS['olho_topo_esq']], pontos[LANDMARKS['olho_base_esq']]) / interpupilar
                    dist_olho_dir = calcular_dist(pontos[LANDMARKS['olho_topo_dir']], pontos[LANDMARKS['olho_base_dir']]) / interpupilar
                    dist_bochecha_esq = calcular_dist(pontos[LANDMARKS['bochecha_esq']], pontos[LANDMARKS['lateral_esq']]) / largura_face
                    dist_bochecha_dir = calcular_dist(pontos[LANDMARKS['bochecha_dir']], pontos[LANDMARKS['lateral_dir']]) / largura_face
                    dist_labios = calcular_dist(pontos[LANDMARKS['labio_sup']], pontos[LANDMARKS['labio_inf']]) / altura_face
                    dist_boca = (calcular_dist(pontos[LANDMARKS['pico_boca']], pontos[LANDMARKS['base_boca']]) / altura_face) / 100

                    if calibrando and tempo_passado < 3:
                        cv2.putText(annotated_frame, "Calibrando. Fique imovel.", (int(frame_w / 2 - 250), int(frame_h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                    elif calibrando and 3 <= tempo_passado < 8:
                        segundos_restantes = int(8 - tempo_passado)
                        msg = f"Calibrando em: {segundos_restantes}"
                        cv2.putText(annotated_frame, msg, (int(frame_w / 2 - 200), int(frame_h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                        valores_sobr_esq.append(dist_sobrancelha_esq)
                        valores_sobr_dir.append(dist_sobrancelha_dir)
                        valores_olho_esq.append(dist_olho_esq)
                        valores_olho_dir.append(dist_olho_dir)
                        valores_bochecha_esq.append(dist_bochecha_esq)
                        valores_bochecha_dir.append(dist_bochecha_dir)
                        valores_labios.append(dist_labios)
                        valores_boca.append(dist_boca)

                    elif calibrando and tempo_passado >= 8:
                        media_sobr_esq = np.mean(valores_sobr_esq)
                        media_sobr_dir = np.mean(valores_sobr_dir)
                        media_olho_esq = np.mean(valores_olho_esq)
                        media_olho_dir = np.mean(valores_olho_dir)
                        media_bochecha_esq = np.mean(valores_bochecha_esq)
                        media_bochecha_dir = np.mean(valores_bochecha_dir)
                        media_labios = np.mean(valores_labios)
                        media_boca = np.mean(valores_boca)
                        calibrando = False
                        print("Calibrado! Médias registradas.")

                    elif not calibrando:
                        var_sobr_esq = variacao_percentual(dist_sobrancelha_esq, media_sobr_esq)
                        var_sobr_dir = variacao_percentual(dist_sobrancelha_dir, media_sobr_dir)
                        var_olho_esq = variacao_percentual(dist_olho_esq, media_olho_esq)
                        var_olho_dir = variacao_percentual(dist_olho_dir, media_olho_dir)
                        var_bochecha_esq = variacao_percentual(dist_bochecha_esq, media_bochecha_esq)
                        var_bochecha_dir = variacao_percentual(dist_bochecha_dir, media_bochecha_dir)
                        var_labios = variacao_percentual(dist_labios, media_labios)
                        var_boca = variacao_percentual(dist_boca, media_boca)

                        cv2.putText(annotated_frame, f"Var. Sobr. Esq: {var_sobr_esq:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Var. Sobr. Dir: {var_sobr_dir:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Var. olho Esq: {var_olho_esq:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Var. olho Dir: {var_olho_dir:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Var. bochecha. Esq: {var_bochecha_esq:.1f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Var. bochecha. Dir: {var_bochecha_dir:.1f}%", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Var. Dist. Labios: {var_labios:.1f}%", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Var. Dist. Boca: {var_boca / 100:.1f}%", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)



                        sobrancelhas = var_sobr_esq >= 8 and var_sobr_dir >= 8
                        olhos = var_olho_esq <= -15 and var_olho_dir <= -15
                        bochechas = var_bochecha_esq <= -8 and var_bochecha_dir <= -8
                        labios = var_labios < -25
                        boca = var_boca > 3500

                        if sum([sobrancelhas, bochechas, labios, olhos, boca]) >= 3:
                            cv2.putText(annotated_frame, "Expressao de DOR detectada!", (150, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        cv2.putText(annotated_frame, f"Sobrancelhas", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if sobrancelhas else (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Olhos", (500, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if olhos else (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Bochechas", (500, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if bochechas else (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Labios", (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if labios else (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Boca", (500, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if boca else (0,255,0), 2)

                    if mostrar_indices:
                        for idx, landmark in enumerate(face_landmarks.landmark):
                            x = int(landmark.x * frame_w)
                            y = int(landmark.y * frame_h)
                            cv2.putText(annotated_frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )
                    last_frame = annotated_frame.copy()

            else:
                last_frame = annotated_frame.copy()
            frame_to_stream = annotated_frame.copy()
        cv2.imshow('Deteccao de expressoes', annotated_frame if not pausado else annotated_frame)
        

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('m') or key == ord('M'):
            mostrar_indices = not mostrar_indices
        elif key == 32:  # SPACE
            pausado = not pausado
        elif key == ord('r') or key == ord('R'):
            iniciar_calibracao()

cap.release()
cv2.destroyAllWindows()
