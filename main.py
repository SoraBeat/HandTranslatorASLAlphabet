import cv2
import torch
import mediapipe as mp
from torchvision import transforms
from PIL import Image
from model import ResNet, ASLDataset

# Función para rastrear la mano izquierda y dibujar el cuadrado verde (ROI)
def track_left_hand(frame, results):
    frame_height, frame_width, _ = frame.shape

    # Determina el centro de la mano izquierda (en este caso, el inicio de la primera falange del dedo medio)
    hand_landmarks = results.multi_hand_landmarks[0]
    hand_center_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame_width)
    hand_center_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame_height)

    # Define el tamaño del cuadrado verde (ROI)
    roi_size = 200

    # Calcula las coordenadas para el cuadrado centrado en la mano izquierda
    top_left_x = max(0, hand_center_x - roi_size // 2)
    top_left_y = max(0, hand_center_y - roi_size // 2)
    bottom_right_x = min(frame_width, hand_center_x + roi_size // 2)
    bottom_right_y = min(frame_height, hand_center_y + roi_size // 2)

    # Dibuja el cuadrado verde en el frame
    green_color = (0, 255, 0)
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), green_color, thickness=2)

    # Obtiene el ROI
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return roi, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Define el número de clases (en este caso, 29 letras del alfabeto y gestos adicionales)
num_classes = 29

# Carga el modelo pre-entrenado de lenguaje de señas ResNet en PyTorch
model = ResNet(num_classes)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Define las transformaciones para preprocesar las imágenes
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Listar cámaras disponibles
def list_cameras():
    camera_list = []
    for i in range(10):  # Prueba con un rango de índices de 0 a 9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_list.append(f'Cámara {i}')
            cap.release()
    return camera_list

# Mostrar la lista de cámaras y permitir al usuario seleccionar una
camera_list = list_cameras()
if not camera_list:
    print("No se encontraron cámaras disponibles.")
else:
    print("Cámaras disponibles:")
    for i, camera in enumerate(camera_list):
        print(f"{i}: {camera}")

    while True:
        camera_index = input("Selecciona el número de la cámara que deseas usar: ")
        try:
            camera_index = int(camera_index)
            if 0 <= camera_index < len(camera_list):
                break
            else:
                print("Índice fuera de rango. Ingresa un número válido.")
        except ValueError:
            print("Ingresa un número válido.")

    selected_camera = camera_list[camera_index]
    print(f"Usando {selected_camera}")

    # Inicializa la cámara seleccionada
    cap = cv2.VideoCapture(camera_index)

    # Variable para controlar la inversión horizontal de la cámara
    flip_frame = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltea horizontalmente el frame si flip_frame es True
        if flip_frame:
            frame = cv2.flip(frame, 1)

        # Dibuja un cuadrado de 25x25 píxeles en el centro del frame
        square_size = 200
        square_color = (0, 255, 0)  # Color en formato BGR (verde en este caso)
        
        # Realiza el seguimiento de manos en la ROI de la mano izquierda
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            roi, top_left, bottom_right = track_left_hand(frame, results)
            
            # Realiza la inferencia con el modelo en la ROI
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            input_image = preprocess(roi_pil).unsqueeze(0)
            output = model(input_image)
            predicted_class = torch.argmax(output, dim=1).item()

            # Mapea la etiqueta a una acción o palabra en lenguaje de señas
            gestos = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}
            palabra_significativa = gestos.get(predicted_class, "Desconocido")

            # Imprime la letra y el ID del gesto en la consola
            print(f"Gesto: {palabra_significativa} (ID: {predicted_class})")

            # Dibuja los puntos de seguimiento de manos utilizando mp_drawing
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # Dibuja el cuadrado verde alrededor de la mano izquierda
            cv2.rectangle(frame, top_left, bottom_right, square_color, thickness=2)

        # Muestra el frame con el cuadrado, el seguimiento de manos y las inferencias
        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1)

        # Si se presiona la tecla "F", cambia el estado de flip_frame
        if key == ord('f') or key == ord('F'):
            flip_frame = not flip_frame

        # Si se presiona la tecla "ESC", cierra el programa
        if key == 27:  # 27 corresponde a la tecla "ESC"
            break

    # Libera la cámara y cierra las ventanas al final del programa
    cap.release()
    cv2.destroyAllWindows()
