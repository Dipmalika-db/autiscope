import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

colors = {
    'black': (0, 0, 0),
    'blue': (255, 0, 0),
    'pink': (203, 192, 255),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'white': (255, 255, 255)
}
current_color = 'black'


canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0
smooth_factor = 5
points = [] 

cap = cv2.VideoCapture(0)


recognizing_numbers = False
recognizing_shapes = False  
last_spoken_number = None


num_smooth = []
num_smooth_factor = 3

def recognize_number(landmarks):
    extended_fingers = 0
    if landmarks[4].x < landmarks[2].x:
        extended_fingers += 1
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            extended_fingers += 1
    return extended_fingers

def smooth_number_detection(num):
    global num_smooth
    num_smooth.append(num)
    if len(num_smooth) > num_smooth_factor:
        num_smooth.pop(0)
    return round(sum(num_smooth) / len(num_smooth))

def recognize_shape():
    global points
    if len(points) < 20:
        return None
    
    contour = np.array(points, dtype=np.int32)
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    num_corners = len(approx)
    
    shape_name = "Unknown"
    if num_corners == 3:
        shape_name = "Triangle"
    elif num_corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"
    elif num_corners > 4:
        shape_name = "Circle"
    
    if shape_name != "Unknown":
        speak(shape_name)
    
    return shape_name

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

   
    palette_x, palette_y = 10, 10
    color_boxes = {}
    for color_name, color_value in colors.items():
        cv2.rectangle(frame, (palette_x, palette_y), (palette_x + 50, palette_y + 50), color_value, -1)
        cv2.putText(frame, color_name, (palette_x + 60, palette_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        color_boxes[color_name] = (palette_x, palette_y, palette_x + 50, palette_y + 50)
        palette_y += 60

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            x, y = int(landmarks[8].x * w), int(landmarks[8].y * h)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            else:
                x = (prev_x * (smooth_factor - 1) + x) // smooth_factor
                y = (prev_y * (smooth_factor - 1) + y) // smooth_factor

            for color_name, (x1, y1, x2, y2) in color_boxes.items():
                if x1 < x < x2 and y1 < y < y2:
                    if current_color != color_name:
                        current_color = color_name
                        speak(color_name)
                    break

            if not recognizing_numbers and not recognizing_shapes:
                if prev_x and prev_y:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), colors[current_color], 5)
                    points.append((x, y))
                prev_x, prev_y = x, y

            if recognizing_numbers:
                num = recognize_number(landmarks)
                num = smooth_number_detection(num)
                cv2.putText(frame, f"Number: {num}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if num != last_spoken_number:
                    speak(str(num))
                    last_spoken_number = num

    if recognizing_shapes:
        shape_name = recognize_shape()
        if shape_name:
            cv2.putText(frame, f"Shape: {shape_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            points.clear()

    frame = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
    cv2.imshow("Hand Gesture Drawing & Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        points.clear()
    elif key == ord('w'):
        recognizing_numbers = not recognizing_numbers
        recognizing_shapes = False
        prev_x, prev_y = 0, 0
        speak("Number recognition mode" if recognizing_numbers else "Drawing mode")
    elif key == ord('s'):
        recognizing_shapes = not recognizing_shapes
        recognizing_numbers = False
        prev_x, prev_y = 0, 0
        speak("Shape recognition mode" if recognizing_shapes else "Drawing mode")

cap.release()
cv2.destroyAllWindows()
