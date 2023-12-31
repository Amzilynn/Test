import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt

def calculate_stress(blink, eyebrow, emotions, lips, hand_movement, gaze_direction, face_orientation):
    final_stress = (
        0.15 * blink +
        0.15 * eyebrow +
        0.15 * emotions +
        0.15 * lips +
        0.15 * hand_movement +
        0.15 * gaze_direction +
        0.10 * face_orientation
    )
    return final_stress

def calculate_hand_movement(hand_landmarks):
    total_distance = 0
    for i in range(1, len(hand_landmarks.landmark)):
        x1, y1, z1 = hand_landmarks.landmark[i-1].x, hand_landmarks.landmark[i-1].y, hand_landmarks.landmark[i-1].z
        x2, y2, z2 = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        total_distance += distance

    return total_distance

def calculate_gaze_direction(eye_landmarks):
    left_eye_horizontal_movement = eye_landmarks.landmark[3].x - eye_landmarks.landmark[0].x
    left_eye_vertical_movement = eye_landmarks.landmark[4].y - eye_landmarks.landmark[1].y

    if left_eye_vertical_movement != 0:
        gaze_direction = left_eye_horizontal_movement / left_eye_vertical_movement
    else:
        gaze_direction = 0

    return gaze_direction

def calculate_face_orientation(pose_landmarks):
    roll_angle = math.degrees(math.asin(pose_landmarks.landmark[33].y - pose_landmarks.landmark[0].y))
    return roll_angle

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection()

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    stress_data = {'blink': [], 'eyebrow': [], 'emotions': [], 'lips': [],
                   'hand_movement': [], 'gaze_direction': [], 'face_orientation': [], 'overall': []}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_detection.process(rgb_frame)

        if face_results.detections:
            blink = analyze_blink(frame, face_results)
            eyebrow = analyze_eyebrow(frame, holistic)
            emotions = analyze_emotions(frame, holistic)
            lips = analyze_lips(frame, holistic)
            hand_movement = analyze_hand_movement(frame, holistic)
            gaze_direction = analyze_direction_of_eyes(frame, holistic)
            face_orientation = analyze_face_orientation(frame, holistic)

            stress_data['blink'].append(blink)
            stress_data['eyebrow'].append(eyebrow)
            stress_data['emotions'].append(emotions)
            stress_data['lips'].append(lips)
            stress_data['hand_movement'].append(hand_movement)
            stress_data['gaze_direction'].append(gaze_direction)
            stress_data['face_orientation'].append(face_orientation)

            overall_stress = calculate_stress(blink, eyebrow, emotions, lips, hand_movement, gaze_direction, face_orientation)
            stress_data['overall'].append(overall_stress)

    cap.release()
    cv2.destroyAllWindows()

    return stress_data

def analyze_blink(frame, face_results):
    return len(face_results.detections)

def analyze_eyebrow(frame, holistic):
    landmarks = holistic.process(frame).face_landmarks
    if landmarks:
        left_eyebrow_points = [(landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYEBROW_INNER].x, landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYEBROW_INNER].y),
                               (landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYEBROW_OUTER].x, landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYEBROW_OUTER].y)]
        right_eyebrow_points = [(landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYEBROW_INNER].x, landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYEBROW_INNER].y),
                                (landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYEBROW_OUTER].x, landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYEBROW_OUTER].y)]
        
        eyebrow_angle = calculate_angle(left_eyebrow_points[0], left_eyebrow_points[1], right_eyebrow_points[0], right_eyebrow_points[1])
        return eyebrow_angle

    return 0

def analyze_emotions(frame, holistic):
    landmarks = holistic.process(frame).face_landmarks
    if landmarks:
        left_eye_inner = (landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYE_INNER].x, landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYE_INNER].y)
        right_eye_inner = (landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYE_INNER].x, landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYE_INNER].y)
        left_eyebrow_outer = (landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYEBROW_OUTER].x, landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYEBROW_OUTER].y)
        right_eyebrow_outer = (landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYEBROW_OUTER].x, landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYEBROW_OUTER].y)
        
        eye_distance = calculate_distance(left_eye_inner, right_eye_inner)
        eyebrow_distance = calculate_distance(left_eyebrow_outer, right_eyebrow_outer)
        

    return 0

def analyze_lips(frame, holistic):
    landmarks = holistic.process(frame).face_landmarks
    if landmarks:
        upper_lip_points = [(landmarks.landmark[mp_holistic.FaceLandmark.LOWER_LIP].x, landmarks.landmark[mp_holistic.FaceLandmark.LOWER_LIP].y) for mp_holistic.FaceLandmark.LOWER_LIP in [10, 11, 12, 13, 14, 15]]
        lower_lip_points = [(landmarks.landmark[mp_holistic.FaceLandmark.UPPER_LIP].x, landmarks.landmark[mp_holistic.FaceLandmark.UPPER_LIP].y) for mp_holistic.FaceLandmark.UPPER_LIP in [10, 11, 12, 13, 14, 15]]
        
        lip_distance = calculate_distance(upper_lip_points[0], lower_lip_points[0])
        

    return 0

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_angle(point1, point2, point3, point4):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4
    
    vector1 = (x2 - x1, y2 - y1)
    vector2 = (x4 - x3, y4 - y3)
    
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    
    if magnitude1 != 0 and magnitude2 != 0:
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    else:
        return 0

def plot_graph(stress_data):
    plt.plot(stress_data['overall'])
    plt.title('Stress Levels Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Stress Level')
    plt.savefig('stress_graph.png')
    plt.show()

if __name__ == "__main__":
    video_path = "C:\\Users\\User\\Downloads\\vd.mp4"
    stress_data = analyze_video(video_path)
    plot_graph(stress_data)
