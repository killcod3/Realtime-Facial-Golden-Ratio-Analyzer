import cv2
import dlib
import math

def draw_lines(image, landmarks):
    colors = [
        (0, 255, 0),   # green
        (255, 0, 0),   # blue
        (0, 0, 255),   # red
        (255, 255, 0), # cyan
        (255, 0, 255), # magenta
        (0, 255, 255), # yellow
        (128, 0, 128), # purple
    ]

    # Draw lines for the facial proportions
    cv2.line(image, landmarks[0], landmarks[16], colors[0], 1) # Face length
    cv2.line(image, landmarks[36], landmarks[45], colors[1], 1) # Eye distance
    cv2.line(image, landmarks[31], landmarks[35], colors[2], 1) # Nose width
    cv2.line(image, landmarks[48], landmarks[54], colors[3], 1) # Mouth width
    cv2.line(image, landmarks[33], landmarks[8], colors[4], 1) # Nose to chin length
    cv2.line(image, landmarks[39], landmarks[42], colors[5], 1) # Nose to center of eyes length
    cv2.line(image, landmarks[21], landmarks[22], colors[6], 1) # Distance between eyebrows

def golden_ratio(face_landmarks):
    horizontal_face_length = face_landmarks[16][0] - face_landmarks[0][0]
    vertical_face_length = face_landmarks[8][1] - face_landmarks[27][1]

    eye_distance = abs(face_landmarks[45][0] - face_landmarks[36][0])
    nose_width = face_landmarks[35][0] - face_landmarks[31][0]
    mouth_width = face_landmarks[54][0] - face_landmarks[48][0]

    nose_chin_length = face_landmarks[8][1] - face_landmarks[33][1]
    nose_center_eyes_length = face_landmarks[33][1] - (face_landmarks[39][1] + face_landmarks[42][1]) // 2
    eyebrows_distance = abs(face_landmarks[21][0] - face_landmarks[22][0])


    face_ratio = vertical_face_length / horizontal_face_length
    eye_nose_ratio = eye_distance / nose_width
    mouth_eye_ratio = mouth_width / eye_distance

    nose_chin_center_eyes_ratio = nose_chin_length / nose_center_eyes_length
    mouth_nose_ratio = mouth_width / nose_width
    eyebrows_mouth_ratio = eyebrows_distance / mouth_width
    eyes_eyebrows_ratio = eye_distance / eyebrows_distance

    golden_ratio = (1 + math.sqrt(5)) / 2

    def calculate_confidence(ratio):
        return 1 - abs(1 - ratio / golden_ratio)

    face_confidence = calculate_confidence(face_ratio)
    eye_nose_confidence = calculate_confidence(eye_nose_ratio)
    mouth_eye_confidence = calculate_confidence(mouth_eye_ratio)

    nose_chin_center_eyes_confidence = calculate_confidence(nose_chin_center_eyes_ratio)
    mouth_nose_confidence = calculate_confidence(mouth_nose_ratio)
    eyebrows_mouth_confidence = calculate_confidence(eyebrows_mouth_ratio)
    eyes_eyebrows_confidence = calculate_confidence(eyes_eyebrows_ratio)

    return (
        face_confidence,
        eye_nose_confidence,
        mouth_eye_confidence,
        nose_chin_center_eyes_confidence,
        mouth_nose_confidence,
        eyebrows_mouth_confidence,
        eyes_eyebrows_confidence,
    )

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        face_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

        (
            face_confidence,
            eye_nose_confidence,
            mouth_eye_confidence,
            nose_chin_center_eyes_confidence,
            mouth_nose_confidence,
            eyebrows_mouth_confidence,
            eyes_eyebrows_confidence,
        ) = golden_ratio(face_landmarks)
        draw_lines(frame, face_landmarks)

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f"Face: {face_confidence*100:.2f}%", (face.left(), face.top() - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Eye-Nose: {eye_nose_confidence*100:.2f}%", (face.left(), face.top() - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Mouth-Eye: {mouth_eye_confidence*100:.2f}%", (face.left(), face.top() - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Nose-Chin/Eyes: {nose_chin_center_eyes_confidence*100:.2f}%", (face.left(), face.top() - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Mouth-Nose: {mouth_nose_confidence*100:.2f}%", (face.left(), face.top() - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Eyebrows-Mouth: {eyebrows_mouth_confidence*100:.2f}%", (face.left(), face.top() - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Eyes-Eyebrows: {eyes_eyebrows_confidence*100:.2f}%", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Golden Ratio", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

