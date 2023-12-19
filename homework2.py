import cv2
import mediapipe as mp
import numpy as np


def find_hands(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    return frame, hand_results


def find_hand_position(frame, hand_results, hand_index=0):
    x_coordinates = []
    y_coordinates = []
    hand_box = []
    landmarks = []

    if hand_results.multi_hand_landmarks:
        detected_hand = hand_results.multi_hand_landmarks[hand_index]
        for point_id, landmark in enumerate(detected_hand.landmark):
            height, width, _ = frame.shape
            x, y = int(landmark.x * width), int(landmark.y * height)
            x_coordinates.append(x)
            y_coordinates.append(y)
            landmarks.append([point_id, x, y])

        min_x, max_x = min(x_coordinates), max(x_coordinates)
        min_y, max_y = min(y_coordinates), max(y_coordinates)
        hand_box = min_x, min_y, max_x, max_y

    return landmarks, hand_box


def detect_rised_fingers(landmarks, tip_ids):
    fingers_status = []

    if landmarks[tip_ids[0]][1] < landmarks[tip_ids[0] - 1][1]:
        fingers_status.append(1)
    else:
        fingers_status.append(0)

    for i in range(1, 5):
        if landmarks[tip_ids[i]][2] < landmarks[tip_ids[i] - 2][2]:
            fingers_status.append(1)
        else:
            fingers_status.append(0)

    return fingers_status


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_radius(finger1, finger2):
    distance = calculate_distance(finger1, finger2)

    return int(distance / 5)


def apply_image_filters(frame):
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    filtered_image = blurred_frame

    return filtered_image


def states_change(curent_state, states):
    for state in states:
        states[state] = False
    states[curent_state] = True
    return states


width_cam, height_cam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(False, 2, 1, 0.5, 0.5)

tip_finger_ids = [4, 8, 12, 16, 20]

states = {"drawing": False, "rubbing": False, "color_change": False}
color_index = 0
colors = [(0, 0, 0, 255), (0, 0, 255, 255), (255, 0, 0, 255), (0, 255, 0, 255)]

center_x, center_y = 0, 0
drawn_frame = np.zeros((height_cam, width_cam, 4), np.uint8)
previous_x, previous_y = None, None
color = colors[color_index]

while True:
    success, current_frame = cap.read()
    current_frame = cv2.flip(current_frame, 1)

    current_frame = apply_image_filters(current_frame)

    current_frame_bgra = cv2.cvtColor(current_frame, cv2.COLOR_BGR2BGRA)
    current_frame, hands_results = find_hands(current_frame, hands_detector)
    hand_landmarks, hand_box = find_hand_position(current_frame, hands_results)

    if hand_landmarks:
        fingers_status_list = detect_rised_fingers(hand_landmarks, tip_finger_ids)
        total_rised_fingers = fingers_status_list.count(1)

        center_x, center_y = (hand_box[0] + hand_box[2]) // 2, (hand_box[1] + hand_box[3]) // 2

        if total_rised_fingers == 1:
            states = states_change('drawing', states)

        elif total_rised_fingers == 3:
            previous_x, previous_y = None, None
            states = states_change('rubbing', states)

        elif total_rised_fingers == 5 and (states["drawing"] or states["rubbing"]):
            previous_x, previous_y = None, None
            states = states_change('color_change', states)

        else:
            previous_x, previous_y = None, None
            states = states_change('None', states)

        if states["drawing"]:
            cv2.circle(drawn_frame, (center_x, center_y), 5, color, -1)
            if previous_x is not None and previous_y is not None:
                cv2.line(drawn_frame, (center_x, center_y), (previous_x, previous_y), color, 10)
            previous_x, previous_y = center_x, center_y

        elif states["rubbing"]:
            index_finger = hand_landmarks[tip_finger_ids[1]][1:]
            ring_finger = hand_landmarks[tip_finger_ids[3]][1:]
            radius = calculate_radius(index_finger, ring_finger)

            cv2.circle(drawn_frame, (center_x, center_y), radius, (0, 0, 0, 0), -1)

        elif states["color_change"]:
            color_index = (color_index + 1) % len(colors)
            color = colors[color_index]
            states["color_change"] = False

    channel_sum = np.sum(drawn_frame, axis=2)
    mask = (channel_sum == 0)
    mask = mask[..., np.newaxis]

    current_frame = current_frame_bgra * mask + drawn_frame
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow("Hand Draw", current_frame)
    cv2.waitKey(1)
