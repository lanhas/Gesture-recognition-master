import cv2
import numpy as np
import mediapipe as mp
from tkinter import messagebox


class HandsKeypointPredict():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7)
        except Exception:
            messagebox.showinfo("提示","系统不支持mediapipe，请切换为备用模式")

    def get_coordinates(self, norm_img):
        coords = []  # shape: (xyz(3), num_keypoints)
        # image = cv2.flip(norm_img, 1)
        results = self.hands.process(cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            coords = np.array([[0] * 21] * 3)
        else:
            for i in range(21):
                coords.append(results.multi_hand_landmarks[0].landmark[i].x)
                coords.append(results.multi_hand_landmarks[0].landmark[i].y)
                coords.append(results.multi_hand_landmarks[0].landmark[i].z)
            coords = np.array(coords)
            coords = coords.reshape((21, 3)).T
        return coords
    
    def drow_coords(self, norm_img):
        # image = cv2.flip(norm_img, 1)
        image_height, image_width, _ = norm_img.shape
        annotated_image = image.copy()
        results = self.hands.process(cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            pass
        else:
            self.mp_drawing.draw_landmarks(
            annotated_image, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
        # annotated_image = cv2.flip(annotated_image, 1)
        return annotated_image

