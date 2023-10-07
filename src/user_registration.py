from time import time

import cv2
import sqlite3

from src.constants import HAARCASCADE_FILE_PATH, ATTENDANCE_DB_PATH


class User:
    def __init__(self, name):
        self.name = name

    def save_profile_image(self, image):
        # Save the user's profile image to the database
        conn = sqlite3.connect(ATTENDANCE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, profile_image) VALUES (?, ?)", (self.name, image))
        conn.commit()
        conn.close()


class VideoCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_FILE_PATH)
        self.best_frame = None
        self.max_accuracy = 0

    def capture_frames(self, duration):
        """
        Capture frames for specified `duration`

        :param duration: integer time in seconds
        :return: None
        """
        for _ in range(duration):
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            accuracy = len(faces)

            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.best_frame = frame.copy()

            cv2.imshow('User Registration', frame)
            cv2.waitKey(20)

    def release(self):
        """
        Release video feed and destroy all cv2 windows.

        :return: None
        """
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    name = input("Enter your name: ")

    user = User(name)
    video_capture = VideoCapture()
    s_time = time()
    video_capture.capture_frames(duration=150)
    print(time() - s_time, 'secs')
    user.save_profile_image(video_capture.best_frame)

    video_capture.release()


if __name__ == "__main__":
    main()
