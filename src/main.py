from time import time

import cv2
import sqlite3

from src.constants import HAARCASCADE_FILE_PATH, ATTENDANCE_DB_PATH
from src.user_registration import VideoCapture, User


class AttendanceSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_FILE_PATH)
        self.create_user_table()

    def create_user_table(self):
        """
        Create the users table in the database if it doesn't exist.

        :return: None
        """
        conn = sqlite3.connect(ATTENDANCE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT NOT NULL, 
            profile_image BLOB)
            """
        )
        conn.commit()
        conn.close()

    def detect_faces(self):
        """
        Detect Faces using the VideoCapture function and default webcam

        :return: None
        """
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def main(self):
        user_registration_choice = input("Do you want to register a new user? (yes/no): ").strip().lower()

        if user_registration_choice == "yes":
            name = input("Enter your name: ")

            user = User(name)
            video_capture = VideoCapture(user)

            # Capture video frames and save the best image
            s_time = time()
            video_capture.capture_frames(duration=150)
            print('Took', time() - s_time, 'secs')

            # Save the user's profile image
            user.save_profile_image(video_capture.best_frame)

            # Release video feed and cleanup
            video_capture.release()


if __name__ == "__main__":
    attendance_system = AttendanceSystem()
    attendance_system.detect_faces()
    attendance_system.main()
