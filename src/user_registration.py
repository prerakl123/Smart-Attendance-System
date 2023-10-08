import os
from time import time

import cv2
import sqlite3

from src.constants import HAARCASCADE_FILE_PATH, ATTENDANCE_DB_PATH, IMAGES_DIR_PATH, VIDEO_CLIPS_DIR_PATH
from src.errors import WebcamError


class User:
    name: str
    user_id: int

    def __init__(self, name):
        self.name = name

    def save_profile_image(self, image):
        """
        Save the user's profile image to the database.

        :param image: binary image data
        :return: None
        """
        conn = sqlite3.connect(ATTENDANCE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, profile_image) VALUES (?, ?)",
            (self.name, image)
        )

        # Get the ID assigned by the database
        self.user_id = cursor.lastrowid

        # Commit and Close DB connection
        conn.commit()
        conn.close()

    @classmethod
    def get_user_data(cls, user_id):
        """
        Gets the user data for the specified `user_id`.

        :param user_id:
        :return:
        """
        conn = sqlite3.connect(ATTENDANCE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
        user_data = cursor.fetchone()
        conn.close()
        return user_data

    def create_user_folders(self):
        """
        Create and set User's media folders.

        :return: None
        """
        user_image_folder = os.path.join(IMAGES_DIR_PATH, str(self.user_id))
        user_video_folder = os.path.join(VIDEO_CLIPS_DIR_PATH, str(self.user_id))

        if not os.path.exists(user_image_folder):
            os.makedirs(user_image_folder)

        if not os.path.exists(user_video_folder):
            os.makedirs(user_video_folder)


class VideoCapture:
    video_writer: cv2.VideoWriter

    def __init__(self, user: User):
        self.video_path = None
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise WebcamError("Cannot open webcam!")
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_FILE_PATH)
        self.best_frame = None
        self.max_accuracy = 0

        # Save user class for folder creation
        self.user = user
        # Create user-specific folders
        self.create_user_folders()

    def capture_frames(self, duration):
        """
        Capture frames for specified `duration`

        :param duration: integer time in seconds
        :return: None
        """
        # Create a VideoWriter object to save the video
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.video_path = f"{VIDEO_CLIPS_DIR_PATH}/{self.user.user_id}/{self.user.name}.mp4"
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 60.0, (640, 480))

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

    def create_user_folders(self):
        user_video_folder = os.path.join(VIDEO_CLIPS_DIR_PATH, str(self.user.user_id))

        if not os.path.exists(user_video_folder):
            os.makedirs(user_video_folder)

    def save_best_image(self):
        if self.best_frame is not None:
            image_path = f"{IMAGES_DIR_PATH}/{self.user.user_id}/{self.user.name}.jpg"
            cv2.imwrite(image_path, self.best_frame)

    def release(self):
        """
        Release video feed and destroy all cv2 windows.

        :return: None
        """
        self.cap.release()
        self.video_writer
        cv2.destroyAllWindows()


def main():
    name = input("Enter your name: ")

    user = User(name)
    video_capture = VideoCapture(user)
    s_time = time()
    video_capture.capture_frames(duration=150)
    print('Took', time() - s_time, 'secs')

    user.save_profile_image(video_capture.best_frame)

    video_capture.release()


if __name__ == "__main__":
    main()
