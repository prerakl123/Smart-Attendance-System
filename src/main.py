import cv2
import sqlite3

from src.constants import HAARCASCADE_FILE_PATH, ATTENDANCE_DB_PATH


class FaceDetection:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_FILE_PATH)
        self.conn = sqlite3.connect(ATTENDANCE_DB_PATH)
        self.cursor = self.conn.cursor()
        self.create_user_table()

    def create_user_table(self):
        """
        Create the users table in the database if it doesn't exist.

        :return: None
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT NOT NULL, 
            profile_image BLOB)
            """
        )
        self.conn.commit()

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
        self.conn.close()


def main():
    face_detection = FaceDetection()
    face_detection.detect_faces()


if __name__ == "__main__":
    main()
