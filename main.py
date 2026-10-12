
# main.py

import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)

from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = str(
    Path.home() / "Downloads" / "ResearchModel" / "FSL_ILY_best.h5"
)

IMG_SIZE = 64
CAMERA_INDEX = 0
PREDICTION_INTERVAL = 5

ROI_WIDTH_RATIO = 0.50
ROI_HEIGHT_RATIO = 0.70

# Must match the alphabetical folder order in train.py.

CLASS_LABELS = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J",
    "K", "L", "M", "Mahal Kita",
    "N", "NOTHING", "O", "P",
    "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]


# =========================================================
# HOME PAGE
# =========================================================

class HomePage(QWidget):

    def __init__(self):
        super().__init__()

        self.camera_window = None

        self.setWindowIcon(QIcon("Icon_fsl.ico"))
        self.setFixedSize(450, 220)

        self.setWindowTitle(
            "Hand Alphabet Neural Detection System"
        )

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.title = QLabel("Home Page")
        self.title.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start")
        self.start_button.setFixedWidth(120)
        self.start_button.clicked.connect(self.start_app)

        self.exit_button = QPushButton("Exit")
        self.exit_button.setFixedWidth(120)
        self.exit_button.clicked.connect(self.close_app)

        left_layout.addWidget(self.title)
        left_layout.addSpacing(15)

        left_layout.addWidget(
            self.start_button,
            alignment=Qt.AlignCenter
        )

        left_layout.addSpacing(10)

        left_layout.addWidget(
            self.exit_button,
            alignment=Qt.AlignCenter
        )

        left_layout.addStretch()

        image_label = QLabel()
        image_label.setPixmap(QPixmap("homepagepic.jpg"))
        image_label.setAlignment(Qt.AlignCenter)

        right_layout.addWidget(image_label)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.apply_style()

    def start_app(self):

        self.camera_window = Camera()

        if self.camera_window.camera_ready:
            self.camera_window.show()
            self.close()

    def close_app(self):
        QApplication.quit()

    def apply_style(self):

        self.setStyleSheet(
            "QWidget { background-color: #eef2f3; }"
        )

        self.title.setStyleSheet("""
            QLabel {
                color: #263942;
                font-size: 32px;
                font-weight: bold;
                padding: 7px;
            }
        """)

        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #263942;
                color: white;
                font-size: 18px;
                padding: 7px;
                border-radius: 8px;
            }

            QPushButton:hover {
                background-color: #3d5964;
            }
        """)

        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #d9534f;
                color: white;
                font-size: 18px;
                padding: 7px;
                border-radius: 8px;
            }

            QPushButton:hover {
                background-color: #c9302c;
            }
        """)


# =========================================================
# CAMERA WINDOW
# =========================================================

class Camera(QWidget):

    def __init__(self):
        super().__init__()

        self.sentence_parts = []
        self.sentence = ""

        self.prediction = None
        self.confidence = 0.0
        self.frame_count = 0

        self.cap = None
        self.timer = None
        self.camera_ready = False

        self.labels = CLASS_LABELS

        # =================================================
        # LOAD MODEL
        # =================================================

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("Model input:", self.model.input_shape)
        print("Model output:", self.model.output_shape)
        print("Labels:", self.labels)

        # =================================================
        # VERIFY MODEL
        # =================================================

        if self.model.output_shape[-1] != len(self.labels):

            raise ValueError(
                f"Model contains "
                f"{self.model.output_shape[-1]} classes "
                f"but main.py has {len(self.labels)} labels. "
                "Retrain using all 28 classes."
            )

        if self.model.input_shape[1:] != (
            IMG_SIZE, IMG_SIZE, 1
        ):

            raise ValueError(
                "Model input must be 64x64 grayscale."
            )

        # =================================================
        # WINDOW SETTINGS
        # =================================================

        self.setWindowTitle(
            "Hand Alphabet Neural Detection System"
        )

        self.setWindowIcon(QIcon("icon_fsl.ico"))
        self.setFixedSize(900, 750)

        layout = QVBoxLayout()

        # VIDEO DISPLAY

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        layout.addWidget(
            self.video_label,
            alignment=Qt.AlignCenter
        )

        # SENTENCE DISPLAY

        self.sentence_label = QLabel("Sentence: ")

        self.sentence_label.setFont(
            QFont("Helvetica", 18)
        )

        layout.addWidget(
            self.sentence_label,
            alignment=Qt.AlignLeft
        )

        # PREDICTION DISPLAY

        self.prediction_label = QLabel("Prediction: ")

        self.prediction_label.setFont(
            QFont("Helvetica", 18)
        )

        layout.addWidget(
            self.prediction_label,
            alignment=Qt.AlignLeft
        )

        # =================================================
        # BUTTONS
        # =================================================

        button_layout = QHBoxLayout()

        self.add_btn = QPushButton("ADD")
        self.add_btn.clicked.connect(self.add_letter)

        self.delete_btn = QPushButton("DELETE")
        self.delete_btn.clicked.connect(self.delete_letter)

        self.speak_btn = QPushButton("SAY")
        self.speak_btn.clicked.connect(self.speak_letter)

        self.space_btn = QPushButton("SPACE")
        self.space_btn.clicked.connect(self.add_space)

        for button in (
            self.add_btn,
            self.delete_btn,
            self.speak_btn,
            self.space_btn
        ):
            button_layout.addWidget(button)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.apply_style()

        # =================================================
        # CAMERA INITIALIZATION
        # =================================================

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():

            self.show_camera_error()

            self.cap.release()
            self.cap = None
            return

        self.camera_ready = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # =====================================================
    # CAMERA ERROR
    # =====================================================

    @staticmethod
    def show_camera_error():

        import subprocess

        subprocess.Popen([
            sys.executable,
            "camera_error.py"
        ])

    # =====================================================
    # CAMERA FRAME AND PREDICTION
    # =====================================================

    def update_frame(self):

        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:

            self.show_camera_error()
            self.close()
            return

        # Mirror camera

        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]

        # =================================================
        # REGION OF INTEREST
        # =================================================

        roi_width = int(
            frame_width * ROI_WIDTH_RATIO
        )

        roi_height = int(
            frame_height * ROI_HEIGHT_RATIO
        )

        x1 = (frame_width - roi_width) // 2
        y1 = (frame_height - roi_height) // 2

        x2 = x1 + roi_width
        y2 = y1 + roi_height

        roi = frame[y1:y2, x1:x2].copy()

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (172, 141, 77),
            2
        )

        # =================================================
        # CNN PREDICTION
        # =================================================

        self.frame_count += 1

        if self.frame_count % PREDICTION_INTERVAL == 0:

            # Resize image

            img = cv2.resize(
                roi,
                (IMG_SIZE, IMG_SIZE)
            )

            # Convert to grayscale

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            # Normalize image

            img = img.astype(np.float32) / 255.0

            # Prepare model input

            img = img.reshape(
                (1, IMG_SIZE, IMG_SIZE, 1)
            )

            # Run prediction

            scores = self.model.predict(
                img,
                verbose=0
            )[0]

            index = int(np.argmax(scores))

            self.confidence = float(
                scores[index]
            )

            detected_class = self.labels[index]

            # =================================================
            # HANDLE NOTHING
            # =================================================

            if detected_class == "NOTHING":

                self.prediction = None

                self.prediction_label.setText(
                    "Prediction: No sign detected"
                )

            # =================================================
            # ACCEPT HIGHEST PREDICTION
            # NO UNCERTAIN STATE
            # =================================================

            else:

                self.prediction = detected_class

                display_label = detected_class.upper()

                self.prediction_label.setText(
                    f"Prediction: {display_label} "
                    f"({self.confidence * 100:.1f}%)"
                )

        # =================================================
        # DISPLAY CAMERA FRAME
        # =================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = rgb_frame.shape

        qt_image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888
        )

        self.video_label.setPixmap(
            QPixmap.fromImage(qt_image.copy())
        )

    # =====================================================
    # UPDATE SENTENCE DISPLAY
    # =====================================================

    def refresh_sentence(self):

        self.sentence = "".join(
            self.sentence_parts
        )

        self.sentence_label.setText(
            "Sentence: " + self.sentence
        )

    # =====================================================
    # ADD LETTER OR MAHAL KITA
    # =====================================================

    def add_letter(self):

        if self.prediction is None:

            self.prediction_label.setText(
                "No valid sign detected."
            )

            return

        if self.prediction.upper() == "MAHAL KITA":

            entry = "MAHAL KITA"

        else:

            entry = self.prediction

        self.sentence_parts.append(entry)

        self.refresh_sentence()

    # =====================================================
    # DELETE LAST ADDED ENTRY
    # =====================================================

    def delete_letter(self):

        if self.sentence_parts:

            self.sentence_parts.pop()

            self.refresh_sentence()

    # =====================================================
    # TEXT TO SPEECH
    # =====================================================

    def speak_letter(self):

        if not self.sentence.strip():
            return

        engine = pyttsx3.init()

        engine.say(self.sentence)

        engine.runAndWait()

    # =====================================================
    # ADD SPACE
    # =====================================================

    def add_space(self):

        self.sentence_parts.append(" ")

        self.refresh_sentence()

    # =====================================================
    # CAMERA CLEANUP
    # =====================================================

    def closeEvent(self, event):

        if self.timer is not None:
            self.timer.stop()

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()

        event.accept()

    # =====================================================
    # BUTTON STYLING
    # =====================================================

    def apply_style(self):

        self.setStyleSheet(
            "QWidget { background-color: #eef2f3; }"
        )

        button_style = """
            QPushButton {
                background-color: #263942;
                color: white;
                font-size: 18px;
                padding: 7px;
                border-radius: 8px;
            }

            QPushButton:hover {
                background-color: #3d5964;
            }
        """

        for button in (
            self.add_btn,
            self.delete_btn,
            self.speak_btn,
            self.space_btn
        ):

            button.setStyleSheet(button_style)


# =========================================================
# MAIN APPLICATION
# =========================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = HomePage()
    window.show()

    sys.exit(app.exec_())
