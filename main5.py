
# =========================================================
# main5.py
# CUSTOMIZABLE HAND ALPHABET DEMONSTRATION TEMPLATE
# =========================================================

import sys
import random
import cv2
import pyttsx3

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout
)

from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt


# =========================================================
# EDIT YOUR LETTER SEQUENCE HERE
# =========================================================

LETTER_SEQUENCE = [
    "H", "E", "L", "L", 
    "NOTHING",
    "O",
    "NOTHING",
    "W", "O", "R", "L", "D"
]

# Edit only this list to change the demonstration.
#
# "A" to "Z"   = Display individual letters
# "MAHAL KITA" = Display entire phrase
# "NOTHING"    = No sign detected for 3 seconds
# "SPACE"      = No sign detected for 3 seconds
#
# Initial and final NOTHING states are automatic.
# You do not need to include them in the sequence.


# =========================================================
# SETTINGS
# =========================================================

CAMERA_INDEX = 0

START_DELAY = 2000           # Initial delay: 2 seconds
LETTER_DURATION = 3000       # Each letter: 3 seconds
NOTHING_DURATION = 3000      # NOTHING: 3 seconds
FLUCTUATION_INTERVAL = 200   # Percentage changes every 0.2 seconds

MIN_PERCENTAGE = 75
MAX_PERCENTAGE = 99

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 750

ROI_WIDTH_RATIO = 0.50
ROI_HEIGHT_RATIO = 0.70


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
        image_label.setPixmap(
            QPixmap("homepagepic.jpg")
        )
        image_label.setAlignment(Qt.AlignCenter)

        right_layout.addWidget(image_label)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.apply_style()

    def start_app(self):

        self.camera_window = Camera()

        QApplication.instance().camera_window = (
            self.camera_window
        )

        self.camera_window.show()
        self.hide()

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

        # =================================================
        # VARIABLES
        # =================================================

        self.sentence_parts = []
        self.sentence = ""

        self.prediction = None
        self.confidence = 0

        self.demo_index = 0
        self.current_letter = None
        self.sequence_finished = False

        self.cap = None
        self.camera_timer = None

        # =================================================
        # TIMERS
        # =================================================

        # Controls the delay before the next entry
        self.sequence_timer = QTimer(self)
        self.sequence_timer.setSingleShot(True)
        self.sequence_timer.timeout.connect(
            self.next_letter
        )

        # Changes percentage while a letter is displayed
        self.percentage_timer = QTimer(self)
        self.percentage_timer.timeout.connect(
            self.update_percentage
        )

        # =================================================
        # WINDOW SETTINGS
        # =================================================

        self.setWindowTitle(
            "Hand Alphabet Neural Detection System"
        )

        self.setWindowIcon(QIcon("icon_fsl.ico"))

        self.setFixedSize(
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )

        layout = QVBoxLayout()

        # =================================================
        # CAMERA DISPLAY
        # =================================================

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(
            self.video_label,
            alignment=Qt.AlignCenter
        )

        # =================================================
        # SENTENCE DISPLAY
        # =================================================

        self.sentence_label = QLabel("Sentence: ")

        self.sentence_label.setFont(
            QFont("Helvetica", 18)
        )

        self.sentence_label.setWordWrap(True)

        layout.addWidget(self.sentence_label)

        # =================================================
        # PREDICTION DISPLAY
        # =================================================

        self.prediction_label = QLabel(
            "Prediction: No sign detected"
        )

        self.prediction_label.setFont(
            QFont("Helvetica", 18)
        )

        self.prediction_label.setWordWrap(True)
        self.prediction_label.setMinimumHeight(45)

        layout.addWidget(self.prediction_label)

        # =================================================
        # BUTTONS
        # =================================================

        button_layout = QHBoxLayout()

        self.add_btn = QPushButton("ADD")
        self.add_btn.clicked.connect(self.add_letter)

        self.delete_btn = QPushButton("DELETE")
        self.delete_btn.clicked.connect(
            self.delete_letter
        )

        self.speak_btn = QPushButton("SAY")
        self.speak_btn.clicked.connect(
            self.speak_letter
        )

        self.space_btn = QPushButton("SPACE")
        self.space_btn.clicked.connect(
            self.add_space
        )

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

        if self.cap.isOpened():

            self.camera_timer = QTimer(self)
            self.camera_timer.timeout.connect(
                self.update_frame
            )
            self.camera_timer.start(30)

        else:
            print("Camera could not be opened.")

        # Initial No sign detected lasts 2 seconds.
        self.show_nothing()
        self.sequence_timer.start(START_DELAY)

    # =====================================================
    # DISPLAY NO SIGN DETECTED
    # =====================================================

    def show_nothing(self):

        self.current_letter = None
        self.prediction = None
        self.confidence = 0

        self.percentage_timer.stop()

        self.prediction_label.setText(
            "Prediction: No sign detected"
        )

    # =====================================================
    # NEXT ENTRY IN CUSTOM SEQUENCE
    # =====================================================

    def next_letter(self):

        # After the final entry, stay on NOTHING.
        if self.demo_index >= len(LETTER_SEQUENCE):

            self.sequence_finished = True
            self.sequence_timer.stop()
            self.show_nothing()
            return

        entry = str(
            LETTER_SEQUENCE[self.demo_index]
        ).strip().upper()

        self.demo_index += 1

        # =================================================
        # NOTHING / SPACE PAUSE: 3 SECONDS
        # =================================================

        if entry in ("NOTHING", "SPACE"):

            self.current_letter = entry
            self.show_nothing()

            self.sequence_timer.start(
                NOTHING_DURATION
            )
            return

        # =================================================
        # REGULAR LETTER OR PHRASE
        # =================================================

        self.current_letter = entry
        self.prediction = entry

        self.update_percentage()

        self.percentage_timer.start(
            FLUCTUATION_INTERVAL
        )

        self.sequence_timer.start(
            LETTER_DURATION
        )

    # =====================================================
    # FLUCTUATING PERCENTAGE
    # =====================================================

    def update_percentage(self):

        if self.prediction is None:
            return

        self.confidence = random.randint(
            MIN_PERCENTAGE,
            MAX_PERCENTAGE
        )

        self.prediction_label.setText(
            f"Prediction: {self.current_letter} "
            f"({self.confidence}%)"
        )

    # =====================================================
    # CAMERA DISPLAY ONLY
    # CNN PREDICTIONS DISABLED
    # =====================================================

    def update_frame(self):

        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()

        if not ret:
            return

        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]

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

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (172, 141, 77),
            2
        )

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
    # REFRESH SENTENCE
    # =====================================================

    def refresh_sentence(self):

        self.sentence = "".join(
            self.sentence_parts
        )

        self.sentence_label.setText(
            "Sentence: " + self.sentence
        )

    # =====================================================
    # ADD CURRENT ENTRY
    # =====================================================

    def add_letter(self):

        if self.current_letter == "SPACE":
            self.add_space()
            return

        if self.prediction is None:
            return

        self.sentence_parts.append(
            self.prediction
        )

        self.refresh_sentence()

    # =====================================================
    # DELETE LAST ENTRY
    # =====================================================

    def delete_letter(self):

        if self.sentence_parts:

            self.sentence_parts.pop()
            self.refresh_sentence()

    # =====================================================
    # TEXT TO SPEECH - SAY BUTTON ONLY
    # =====================================================

    def speak_letter(self):

        if not self.sentence.strip():
            return

        engine = pyttsx3.init()

        engine.say(self.sentence)
        engine.runAndWait()
        engine.stop()

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

        self.sequence_timer.stop()
        self.percentage_timer.stop()

        if self.camera_timer is not None:
            self.camera_timer.stop()

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
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = HomePage()
    window.show()

    sys.exit(app.exec_())
