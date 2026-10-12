
# main4.py

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
# SETTINGS
# =========================================================

LETTER_SEQUENCE = [
    "MAHAL KITA", "J", "I", "R", "A", "H", "NOTHING"
]

CAMERA_INDEX = 0

START_DELAY = 2000
LETTER_DURATION = 3000
DELAY_AFTER_MAHAL_KITA = 3000
FLUCTUATION_INTERVAL = 200

MIN_PERCENTAGE = 75
MAX_PERCENTAGE = 99

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
        image_label.setPixmap(QPixmap("homepagepic.jpg"))
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

        self.sentence_parts = []
        self.sentence = ""

        self.prediction = None
        self.confidence = 0

        self.cap = None
        self.camera_timer = None

        self.demo_index = 0
        self.current_letter = None
        self.pause_done = False

        # =================================================
        # TIMERS
        # =================================================

        self.demo_timer = QTimer(self)
        self.demo_timer.timeout.connect(self.next_letter)

        self.percentage_timer = QTimer(self)
        self.percentage_timer.timeout.connect(
            self.update_percentage
        )

        # Timer for pause after MAHAL KITA
        self.pause_timer = QTimer(self)
        self.pause_timer.setSingleShot(True)
        self.pause_timer.timeout.connect(
            self.resume_sequence
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

        # =================================================
        # VIDEO DISPLAY
        # =================================================

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

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

        layout.addWidget(
            self.sentence_label,
            alignment=Qt.AlignLeft
        )

        # =================================================
        # PREDICTION DISPLAY
        # =================================================

        # Show No sign detected during initial delay
        self.prediction_label = QLabel(
            "Prediction: No sign detected"
        )
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

        # Start after 2 seconds
        QTimer.singleShot(
            START_DELAY,
            self.start_sequence
        )

    # =====================================================
    # START SEQUENCE
    # =====================================================

    def start_sequence(self):

        self.demo_index = 0
        self.current_letter = None
        self.pause_done = False

        self.next_letter()

        self.demo_timer.start(LETTER_DURATION)
        self.percentage_timer.start(
            FLUCTUATION_INTERVAL
        )

    # =====================================================
    # NEXT LETTER OR PHRASE
    # =====================================================

    def next_letter(self):

        # Pause after MAHAL KITA
        if (
            self.current_letter == "MAHAL KITA"
            and not self.pause_done
        ):
            self.pause_done = True

            self.demo_timer.stop()
            self.percentage_timer.stop()

            self.current_letter = None
            self.prediction = None
            self.confidence = 0

            self.prediction_label.setText(
                "Prediction: No sign detected"
            )

            self.pause_timer.start(
                DELAY_AFTER_MAHAL_KITA
            )
            return

        # End of sequence
        if self.demo_index >= len(LETTER_SEQUENCE):
            self.demo_timer.stop()
            self.percentage_timer.stop()
            return

        # Display next entry
        self.current_letter = LETTER_SEQUENCE[self.demo_index]
        self.demo_index += 1

        if self.current_letter == "NOTHING":
            self.prediction = None
        else:
            self.prediction = self.current_letter

        self.update_percentage()

    # =====================================================
    # RESUME AFTER MAHAL KITA
    # =====================================================

    def resume_sequence(self):

        self.next_letter()

        self.demo_timer.start(LETTER_DURATION)
        self.percentage_timer.start(
            FLUCTUATION_INTERVAL
        )

    # =====================================================
    # FLUCTUATING PERCENTAGE
    # =====================================================

    def update_percentage(self):

        if self.current_letter is None:
            return

        # Handle NOTHING
        if self.current_letter == "NOTHING":
            self.confidence = 0
            self.prediction_label.setText(
                "Prediction: No sign detected"
            )
            return

        # Random percentage for other entries
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

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (172, 141, 77),
            2
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
    # SENTENCE MANAGEMENT
    # =====================================================

    def refresh_sentence(self):

        self.sentence = "".join(
            self.sentence_parts
        )

        self.sentence_label.setText(
            "Sentence: " + self.sentence
        )

    # =====================================================
    # ADD LETTER OR PHRASE
    # =====================================================

    def add_letter(self):

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

        self.demo_timer.stop()
        self.percentage_timer.stop()
        self.pause_timer.stop()

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
