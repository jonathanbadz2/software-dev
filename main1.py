
# main1.py

import sys
import random
import cv2

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout
)

from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt


# =========================================================
# SETTINGS
# =========================================================

CAMERA_INDEX = 0

# Total display time for each letter: 5 seconds
LETTER_DURATION = 3000

# First 2 seconds: fluctuate between similar letters
CONFUSION_DURATION = 2000

# Last 3 seconds: display the correct letter
STABLE_DURATION = 3000

# Update displayed prediction every 80 milliseconds
FLUCTUATION_INTERVAL = 80

MIN_PERCENTAGE = 75
MAX_PERCENTAGE = 99

ROI_WIDTH_RATIO = 0.50
ROI_HEIGHT_RATIO = 0.70


# =========================================================
# LETTER CONFUSION SETTINGS
# =========================================================

CONFUSION_PAIRS = {
    "A": "S",
    "S": "A",
    "M": "N",
    "N": "M",
    "P": "Q",
    "Q": "P"
}

# J initially appears as I
INITIAL_GUESSES = {
    "J": "I"
}


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

        self.cap = None
        self.camera_timer = None

        self.demo_index = 0
        self.current_letter = None

        # Prediction phases:
        # fluctuating, stable, normal, finished
        self.phase = None

        # Alternates between correct and confused letter
        self.show_wrong_next = True

        # =================================================
        # TIMERS
        # =================================================

        # Controls transition between letters
        self.demo_timer = QTimer(self)
        self.demo_timer.setSingleShot(True)
        self.demo_timer.timeout.connect(
            self.advance_sequence
        )

        # Updates letter and percentage
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
        self.setFixedSize(900, 750)

        layout = QVBoxLayout()

        # =================================================
        # VIDEO DISPLAY
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

        layout.addWidget(
            self.sentence_label,
            alignment=Qt.AlignLeft
        )

        # =================================================
        # PREDICTION DISPLAY
        # =================================================

        self.prediction_label = QLabel("Prediction: ")

        self.prediction_label.setFont(
            QFont("Helvetica", 18)
        )

        self.prediction_label.setWordWrap(True)

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

        self.space_btn = QPushButton("SPACE")
        self.space_btn.clicked.connect(
            self.add_space
        )

        for button in (
            self.add_btn,
            self.delete_btn,
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

        # Automatically start A-Z
        QTimer.singleShot(0, self.start_sequence)

    # =====================================================
    # START AUTOMATIC A-Z SEQUENCE
    # =====================================================

    def start_sequence(self):

        self.demo_index = 0
        self.next_letter()

    # =====================================================
    # DISPLAY NEXT LETTER
    # =====================================================

    def next_letter(self):

        # =================================================
        # END OF A-Z SEQUENCE
        # =================================================

        if self.demo_index >= 26:

            self.demo_timer.stop()
            self.percentage_timer.stop()

            self.phase = "finished"
            return

        # =================================================
        # GET NEXT LETTER
        # =================================================

        self.current_letter = chr(
            ord("A") + self.demo_index
        )

        self.demo_index += 1

        # =================================================
        # SPECIAL LETTERS
        # A/S, M/N, J, P/Q
        # =================================================

        if (
            self.current_letter in CONFUSION_PAIRS
            or self.current_letter in INITIAL_GUESSES
        ):

            self.phase = "fluctuating"

            self.show_wrong_next = True

            # Immediately show the first guess
            self.update_percentage()

            # Rapid prediction changes
            self.percentage_timer.start(
                FLUCTUATION_INTERVAL
            )

            # Fluctuate for 2 seconds
            self.demo_timer.start(
                CONFUSION_DURATION
            )

            return

        # =================================================
        # NORMAL LETTERS
        # =================================================

        self.phase = "normal"

        self.display_prediction(
            self.current_letter
        )

        self.percentage_timer.start(
            FLUCTUATION_INTERVAL
        )

        self.demo_timer.start(
            LETTER_DURATION
        )

    # =====================================================
    # ADVANCE BETWEEN PHASES
    # =====================================================

    def advance_sequence(self):

        # =================================================
        # AFTER 2 SECONDS:
        # LOCK THE CORRECT LETTER FOR 3 SECONDS
        # =================================================

        if self.phase == "fluctuating":

            self.phase = "stable"

            self.display_prediction(
                self.current_letter
            )

            self.demo_timer.start(
                STABLE_DURATION
            )

            return

        # =================================================
        # AFTER 3 SECONDS:
        # MOVE TO NEXT LETTER
        # =================================================

        self.percentage_timer.stop()
        self.next_letter()

    # =====================================================
    # DISPLAY PREDICTION WITH RANDOM CONFIDENCE
    # =====================================================

    def display_prediction(self, letter):

        self.prediction = letter

        self.confidence = random.randint(
            MIN_PERCENTAGE,
            MAX_PERCENTAGE
        )

        self.prediction_label.setText(
            f"Prediction: {letter} "
            f"({self.confidence}%)"
        )

    # =====================================================
    # UPDATE PREDICTION EVERY 80 MILLISECONDS
    # =====================================================

    def update_percentage(self):

        if self.current_letter is None:
            return

        # =================================================
        # CONFUSION PHASE
        # =================================================

        if self.phase == "fluctuating":

            # J initially appears as I
            if self.current_letter in INITIAL_GUESSES:

                wrong_letter = INITIAL_GUESSES[
                    self.current_letter
                ]

                self.display_prediction(
                    wrong_letter
                )

                return

            # Alternate between similar letters
            if self.current_letter in CONFUSION_PAIRS:

                wrong_letter = CONFUSION_PAIRS[
                    self.current_letter
                ]

                if self.show_wrong_next:

                    displayed_letter = wrong_letter

                else:

                    displayed_letter = self.current_letter

                self.show_wrong_next = (
                    not self.show_wrong_next
                )

                self.display_prediction(
                    displayed_letter
                )

                return

        # =================================================
        # STABLE PHASE
        # CORRECT LETTER ONLY
        # =================================================

        if self.phase in ("stable", "normal"):

            self.display_prediction(
                self.current_letter
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
    # ADD CURRENT DISPLAYED LETTER
    # =====================================================

    def add_letter(self):

        if self.prediction is None:
            return

        self.sentence_parts.append(
            self.prediction
        )

        self.refresh_sentence()

    # =====================================================
    # DELETE LAST LETTER
    # =====================================================

    def delete_letter(self):

        if self.sentence_parts:

            self.sentence_parts.pop()
            self.refresh_sentence()

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
