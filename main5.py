
# =========================================================
# main5.py
# CUSTOMIZABLE HAND ALPHABET DEMONSTRATION
# WITH PREDICTION FLUCTUATION
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
# EDIT ONLY THIS SECTION TO CHANGE YOUR WORDS
# =========================================================

LETTER_SEQUENCE = [
    "S", "I", 
    "L",
    "L", "I",
    "M",
    "A", "N",
    "NOTHING"
]


# =========================================================
# PREDICTION CONFUSION SETTINGS
# =========================================================

# Letters that initially fluctuate between each other.
# The dictionary key is the CORRECT letter.
# The value is the letter it may be confused with.

CONFUSION_PAIRS = {
    "A": "S",
    "S": "A",
    "M": "N",
    "N": "M",
    "P": "Q",
    "Q": "P"
}

# J initially displays I, then settles on J.
INITIAL_WRONG_GUESSES = {
    "J": "I"
}


# =========================================================
# TIMING SETTINGS
# =========================================================

CAMERA_INDEX = 0

START_DELAY = 2000

# Normal letters
LETTER_DURATION = 3000

# Confusing letters
FLUCTUATION_DURATION = 2000
STABLE_DURATION = 3000

# No sign detected
NOTHING_DURATION = 3000

# Prediction percentage/letter update
FLUCTUATION_INTERVAL = 80

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

        # Correct target from LETTER_SEQUENCE
        self.target_letter = None

        # Current entry, including NOTHING or SPACE
        self.current_entry = None

        # Current display phase
        self.phase = "initial"

        # Controls alternating prediction guesses
        self.show_wrong_next = True

        self.sequence_finished = False

        self.cap = None
        self.camera_timer = None

        # =================================================
        # TIMERS
        # =================================================

        # Moves from one entry or phase to the next
        self.sequence_timer = QTimer(self)
        self.sequence_timer.setSingleShot(True)
        self.sequence_timer.timeout.connect(
            self.advance_sequence
        )

        # Updates predictions and their percentages
        self.fluctuation_timer = QTimer(self)
        self.fluctuation_timer.timeout.connect(
            self.update_prediction
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

        # Initial 2-second No sign detected
        self.show_nothing()
        self.sequence_timer.start(START_DELAY)

    # =====================================================
    # DISPLAY NO SIGN DETECTED
    # =====================================================

    def show_nothing(self):

        self.prediction = None
        self.confidence = 0

        self.fluctuation_timer.stop()

        self.prediction_label.setText(
            "Prediction: No sign detected"
        )

    # =====================================================
    # DISPLAY A GIVEN PREDICTION
    # =====================================================

    def display_letter(self, letter):

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
    # BEGIN THE NEXT LETTER
    # =====================================================

    def next_letter(self):

        # =================================================
        # END OF SEQUENCE - NOTHING FOREVER
        # =================================================

        if self.demo_index >= len(LETTER_SEQUENCE):

            self.sequence_finished = True
            self.phase = "finished"

            self.current_entry = None
            self.target_letter = None

            self.sequence_timer.stop()
            self.show_nothing()

            return

        # =================================================
        # READ NEXT ENTRY
        # =================================================

        entry = str(
            LETTER_SEQUENCE[self.demo_index]
        ).strip().upper()

        self.demo_index += 1

        self.current_entry = entry
        self.target_letter = entry

        # =================================================
        # NOTHING / SPACE
        # =================================================

        if entry in ("NOTHING", "SPACE"):

            self.phase = "nothing"
            self.show_nothing()

            self.sequence_timer.start(
                NOTHING_DURATION
            )

            return

        # =================================================
        # LETTERS THAT REQUIRE FLUCTUATION
        # =================================================

        if (
            entry in CONFUSION_PAIRS
            or entry in INITIAL_WRONG_GUESSES
        ):

            self.phase = "fluctuating"

            self.show_wrong_next = True

            # Show the first guess immediately
            self.update_prediction()

            # Keep updating guesses during first 2 sec
            self.fluctuation_timer.start(
                FLUCTUATION_INTERVAL
            )

            # Finish fluctuation after 2 sec
            self.sequence_timer.start(
                FLUCTUATION_DURATION
            )

            return

        # =================================================
        # NORMAL LETTERS
        # =================================================

        self.phase = "normal"

        self.display_letter(entry)

        self.fluctuation_timer.start(
            FLUCTUATION_INTERVAL
        )

        self.sequence_timer.start(
            LETTER_DURATION
        )

    # =====================================================
    # ADVANCE BETWEEN LETTERS AND PHASES
    # =====================================================

    def advance_sequence(self):

        # =================================================
        # AFTER FLUCTUATION:
        # SHOW THE CORRECT LETTER FOR 3 SECONDS
        # =================================================

        if self.phase == "fluctuating":

            self.phase = "stable"

            # Correct letter is now locked
            self.display_letter(
                self.target_letter
            )

            # Percentage can still fluctuate,
            # but the LETTER cannot change.
            self.fluctuation_timer.start(
                FLUCTUATION_INTERVAL
            )

            self.sequence_timer.start(
                STABLE_DURATION
            )

            return

        # =================================================
        # AFTER STABLE / NORMAL / NOTHING:
        # MOVE TO NEXT ENTRY
        # =================================================

        self.fluctuation_timer.stop()
        self.next_letter()

    # =====================================================
    # UPDATE PREDICTION
    # =====================================================

    def update_prediction(self):

        if self.target_letter is None:
            return

        # =================================================
        # FLUCTUATION PHASE
        # =================================================

        if self.phase == "fluctuating":

            # J starts as I, then becomes J
            if self.target_letter in INITIAL_WRONG_GUESSES:

                wrong_guess = INITIAL_WRONG_GUESSES[
                    self.target_letter
                ]

                # During the initial 2 seconds,
                # J is shown as I.
                self.display_letter(wrong_guess)

                return

            # Alternate between similar letters
            if self.target_letter in CONFUSION_PAIRS:

                wrong_guess = CONFUSION_PAIRS[
                    self.target_letter
                ]

                if self.show_wrong_next:

                    displayed_letter = wrong_guess

                else:

                    displayed_letter = self.target_letter

                # Switch the next displayed letter
                self.show_wrong_next = (
                    not self.show_wrong_next
                )

                self.display_letter(
                    displayed_letter
                )

                return

        # =================================================
        # STABLE / NORMAL PHASE
        # =================================================

        if self.phase in ("stable", "normal"):

            # Correct letter stays consistent.
            # Only the percentage fluctuates.
            self.display_letter(
                self.target_letter
            )

    # =====================================================
    # CAMERA DISPLAY
    # CNN MODEL IS NOT USED IN THIS TEMPLATE
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
        # CONVERT FRAME FOR PYQT
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
    # ADD CURRENT DISPLAYED LETTER
    # =====================================================

    def add_letter(self):

        if (
            self.current_entry == "SPACE"
            and self.phase == "nothing"
        ):
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
        self.fluctuation_timer.stop()

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
