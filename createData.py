# createData.py
import cv2
import os
import time
import re

# =========================================================
# SETTINGS
# =========================================================

OUTPUT_FOLDER = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "ResearchImage"
)

CAMERA_INDEX = 0

CAPTURE_INTERVAL = 0.5

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320

MAX_IMAGES = 3000

# ROI as a proportion of the camera frame
ROI_WIDTH_RATIO = 0.50
ROI_HEIGHT_RATIO = 0.70

# =========================================================
# CREATE MAHAL KITA FOLDER
# =========================================================

letters = ["Mahal Kita"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for letter in letters:
    os.makedirs(
        os.path.join(OUTPUT_FOLDER, letter),
        exist_ok=True
    )

# =========================================================
# COUNT EXISTING IMAGES AND FIND NEXT FILE NUMBER
# =========================================================

image_counts = {}
next_numbers = {}

for letter in letters:

    folder = os.path.join(OUTPUT_FOLDER, letter)

    existing_images = [
        filename for filename in os.listdir(folder)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_counts[letter] = len(existing_images)

    highest_number = 0

    pattern = re.compile(
        rf"^{re.escape(letter)}_(\d+)\.(jpg|jpeg|png)$",
        re.IGNORECASE
    )

    for filename in existing_images:

        match = pattern.match(filename)

        if match:
            number = int(match.group(1))
            highest_number = max(highest_number, number)

    next_numbers[letter] = highest_number + 1

# =========================================================
# OPEN CAMERA
# =========================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================================================
# VARIABLES
# =========================================================

current_index = 0
capturing = False
last_capture_time = 0.0

# =========================================================
# SAVE IMAGE FUNCTION
# =========================================================

def save_image(image, letter):

    if image_counts[letter] >= MAX_IMAGES:
        print(f"{letter} already has {MAX_IMAGES} images.")
        return False

    folder = os.path.join(OUTPUT_FOLDER, letter)

    while True:

        number = next_numbers[letter]

        filename = os.path.join(
            folder,
            f"{letter}_{number:05d}.jpg"
        )

        next_numbers[letter] += 1

        if not os.path.exists(filename):
            break

    success = cv2.imwrite(filename, image)

    if success:
        image_counts[letter] += 1

        print(
            f"Saved {letter}: "
            f"{image_counts[letter]}/{MAX_IMAGES}"
        )

        return True

    print("Error saving:", filename)
    return False

# =========================================================
# INSTRUCTIONS
# =========================================================

print("=" * 50)
print("MAHAL KITA DATASET COLLECTOR")
print("=" * 50)
print("SPACE : Start / Stop automatic capture")
print("S     : Capture one image")
print("Q/ESC : Quit")
print("=" * 50)

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error reading camera frame.")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]

    # =====================================================
    # REGION OF INTEREST
    # =====================================================

    roi_width = int(frame_width * ROI_WIDTH_RATIO)
    roi_height = int(frame_height * ROI_HEIGHT_RATIO)

    x1 = (frame_width - roi_width) // 2
    y1 = (frame_height - roi_height) // 2

    x2 = x1 + roi_width
    y2 = y1 + roi_height

    roi = frame[y1:y2, x1:x2].copy()

    # Resize only the ROI for saving
    clean_image = cv2.resize(
        roi,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    current_letter = letters[current_index]

    # =====================================================
    # AUTOMATIC CAPTURE
    # =====================================================

    current_time = time.monotonic()

    if capturing:

        if image_counts[current_letter] >= MAX_IMAGES:

            capturing = False
            print(
                f"Completed {current_letter}: "
                f"{MAX_IMAGES} images."
            )

        elif current_time - last_capture_time >= CAPTURE_INTERVAL:

            save_image(clean_image, current_letter)
            last_capture_time = current_time

            if image_counts[current_letter] >= MAX_IMAGES:
                capturing = False
                print(f"Completed {current_letter}.")

    # =====================================================
    # DISPLAY
    # =====================================================

    display = frame.copy()

    # Draw ROI rectangle only on display
    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    status = "CAPTURING" if capturing else "PAUSED"

    cv2.putText(
        display,
        f"CLASS: {current_letter}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"STATUS: {status}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255) if capturing else (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"IMAGES: {image_counts[current_letter]}/{MAX_IMAGES}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Progress bar
    progress = min(
        image_counts[current_letter] / MAX_IMAGES,
        1.0
    )

    bar_x = 10
    bar_y = 115
    bar_width = 220
    bar_height = 15

    cv2.rectangle(
        display,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (100, 100, 100),
        -1
    )

    cv2.rectangle(
        display,
        (bar_x, bar_y),
        (
            bar_x + int(bar_width * progress),
            bar_y + bar_height
        ),
        (0, 200, 0),
        -1
    )

    cv2.putText(
        display,
        "SPACE: Auto | S: Save | Q: Quit",
        (10, frame_height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "Mahal Kita Dataset Collector",
        display
    )

    # =====================================================
    # KEYBOARD CONTROLS
    # =====================================================

    key = cv2.waitKeyEx(1)

    # Quit
    if key in (ord("q"), ord("Q"), 27):
        break

    # Automatic capture
    elif key == 32:

        if capturing:

            capturing = False
            print("Capture paused.")

        elif image_counts[current_letter] >= MAX_IMAGES:

            print(
                f"{current_letter} already reached "
                f"{MAX_IMAGES} images."
            )

        else:

            capturing = True
            last_capture_time = (
                time.monotonic() - CAPTURE_INTERVAL
            )

            print(
                f"Started capturing {current_letter}"
            )

    # Manual capture
    elif key in (ord("s"), ord("S")):

        save_image(clean_image, current_letter)

# =========================================================
# CLEAN UP
# =========================================================

cap.release()
cv2.destroyAllWindows()

print("=" * 50)
print("Dataset collection finished.")
print("Saved to:", OUTPUT_FOLDER)

print("\nIMAGE COUNTS:")

total = 0

for letter in letters:

    count = image_counts[letter]
    total += count

    print(f"{letter}: {count}/{MAX_IMAGES}")

print("-" * 50)
print(f"TOTAL IMAGES: {total}")
print("=" * 50)