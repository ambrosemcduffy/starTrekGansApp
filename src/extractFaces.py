import cv2
import os
import glob
import dlib

input_folder = "/mnt/e/gansStudy/videosToExtract/"  # Update with your path
output_folder = "newExtract"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Get a list of all video files in the input folder
videos = glob.glob(os.path.join(input_folder, "*.mp4"))

for video_path in videos:
    video_capture = cv2.VideoCapture(video_path)
    face_count = 0

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for rect in faces:
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            face_image = frame[y : y + h, x : x + w]

            # Check if the image is not empty
            if (
                face_image.size == 0
                or face_image.shape[0] == 0
                or face_image.shape[1] == 0
            ):
                continue  # Skip this iteration

            output_path = os.path.join(
                output_folder, f"{os.path.basename(video_path)}_face_{face_count}.jpg"
            )
            cv2.imwrite(output_path, face_image)
            face_count += 1

    video_capture.release()

print("Face extraction completed.")
