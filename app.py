import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("👕 AI Image-Based Virtual Try-On")

# Upload files
person_file = st.file_uploader("Upload Person Image", type=["jpg", "png"])
cloth_file = st.file_uploader("Upload Cloth Image (PNG with transparent background)", type=["png"])

if person_file and cloth_file:

    # Convert images
    person_img = np.array(Image.open(person_file).convert("RGB"))
    cloth_img = np.array(Image.open(cloth_file).convert("RGBA"))

    person_height, person_width, _ = person_img.shape

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    results = pose.process(cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # Get shoulder landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Convert to pixel coordinates
        left_x = int(left_shoulder.x * person_width)
        left_y = int(left_shoulder.y * person_height)
        right_x = int(right_shoulder.x * person_width)
        right_y = int(right_shoulder.y * person_height)

        # Calculate cloth width based on shoulder distance
        shoulder_width = abs(right_x - left_x)
        cloth_width = shoulder_width
        cloth_height = int(cloth_width * cloth_img.shape[0] / cloth_img.shape[1])

        # Resize cloth
        cloth_resized = cv2.resize(cloth_img, (cloth_width, cloth_height))

        # Position cloth
        center_x = (left_x + right_x) // 2
        top_y = left_y - int(cloth_height * 0.2)

        x1 = center_x - cloth_width // 2
        y1 = top_y

        # Overlay cloth
        output = person_img.copy()

        for i in range(cloth_height):
            for j in range(cloth_width):
                if 0 <= y1 + i < person_height and 0 <= x1 + j < person_width:
                    alpha = cloth_resized[i, j, 3] / 255.0
                    for c in range(3):
                        output[y1 + i, x1 + j, c] = (
                            alpha * cloth_resized[i, j, c] +
                            (1 - alpha) * output[y1 + i, x1 + j, c]
                        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(person_img, caption="Original Image", use_column_width=True)

        with col2:
            st.image(output, caption="Virtual Try-On Result", use_column_width=True)

    else:
        st.error("⚠️ Could not detect body pose. Please upload a clear front-facing image.")