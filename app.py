import cv2
import numpy as np
import streamlit as st

from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="😴",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_drowsiness_model():
    return load_model("models/drowsiness_cnn_final.keras")


model = load_drowsiness_model()


# --------------------------------------------------
# Load Haar cascades
# --------------------------------------------------

FACE_CASCADE = cv2.CascadeClassifier(
    "haar cascade files/haarcascade_frontalface_alt.xml"
)

LEFT_EYE_CASCADE = cv2.CascadeClassifier(
    "haar cascade files/haarcascade_lefteye_2splits.xml"
)

RIGHT_EYE_CASCADE = cv2.CascadeClassifier(
    "haar cascade files/haarcascade_righteye_2splits.xml"
)


# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("😴 Driver Drowsiness Detection")

st.markdown(
    """
    Real-time driver monitoring using **OpenCV + CNN**.

    The system detects eye state from webcam frames and
    calculates a drowsiness score based on consecutive
    closed-eye frames.
    """
)


# --------------------------------------------------
# Model performance
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "93.27%")
col2.metric("Precision", "94.06%")
col3.metric("Recall", "93.27%")
col4.metric("ROC-AUC", "99.89%")


st.divider()


# --------------------------------------------------
# Video processor
# --------------------------------------------------

class DrowsinessProcessor(VideoProcessorBase):

    def __init__(self):
        self.score = 0
        self.closed_frames = 0
        self.status = "Open"

    def predict_eye(self, eye):

        eye = cv2.cvtColor(
            eye,
            cv2.COLOR_BGR2GRAY
        )

        eye = cv2.resize(
            eye,
            (24, 24)
        )

        eye = eye / 255.0

        eye = eye.reshape(
            24,
            24,
            1
        )

        eye = np.expand_dims(
            eye,
            axis=0
        )

        prediction = np.argmax(
            model.predict(
                eye,
                verbose=0
            ),
            axis=-1
        )

        return prediction[0]


    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        # ------------------------------------------
        # Detect face
        # ------------------------------------------

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25)
        )


        # ------------------------------------------
        # Detect eyes
        # ------------------------------------------

        left_eyes = LEFT_EYE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        right_eyes = RIGHT_EYE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )


        left_prediction = None
        right_prediction = None


        # ------------------------------------------
        # Right eye
        # ------------------------------------------

        for (x, y, w, h) in right_eyes:

            eye = img[y:y+h, x:x+w]

            if eye.size == 0:
                continue

            right_prediction = self.predict_eye(eye)

            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (255, 255, 0),
                2
            )

            break


        # ------------------------------------------
        # Left eye
        # ------------------------------------------

        for (x, y, w, h) in left_eyes:

            eye = img[y:y+h, x:x+w]

            if eye.size == 0:
                continue

            left_prediction = self.predict_eye(eye)

            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (255, 255, 0),
                2
            )

            break


        # ------------------------------------------
        # Face rectangle
        # ------------------------------------------

        for (x, y, w, h) in faces:

            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (180, 180, 180),
                2
            )

            break


        # ------------------------------------------
        # Determine eye state
        # ------------------------------------------

        if (
            left_prediction == 0
            and right_prediction == 0
        ):

            self.score += 1
            self.closed_frames += 1
            self.status = "Closed"

        elif (
            left_prediction is not None
            or right_prediction is not None
        ):

            self.score = max(
                0,
                self.score - 1
            )

            self.closed_frames = 0
            self.status = "Open"


        # ------------------------------------------
        # Drowsiness threshold
        # ------------------------------------------

        if self.score > 15:

            alert_text = "DROWSINESS ALERT!"

            cv2.rectangle(
                img,
                (0, 0),
                (img.shape[1], img.shape[0]),
                (0, 0, 255),
                8
            )

            cv2.putText(
                img,
                alert_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        else:

            cv2.putText(
                img,
                "Monitoring...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


        # ------------------------------------------
        # Display status
        # ------------------------------------------

        cv2.putText(
            img,
            f"Eye State: {self.status}",
            (30, img.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            f"Drowsiness Score: {self.score}",
            (30, img.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        return frame.from_ndarray(
            img,
            format="bgr24"
        )


# --------------------------------------------------
# WebRTC configuration
# --------------------------------------------------

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    }
)


# --------------------------------------------------
# Start webcam
# --------------------------------------------------

st.subheader("📷 Live Detection")

st.info(
    "Allow camera access when your browser asks for permission."
)

webrtc_ctx = webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# --------------------------------------------------
# Project information
# --------------------------------------------------

st.divider()

st.subheader("📊 Model Performance")

st.write(
    "The CNN was evaluated on a test set containing "
    "**3,223 images**."
)

st.markdown(
    """
    - **Accuracy:** 93.27%
    - **Precision:** 94.06%
    - **Recall:** 93.27%
    - **F1 Score:** 93.25%
    - **ROC-AUC:** 99.89%
    """
)

st.caption(
    "This is a computer-vision prototype and is not a "
    "safety-certified driver monitoring system."
)