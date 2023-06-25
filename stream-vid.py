import streamlit as st
import cv2
from ultralytics import YOLO


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = 0
    try:
        vid_cap = cv2.VideoCapture(source_webcam)
        st_frame = st.empty()
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                print(image)
                image = cv2.resize(image, (720, int(720 * (9 / 16))))
                # img = frame.to_ndarray(format="bgr24")
                res = model.predict(image, conf=conf)
                ## Tracking
                # res = model.track(image, conf=conf, persist=True, tracker="bytetrack.yaml")
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.sidebar.error("Error loading video: " + str(e))

    return image


model = YOLO('yolov8n-seg.pt')
play_webcam(0.15, model)
st.write(play_webcam(0.15, model))