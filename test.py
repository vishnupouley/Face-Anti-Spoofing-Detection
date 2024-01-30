import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings("ignore")

SAMPLE_IMAGE_PATH = "./images/sample/"

model_dir = "./resources/anti_spoof_models"


def process_frame(frame, model_test, image_cropper):

    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    test_speed = 0

    # Sum the prediction from single model's result Hotel Seganas - I
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # Draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Real",
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5 * frame.shape[0] / 1024,
            (0, 255, 0),
        )
    else:
        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "Spoof",
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5 * frame.shape[0] / 1024,
            (0, 0, 255),
        )

    # print("Prediction cost {:.2f} s".format(test_speed))
    return frame


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test",
    )
    args = parser.parse_args()

    # Load the anti-spoofing models
    model_test = AntiSpoofPredict(args.device_id)
    image_cropper = CropImage()

    # Open the webcam for capturing frames
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Process the frame to detect face spoofing
        processed_frame = process_frame(frame, model_test, image_cropper)

        # Display the processed frame
        cv2.imshow("Face Anti-Spoofing", processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()
