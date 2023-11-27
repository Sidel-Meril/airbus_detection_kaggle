from LightUNet.model import LightUNet
import argparse
import cv2
from LightUNet.preprocessing import preprocess_sample
from config import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Model Prediction Script')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('out', type=str, help='Path to save the result image')
    args = parser.parse_args()

    model = LightUNet(base_model_name=BASE_MODEL,
                      checkpoints_path=CHECKPOINTS_PATH,
                      name = NAME)
    model.load()

    sample = preprocess_sample(args.image_path)
    pred = model.predict(sample).numpy()
    pred = pred.reshape(*TARGET_IM_SIZE, 1)
    pred = cv2.resize(pred, IM_SIZE)

    cv2.imwrite(args.out, pred)
    print('Image saved to', args.output_path)