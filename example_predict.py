import argparse
import numpy as np
from src.models.predict_model import load_model, predict_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Where to load the pretrained model from ? Default: random pick from inside models folder')
    parser.add_argument('--image', type=str, default=None,
                        help='Which image to classify (full path) ? Default: no default value, will throw error')

    args = parser.parse_args()

    # get path where trained model resides
    model_path = args.checkpoint

    # load model
    model = load_model(model_path)

    # predict for new image
    preds, class_names = predict_model(model, args.image, model_path)

    # output prediction results
    for i, p in enumerate(preds[0]):
        print(f'p:{p:.05f}\t{class_names[i]}')
    print(f'The most likely class for this image is: {class_names[np.argmax(preds)]}, p={np.max(preds):.02f}')


if __name__ == '__main__':
    main()


