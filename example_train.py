import argparse
from src.utils.basic_functions import atuple, aninteger, afloat
from src.models.train_model import do_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_split', type=afloat, default=None,
                        help='How much training data to use for validation ? Default value: 0.2')

    parser.add_argument('--batch_size', type=aninteger, default=None,
                        help='What batch size to use for training. Default value: 32')

    parser.add_argument('--epochs', type=aninteger, default=None,
                        help='How many epochs to train for ? Default: 40, with early stopping callback')

    parser.add_argument('--input_size', type=atuple, default=None,
                        help='What input size to set for the ResNet50 model architecture ? Default: \'(64, 64)\'')

    parser.add_argument('--channels', type=aninteger, default=None,
                        help='How many channels do training images have ? Assumed RGB images by default. Default: 3')

    parser.add_argument('--log_level', type=aninteger, default=None,
                        help='How verbose do you want the logging level ?  DEBUG: 10, INFO: 20, WARNING: 30')

    parser.add_argument('--fld', type=str, default=None,
                        help='Name of the training data folder. Must be placed inside images/processed. Default: Animals-10')

    args = parser.parse_args()

    # train ResNet50 model on images from given folder for two epochs
    model, history = do_train(fld=args.fld, epochs=args.epochs, batch_size=args.batch_size,
                              image_size=args.input_size, channels=args.channels, log_level=args.log_level,
                              validation_split=args.validation_split)


if __name__ == '__main__':
    main()


