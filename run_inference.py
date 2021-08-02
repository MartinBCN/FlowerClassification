import json
from argparse import ArgumentParser

from PIL import Image

from flower_classification.flower_data import get_loader
from flower_classification.flower_inference import FlowerInference
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='flower_classifier_resnet', type=str,
                        help='Identifier for the saved model')

    # Names and directories
    parser.add_argument('--cat_to_name_file', type=str, default='data/flowers/cat_to_name.json',
                        help='Filename for category to name dictionary')
    parser.add_argument('--data_dir', default='flowers', type=str,
                        help='Data directory. Needs to contain the sub-folders train, test, valid')
    parser.add_argument('--model_dir', default='models', type=str, help='Model directory')
    parser.add_argument('--figure_dir', default='figures', type=str, help='Figure directory')
    parser.add_argument('--test_image_path', default='data/flowers/test/10/image_07090.jpg')

    return parser


def main(model_name: str, model_dir: str, cat_to_name_file: str, test_image_path: str) -> None:
    model = FlowerInference.load(f'{model_dir}/{model_name}.ckpt')

    with open(cat_to_name_file) as file:
        cat_to_name = json.load(file)

    # model.set_label_dictionary(cat_to_name)

    batch_size = 1
    use_fraction = None
    test_loader = get_loader('data/flowers', 'test', batch_size=batch_size, use_fraction=use_fraction)

    # Get a single image tensor and the label as integer
    image, label = next(iter(test_loader))
    image = image[0]
    label = int(label[0])
    print(label)

    prediction = model.tensor_to_probability(image)
    print(prediction)

    prediction = model.tensor_to_label(image)
    print(prediction)

    fn = 'data/flowers/test/10/image_07090.jpg'
    image = Image.open(fn).convert("RGB")
    prediction = model.image_to_probability(image)
    print(prediction)

    model.set_label_dictionary(cat_to_name)
    fig = model.plot_topk(image, 10)
    fig.savefig('figures/probability.png')


if __name__ == '__main__':
    p = get_argparser()
    arguments = p.parse_args()
    main(model_name=arguments.model_name, model_dir=arguments.model_dir, cat_to_name_file=arguments.cat_to_name_file,
         test_image_path=arguments.test_image_path)

