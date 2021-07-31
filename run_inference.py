import json

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


def main():
    model = FlowerInference.load('models/test.ckpt')

    with open('data/flowers/cat_to_name.json') as file:
        cat_to_name = json.load(file)

    model.set_label_dictionary(cat_to_name)

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
    print(image)
    prediction = model.image_to_probability(image)
    print(prediction)

    fig = model.plot(image, 10)
    fig.savefig('figures/probability.png')


if __name__ == '__main__':
    main()
