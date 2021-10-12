from utils import get_train_input
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def recenter_image(image):
    # ssd preprocessing
    image += [123.68, 116.779, 103.939]
    return image


def display_instances(image, bboxes, classes):
    image = recenter_image(image)
    w, h, _ = image.shape
    # resize the bboxes
    bboxes[:, [0, 2]] *= w
    bboxes[:, [1, 3]] *= h

    colormap = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    f, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image.astype(np.uint8))
    for bb, cl in zip(bboxes, classes):
        index = 0
        for idx in range(len(cl)):
            if cl[idx] == 1:
                index = idx
                break

        y1, x1, y2, x2 = bb
        rec = Rectangle((x1, y1), x2-x1, y2-y1,
                        facecolor='none', edgecolor=colormap[idx])
        ax.add_patch(rec)
    plt.show()


def display_batch(batch):
    # get images, bboxes and classes
    batched_images = batch[0]['image'].numpy()
    batched_bboxes = batch[1]['groundtruth_boxes'].numpy()
    batched_classes = batch[1]['groundtruth_classes'].numpy()
    num_bboxes = batch[1]['num_groundtruth_boxes'].numpy()
    batch_size = batched_images.shape[0]
    for idx in range(batch_size):
        display_instances(batched_images[idx, ...],
                          batched_bboxes[idx, :num_bboxes[idx], :],
                          batched_classes[idx, ...])


# ex: python3 explore_augmentation.py --config_path training/reference/pipeline_new.config
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='visualization of data augmentation')
    parser.add_argument('--config_path', required=True,
                        type=str, help='configuration path')
    args = parser.parse_args()

    train_dataset = get_train_input(args.config_path)
    for batch in train_dataset.take(1):
        display_batch(batch)
