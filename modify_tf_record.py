import os
import argparse
import logging
import glob

from utils import get_dataset
import tensorflow.compat.v1 as tf
from utils import int64_feature, int64_list_feature, bytes_feature
from utils import bytes_list_feature, float_list_feature


def create_tf_example(batch):
    """
    returns:
        - tf_example [tf.Train.Example]: tf example in the objection detection api format.
    """
    mapping = {1: 'vehicle', 2: 'pedestrian', 4: 'cyclist'}
    image_format = b'jpg'
    height = 640
    width = 640
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    filename = batch['filename'].numpy()
    bboxes = batch['groundtruth_boxes'].numpy()
    labels = batch['groundtruth_classes'].numpy()
    encoded_jpeg = tf.io.encode_jpeg(batch['image']).numpy()

    for bbox, label in zip(bboxes, labels):
        y1, x1, y2, x2 = bbox
        # ex: correct bounding box region
        ymin = y1 * (640/1280)
        xmin = x1 * (640/1920)
        ymax = y2 * (640/1280)
        xmax = x2 * (640/1920)
        ymins.append(ymin)
        xmins.append(xmin)
        ymaxs.append(ymax)
        xmaxs.append(xmax)
        classes.append(label)
        classes_text.append(mapping[label].encode('utf8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpeg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def process_tfr(path, batch_data, dst_dir):
    """
    correct tf api tf record
    """
    file_name = os.path.basename(path)

    logging.info(f'Processing {path}')
    writer = tf.python_io.TFRecordWriter(f'{dst_dir}/{file_name}')

    for key in batch_data:
        batch = batch_data[key]
        tf_example = create_tf_example(batch)
        writer.write(tf_example.SerializeToString())
    writer.close()


# ex: python3 modify_tf_record.py --src_dir data/tfrecord_to_be_modified --dst_dir data/processed_data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='modify tfrecord and re-export it')
    parser.add_argument('-s', '--src_dir', required=True, type=str,
                        help='input directory of tf records')
    parser.add_argument('-d', '--dst_dir', required=True, type=str,
                        help='output directory of tf records')
    args = parser.parse_args()
    paths = glob.glob(f'{args.src_dir}/*.tfrecord')

    for path in paths:
        tfrecord_count = 0
        # check size
        tf_dataset = tf.data.TFRecordDataset(path)
        for idx, data in enumerate(tf_dataset):
            tfrecord_count += 1

        dataset = get_dataset(path)
        image_count = 0
        batch_data = {}
        for idx, data in enumerate(dataset):
            filename = data['filename'].numpy().decode('UTF8')
            if filename not in batch_data.keys():
                batch_data[filename] = data
                image_count += 1
                if image_count == tfrecord_count:
                    break

        print("processing",filename)
        process_tfr(path, batch_data, args.dst_dir)
