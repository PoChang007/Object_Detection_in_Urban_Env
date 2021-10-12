import argparse
import glob
import os
import random
from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # 75% for training. 15% for validation. 10% for testing
    paths = glob.glob(f'{data_dir}/*.tfrecord')
    paths.sort()
    random.seed(100)
    random.shuffle(paths)

    split_1 = int(0.75 * len(paths))
    split_2 = int(0.9 * len(paths))
    train_data = paths[:split_1]
    val_data = paths[split_1:split_2]
    test_data = paths[split_2:]

    train_dir = f'{data_dir}/../train'
    val_dir = f'{data_dir}/../val'
    test_dir = f'{data_dir}/../test'

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for path in train_data:
        file_name = os.path.basename(path)
        new_path = f'{train_dir}/{file_name}'
        os.rename(path, new_path)

    for path in val_data:
        file_name = os.path.basename(path)
        new_path = f'{val_dir}/{file_name}'
        os.rename(path, new_path)

    for path in test_data:
        file_name = os.path.basename(path)
        new_path = f'{test_dir}/{file_name}'
        os.rename(path, new_path)


# ex: python3 create_splits.py --data_dir data/processed_data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
