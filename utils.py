
import os
import pickle
import tarfile

import zipfile

import numpy as np
import requests
from tqdm import tqdm

data_path = os.getcwd()
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels


num_classes = 10

_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file

def _get_file_path(filename=""):
    return os.path.join(data_path, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

def _convert_images(raw):
    raw_float = np.array(raw, dtype=int)
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])

    return images

def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls

def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data():
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls


def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch")

    return images, cls

def download_and_extract(
        url='https://www.cs.toronto.edu/%7Ekriz/cifar-10-python.tar.gz',
        target_dir=None):
    """Download and extract CIFAR-10"""
    target_dir = target_dir or os.getcwd()

    filename = url.split('/')[-1]
    print(target_dir)
    r = requests.get(url, stream=True)

    # total size in bytes
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as file_handle:
        for data in tqdm(
                r.iter_content(block_size),
                total=math.ceil(total_size // block_size),
                unit='KB',
                unit_scale=True
        ):
            file_handle.write(data)

    # extract if necessary
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, "r") as zip_handle:
            zip_handle.extractall(target_dir)
            # since data was extracted, remove the zip file
            os.remove(filename)
    elif filename.endswith((".tar.gz", ".tgz")):
        with tarfile.open(filename, "r:gz") as tar_handle:
            tar_handle.extractall(target_dir)
            # since data was extracted, remove the tar file
            os.remove(filename)

def load_dataset():
    x_train, y_train = load_training_data()
    x_test, y_test = load_test_data()
    x_train, y_train = np.array(x_train, dtype='int'), np.array(y_train, dtype='int')
    x_test, y_test = np.array(x_test, dtype='int'), np.array(y_test, dtype='int')
    return x_train, y_train, x_test, y_test


