import numpy as np
import os
from scipy import ndimage
from six.moves import cPickle as pickle
import sys
import tarfile

from dataset import utils

train_data_url = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz'
test_data_url = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz'
train_data_file_size = 247336696
test_data_file_size = 8458043
local_train_data_filename = 'notMNIST_large.tar.gz'
local_test_data_filename = 'notMNIST_small.tar.gz'
pickle_file = 'notMNIST.pickle'

num_classes = 10
str_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def get_local_filenames(folder):
    return os.path.join(folder, local_train_data_filename), os.path.join(folder, local_test_data_filename)

def get_pickle_filename(folder):
    return os.path.join(folder, pickle_file)

def extract_file(filename):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz

    if os.path.isdir(root):
        print('Directory already exists, assuming all the data is there.')
    else:
        tar = tarfile.open(filename)
        extract_path = os.path.dirname(filename)

        print('Extracting data for %s. This may take a while. Please wait.' % root)
        sys.stdout.flush()

        tar.extractall(extract_path)
        tar.close()

    data_folders = [os.path.join(root, d)
        for d in sorted(os.listdir(root)) if d != '.DS_Store']
    if len(data_folders) != num_classes:
        raise Exception('Expected %d folders, one per class. Found %d instead.' %
            (num_classes, len(data_folders)))

    return data_folders

def load_from_folders(data_folders, min_num_images, max_num_images):
    dataset = np.ndarray(shape=(max_num_images, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
    label_index = 0
    image_index = 0

    for folder in data_folders:
        print(folder)

        for image in os.listdir(folder):
            if image_index >= max_num_images:
                raise Exception('More images than expected: %d >= %d' %
                    (image_index, max_num_images))

            image_file = os.path.join(folder, image)
            try:
                image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))

                dataset[image_index, :, :] = image_data
                labels[image_index] = label_index
                image_index += 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

        label_index += 1

    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    labels = labels[0:num_images]

    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
            (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))

    return dataset, labels

def save_to_pickle(train_dataset, train_labels, valid_dataset, valid_labels,
         test_dataset, test_labels, pickle_file):
    try:
        f = open(pickle_file, 'wb')
        save_dic = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save_dic, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

def read_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def prepare_dataset(train_size, valid_size, folder):
    pickle_file = get_pickle_filename(folder)

    if os.path.isfile(pickle_file):
        print('Pickle file already exists, assuming everything is in there.\n')
        return read_from_pickle(pickle_file)
    else:
        train_file, test_file = get_local_filenames(folder)

        print('Downloading / checking archives...')
        utils.maybe_download(train_data_url, train_file, train_data_file_size)
        utils.maybe_download(test_data_url, test_file, test_data_file_size)

        print('Extracting dataset...')
        train_folders = extract_file(train_file)
        test_folders = extract_file(test_file)

        print('Creating numpy dataset...')
        train_dataset, train_labels = load_from_folders(train_folders, 450000, 550000)
        test_dataset, test_labels = load_from_folders(test_folders, 18000, 20000)

        print('Randomising input...')
        train_dataset, train_labels = utils.randomize(train_dataset, train_labels)
        test_dataset, test_labels = utils.randomize(test_dataset, test_labels)

        valid_dataset = train_dataset[:valid_size,:,:]
        valid_labels = train_labels[:valid_size]
        train_dataset = train_dataset[valid_size:valid_size+train_size,:,:]
        train_labels = train_labels[valid_size:valid_size+train_size]

        print('Saving dataset to pickle file...\n')
        save_to_pickle(train_dataset, train_labels, valid_dataset, valid_labels,
             test_dataset, test_labels, pickle_file)

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
