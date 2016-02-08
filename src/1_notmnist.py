#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression

from dataset import not_mnist, utils

train_size = 200000
valid_size = 10000
data_folder = '../data'
pickle_sanitised_file = os.path.join(data_folder, 'notMNIST_sanitised.pickle')

# Creating the dataset (or retrieving it).
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
    not_mnist.prepare_dataset(train_size, valid_size, data_folder)

# Randomly plotting some of the training data.
num_plots = 10
for i in range(num_plots):
    plt.subplot(1, num_plots, i + 1)
    j = np.random.randint(0, len(train_labels))
    plt.imshow(train_dataset[j])
    plt.title(not_mnist.str_labels[train_labels[j]])
plt.show()

# Checking that the classes are equally reprensented.
print('Distribution of the samples in the {0} classes:'.format(not_mnist.num_classes))
n = plt.hist(train_labels, bins=not_mnist.num_classes, range=(-0.5, not_mnist.num_classes-0.5))
n = np.array(n[0], dtype=float)
print(n / np.sum(n))
print('')
plt.show()

# Looking for duplicate samples in the data.
print('Looking for duplicate samples in the data.')

# There is a small caveat when using hashes:
# Duplicates will be removed in each set, so this
# might reduce the estimated overlap between sets.
# Also, there can be colisions between hashes.

def hash_img(sample):
    return hash(sample.data.tobytes())

def hash_set(dataset):
    return {hash_img(dataset[i]): i for i in range(len(dataset))}

print('Creating hashtable for training set...')
tr_set = hash_set(train_dataset)
print('Creating hashtable for testing set...')
te_set = hash_set(test_dataset)
print('Creating hashtable for validation set...\n')
va_set = hash_set(valid_dataset)

sets_names = ("training", "test", "validation")
sets_sizes = (len(train_dataset), len(test_dataset), len(valid_dataset))
sets = [set(s.keys()) for s in (tr_set, te_set, va_set)]

for s1 in range(len(sets)):
    h1 = sets[s1]

    # The duplicates inside a single set are removed simply because
    # the same hash is not added twice to the set.
    num_duplicates = sets_sizes[s1] - len(h1)
    print('Number of duplicates in the {0} set: {1}/{2}'.format(sets_names[s1], num_duplicates, sets_sizes[s1]))

    for s2 in range(s1+1, len(sets)):
        h2 = sets[s2]

        # This is the ratio of common samples between the sets from
        # which duplicates were already removed. This might slightly
        # underestimates the true overlap in the original data.
        num_common = len(h1.intersection(h2))
        ratio_common = num_common / min(len(h1), len(h2))
        print('Between {0} and {1}: {2}'.format(sets_names[s1], sets_names[s2], ratio_common))

# Removing duplicates from validation and test set.
def get_unique_idx(original_dic, others):
    """Returns the list of values in original_dic that correspond to keys
    that are only contained in original_dic, and not in any of the dictionaries
    in 'others'."""
    unique_keys = set(original_dic.keys())
    for other_dic in others:
        # Simply uses the set operations to remove all the samples from
        # original_dic also contained in 'others' dictionaries.
        unique_keys = unique_keys - set(other_dic.keys())
    return [original_dic[i] for i in unique_keys]

print('\nCreating sanitised dataset...')
san_train_idx = get_unique_idx(tr_set, [])
san_train_dataset = train_dataset[san_train_idx, :, :]
san_train_labels = train_labels[san_train_idx]
print('Number of training samples in the sanitised set: {0}'.format(len(san_train_dataset)))

san_test_idx = get_unique_idx(te_set, (tr_set, va_set))
san_test_dataset = test_dataset[san_test_idx, :, :]
san_test_labels = test_labels[san_test_idx]
print('Number of test samples in the sanitised set: {0}'.format(len(san_test_dataset)))

san_valid_idx = get_unique_idx(va_set, (tr_set, te_set))
san_valid_dataset = valid_dataset[san_valid_idx, :, :]
san_valid_labels = valid_dataset[san_valid_idx]
print('Number of validation samples in the sanitised set: {0}'.format(len(san_valid_dataset)))

# Saving sanitised data in a separate file.
print('Saving...')
not_mnist.save_to_pickle(san_train_dataset, san_train_labels, san_valid_dataset, san_valid_labels,
    san_test_dataset, san_test_labels, pickle_sanitised_file)
print('')

# Finally, near duplicates could be found by quantising the samples,
# so that each pixel can take a limited number of values. We can then
# look for exact matches in the quantised images.

# Training a simple logistic regression on the data.
num_samples_list = [50, 100, 1000, 5000, len(train_dataset)]

flat_test_dataset = utils.flatten_batch(test_dataset)
flat_valid_dataset = utils.flatten_batch(valid_dataset)
flat_train_dataset = utils.flatten_batch(train_dataset)

for num_samples in num_samples_list:
    if num_samples > 10000:
        train = input(
            'Are you sure you want to train a logistic regression on {0} samples? [y/n]'.format(num_samples))
        if train != 'y':
            continue

    logistic_model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    logistic_model.fit(flat_train_dataset[:num_samples, :], train_labels[:num_samples])

    valid_score = logistic_model.score(flat_valid_dataset, valid_labels)
    print('Score on validation data when training on {0} samples: {1}'.format(num_samples, valid_score))

    test_score = logistic_model.score(flat_test_dataset, test_labels)
    print('Score on test data when training on {0} samples: {1}\n'.format(num_samples, test_score))
