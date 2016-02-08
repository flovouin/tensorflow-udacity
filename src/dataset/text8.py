import collections
import os
from six.moves import cPickle as pickle
import zipfile

from dataset import utils

data_url = 'http://mattmahoney.net/dc/text8.zip'
local_data_filename = 'text8.zip'
pickle_file = 'text8.pickle'

def get_local_filenames(folder):
    return os.path.join(folder, local_data_filename)

def get_pickle_filename(folder):
    return os.path.join(folder, pickle_file)

def read_data(data_file):
    f = zipfile.ZipFile(data_file)
    for name in f.namelist():
        return f.read(name)
    f.close()

def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def save_to_pickle(letters, data, count, dictionary, reverse_dictionary, pickle_file):
    try:
        f = open(pickle_file, 'wb')
        save_dic = {
            'letters': letters,
            'data': data,
            'count': count,
            'dictionary': dictionary,
            'reverse_dictionary': reverse_dictionary
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
        letters = save['letters']
        data = save['data']
        count = save['count']
        dictionary = save['dictionary']
        reverse_dictionary = save['reverse_dictionary']
        del save
        return letters, data, count, dictionary, reverse_dictionary

def prepare_dataset(vocabulary_size, folder):
    pickle_file = get_pickle_filename(folder)

    if os.path.isfile(pickle_file):
        print('Pickle file already exists, assuming everything is in there.\n')
        return read_from_pickle(pickle_file)
    else:
        print('Downloading archive (if necessary)...')
        data_file = get_local_filenames(folder)
        utils.maybe_download(data_url, data_file, 31344016)

        print('Reading data from archive...')
        letters = read_data(data_file)
        words = letters.split()

        print('Converting letters to integers...')
        letters = [utils.char2id(chr(c)) for c in letters]

        print('Building dataset using a vocabulary of {0} words...'.format(
            vocabulary_size))
        data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
        del words

        print('Saving dataset...')
        save_to_pickle(letters, data, count, dictionary, reverse_dictionary, pickle_file)

        return letters, data, count, dictionary, reverse_dictionary
