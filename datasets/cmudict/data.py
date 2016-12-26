START_LINE = 126
END_LINE = 133905
FILENAME = 'data/cmudict-0.7b'
ALPHA = '_abcdefghijklmnopqrstuvwxyz'

limit = {
        'maxw'  : 16,
        'minw'  : 5,
        'maxph' : 16,
        'minph' : 5
        }

import random
import numpy as np
import pickle


'''
 read lines from file
     return [list of lines]

'''
def read_line(filename=FILENAME):
    return open(filename, 'r', encoding='utf-8', errors='ignore').read().split('\n')[START_LINE:END_LINE]


'''
 separate lines into lines of words and pronunciation
    return tuple( [words], [phoneme lists] )

'''
def split_data(lines):

    words, phoneme_lists = [], []
    for line in lines:
        word, phonemes = line.split('  ')
        phoneme_lists.append(phonemes.split(' '))
        words.append(word.lower())
    return words, phoneme_lists


'''
 read the phoneme_lists, create index to phoneme, 
  phoneme to index dictionaries
    return tuple( idx2pho, pho2idx )

'''
def index_phonemes(phoneme_lists):
    phoneme_vocab = set([phoneme for phonemes in phoneme_lists 
        for phoneme in phonemes])
    idx2pho = dict(enumerate(['_'] + sorted(list(phoneme_vocab))))
    # we add an extra dummy element to make sure 
    #  we dont touch the zero index (zero padding)
    pho2idx = dict(zip(idx2pho.values(), idx2pho.keys()))
    return idx2pho, pho2idx


'''
 generate index english alphabets
    return tuple( idx2alpha, alpha2idx )

'''
def index_alphabets(alpha = ALPHA):
    idx2alpha = dict(enumerate(alpha))
    alpha2idx = dict(zip(idx2alpha.values(), idx2alpha.keys()))
    return idx2alpha, alpha2idx


'''
 filter too long and too short sequences
    return tuple( filtered_words, filtered_phoneme_lists )

'''
def filter_data(words, phoneme_lists):
    # need a threshold
    #  say max : 16, 16
    #      min : 5, 5
    '''
        limit = {
                'maxw'  : 16,
                'minw'  : 5,
                'maxph' : 16,
                'minph' : 5
                }
    '''

    # also make sure, every character in words is an alphabet
    #  if not, skip the row
    filtered_words, filtered_phoneme_lists = [], []
    raw_data_len = len(words)
    for word, phonemes in zip(words, phoneme_lists):
        if word.isalpha():
            if len(word) < limit['maxw'] and len(word) > limit['minw']:
                if len(phonemes) < limit['maxph'] and len(phonemes) > limit['minph']:
                    filtered_words.append(word)
                    filtered_phoneme_lists.append(phonemes)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_words)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_words, filtered_phoneme_lists


'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_words([indices]), array_phonemes([indices]) )
 
'''
def zero_pad(words, phoneme_lists, alpha2idx, pho2idx):
    # num of rows
    data_len = len(words)
    idx_words = np.zeros([data_len, limit['maxw']], dtype=np.int32)
    idx_phonemes = np.zeros([data_len, limit['maxph']], dtype=np.int32)

    for i in range(data_len):
        #print(words[i], phoneme_lists[i], i)
        word_indices = [ alpha2idx[alpha] for alpha in words[i] ] \
                            + [0]*(limit['maxw'] - len(words[i]))
        pho_indices  = [ pho2idx[phoneme] for phoneme in phoneme_lists[i] ] \
                            + [0]*(limit['maxph'] - len(phoneme_lists[i]))

        idx_words[i] = np.array(word_indices)
        idx_phonemes[i] = np.array(pho_indices)

    return idx_words, idx_phonemes


def process_data():

    print('>> Read from file')
    lines = read_line()
    print('\n:: random.choice(lines)')
    print(random.choice(lines))
    print(random.choice(lines))

    print('\n>> Separate data')
    # shuffle data
    random.shuffle(lines)
    # separate into words and phoneme lists
    words, phoneme_lists = split_data(lines)
    print('\n:: random.choice(words)')
    print(random.choice(words))
    print(random.choice(words))
    print('\n:: random.choice(phoneme_lists)')
    print(random.choice(phoneme_lists))
    print(random.choice(phoneme_lists))

    print('\n>> Index phonemes')
    idx2pho, pho2idx = index_phonemes(phoneme_lists)
    print('\n:: random.choice(idx2pho)')
    print(idx2pho[random.choice(list(idx2pho.keys()))])
    print(idx2pho[random.choice(list(idx2pho.keys()))])
    print('\n:: random.choice(pho2idx)')
    print(pho2idx[random.choice(list(pho2idx.keys()))])
    print(pho2idx[random.choice(list(pho2idx.keys()))])

    # index alphabets
    idx2alpha, alpha2idx = index_alphabets()

    print('\n>> Filter data')
    words, phoneme_lists = filter_data(words, phoneme_lists)

    # we know the maximum length of both sequences : 15
    #  we should create zero-padded numpy arrays
    idx_words, idx_phonemes = zero_pad(words, phoneme_lists, alpha2idx, pho2idx)
    # save them
    np.save('idx_words.npy', idx_words)
    np.save('idx_phonemes.npy', idx_phonemes)

    # let us now save the necessary dictionaries
    data_ctl = {
            'idx2alpha' : idx2alpha,
            'idx2pho' : idx2pho,
            'pho2idx' : pho2idx,
            'alpha2idx' : alpha2idx,
            'limit' : limit
                }
    # write to disk : data control dictionaries
    with open('data_ctl.pkl', 'wb') as f:
        pickle.dump(data_ctl, f)


def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'data_ctl.pkl', 'rb') as f:
        data_ctl = pickle.load(f)
    # read numpy arrays
    idx_words = np.load(PATH + 'idx_words.npy')
    idx_phonemes = np.load(PATH + 'idx_phonemes.npy')
    return data_ctl, idx_words, idx_phonemes



if __name__ == '__main__':
    process_data()
