import torchtext

from torchtext.legacy import data, datasets
from torchtext.vocab import GloVe

import csv
import pickle


class flickr8k_Data:

    def __init__(self, emb_dim=300, mbsize=32):
        # Use the original filtered dataset to rebuild vocabulary, since we are loading the pre-trained model
        with open('built_vocab.pkl', 'rb') as file:
            built_vocab = pickle.load(file)

        self.TEXT_8k = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=35)
        self.LABEL_8k = data.Field(sequential=False, unk_token=None)

        self.train_8k, self.valid_8k, self.test_8k = data.TabularDataset.splits(
            path='FLICKR/flickr8k_data/',
            train='flickr8k_sup_train.csv',
            validation='flickr8k_sup_valid.csv',
            test='flickr8k_sup_test.csv',
            format='csv',
            fields=[('text', self.TEXT_8k), ('label', self.LABEL_8k)],
            csv_reader_params={'dialect': 'excel', 'delimiter': '|'},
            skip_header=False
        )

        self.valid_iter = data.BucketIterator(
            self.valid_8k, batch_size=mbsize, device='cuda',
            shuffle=False, repeat=True
        )
        self.valid_iter = iter(self.valid_iter)

        self.test_iter = data.BucketIterator(
            self.test_8k, batch_size=mbsize, device='cuda',
            shuffle=False, repeat=True
        )
        self.test_iter = iter(self.test_iter)

        self.TEXT_8k.vocab = built_vocab  # Both share same vocab
        self.LABEL_8k.build_vocab(self.train_8k)

        self.n_vocab = len(self.TEXT_8k.vocab.itos)

        print('Label vocab mapping: ')
        print(self.LABEL_8k.vocab.stoi)

    def get_vocab_vectors(self):
        return self.TEXT_8k.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def create_batch_iterable(self, b_size=32):

        self.train_iter = data.BucketIterator(
            self.train_8k, batch_size=b_size, device='cuda',
            shuffle=True, repeat=True
        )
        self.train_iter = iter(self.train_iter)

    def next_validation_batch(self, gpu=False):
        batch = next(self.valid_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT_8k.vocab.itos[i] for i in idxs])

    def idxs2sentence_val(self, model, idxs):
        display_list = []

        for i in idxs:

            if i == model.PAD_IDX or i == model.START_IDX or i == model.EOS_IDX:
                continue

            display_list.append(i)

        return ' '.join([self.TEXT_8k.vocab.itos[i] for i in display_list])

    def idx2label(self, idx):
        return self.LABEL_8k.vocab.itos[idx]


