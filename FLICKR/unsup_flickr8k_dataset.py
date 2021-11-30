import torchtext

from torchtext.legacy import data, datasets
from torchtext.vocab import GloVe

import csv
import pickle


class flickr8k_Data:

    def __init__(self, emb_dim=300, mbsize=32):
        self.TEXT_vocab = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=35)

        self.train_vocab = data.TabularDataset(
            path='FLICKR/flickr8k_data/flickr8k_sup.csv',
            format='csv',
            fields=[('text', self.TEXT_vocab)],
            csv_reader_params={'dialect': 'excel', 'delimiter': '|'},
            skip_header=False
        )

        self.TEXT_vocab.build_vocab(self.train_vocab, vectors=GloVe('6B', dim=emb_dim))

        self.n_vocab = len(self.TEXT_vocab.vocab.itos)
        self.emb_dim = emb_dim

        with open('built_vocab.pkl', 'wb') as file:
            pickle.dump(self.TEXT_vocab.vocab, file)

    def get_vocab_vectors(self):
        return self.TEXT_vocab.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda()

        return batch.text

    def create_batch_iterable(self, b_size=32):

        self.train_iter = data.BucketIterator(
            self.train_vocab, batch_size=b_size, device='cuda',
            shuffle=True, repeat=True  # Should I set repeat to false? is it creating an infinite iterator?
        )
        self.train_iter = iter(self.train_iter)

    # def next_validation_batch(self, gpu=False):
    #     batch = next(self.val_iter)
    #
    #     if gpu:
    #         return batch.text.cuda(), batch.label.cuda()
    #
    #     return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT_vocab.vocab.itos[i] for i in idxs])

    # def idx2label(self, idx):
    #     return self.LABEL_8k.vocab.itos[idx]


