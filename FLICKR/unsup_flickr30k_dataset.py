import torchtext

from torchtext.legacy import data, datasets
from torchtext.vocab import GloVe

import csv

class flickr30k_Data:

    def __init__(self, emb_dim=300, mbsize=32):  # emb_dim=50
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=20)
        # self.LABEL = data.Field(sequential=False, unk_token=None)

        # train = data.TabularDataset(
        #     path='FLICKR/flickr30k_results_rem_commas.csv',
        #     format='csv',
        #     fields=[('image_name', None), ('comment_num', None), ('text', self.TEXT)],
        #     csv_reader_params={'dialect': 'excel', 'delimiter': '|', 'quotechar': '"', 'quoting': csv.QUOTE_NONE, 'skipinitialspace': True},
        #     skip_header=True
        # )

        self.train = data.TabularDataset(
            path='FLICKR/flickr30k_data/unsup_flickr30k_filtered.csv',
            format='csv',
            fields=[('text', self.TEXT)],
            csv_reader_params={'dialect': 'excel'},
            skip_header=True
        )

        self.TEXT.build_vocab(self.train, vectors=GloVe('6B', dim=emb_dim))
        # self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        # self.train_iter = data.BucketIterator(
        #     train, batch_size=mbsize, device=-1,
        #     shuffle=True, repeat=False
        # )
        # self.train_iter = iter(self.train_iter)
        # self.val_iter = iter(self.val_iter)

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda()

        return batch.text

    def create_batch_iterable(self, b_size=32):

        self.train_iter = data.BucketIterator(
            self.train, batch_size=b_size, device=-1,
            shuffle=True, repeat=False
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
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]


