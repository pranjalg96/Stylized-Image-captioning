
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from FLICKR.sup_flickr8k_dataset import *
from FLICKR.model_flickr import RNN_VAE

import matplotlib.pyplot as plt

import argparse

# Check the accuracy of conditional text generation

parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c)'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')

args = parser.parse_args()

mb_size = 32  # Mini-batch size
h_dim = 300
z_dim = 100
c_dim = 2

dataset = flickr8k_Data(emb_dim=300)

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=False,
    gpu=args.gpu
)

num_gen_examples = 100

# Load pretrained base VAE with c ~ p(c)
model.load_state_dict(torch.load('models/ctextgen_flickr8k_final.bin'))
model.eval()

def calc_accuracy(y_scores_raw, y_labels):

    y_scores = F.softmax(y_scores_raw, dim=1)
    _, y_preds = torch.max(y_scores, dim=1)

    cur_batch_size = y_scores_raw.size(0)

    disc_acc = torch.sum(y_preds == y_labels)
    disc_acc = disc_acc.item()/cur_batch_size

    return disc_acc

def main():

    x_gen, c_gen = model.generate_sentences(num_gen_examples)
    _, target_c = torch.max(c_gen, dim=1)

    y_disc_fake = model.forward_discriminator(x_gen)
    disc_acc = calc_accuracy(y_disc_fake, target_c)

    print('Accuracy of conditional text generation: ', disc_acc)


if __name__ == '__main__':
    main()

