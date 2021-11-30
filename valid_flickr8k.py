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

import argparse

# Reduce temperature parameter to produce more confident sentences


parser = argparse.ArgumentParser(
    description='Validation set Conditional Text Generation'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--model', default='ctextgen_flickr8k', metavar='',
                    help='choose the model')

args = parser.parse_args()


mb_size = 32
h_dim = 300
z_dim = 100
c_dim = 2

dataset = flickr8k_Data(emb_dim=300, mbsize=mb_size)

# torch.manual_seed(int(time.time()))

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=args.gpu
)

if args.gpu:
    model.load_state_dict(torch.load('models/{}.bin'.format(args.model)))
else:
    model.load_state_dict(torch.load('models/{}.bin'.format(args.model), map_location=lambda storage, loc: storage))

num_valid_rows = 550
num_valid_batches = math.ceil(num_valid_rows/mb_size)

num_gen_instances = 5

model.eval()

# for batch_idx_i in range(num_valid_batches):
#
#     valid_inputs, valid_labels = dataset.next_validation_batch(args.gpu)
#     val_batch_size = valid_inputs.size(dim=1)
#
#     print('--------------------------------------Validation Batch {}---------------------------------'.format(batch_idx_i+1))
#
#     for sent_idx_j in range(val_batch_size):
#
#         val_sent_j, val_label_j = valid_inputs[:, sent_idx_j], valid_labels[sent_idx_j]
#
#         print('Original sentence {}: {}, "{}"'.format(sent_idx_j+1, dataset.idxs2sentence_val(model, val_sent_j), dataset.idx2label(val_label_j)))
#
#         val_sent_j = val_sent_j.unsqueeze(dim=1)
#         enc_inputs = val_sent_j.repeat(1, num_gen_instances)
#
#         mu_val, logvar_val = model.forward_encoder(enc_inputs)
#         z_val = model.sample_z(mu_val, logvar_val)
#
#         c_val = model.sample_c_prior(num_gen_instances)
#
#         print('+++++Generated Instances++++++')
#
#         for gen_k in range(num_gen_instances):
#
#             z_val_k = z_val[gen_k].unsqueeze(dim=0)
#             c_val_k = c_val[gen_k].unsqueeze(dim=0)
#
#             val_idx_gen_k = model.sample_sentence(z_val_k, c_val_k, raw=False, temp=1)
#             val_sent_gen_k = dataset.idxs2sentence(val_idx_gen_k)
#
#             _, c_val_idx_k = torch.max(c_val_k, dim=1)
#
#             print('Gen {}: {}, "{}"'.format(gen_k+1, val_sent_gen_k, dataset.idx2label(int(c_val_idx_k))))
#
#         print()
#         print()



# Samples latent and conditional codes randomly from prior
z = model.sample_z_prior(1)
c = model.sample_c_prior(1)

# Generate positive sample given z
c[0, 0], c[0, 1] = 1, 0

_, c_idx = torch.max(c, dim=1)
sample_idxs = model.sample_sentence(z, c, temp=1)

print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx))))
print('Generated: {}'.format(dataset.idxs2sentence(sample_idxs)))

# Generate negative sample from the same z
c[0, 0], c[0, 1] = 0, 1

_, c_idx = torch.max(c, dim=1)
sample_idxs = model.sample_sentence(z, c, temp=1)

print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx))))
print('Generated: {}'.format(dataset.idxs2sentence(sample_idxs)))

print()

# Interpolation
c = model.sample_c_prior(1)

_, c_idx = torch.max(c, dim=1)

z1 = model.sample_z_prior(1).view(1, 1, z_dim)
z1 = z1.cuda() if args.gpu else z1

z2 = model.sample_z_prior(1).view(1, 1, z_dim)
z2 = z2.cuda() if args.gpu else z2

# Interpolation coefficients
alphas = np.linspace(0, 1, 5)

print('Interpolation of z:')
print('-------------------')

for alpha in alphas:
    z = float(1-alpha)*z1 + float(alpha)*z2

    sample_idxs = model.sample_sentence(z, c, temp=1)
    sample_sent = dataset.idxs2sentence(sample_idxs)

    print("alpha={}, {}, '{}'".format(alpha, sample_sent, dataset.idx2label(c_idx)))

print()
