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

import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')

args = parser.parse_args()

print(args.save)


mb_size = 32
# z_dim = 20
h_dim = 300
lr = 1e-3
lr_decay_every = 1000000
n_epochs = 100
log_interval = 100
z_dim = 100
c_dim = 2
kl_weight_max = 0.4

# Specific hyperparams
beta = 0.1
lambda_c = 0.1
lambda_z = 0.1
lambda_u = 0.1

dataset = flickr8k_Data(emb_dim=300, mbsize=mb_size)

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=args.gpu
)

num_rows = 14000
num_batches = math.ceil(num_rows/mb_size)

n_iter = n_epochs * num_batches

# Load pretrained base VAE with c ~ p(c)
model.load_state_dict(torch.load('models/ctextgen_flickr8k.bin'))
model.train()

dataset.create_batch_iterable()


def kl_weight(it):
    """
    Credit to: https://github.com/kefirski/pytorch_RVAE/
    0 -> 1
    """
    return (math.tanh((it - 11000)/1000) + 1)/2


def temp(it):
    """
    Softmax temperature annealing
    1 -> 0
    """
    return 1-kl_weight(it) + 1e-5  # To avoid overflow


def calc_accuracy(y_scores_raw, y_labels):

    y_scores = F.softmax(y_scores_raw, dim=1)
    _, y_preds = torch.max(y_scores, dim=1)

    cur_batch_size = y_scores_raw.size(0)

    disc_acc = torch.sum(y_preds == y_labels)
    disc_acc = disc_acc.item()/cur_batch_size

    return disc_acc


disc_losses = []
gen_losses = []
enc_losses = []

disc_accuracies = []

def main():
    trainer_D = optim.Adam(model.discriminator_params, lr=lr)
    trainer_G = optim.Adam(model.decoder_params, lr=lr)
    trainer_E = optim.Adam(model.encoder_params, lr=lr)

    it = 0

    trainer_D.zero_grad()
    trainer_G.zero_grad()
    trainer_E.zero_grad()

    for epoch in range(n_epochs):

        # dataset.create_batch_iterable()

        avg_disc_loss = 0.0
        avg_gen_loss = 0.0
        avg_enc_loss = 0.0

        avg_disc_acc = 0.0

        for batch_idx in range(num_batches):
        # The word embeddings are not being trained. Could consider doing that as well.
            inputs, labels = dataset.next_batch(args.gpu)

            """ Update discriminator, eq. 11 """
            batch_size = inputs.size(1)
            # get sentences and corresponding z
            x_gen, c_gen  = model.generate_sentences(batch_size)
            _, target_c = torch.max(c_gen, dim=1)

            y_disc_real = model.forward_discriminator(inputs.transpose(0, 1))
            y_disc_fake = model.forward_discriminator(x_gen)

            log_y_disc_fake = F.log_softmax(y_disc_fake, dim=1)
            # y_disc_fake_probs = F.softmax(y_disc_fake, dim=1)
            entropy = -log_y_disc_fake.mean()  # Seems to be calculated incorrectly
            # entropy = torch.sum(torch.mul(y_disc_fake_probs, log_y_disc_fake), dim=1)
            # entropy = torch.sum(entropy)/entropy.size(0)

            loss_s = F.cross_entropy(y_disc_real, labels)
            loss_u = F.cross_entropy(y_disc_fake, target_c) + beta*entropy

            disc_acc = calc_accuracy(y_disc_real, labels)
            avg_disc_acc += disc_acc

            loss_D = loss_s + lambda_u*loss_u

            avg_disc_loss += loss_D.item()

            loss_D.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.discriminator_params, 5)
            trainer_D.step()
            trainer_D.zero_grad()
            trainer_E.zero_grad()
            trainer_G.zero_grad()

            """ Update generator, eq. 8 """
            # Forward VAE with c ~ q(c|x) instead of from prior
            recon_loss, kl_loss = model.forward(inputs, use_c_prior=False)
            # x_gen: mbsize x seq_len x emb_dim
            x_gen_attr, target_z, target_c = model.generate_soft_embed(batch_size, temp=temp(it))

            # y_z: mbsize x z_dim
            y_z, _ = model.forward_encoder_embed(x_gen_attr.transpose(0, 1).detach())
            y_c = model.forward_discriminator_embed(x_gen_attr.detach())

            loss_vae = recon_loss + kl_weight_max * kl_loss
            loss_attr_c = F.cross_entropy(y_c, target_c)
            loss_attr_z = F.mse_loss(y_z, target_z)  # Replacing this with KL divergence might help

            loss_G = loss_vae + lambda_c*loss_attr_c + lambda_z*loss_attr_z

            avg_gen_loss += loss_G.item()

            loss_G.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.decoder_params, 5)
            trainer_G.step()
            trainer_G.zero_grad()
            trainer_E.zero_grad()  # Encoder gradients doubled otherwise?
            trainer_D.zero_grad()

            """ Update encoder, eq. 4 """
            recon_loss, kl_loss = model.forward(inputs, use_c_prior=False)

            loss_E = recon_loss + kl_weight_max * kl_loss

            avg_enc_loss += loss_E.item()

            loss_E.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.encoder_params, 5)
            trainer_E.step()
            trainer_E.zero_grad()
            trainer_G.zero_grad()
            trainer_D.zero_grad()

            if it % log_interval == 0:
                z = model.sample_z_prior(1) # Obtain z by passing validation sentences through encoder
                c = model.sample_c_prior(1) # Obtain c by passing opposite label as validation label

                sample_idxs = model.sample_sentence(z, c, temp=1)
                sample_sent = dataset.idxs2sentence(sample_idxs)


                print('Iter-{}; loss_D: {:.4f}; loss_G: {:.4f}; loss_E: {:.4f}; disc_acc: {:.4f}'
                      .format(it, float(loss_D), float(loss_G), float(loss_E), disc_acc))

                _, c_idx = torch.max(c, dim=1)

                print('c = {}'.format(dataset.idx2label(int(c_idx))))
                print('Sample: "{}" (temp=1)'.format(sample_sent))

                model.train()
                print()

            it += 1

        avg_disc_loss /= num_batches
        avg_gen_loss /= num_batches
        avg_enc_loss /= num_batches

        avg_disc_acc /= num_batches

        disc_losses.append(avg_disc_loss)
        gen_losses.append(avg_gen_loss)
        enc_losses.append(avg_enc_loss)

        disc_accuracies.append(avg_disc_acc)


def save_model():
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/ctextgen_flickr8k_final.bin')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        with open('Results/supervised/disc_losses.pkl', 'wb') as file:
            pickle.dump(disc_losses, file)

        with open('Results/supervised/gen_losses.pkl', 'wb') as file:
            pickle.dump(gen_losses, file)

        with open('Results/supervised/enc_losses.pkl', 'wb') as file:
            pickle.dump(enc_losses, file)

        with open('Results/supervised/disc_accuracies.pkl', 'wb') as file:
            pickle.dump(disc_accuracies, file)

        plt.plot(disc_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Disc loss')
        plt.show()

        plt.plot(gen_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Gen. loss')
        plt.show()

        plt.plot(enc_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Enc. loss')
        plt.show()

        plt.plot(disc_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Disc. Accuracies')
        plt.show()

        if args.save:
            save_model()

        exit(0)

    if args.save:
        save_model()
