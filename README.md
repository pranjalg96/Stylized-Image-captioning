# Stylized Image captioning
Reproducing Hu, et. al., ICML 2017's "Toward Controlled Generation of Text" in PyTorch to solve
the Stylized captioning problem. Only perform romantic style captioning from factual.

(wise0dd's implementation, but with actual dataset rather than inbuilt SST dataset. Some minor
structural changes as well.)

## Requirements
1. Python 3.5+
2. PyTorch
3. TorchText <https://github.com/pytorch/text>

## Dataset
1. FLICKR/flickr8k_data contains a factual style dataset
2. FLICKR/flickr7k_style_data contains the romantic style dataset

## Set up the data
Check FLICKR/file_loaders to create the train, validation and test sets

## Training
1. Run `python train_vae_flickr8k.py --save {--gpu}`. This will create `built_vocab.pkl`, which is a vocabulary usuful in the second part. Also creates 'models/flickr8k_vae.bin'. Essentially this is the base VAE as in Bowman, 2015 [2].
2. Run `python train_discriminator_flickr8k.py --save {--gpu}`. This will create `ctextgen_flickr8k.bin`. The discriminator is using Kim, 2014 [3] architecture and the training procedure is as in Hu, 2017 [1].

# Evaluation
1. Run 'python valid_flickr8k.py', to generate sentences from validation set. Also contains 
latent space interpolation and tests for disentanglement.
2. Run 'compute_perplexity_vae_val_flickr8k.py' to evaluate perplexity on validation set.
3. Run 'conditional_text_generation_accuracy.py' to test the accuracy on conditional generation. 

## References
1. Hu, Zhiting, et al. "Toward controlled generation of text." International Conference on Machine Learning. 2017. [[pdf](http://proceedings.mlr.press/v70/hu17e/hu17e.pdf)]
2. Bowman, Samuel R., et al. "Generating sentences from a continuous space." arXiv preprint arXiv:1511.06349 (2015). [[pdf](https://arxiv.org/pdf/1511.06349.pdf?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue)]
3. Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014). [[pdf](https://arxiv.org/pdf/1408.5882)]