import csv
import numpy as np
import torch
from torch.autograd import Variable

recover_unsuper = []

with open('FLICKR/flickr30k_data/unsup_flickr30k_filtered.csv', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, dialect='excel')

    for row in csvreader:
        print(row)




