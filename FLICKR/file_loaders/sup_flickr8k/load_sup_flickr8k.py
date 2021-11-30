import csv
import numpy as np
import torch
from torch.autograd import Variable

recover_unsuper = []

with open('FLICKR/flickr8k_data/flickr8k_sup_train.csv', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, dialect='excel', delimiter='|')

    for row in csvreader:
        print(row)




