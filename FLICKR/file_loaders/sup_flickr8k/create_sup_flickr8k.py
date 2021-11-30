import csv
import random

# Will store the factual examples, with label 0. To start with, use all first examples
# in flickr8k dataset
supervised_examples = []
img_names = []

avg_length = 0.0
max_length = 0

num_selected = 0
first_row = True

num_train = 14000
num_valid = 550

with open('FLICKR/flickr8k_data/flickr8k_captions.txt', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, dialect='excel')

    for row in csvreader:

        if first_row:
            first_row = False
            continue

        row_8k_comment = row[1]
        row_8k_img_name = row[0]

        if row_8k_img_name in img_names:
            continue

        img_names.append(row_8k_img_name)

        row_8k_word_list = row_8k_comment.split(' ')
        len_i = len(row_8k_word_list)

        # if len_i > trunc_len:
        #     continue

        if len_i > max_length:
            max_length = len_i

        avg_length += len_i

        supervised_examples.append([row_8k_comment, 'factual'])

        num_selected += 1


with open('FLICKR/flickr7k_style_data/romantic_train.txt') as csvfile:
    csvreader = csv.reader(csvfile, dialect='excel', delimiter='|')

    for row in csvreader:

        row_7k_comment = row[0]

        row_7k_word_list = row_7k_comment.split(' ')
        len_i = len(row_7k_word_list)

        # if len_i > trunc_len:
        #     continue

        if len_i > max_length:
            max_length = len_i

        avg_length += len_i

        supervised_examples.append([row_7k_comment, 'romantic'])

        num_selected += 1

avg_length /= num_selected

print('Average length of captions: ', avg_length)
print('Max length of captions: ', max_length)

print('Number of supervised examples: ', num_selected)

num_test = num_selected - num_train - num_valid

print('Number of test examples: ', num_test)

random.shuffle(supervised_examples)

train_examples = supervised_examples[:num_train]
valid_examples = supervised_examples[num_train: num_train+num_valid]
test_examples = supervised_examples[num_train+num_valid:]


# Create training file
with open('FLICKR/flickr8k_data/flickr8k_sup_train.csv', 'w', newline='', encoding='utf-8') as result_file:
    wr = csv.writer(result_file, dialect='excel', delimiter='|')

    for row in train_examples:

        wr.writerow(row)

# Create validation file
with open('FLICKR/flickr8k_data/flickr8k_sup_valid.csv', 'w', newline='', encoding='utf-8') as result_file:
    wr = csv.writer(result_file, dialect='excel', delimiter='|')

    for row in valid_examples:

        wr.writerow(row)

# Create test file
with open('FLICKR/flickr8k_data/flickr8k_sup_test.csv', 'w', newline='', encoding='utf-8') as result_file:
    wr = csv.writer(result_file, dialect='excel', delimiter='|')

    for row in test_examples:

        wr.writerow(row)








