# Use this to create a dataset for the language model part (basically, train a caption generator). Just load in all
# the captions from flickr30k. Split this into a train and validation file.

# import pandas as pd
# import csv
#
# flickr30k_data = pd.read_csv('FLICKR/flickr30k_results_rem_commas.csv', delimiter='|')
#
# unsupervised_examples = []  # Will store the unsupervised examples
#
# avg_length = 0.0
# max_length = 0
#
#     # See example 182, has some double quotes that makes it a NaN. Skip for now.
#     if isinstance(flickr30k_data[' comment'][i], float):
#         continue
#
#     row_30k_comment = flickr30k_data[' comment'][i][1:] # Removes extra space at the beginning
#
#     row_30k_word_list = row_30k_comment.split(' ')
#     len_i = len(row_30k_word_list)
#
#     if len_i <= trunc_len:
#         unsupervised_examples.append([row_30k_comment])
#
#     if len_i > max_length:
#         max_length = len_i
#
#     avg_length += len_i
#
# # avg_length /= len(unsupervised_examples)
#
# # print('Average length of captions: ', avg_length)
# # print('Max length of captions: ', max_length)
#
# print('Number of unsupervised examples under or equal to length ' + str(trunc_len) + ': ', len(unsupervised_examples))
#
# with open('FLICKR/unsup_flickr30k_filtered.csv', 'w', newline='') as result_file:
#     wr = csv.writer(result_file, dialect='excel')
#
#     for unsup_ex in unsupervised_examples:
#         wr.writerow(unsup_ex)

import csv

unsupervised_examples = []  # Will store the unsupervised examples

avg_length = 0.0
max_length = 0

trunc_len = 20

num_selected = 0

with open('FLICKR/flickr30k_data/unsup_flickr30k_filtered.csv', 'w', newline='', encoding='utf-8') as result_file:
    wr = csv.writer(result_file, dialect='excel')

    with open('FLICKR/flickr30k_data/flickr30k_results_rem_commas.csv', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, quotechar='"', dialect='excel', quoting=csv.QUOTE_NONE, delimiter='|', skipinitialspace=True)

        for row in csvreader:
            row_30k_comment = row[2]

            row_30k_word_list = row_30k_comment.split(' ')
            len_i = len(row_30k_word_list)

            if len_i > trunc_len:
                continue

            if len_i > max_length:
                max_length = len_i

            avg_length += len_i

            num_selected += 1

            wr.writerow([row_30k_comment])

    avg_length /= num_selected

    print('Average length of captions: ', avg_length)
    print('Max length of captions: ', max_length)

    print('Number of unsupervised examples under or equal to length ' + str(trunc_len) + ': ', num_selected)




