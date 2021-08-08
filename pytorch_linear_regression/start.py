import sys
import os

INPUT_FOLDER = r'E:\Development\corpora\yelp_dataset'
OUTPUT_FOLDER = '.'

import pandas as pd

def load_yelp_orig_data():
    PATH_TO_YELP_REVIEWS = INPUT_FOLDER + '/yelp_academic_dataset_review.json'

    # read the entire file into a python array
    with open(PATH_TO_YELP_REVIEWS, 'r', encoding='utf-8') as f:
        data = f.readlines()

    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"

    # now, load it into pandas
    data_df = pd.read_json(data_json_str)
    print(data_df.head())
    a = 1


    data_df.head(100000).to_csv(OUTPUT_FOLDER + '/output_reviews_top.csv')

if not os.path.exists(OUTPUT_FOLDER + '/output_reviews_top.csv'):
    load_yelp_orig_data()

top_data_df = pd.read_csv(OUTPUT_FOLDER + '/output_reviews_top.csv')
print("Columns in the original dataset:\n")
print(top_data_df.columns)


import matplotlib.pyplot as plt 

print("Number of rows per star rating:")
print(top_data_df['stars'].value_counts())

# Function to map stars to sentiment
def map_sentiment(stars_received):
    if stars_received <= 2:
        return -1
    elif stars_received == 3:
        return 0
    else:
        return 1
# Mapping stars to sentiment into three categories
top_data_df['sentiment'] = [ map_sentiment(x) for x in top_data_df['stars']]
# Plotting the sentiment distribution
plt.figure()
pd.value_counts(top_data_df['sentiment']).plot.bar(title="Sentiment distribution in df")
plt.xlabel("Sentiment")
plt.ylabel("No. of rows in df")
plt.show()


# Function to retrieve top few number of each category
def get_top_data(top_n = 5000):
    top_data_df_positive = top_data_df[top_data_df['sentiment'] == 1].head(top_n)
    top_data_df_negative = top_data_df[top_data_df['sentiment'] == -1].head(top_n)
    top_data_df_neutral = top_data_df[top_data_df['sentiment'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative, top_data_df_neutral])
    return top_data_df_small

# Function call to get the top 10000 from each sentiment
top_data_df_small = get_top_data(top_n=10000)

# After selecting top few samples of each sentiment
print("After segregating and taking equal number of rows for each sentiment:")
print(top_data_df_small['sentiment'].value_counts())
top_data_df_small.head(10)


from gensim.utils import tokenize
from gensim.utils import simple_preprocess
import re
# Running this for the dataframe as this will be used for next steps
top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df_small['text']]
print(top_data_df_small['tokenized_text'].head(10))

a = 1








from gensim.utils import simple_preprocess
# Tokenize the text column to get the new column 'tokenized_text'
top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df_small['text']]
print(top_data_df_small['tokenized_text'].head(10))

# Step 5
# Example of Porter stemming
from gensim.parsing.porter import PorterStemmer
import nltk
# Uncomment the following for the first time run
# nltk.download('stopwords')
porter_stemmer = PorterStemmer()
# Get the stemmed_tokens required for next steps
top_data_df_small['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in top_data_df_small['tokenized_text'] ]
top_data_df_small['stemmed_tokens'].head(10)



# Step 9
from gensim import corpora
# Function to return the dictionary either with padding word or without padding
def make_dict(top_data_df_small, padding=True):
    if padding:
        print("Dictionary with padded token added")
        mydict = corpora.Dictionary([['pad']])
        mydict.add_documents(top_data_df_small['stemmed_tokens'])
    else:
        print("Dictionary without padding")
        mydict = corpora.Dictionary(top_data_df_small['stemmed_tokens'])
    return mydict

# Make the dictionary without padding for the basic models ( Padding is required for CNN )
mydict = make_dict(top_data_df_small, padding=False)

# Getting the mapping from word to unique id
from collections import Counter
d= Counter(mydict.token2id)
print(d.most_common()[:10])
print(len(mydict.token2id))



from sklearn.model_selection import train_test_split
# Train Test Split Function
def split_train_test(top_data_df_small, test_size=0.3, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df_small[['business_id', 'cool', 'date', 'funny', 'review_id', 'stars', 'text', 'useful', 'user_id', 'stemmed_tokens']],
                                                        top_data_df_small['sentiment'],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)
    print("Value counts for Train sentiments")
    print(Y_train.value_counts())
    print("Value counts for Test sentiments")
    print(Y_test.value_counts())
    print(type(X_train))
    print(type(Y_train))
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    print(X_train.head())
    return X_train, X_test, Y_train, Y_test

# Call the train_test_split
X_train, X_test, Y_train, Y_test = split_train_test(top_data_df_small)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
# Use cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)

from gensim import corpora
# Function to return the dictionary either with padding word or without padding
def make_dict(top_data_df_small, padding=True):
    if padding:
        print("Dictionary with padded token added")
        review_dict = corpora.Dictionary([['pad']])
        review_dict.add_documents(top_data_df_small['stemmed_tokens'])
    else:
        print("Dictionary without padding")
        review_dict = corpora.Dictionary(top_data_df_small['stemmed_tokens'])
    return review_dict

# Make the dictionary without padding for the basic models
review_dict = make_dict(top_data_df_small, padding=False)

VOCAB_SIZE = len(mydict)
NUM_LABELS = 3

# Function to make bow vector to be used as input to network
def make_bow_vector(review_dict, sentence):
    vec = torch.zeros(VOCAB_SIZE, dtype=torch.float64, device=device)
    for word in sentence:
        vec[review_dict.token2id[word]] += 1
    return vec.view(1, -1).float()

# Function to get the output tensor
def make_target(label):
    if label == -1:
        return torch.tensor([0], dtype=torch.long, device=device)
    elif label == 0:
        return torch.tensor([1], dtype=torch.long, device=device)
    else:
        return torch.tensor([2], dtype=torch.long, device=device)


# Defining neural network structure
class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # needs to be done everytime in the nn.module derived class
        super(BoWClassifier, self).__init__()

        # Define the parameters that are needed for linear model ( Ax + b)
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec): # Defines the computation performed at every call.
        # Pass the input through the linear layer,
        # then pass that through log_softmax.

        return F.log_softmax(self.linear(bow_vec), dim=1)

#  Initialize the model
bow_nn_model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
bow_nn_model.to(device)

# Loss Function
loss_function = nn.NLLLoss()
# Optimizer initlialization



optimizer = optim.SGD(bow_nn_model.parameters(), lr=0.01)

import time
start_time = time.time()

# Train the model
for epoch in range(100):
    print ("EPOCH", epoch)

    for index, row in X_train.iterrows():
        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        bow_nn_model.zero_grad()

        # Step 2. Make BOW vector for input features and target label
        bow_vec = make_bow_vector(review_dict, row['stemmed_tokens'])
        target = make_target(Y_train['sentiment'][index])

        # Step 3. Run the forward pass.
        probs = bow_nn_model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(probs, target)
        loss.backward()
        optimizer.step()
print("Time taken to train the model: " + str(time.time() - start_time))





