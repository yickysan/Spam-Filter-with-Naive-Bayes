import pandas as pd
import re
import gradio as gr

sms = pd.read_csv('SMSSpamCollection',
                 sep ='\t',
                 header = None,
                 names = ['Label', 'SMS']
                 )
sample = sms.sample(frac=1, random_state=1) # randomizes the dataset
training_set = sample.iloc[0:4458].reset_index(drop=True) # (drop=True) drops old index after resetting index 
test_set = sample.iloc[4458:].reset_index(drop=True)

#removing punctuations
training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ', regex=True)

#turning every text to lower case
training_set['SMS'] = training_set['SMS'].str.lower()

#extracting every individual word in the sms column
training_set['SMS'] = training_set['SMS'].str.split()
vocabulary = []
for value in training_set['SMS']:
    for i in value:
        vocabulary.append(i)
vocabulary = set(vocabulary) # using the set function gets rid of duplicate values
vocabulary = list(vocabulary) 

word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, message in enumerate(training_set['SMS']):
    for word in message:
        word_counts_per_sms[word][index] += 1

clean_training_set = pd.concat([training_set, word_counts], axis=1)
spam_messages = clean_training_set[clean_training_set['Label']=='spam'].shape[0] # returns the number of rows with spam messages

ham_messages = clean_training_set[clean_training_set['Label']=='ham'].shape[0] # returns the number of rows with ham messages

total_messages = clean_training_set.shape[0]

probability_spam = spam_messages / total_messages
probability_ham = ham_messages / total_messages

spam_words = clean_training_set[clean_training_set['Label']=='spam']['SMS'].apply(len) # len function counts the individual words in the sms column
n_spam = spam_words.sum() # sums them to get the total amount of spam words in the SMS column

ham_words = clean_training_set[clean_training_set['Label']=='ham']['SMS'].apply(len)
n_ham = ham_words.sum()

n_vocabulary = len(vocabulary)
alpha = 1 #laplace smoothing

spam_dict = {unique_word:0 for unique_word in vocabulary} # initializes  a dictionary with every word in the vocabulary list as a key.
ham_dict = {unique_word:0 for unique_word in vocabulary}

# creating new DataFrames for both Spam and Ham messages
spam_df = clean_training_set[clean_training_set['Label']=='spam'].copy() 
ham_df = clean_training_set[clean_training_set['Label']=='ham'].copy()

#calculating the probability for each word in spam messages
for word in vocabulary:
    n_word_spam_messages = spam_df[word].sum() # the number of times the word occurs in the spam DataFrame
    p_word_spam_messages = (n_word_spam_messages + alpha) / (n_spam + alpha * n_vocabulary)
    spam_dict[word] =  p_word_spam_messages # updates the dictionary values with the probability of each unique word
    

#calculating the probability for each word in ham messages
for word in vocabulary:
    n_word_ham_messages = ham_df[word].sum() # the number of times the word occurs in the spam DataFrame
    p_word_ham_messages = (n_word_ham_messages + alpha) / (n_ham + alpha * n_vocabulary)
    ham_dict[word] =  p_word_ham_messages # updates the dictionary values with the probability of each unique word
        
def classify(message):
    """function to classify sms as spam or non-spam messages"""

    message = re.sub('\W', ' ', message) # replaces any special character with an empty string
    message = message.lower()
    message = message.split()


    p_spam_given_message = probability_spam
    p_ham_given_message = probability_ham
    
    for word in message: # checks if the word is in ham_dict and spam_dict and gets their probability
        if word in spam_dict:
            p_spam_given_message *= spam_dict[word] 

            
        if word in ham_dict:
            p_ham_given_message *= ham_dict[word]
            
   

    if p_ham_given_message > p_spam_given_message:
        return 'This is not a spam message.'
    elif p_ham_given_message < p_spam_given_message:
        return 'This is a spam message.'
    else:
        return 'Unsure, have a human classify this!'

interface = gr.Interface(fn=classify,
                         inputs = gr.Textbox(lines=10,
                                             placeholder = 'enter your message here....'),
                         outputs='text',
                         description = 'If your message is wrongly classified, please click the flag button.'
                        )

interface.launch()