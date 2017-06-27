import pandas as pd

# read data
df = pd.read_table('SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

# Data Prepossessing
"""\Convert the values in the 'label' colum to numerical values using map method as follows:
{'ham':0, 'spam':1} This maps the 'ham' value to 0 and the 'spam' value to 1.
Also, to get an idea of the size of the dataset we are dealing with, print out
number of rows and columns using 'shape'.
"""

df['label'] = df.label.map({'ham': 0, 'spam': 1})
#print(df.head())

"""
#Practice understand
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())

# print(lower_case_documents)

sans_punctuation_documents = []
import string

# removing panctuation


for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))

# tocakenize
preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))

# print(preprocessed_documents)
# Count frequencies

frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)

# pprint.pprint(frequency_list)

# Implementing Bag of Words in scikit-learn

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
count_vector.fit(documents)


doc_array = count_vector.transform(documents).toarray()
print(doc_array)

frequency_matrix = pd.DataFrame(doc_array,
                                columns=count_vector.get_feature_names())

print(frequency_matrix)

"""

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


x_train, x_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

# print('Number of rows in the total set: {}'.format(df.shape[0]))
# print('Number of rows in the training set: {}'.format(x_train.shape[0]))
# print('Number of rows in the test set: {}'.format(x_test.shape[0]))

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(x_train)
# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(x_test)


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
print(naive_bayes.fit(training_data, y_train))
predictions = naive_bayes.predict(testing_data)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))