"""
    Sean Corbett
    04/16/2018
    M462
    Chapter 10: Tutorial 1 - Federalist Paper Prediction

    NOTE: PLEASE INSTALL PANDAS BEFORE RUNNING THIS SCRIPT.
"""

from stop_words import get_stop_words
from collections import namedtuple, Counter
from re import sub, finditer
import numpy as np
import pandas as pd

def clean_paper(paper_in, stopwords_set):
    """
        INPUT: paper_in = The raw Federalist Papers data.

               stopwords_set = A set of predetermined stopwords to remove from
                               the raw data in the Federalist Papers.


        RETURN: paper_in = Federalist Papers set with stopwords removed.
    """

    paper_in = [word for word in sub('[^A-Za-z0-9]+', ' ', paper_in).lower().split(' ') if word not in stopwords_set]
    return paper_in

def load_stopwords():
    """
        INPUT: None


        RETURN: stopwords_set = set of stopwords (English) to use for data
                                cleaning
    """
    stopwords = get_stop_words('en')
    stopwords_set = set(stopwords) | set(' ')
    return stopwords_set

def load_data(path):
    """
        INPUT: path = Path to Federalist Papers data.


        RETURN: word_dict = Dictionary of words in cleaned Federalist Papers and
                            their associated frequency.

                user_dict = Dictionary of words and authors associated with a
                            given word.

                train_labels = Vectors of labels based upon probability of
                               whether or not author is associated with a
                               given word in a given paper.

                common_list = List of words common in Federalist Papers.
    """


    stopwords_set = load_stopwords()
    paper_dict = {}
    sentence_dict = {}
    number_of_sentences = 0
    current_sentence = ''
    sentence_string = ''

    with open('./' + path + '/owners.txt') as owners_file:
        for line in owners_file:
            key, value = line.replace('\n', ' ').split(',')
            paper_dict[int(key)] = value

    with open('./' + path + '/fed_papers.txt') as papers_file:
        for line in papers_file:
            sentence_string += line.replace('\n', ' ')

    position_dict = {}

    opening = 'To the People of the State of New York'
    counter = 0

    for m in finditer(opening, sentence_string):
        counter += 1
        position_dict[counter] = [m.end()]

    close = 'PUBLIUS'
    end_counter = 0

    for m in finditer(close, sentence_string):
        end_counter += 1
        position_dict[end_counter].append(m.start())

    del position_dict[counter]

    word_dict = {}

    for paper_number in position_dict:
        beginning, end = position_dict[paper_number]
        author = paper_dict[paper_number]
        label = identifier(paper_number, author)
        paper = clean_paper(sentence_string[beginning+1:end-1], stopwords_set)
        word_dict[label] = Counter(paper)

    table = dict.fromkeys(set(paper_dict.values()), 0)
    for label in word_dict:
        table[label.author] += 1

    disputed_list = [18, 19, 20, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63]

    train_labels = [label for label in word_dict if label.index not in disputed_list]

    user_dict = {}

    for label in train_labels:
        words = list(word_dict[label].keys())
        for word in words:
            value = user_dict.get(word)

            if value is None:
                user_dict[word] = set([label.author])
            else:
                user_dict[word] = value | set([label.author])

    common_list = [word for word in user_dict if len(user_dict[word]) == 3]

    for label in word_dict:
        D = word_dict[label]
        new_dict = {}
        for word in D:
            if word in common_list:
                new_dict[word] = D[word]

        word_dict[label] = new_dict

    return word_dict, user_dict, train_labels, common_list

def create_log_priors(word_dict, user_dict, train_labels):
    """
        INPUT: word_dict = Dictionary of words in cleaned Federalist Papers and
                           their associated frequency.

               user_dict = Dictionary of words and authors associated with a
                           given word.

               train_labels = Vectors of labels based upon probability of
                              whether or not author is associated with a
                              given word in a given paper.


        RETURN: log_priors = The probabilities that a common set word will be
                             randomly sampled from among the non-stopwords.

                freq_distribution_dict = Contains a dictionary for each the
                                         three authors. The value associated
                                         with a particular author is a frequency
                                         dictionary, hence, each key is a word
                                         and the associated value is the
                                         frequency of occurrence of the word.
    """

    authors = ['MADISON ', 'JAY ', 'HAMILTON ']
    log_priors = dict.fromkeys(authors, 0)
    freq_distribution_dict = dict.fromkeys(authors)

    for label in train_labels:
        number, author = label
        D = word_dict[label]
        distribution = freq_distribution_dict.get(author)

        if distribution is None:
            distribution = D
        else:
            for word in D:
                value = distribution.get(word)
                if value is not None:
                    distribution[word] += D[word]
                else:
                    distribution[word] = D[word]

        freq_distribution_dict[author] = distribution
        log_priors[author] += 1

    return log_priors, freq_distribution_dict

def relative_frequency(train_labels, freq_distribution_dict, log_priors, common_list):
    """
        INPUT: train_labels = Vectors of labels based upon probability of
                              whether or not author is associated with a
                              given word in a given paper.

                freq_distribution_dict = Contains a dictionary for each the
                                         three authors. The value associated
                                         with a particular author is a frequency
                                         dictionary, hence, each key is a word
                                         and the associated value is the
                                         frequency of occurrence of the word.

                log_priors = The probabilities that a common set word will be
                                     randomly sampled from among the non-stop.

                common_list = List of words common in Federalist Papers.


        RETURN: distribution_dict = Contains a dictionary with the relative
                                         frequency of a given word for each
                                         author.

                log_priors = Dictionary of the log-probability of an author
                             using a given word given all words in the
                             documents.
    """


    number_rows = len(train_labels)
    authors = ['MADISON ', 'JAY ', 'HAMILTON ']

    log_probability_dict = dict.fromkeys(authors, {})
    distribution_dict = dict.fromkeys(authors)

    for author in authors:
        author_dict = {}
        log_priors[author] = np.log(log_priors[author]/number_rows)

        number_of_words = sum([freq_distribution_dict[author][word] for word in common_list])

        for word in common_list:
            relative_frequency = freq_distribution_dict[author][word]/number_of_words
            author_dict[word] = np.log(relative_frequency)

        distribution_dict[author] = [log_priors[author], author_dict]

    return log_probability_dict, distribution_dict

def assign_author(test_author, predicted_author, confusion_matrix):
    """
        INPUT: test_author = Actual author of the given document.

               predicted_author = Predicted author of the given document.

               confusion_matrix = The n by n confusion matrix containing the
                                  true positive, falst positive, true negative
                                  and false negative assignments for each
                                  author.


        RETURN: confusion_matrix = The n by n confusion matrix containing the
                           true positive, falst positive, true negative
                           and false negative assignments for each
                           author after incrementing the document count for
                           a test author vs the predicted author by one.
    """

    authors = ['MADISON ', 'JAY ', 'HAMILTON ']
    i, j = authors.index(test_author), authors.index(predicted_author)
    confusion_matrix[i, j] += 1

    return confusion_matrix

def multinomial_bayes(word_dict, distribution_dict):
    """
        INPUT: word_dict = Dictionary of words in cleaned Federalist Papers and
                           their associated frequency.

               distribution_dict = Contains a dictionary with the relative
                                   frequency of a given word for each
                                   author.


        RETURN: confusion_matrix = The n by n confusion matrix containing the
                                   true positive, falst positive, true negative
                                   and false negative assignments for each
                                   author.

                accuracy = The final accuracy of all predictions made for
                           authors of papers within the Federalist Papers.
    """

    authors = ['MADISON ', 'JAY ', 'HAMILTON ']
    number_of_groups = len(authors)
    confusion_matrix = np.zeros(shape = (number_of_groups, number_of_groups))

    skip = [18, 19, 20, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63]

    for label in word_dict:
        test_number, test_author = label

        if test_number not in skip:
            x_i = word_dict[label]
            posterior_probability = dict.fromkeys(authors, 0)
            for author in authors:
                log_prior, log_probability_dict = distribution_dict[author]
                posterior_probability[author] = log_prior + sum([x_i[word]*log_probability_dict[word] for word in x_i])

        posterior_probability_list = list(posterior_probability.values())
        posterior_probability_authors = list(posterior_probability.keys())
        maximum_index = np.argmax(posterior_probability_list)
        prediction = posterior_probability_authors[maximum_index]

        if test_author == 'H and M ':
            confusion_matrix = assign_author(test_author='HAMILTON ', predicted_author=prediction, confusion_matrix=confusion_matrix)
            confusion_matrix = assign_author(test_author='MADISON ', predicted_author=prediction, confusion_matrix=confusion_matrix)
        else:
            confusion_matrix = assign_author(test_author, prediction, confusion_matrix)

    accuracy = sum(np.diag(confusion_matrix)) / sum(sum(confusion_matrix))
    confusion_matrix = pd.DataFrame(data=confusion_matrix, columns=['Hamilton', 'Madison', 'Jay']).rename({0: 'Hamilton', 1: 'Madison', 2: 'Jay'}, axis='index')
    return confusion_matrix, accuracy


identifier = namedtuple('label', 'index author')
word_dict, user_dict, train_labels, common_list = load_data('data')
log_priors, freq_distribution_dict = create_log_priors(word_dict, user_dict, train_labels)
log_probability_dict, distribution_dict = relative_frequency(train_labels, freq_distribution_dict, log_priors, common_list)
confusion_matrix, accuracy = multinomial_bayes(word_dict, distribution_dict)

print('Confusion Matrix:\n', confusion_matrix)
print('\nacc = ', accuracy)
