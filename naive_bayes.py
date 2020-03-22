import math
import collections
import pdb
import time
from nltk.stem import PorterStemmer

class naive_bayes():
    def __init__(self):
        self.total_articles = 0
        self.word_counts = collections.defaultdict(collections.Counter)
        self.class_probabilities = collections.defaultdict(int)
        self.stop_words = [word.rstrip('\n') for word in open('data/stopwords.data', 'r')]
        
        self.load_word_counts()        
        
    def load_word_counts(self):
        train = [line.rstrip('\n').split(' ') for line in open('data/forumTraining.data', 'r')]
        test = [line.rstrip('\n').split(' ') for line in open('data/forumTest.data', 'r')]
        self.articles = train + test

        #Remove stopwords and stem
        porter = PorterStemmer()
        for article in self.articles:
            for word in range(len(article) - 1, 1, -1):
                if article[word] in self.stop_words:
                    del article[word]
                else:
                    #article[word] = porter.stem(article[word])
                    #do nothing - this takes quite a bit longer and has worse accuracy
                    pass
                    
        #Get word counts per article
        for article in self.articles:
            a_type = article[0]
            self.total_articles += 1
            self.class_probabilities[a_type] += 1
            for word in article[1:]:
                self.word_counts[a_type][word] += 1
        
    def classify(self):
        types = self.word_counts.keys()
        correct_count = 0

        #Loop through each article
        for article in self.articles:

            #Get the likelihood of each class
            likelihoods = collections.defaultdict(int)
            normalizer = collections.defaultdict(int)
            posterior = collections.defaultdict(int)
            correct_type = article[0]

            #Check each word and append likelihood for each type
            for word in article[1:]:
                for a_type in types:
                        likelihoods[a_type] += math.log((self.word_counts[a_type][word] + 0.5) / (self.class_probabilities[a_type] + 1))

            """
            #Better results without normalizer
            for a_type in types:
                p_not_type = sum([value for key, value in self.class_probabilities.items() if key != a_type])
                p_words_not_type = sum([value for key, value in likelihoods.items() if key != a_type])
                normalizer[a_type] = likelihoods[a_type] * math.log(self.class_probabilities[a_type])
                normalizer[a_type] += (p_not_type * p_words_not_type)
            """            
                
            #Multiply by the actual count
            for a_type in types:
                normalizer[a_type] = 0
                posterior[a_type] = (likelihoods[a_type] * math.log(self.class_probabilities[a_type])) / (normalizer[a_type] + 1)
            
            #Get the max
            maximum = max(posterior, key=posterior.get)
            if correct_type == maximum:
                correct_count += 1

        print(correct_count / len(self.articles))        

start = time.time()
temp = naive_bayes()
temp.classify()
print(time.time() - start)
