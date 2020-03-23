import math
import collections
import pdb
import time

class naive_bayes():
    def __init__(self):
        self.word_counts = collections.defaultdict(collections.Counter)
        self.class_counts = collections.defaultdict(int)
        self.class_probabilities = collections.defaultdict(int)
        self.stop_words = [word.rstrip('\n') for word in open('data/stopwords.data', 'r')]
        
        self.load_word_counts()        
        
    def load_word_counts(self):
        train = [line.rstrip('\n').split(' ') for line in open('data/forumTraining.data', 'r')]
        test = [line.rstrip('\n').split(' ') for line in open('data/forumTest.data', 'r')]
        self.articles = train + test

        #Remove stopwords and stem
        for article in self.articles:
            for word in range(len(article) - 1, 1, -1):
                if article[word] in self.stop_words:
                    del article[word]
                    
        #Get word counts per article
        for article in self.articles:
            a_type = article[0]
            self.class_probabilities[a_type] += 1
            for word in article[1:]:
                self.word_counts[a_type][word] += 1
        
        for a_type, counter in self.word_counts.items():
            self.class_counts[a_type] = sum(counter.values())
                
    def classify(self):
        types = self.word_counts.keys()
        correct_count = 0

        #Loop through each article
        for article in self.articles:

            #Get the likelihood of each class
            likelihoods = collections.defaultdict(int)
            posterior = collections.defaultdict(int)
            correct_type = article[0]

            #Check each word and append likelihood for each type
            for word in article[1:]:
                for a_type in types:
                    #Scalar additions are for smoothing. Small amt to numerator so weight is all on word frequency
                    likelihoods[a_type] += math.log((self.word_counts[a_type][word] + 1e-50) / (self.class_counts[a_type] + 1))
            
            #Multiply by the actual count
            for a_type in types:
                posterior[a_type] = likelihoods[a_type] * math.log(self.class_probabilities[a_type])

            #Get the max
            maximum = max(posterior, key=posterior.get)
            if correct_type == maximum:
                correct_count += 1

        print("Accuracy: {}%".format(round((correct_count * 100) / len(self.articles), 4)))
        print("Right: {}/{}\nWrong: {}/{}".format(correct_count, len(self.articles), len(self.articles) - correct_count, len(self.articles)))

start = time.time()
temp = naive_bayes()
temp.classify()
print(time.time() - start)
