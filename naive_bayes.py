import math
import collections
import pdb
import time
"""
P(article type | words)
Need -
Numerator:
    P(words | article type)
        P(word 1 | article type), P(word n | article type)
            P(Ai|Ck) = # instances of Ai belonging to class Ck
            total word count for that word?
        P(article type) * PI(P(words | article type))
    P(article type)

Denominator:
    P(A1, A1, An)
        Dissolves to numerator + (P(not article type) * P(words | not article type)


Each line is article, first word is type.
Need the count of each word for each article type
Memory concerns here
Sacrifice speed - just keep in line and search each line after?
"""

class naive_bayes():
    def __init__(self):
        self.articles = [line.rstrip('\n').split(' ') for line in open('data/forumTraining.data', 'r')]
        self.total_articles = 0
        self.word_counts = collections.defaultdict(collections.Counter)
        self.total_word_counts = collections.Counter()
        self.class_probabilities = collections.defaultdict(int)
        self.stop_words = [word.rstrip('\n') for word in open('data/stopwords.data', 'r')]
        
        self.load_word_counts()
        self.train()

        
    def load_word_counts(self):
        

        for article in self.articles:
            for word in range(len(article) - 1, 1, -1):
                if article[word] in self.stop_words:
                    del article[word]
                    
        #Get word counts per article
        for article in self.articles:
            a_type = article[0]
            self.total_articles += 1
            self.class_probabilities[a_type] += 1
            for word in article[1:]:
                self.word_counts[a_type][word] += 1
                
        #Get probability of each class
        #for key in self.class_probabilities.keys():
        #    self.class_probabilities[key] /= self.total_articles

        #Get total count for each word
        for article_type in self.word_counts.values():
            for word in article_type:
                self.total_word_counts[word] += article_type[word]

        
    def train(self):
        types = self.word_counts.keys()
        correct_count = 0
        start = time.time()
        for article in self.articles[:30]:
            #Get likelihooh of each class
            likelihoods = collections.defaultdict(int)
            correct_type = article[0]
            for word in article[1:]:
                for a_type in types:
                        likelihoods[a_type] += math.log((self.word_counts[a_type][word] + 0.5) / (self.class_probabilities[a_type] + 1))
            for a_type in types:
                likelihoods[a_type] *= math.log(self.class_probabilities[a_type])
            maximum = max(likelihoods, key=likelihoods.get)
            maxes = {k: v for k, v in sorted(likelihoods.items(), key=lambda item: item[1])}
            #keys = list(maxes.keys())[-3]
                
            if correct_type == maximum:
                correct_count += 1
            else:
                print(maxes)
        print(correct_count / len(self.articles))
        print(time.time() - start)
        return
        

temp = naive_bayes()
