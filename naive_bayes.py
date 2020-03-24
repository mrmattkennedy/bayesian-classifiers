import pdb
import math
import time
import random
import collections

class naive_bayes():
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.stop_words = [word.rstrip('\n') for word in open('data/stopwords.data', 'r')]
        
    def load_word_counts(self, split=None):
        #Initialize data
        train = [line.rstrip('\n').split(' ') for line in open('data/forumTraining.data', 'r')]
        test = [line.rstrip('\n').split(' ') for line in open('data/forumTest.data', 'r')]
        self.articles = train + test

        #Initialize counters
        self.word_counts = collections.defaultdict(collections.Counter)
        self.class_counts = collections.defaultdict(int)
        self.class_probabilities = collections.defaultdict(int)
        
        #Remove stopwords and stem
        for article in self.articles:
            for word in range(len(article) - 1, 1, -1):
                if article[word] in self.stop_words:
                    del article[word]
                    
        #Split into train and test
        random.shuffle(self.articles)
        if split is not None:
            self.train = self.articles[:split]
            self.test = self.articles[split:]
        else:
            self.train = self.articles
            self.test = self.articles
        
        #Get word counts per article
        for article in self.train:
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
        for article in self.test:
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
        if self.verbose:
            print("Accuracy: {}%".format(round((correct_count * 100) / len(self.test), 4)))
            print("Right: {}/{}\nWrong: {}/{}".format(correct_count, len(self.test), len(self.test) - correct_count, len(self.test)))
            
        return round((correct_count * 100) / len(self.test), 4)

    def get_results(self, start, stop, step=500, avg_iters=5):
        data = []
        #Classify for each train size from start to stop + 1 with step.
        for i in range(start, stop + 1, step):
            #Reset the average
            avg = 0
            for _ in range(avg_iters):
                #See how long it takes
                start = time.time()
                self.load_word_counts(i)
                results = self.classify()
                avg += results

                #If verbose, print info
                if self.verbose:
                    print("Time: {}".format(round(time.time() - start, 3)))
                    print(i)
                    print("---------------\n")

            #Append i and average for i
            data.append([str(i), str(round(avg / avg_iters, 4))])

        with open("data/results.data", "w") as file:
            for row in data:
                line = ",".join(row)
                file.write(line + '\n')

                    
    def visualize(self):
        #No need to import these unless doing visualization
        import numpy as np
        import matplotlib.pyplot as plt

        #Get results
        results = [line.rstrip("\n").split(',') for line in open('data/results.data', 'r')]
        train_size = [int(elem[0]) for elem in results]
        accuracy = [float(elem[1]) for elem in results]

        #Get best quadratic fit
        x = np.array(train_size)
        y = np.array(accuracy)
        fit_coefficients = np.polyfit(x, y, 4)
        fit = np.poly1d(fit_coefficients)

        #Plot
        plt.figure().canvas.set_window_title("Train size vs accuracy")
        print(fit_coefficients)
        plt.plot(train_size, accuracy, '-o')
        plt.plot(x, fit(x))
        plt.grid()
        plt.show()

data = []
temp = naive_bayes(verbose=True)
temp.get_results(start = 1000, stop = 18000, step = 500, avg_iters=5)
#temp.load_word_counts()
#temp.visualize()
#results = temp.classify()

