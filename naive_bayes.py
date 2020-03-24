import pdb
import math
import time
import random
import collections
from nltk.stem.snowball import SnowballStemmer

class naive_bayes():
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.stop_words = [word.rstrip('\n') for word in open('data/stopwords.data', 'r')]
        
    def load_word_counts(self, split=None, stem=False):
        #Initialize data
        train = [line.rstrip('\n').split(' ') for line in open('data/forumTraining.data', 'r')]
        test = [line.rstrip('\n').split(' ') for line in open('data/forumTest.data', 'r')]
        self.articles = train + test

        #Initialize counters
        self.word_counts = collections.defaultdict(collections.Counter)
        self.class_counts = collections.defaultdict(int)
        self.class_probabilities = collections.defaultdict(int)
        
        #Remove stopwords and stem
        stemmer = SnowballStemmer("english")
        for article in self.articles:
            for word in range(len(article) - 1, 1, -1):
                if article[word] in self.stop_words:
                    del article[word]
                elif stem:
                    article[word] = stemmer.stem(article[word])
                    
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

    def get_results(self, start, stop, step=500, avg_iters=5, stem=False):
        data = []
        
        for i in range(start, stop + 1, step):
            #Reset the average
            avg = 0
            avg_time = 0
            
            for _ in range(avg_iters):
                #See how long it takes
                start_time = time.time()
                self.load_word_counts(i, stem)
                results = self.classify()

                total_time = round(time.time() - start_time, 4)
                avg += results
                avg_time += total_time
                
                #If verbose, print info
                if self.verbose:
                    print("Time: {}".format(total_time))
                    print(i)
                    print("---------------\n")

            #Append i and average for i as well as time
            data.append([str(i), str(round(avg / avg_iters, 4)), str(round(avg_time / avg_iters, 4))])
            
        #avg_time = round(avg_time / ((math.floor((stop - start) / step) + 1) * avg_iters), 4)
        results_path = "data/results.data" if not stem else "data/results_stem.data"
        with open(results_path, "w") as file:
            #file.write(str(avg_time) + '\n')
            for row in data:
                line = ",".join(row)
                file.write(line + '\n')

                    
    def visualize(self):
        #No need to import these unless doing visualization
        import numpy as np
        import matplotlib.pyplot as plt

        #Get results
        results = [line.rstrip("\n").split(',') for line in open('data/results.data', 'r')]
        stem_results = [line.rstrip("\n").split(',') for line in open('data/results_stem.data', 'r')]
        
        train_size = [int(elem[0]) for elem in results]
        accuracy = [float(elem[1]) for elem in results]
        times = [float(elem[2]) for elem in results]
        stem_train_size = [int(elem[0]) for elem in stem_results]
        stem_accuracy = [float(elem[1]) for elem in stem_results]
        stem_times = [float(elem[2]) for elem in stem_results]

        #Get best quadratic fit
        x = np.array(train_size)
        y = np.array(accuracy)
        fit = np.poly1d(np.polyfit(x, y, 4))

        x_stem = np.array(stem_train_size)
        y_stem = np.array(stem_accuracy)
        fit_stem = np.poly1d(np.polyfit(x_stem, y_stem, 4))

        #Plot
        fig, (axs1, axs2) = plt.subplots(2)
        fig.canvas.set_window_title("Train size vs accuracy")
        fig.suptitle("Train size vs accuracy")
        axs1.plot(train_size, accuracy, '-o')
        axs1.plot(train_size, times, '-s')
        axs1.plot(x, fit(x))
        axs1.grid()

        axs2.plot(stem_train_size, stem_accuracy, '-o')
        axs2.plot(stem_train_size, stem_times, '-s')
        axs2.plot(x_stem, fit_stem(x_stem))
        axs2.grid()
        fig.show()

data = []
temp = naive_bayes(verbose=True)
temp.get_results(start = 1000, stop = 18000, step = 500, avg_iters=4, stem=False)
temp.visualize()

