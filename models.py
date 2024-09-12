# models.py

from sentiment_data import *
from utils import *
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import math
import random
import matplotlib.pyplot as plt

class FeatureExtractor(object):
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english')) # Load stopwords in the initializer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_vector = Counter()
        for word in sentence:
            word = word.lower()
            
            # If we're adding to the indexer, get the index of the word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else:
                idx = self.indexer.index_of(word)
                if idx == -1:
                    continue
            
            # Increment the count of the word in the feature vector
            feature_vector[idx] += 1
        
        return feature_vector


class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_vector = Counter()
        for i in range(len(sentence) - 1):
            bigram = sentence[i] + " " + sentence[i+1]
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                idx = self.indexer.index_of(bigram)
                if idx == -1:
                    continue
            feature_vector[idx] += 1
        return feature_vector


class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_vector = Counter()
        for word in sentence:
            word = word.lower()
            if word in self.stop_words:
                continue
            
            # If we're adding to the indexer, get the index of the word
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else:
                idx = self.indexer.index_of(word)
                if idx == -1:
                    continue
            
            # Increment the count of the word in the feature vector
            feature_vector[idx] += 1

        # Experiment: removing features with count 1 resulted in a lower accuracy (0.494)
        # feature_vector = Counter({k: v for k, v in feature_vector.items() if v > 1})

        return feature_vector


class SentimentClassifier(object):
    def predict(self, sentence: List[str]) -> int:
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    def __init__(self, feat_extractor: FeatureExtractor):
        self.weights = Counter()
        self.indexer = feat_extractor.get_indexer()
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.feat_extractor.extract_features(sentence)
        score = sum(self.weights[feature] * count for feature, count in feature_vector.items())
        return 1 if score > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, feat_extractor: FeatureExtractor):
        self.weights = Counter()
        self.indexer = feat_extractor.get_indexer()
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.feat_extractor.extract_features(sentence)
        score = sum(self.weights[feature] * count for feature, count in feature_vector.items())
        probability = 1 / (1 + math.exp(-score))
        return 1 if probability >= 0.5 else 0
    
    def update_weights(self, feature_vector, label: int):
        learning_rate = 0.1
        score = sum(self.weights[feature] * count for feature, count in feature_vector.items())
        probability = 1 / (1 + math.exp(-score))
        error = label - probability
        for feature, count in feature_vector.items():
            self.weights[feature] +=  learning_rate * error * count


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    perceptron = PerceptronClassifier(feat_extractor)

    random.seed(60)
    learning_rate = 0.1

    for _ in range(10):  # Number of epochs
        random.shuffle(train_exs)
        for example in train_exs:
            # Extract features and get the current score (weighted sum)
            feature_vector = feat_extractor.extract_features(example.words, add_to_indexer=True)
            score = sum(perceptron.weights[feature] * count for feature, count in feature_vector.items())
            
            prediction = 1 if score > 0 else 0

            # Q2: Experiment with step size. Dev Accuracy dropped to 0.524
            # learning_rate = 4 * learning_rate / 5
            
            # Update weights if prediction is wrong
            for feature, count in feature_vector.items():
                perceptron.weights[feature] += learning_rate*(example.label - prediction) * count

    # List the top 10 words with the most positive and negative weights
    # positive_weights = {k: v for k, v in perceptron.weights.items() if v > 0}
    # top_ten_positive = sorted(positive_weights.items(), key=lambda item: item[1], reverse=True)[:10]
    # top_ten_positive_words = [(perceptron.indexer.get_object(k), f"{v:.3f}") for k, v in top_ten_positive]
    # [print(f"{word}: {weight}") for word, weight in top_ten_positive_words]

    # negative_weights = {k: v for k, v in perceptron.weights.items() if v < 0}
    # top_ten_negative = sorted(negative_weights.items(), key=lambda item: item[1])[:10]
    # top_ten_negative_words = [(perceptron.indexer.get_object(k), f"{v:.3f}") for k, v in top_ten_negative]
    # [print(f"{word}: {weight}") for word, weight in top_ten_negative_words]


    
    return perceptron



def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    logistic_regression = LogisticRegressionClassifier(feat_extractor)
    weights = logistic_regression.weights

    random.seed(21)

    for _ in range(11):  # Number of epochs
        random.shuffle(train_exs)
        for example in train_exs:
            feature_vector = feat_extractor.extract_features(example.words, add_to_indexer=True) 
            logistic_regression.update_weights(feature_vector, example.label)
    return logistic_regression


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    # elif args.model == "LR_STEPSIZES":
    #     plot_lr_different_learning_rates(train_exs, dev_exs)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")

    return model