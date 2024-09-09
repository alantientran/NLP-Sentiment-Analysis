# models.py

from sentiment_data import *
from utils import *
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import math

class FeatureExtractor(object):
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_vector = Counter()
        for word in sentence:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
            else:
                idx = self.indexer.index_of(word)
                if idx == -1:
                    continue
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

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_vector = Counter()
        stop_words = set(stopwords.words('english'))
        for word in sentence:
            if word not in stop_words:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(word)
                else:
                    idx = self.indexer.index_of(word)
                    if idx == -1:
                        continue
                feature_vector[idx] += 1
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
        return 1 if score >= 0 else 0


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


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    perceptron = PerceptronClassifier(feat_extractor)
    weights = perceptron.weights

    for _ in range(10):  # Number of epochs
        for example in train_exs:
            feature_vector = feat_extractor.extract_features(example.words)
            score = sum(weights[feature] * count for feature, count in feature_vector.items())
            prediction = 1 if score >= 0 else 0
            if prediction != example.label:
                for feature, count in feature_vector.items():
                    weights[feature] += (example.label - prediction) * count
    return perceptron


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    logistic_regression = LogisticRegressionClassifier(feat_extractor)
    weights = logistic_regression.weights

    for _ in range(10):  # Number of epochs
        for example in train_exs:
            feature_vector = feat_extractor.extract_features(example.words)
            score = sum(weights[feature] * count for feature, count in feature_vector.items())
            probability = 1 / (1 + math.exp(-score))
            for feature, count in feature_vector.items():
                weights[feature] += (example.label - probability) * count
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
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")

    return model
