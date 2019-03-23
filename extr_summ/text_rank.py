#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
import numpy as np
import math

class TextRankSummarizer:

    epsilon = 1e-4
    damping = 0.85

    def __init__(self, n_sentences):
        self.n_sentences = n_sentences
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lmtzr = RussianStemmer()
        self.stop_words = stopwords.words('russian')

    def __call__(self, text):
        sentences = sent_tokenize(text)

        rating = self.rate_sentences(sentences)
        return self._extract_best_sentences(sentences, rating)

    def rate_sentences(self, sentences):
        matrix = self._create_matrix(sentences)
        ranks = self._power_method(matrix)
        return {sent: rank for sent, rank in zip(sentences, ranks)}

    def _create_matrix(self, sentences):
        sent_words = [set(self.lmtzr.stem(word) for word in self.tokenizer.tokenize(sentence.lower())
                           if word not in self.stop_words) for sentence in sentences]

        weights = np.zeros((len(sentences), len(sentences)))

        for i, wi, in enumerate(sent_words):
            for j, wj in enumerate(sent_words):
                weights[i, j] = self._rate_sent_edge(wi, wj)
        weights /= weights.sum(axis=1)[:, np.newaxis]

        return np.full((len(sentences), len(sentences)), (1.-self.damping) / len(sentences)) + self.damping * weights

    @staticmethod
    def _rate_sent_edge(words1, words2):
        rank = 0
        for w1 in words1:
            for w2 in words2:
                rank += int(w1 == w2)

        if rank == 0:
            return 0.0

        norm = math.log(len(words1)) + math.log(len(words2))
        if np.isclose(norm, 0.):
            return rank * 1.0

        else:
            return rank / norm

    def _power_method(self, matrix):
        t_matrix = matrix.T

        # len(matrix) = sentence count
        p_vec = np.array([1.0 / len(matrix)] * len(matrix))
        lambda_val = 1.0

        while lambda_val > self.epsilon:
            next_p = np.dot(t_matrix, p_vec)
            lambda_val = np.linalg.norm(np.subtract(next_p, p_vec))
            p_vec = next_p

        return p_vec

    def _extract_best_sentences(self, sentences, rating):
        sorted_by_rating =  sorted([(sent, rating) for sent, rating in rating.items()],
                                   key=lambda x: x[1], reverse=True)[:self.n_sentences]

        top_sentences = [x[0] for x in sorted_by_rating]
        ordered_top_sentences = [s for s in sentences if s in top_sentences]
        return '\n\n'.join(ordered_top_sentences)
