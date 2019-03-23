from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import math

class LexRankSummarizer:

    threshold = 0.1
    epsilon = 0.1

    def __init__(self, n_sentences):
        self.n_sentences = n_sentences
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lmtzr = RussianStemmer()
        self.stop_words = stopwords.words('russian')

    def __call__(self, text):
        sentences = sent_tokenize(text)
        sent_words = [set(self.lmtzr.stem(word) for word in self.tokenizer.tokenize(sentence.lower())
                           if word not in self.stop_words) for sentence in sentences]


        tf_vals = self._eval_tf(sent_words)
        idf_vals = self._eval_idf(sent_words)

        matrix = self._eval_matrix(sent_words, tf_vals, idf_vals)
        scores = self._power_method(matrix)
        rating = dict(zip(sentences, scores))

        return self._extract_best_sentences(rating, sentences)


    @staticmethod
    def _cosine_similarity(s1, s2, tf1, tf2, idfs):
        unique_w1 = set(s1)
        unique_w2 = set(s2)
        common_ws = unique_w1 & unique_w2

        numerator = 0.0
        for term in common_ws:
            numerator += tf1[term]*tf2[term] * idfs[term]**2

        denominator1 = sum((tf1[t]*idfs[t])**2 for t in unique_w1)
        denominator2 = sum((tf2[t]*idfs[t])**2 for t in unique_w2)

        if denominator1 > 0 and denominator2 > 0:
            return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        else:
            return 0.0

    @staticmethod
    def _eval_tf(sentences):
        tf_vals = map(Counter, sentences)
        tf_metrics = []

        for sentence in tf_vals:
            metrics = {}

            for term, tf in sentence.items():
                metrics[term] = tf / len(sentence)

            tf_metrics.append(metrics)

        return tf_metrics

    @staticmethod
    def _eval_idf(sentences):
        idf_metrics = {}

        for sentence in sentences:
            for term in sentence:
                if term not in idf_metrics:
                    n_j = sum(1 for s in sentence if term in s)
                    idf_metrics[term] = math.log(len(sentences) / (1 + n_j))

        return idf_metrics


    def _eval_matrix(self, sentences, tf_metrics, idf_metrics):
        matrix = np.zeros((len(sentences), len(sentences)))
        degrees = np.zeros((len(sentences), ))

        for i, (s1, tf1) in enumerate(zip(sentences, tf_metrics)):
            for j, (s2, tf2) in enumerate(zip(sentences, tf_metrics)):
                matrix[i][j] = self._cosine_similarity(s1, s2, tf1, tf2, idf_metrics)

                if matrix[i][j] > self.threshold:
                    matrix[i][j] = 1.0
                    degrees[i] += 1
                else:
                    matrix[i][j] = 0



        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if degrees[i] == 0:
                    degrees[i] == 1

                matrix[i][j] = matrix[i][j] / degrees[i]

        return matrix

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

    def _extract_best_sentences(self, rating, sentences):
        sorted_by_rating =  sorted([(sent, rating) for sent, rating in rating.items()],
                                       key=lambda x: x[1], reverse=True)[:self.n_sentences]

        top_sentences = [x[0] for x in sorted_by_rating]
        ordered_top_sentences = [s for s in sentences if s in top_sentences]
        return '\n\n'.join(ordered_top_sentences)
