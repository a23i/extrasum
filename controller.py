#!/usr/bin/env python3
from extr_summ import text_rank, lex_rank
import meta

def extractive_summarization(text, n_sentences, algorithm):
    if text == "":
        return ""
    if algorithm not in meta.available_algorithms:
        raise NotImplemented

    else:
        if algorithm == "Text Rank":
            tr = text_rank.TextRankSummarizer(n_sentences)
            return tr(text)

        elif algorithm == "Lex Rank":
            lr = lex_rank.LexRankSummarizer(n_sentences)
            return lr(text)

if __name__ == "__main__":
    print("This module provides summarization function")
