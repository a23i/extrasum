#!/usr/bin/env python3
from flask import Flask, render_template
from flask import request
from extr_summ import text_rank
import controller
import nltk
app = Flask(__name__)

@app.route("/")
def main_form():
    return render_template("main.html")

@app.route("/faq")
def faq_form():
    return render_template("faq.html")

@app.route("/submit", methods=["POST"])
def submit():
    # Request text to summarize
    text = request.form.get("text")

    # Request summarization parameters
    n_sentences = int(request.form.get("sentences"))
    algorithm = request.form.get("algo")

    # Invoke extractive summarization controller
    summary = controller.extractive_summarization(text, n_sentences, algorithm)

    return render_template("main.html", text=text, summary=summary)

if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('punkt')
    app.run(host="0.0.0.0")
