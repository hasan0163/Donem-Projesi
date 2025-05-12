from django.shortcuts import render
from django.http import HttpResponse
from googletrans import Translator

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Turkish Sentiment Analysis
from transformers import AutoModelForSequenceClassification, AutoTokenizer as SentimentTokenizer
sentiment_model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sentiment_tokenizer = SentimentTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sentiment_pipe = pipeline("sentiment-analysis", tokenizer=sentiment_tokenizer, model=sentiment_model)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model ve tokenizer'ı yükle
summarizer_ckpt = "mrm8488/bert2bert_shared-turkish-summarization"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_ckpt)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_ckpt)

def generate_summary(text):
    # Giriş metnini tokenize et
    inputs = summarizer_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    # Özetleme işlemini gerçekleştir
    summary_ids = summarizer_model.generate(inputs, max_length=150, min_length=30, num_beams=4, early_stopping=True)
    # Özet metni decode et
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# New: BlenderBot Chatbot
chatbot_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-1B-distill")
chatbot_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")

translator = Translator()


def index(request):
    return render(request, "index.html", {"name": "Hasan"})


def counter(request):
    text = request.POST['text']
    number_of_words = len(text.split(" "))
    return render(request, "counter.html", {"text": text, "amount": number_of_words})


def metin_summary(request):
    return render(request, "metin_summary.html")


def generate_summary(text):
    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(**inputs, max_length=150, min_length=30, num_beams=4, early_stopping=True)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def metin_summary_sonuc(request):
    text = request.POST['metin']
    summary = generate_summary(text)
    return render(request, "metin_summary.html", {"text": summary})


def chatbot(request):
    return render(request, "chatbot.html")


def chatbot_sonuc(request):
    input_text = request.POST['metin']
    
    # Translate Turkish to English
    translated_en = translator.translate(input_text, src='tr', dest='en').text
    
    # Generate response in English
    inputs = chatbot_tokenizer([translated_en], return_tensors="pt", truncation=True)
    reply_ids = chatbot_model.generate(**inputs, max_length=100)
    response_en = chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
    # Translate back to Turkish
    response_tr = translator.translate(response_en, src='en', dest='tr').text

    return render(request, "chatbot.html", {"result": response_tr})


def sentiment(request):
    return render(request, "sentiment.html")


def sentiment_sonuc(request):
    text = request.POST['metin']
    result = sentiment_pipe(text)
    label = result[0]['label']
    score = round(result[0]['score'] * 100, 2)
    return render(request, "sentiment.html", {"result": label, "score": score})
