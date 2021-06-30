from django.shortcuts import render,HttpResponse
import spacy

from spacy.lang.en.stop_words import  STOP_WORDS
from string import  punctuation
from googletrans import Translator

from heapq import nlargest

from .models import news


# Create your views here.
punctuation=punctuation+"\n"
def textsummary(text):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')

    # model intilization
    doc = nlp(text)

    tokens = [token.text for token in doc]

    # step1:text cleaning: removing stop words and punctuations and count worf frequencies

    word_frequenies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequenies.keys():
                    word_frequenies[word.text] = 1
                else:
                    word_frequenies[word.text] += 1
    max_frequency = max(word_frequenies.values())

    for word in word_frequenies.keys():
        word_frequenies[word] = word_frequenies[word] / max_frequency

    # sentence tokenisation
    sentence_tokens = [sent for sent in doc.sents]

    # To calculate sentence score
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequenies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequenies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequenies[word.text.lower()]
    # to get 30% of the score is maximizing score

    select_length = int(len(sentence_tokens) * 0.3)

    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    final_summary = [word.text for word in summary]
    return " ".join(final_summary)

def createnews(request):
    return render(request, 'controllers/createnews.html')
def addnews(request):
    if request.method == "POST":
        try:
            news_type = request.POST['news_type']
            title = request.POST['title']
            image = request.FILES['profileimage']
            date = request.POST['newsdate']
            location = request.POST['location']
            description = request.POST['description']
            summary = request.POST['summary']
            submit_details = news(news_type=news_type, title=title, image=image, news_date=date, location=location,
                                  description=description, summary=summary)
            submit_details.save()
            return render(request, 'controllers/createnews.html', {"success": True})
        except:
            return render(request, 'controllers/createnews.html', {"error": True})



def trans(text,langu):
    translator = Translator()
    translation = translator.translate(text, dest=langu)
    return translation

def getsummary(request):
    text=request.POST['description']
    ts = textsummary(text)

    return HttpResponse(ts)

