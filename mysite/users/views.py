from django.shortcuts import render,redirect
from django.http import HttpResponse
from .models import users,WebcamPictures,temp,demoposts,posts,notifications,newscomments,likes,supportnews
from  controllers.models import news
import spacy

from spacy.lang.en.stop_words import  STOP_WORDS
from string import  punctuation
from googletrans import Translator
import googletrans

from heapq import nlargest

import cv2
import base64
from django.core.files.base import ContentFile
import face_recognition
import sys
import shutil
import os
import socket
import wmi

import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords


from nltk.stem.porter import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier
import re

from django.core.files.uploadedfile import InMemoryUploadedFile
import io
'''
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

import numpy as np
from numpy import load
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.models import load_model

from scipy.spatial.distance import cosine
'''
# Create your views here.
# ------------------------------get embeddingd--------------
'''
detector=MTCNN()
def extract_face(image,resize=(224,224)):
  image=cv2.imread(image)
  faces=detector.detect_faces(image)
  x1,y1,width,height=faces[0]['box']
  x2,y2=x1+width,y1+height

  face_boundary=image[y1:y2,x1:x2]
  face_image=cv2.resize(face_boundary,resize)
  return face_image
def get_embeddings(faces):
  face=np.asarray(faces,'float32')
  face=preprocess_input(face,version=2)
  model=VGGFace(model='vgg16',include_top=False,input_shape=(224,224,3),pooling='avg')
  return model.predict(face)
def get_similarity(faces):
  embeddings=get_embeddings(faces)
  score=cosine(embeddings[0],embeddings[1])
  return score
'''
# --------------------------end embeddings-------------------------------
from PIL import Image
import io

def decodeDesignImage(data):
    try:
        data = base64.b64decode(data.encode('UTF-8'))
        buf = io.BytesIO(data)
        img = Image.open(buf)
        return img
    except:
        return None

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


def getsummary(request):
    text=request.POST['description']
    # length=request.POST['l']
    # ts = textsummary(text,length)
    ts = textsummary(text)

    return HttpResponse(ts)

def trans(text,langu):
    translator = Translator()
    translation = translator.translate(text, dest=langu)
    return translation

def langconverter(request):
    text=request.POST['a']
    lang=request.POST['l']
    ts = trans(text,lang)
    return HttpResponse(ts.text)
def login(request):
    return render(request, 'users/login.html')

def faceauthentication(request):
    name = request.POST['name']
    email = request.POST['email']
    password = request.POST['password']
    profile = request.FILES['profile']
    # prof_img=str(profile.read())

    submit_details = temp(username=name, email=email, password=password, photo=profile)
    submit_details.save()
    users_data = submit_details.id
    print("id", users_data)
    request.session['temp_id'] = users_data
    return render(request, 'users/faceverification.html',{"temp_data":users_data})


def analyse_user(request):
    print("Anlyxing face")
    temp_data = request.session['temp_id']
    #temp_data=22
    temp_users=temp.objects.get(id=temp_data)

    profile_image=temp_users.photo.url
    print(profile_image)
    fp="C:/Users/HP/PycharmProjects/Bulletin/mysite/users"+profile_image
    baseimg = face_recognition.load_image_file(fp)
    baseimg=cv2.cvtColor(baseimg,cv2.COLOR_BGR2RGB)
    try:
        myface=face_recognition.face_locations(baseimg)[0]
        encodemyface=face_recognition.face_encodings(baseimg)[0]
    except IndexError as e:
        return render(request, 'users/login.html', {"message": "Please Upload Straight and Clear Image"})


    cv2.rectangle(baseimg,(myface[3],myface[0]),(myface[1],myface[2]),(255,0,255),2)

    webcampicture=temp_users.webcampic.url
    print('webcampicture',webcampicture)
    wp = "C:/Users/HP/PycharmProjects/Bulletin/mysite/users" + webcampicture

    sampleimg = face_recognition.load_image_file(wp)
    sampleimg = cv2.cvtColor(sampleimg, cv2.COLOR_BGR2RGB)


    try:
        samplefacetest = face_recognition.face_locations(sampleimg)[0]
        encodesamplefacetest = face_recognition.face_encodings(sampleimg)[0]
    except IndexError as e:
        return render(request, 'users/faceverification.html', {"message": "No Face Detected Please Recapture the photo"})
        print("index error . Authrntication Faled")
        sys.exit()
    result=face_recognition.compare_faces([encodemyface],encodesamplefacetest)
    print(result)
    output=str(result)
    if output=="[True]":
        username=temp_users.username
        email=temp_users.email
        password=temp_users.password
        temp_image=temp_users.photo
        photo=temp_image
        # file_name=str(temp_image).split('/')[-1]
        # print(file_name)
        # # data = ContentFile(temp_image, name=file_name)
        # # print("data",data)
        print("imageurl",photo,temp_image)
        submit_users=users(username=username,email=email,password=password,photo=photo)
        submit_users.save()

        return render(request, 'users/ProfileVerification.html', {"users_data": temp_users})
        print("user authenticate")
    else:

        return render(request, 'users/ProfileVerificationUnsucess.html', {"users_data": temp_users})
        print("authentication failed")


def index(request):
    sports=news.objects.filter(news_type='1').order_by('news_date')[::-1]
    business= news.objects.filter(news_type='3')
    technology = news.objects.filter(news_type='2')
    education= news.objects.filter(news_type='4')
    entertainment = news.objects.filter(news_type='5')
    mainpage=news.objects.filter(news_type='2').order_by('news_date')[0:1]
    flash_news=news.objects.filter(news_type='1').order_by('news_date')[4:6]
    flash_news_2 = news.objects.filter(news_type='4')[1:3]
    socialproblems=posts.objects.all().order_by('created_date')[::-1]
    if(request.session.get('id')):
        uid=request.session['id']
        users_details=users.objects.get(id=uid)
        notifications_details=notifications.objects.filter(userid=uid,status=0)
        return render(request, 'users/index.html',
                      {"mainpage": mainpage, "flash_news": flash_news, "flash_news_2": flash_news_2, "sports": sports,
                       "business": business, "technology": technology, "education": education,
                       "entertainment": entertainment,"socialproblems":socialproblems,"users_details":users_details,"notifications":notifications_details})

    return render(request, 'users/index.html',
                  {"mainpage":mainpage,"flash_news":flash_news,"flash_news_2":flash_news_2,"sports":sports,"business":business,"technology":technology,"socialproblems":socialproblems,"education":education,"entertainment":entertainment})

def faceverification(request):
    return render(request, 'users/faceverification.html')
def file_put_contents(filename, text):
    file1 = open(filename, "w")
    file1.write(text)
    file1.close()
def getfacename(request):
    img_data = request.POST['imageprev']
    temp_data= request.session['temp_id']
    # image = data.replace("data:image/png;base64,", "")
    format, imgstr = img_data.split(';base64,')
    ext = format.split('/')[-1]
    data = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)
    # submit_details = WebcamPictures(email="pavankumar@gmail.com",photo=data)
    # submit_details.save()
    temp_table=temp.objects.get(id=temp_data)
    temp_table.webcampic=data
    temp_table.save()

    return HttpResponse("1")


def loginauth(request):
    mail=request.POST['a']
    passw=request.POST['b']
    users_table=users.objects.all()
    for i in users_table:
        if(i.email==mail and i.password==passw):
            request.session['id']=i.id
            return  HttpResponse(1)
    return HttpResponse(0)

def sportsNews(request):

    # news.objects.filter(news_type='1').exclude(id=sport_id).order_by('news_date')[::-1]

    sports_count = news.objects.filter(news_type='1').order_by('news_date').count()
    main_sports = news.objects.filter(news_type='1').order_by('news_date')[sports_count-1:sports_count]
    pid = main_sports[0].id
    side_news = news.objects.filter(news_type='1').exclude(id=pid).order_by('created_date')[::-1]

    # side_news=news.objects.filter(news_type='1').order_by('news_date')[::-1]
    lang_list = googletrans.LANGUAGES
    #comments = newscomments.objects.filter(news_id=pid).order_by('created_date')[::-1]
    comments = users.objects.raw(
        'select u.id,n.id as commentid, n.comment,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [pid])
    news_likes = likes.objects.filter(news_id=pid)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == pid and l.userid == int(uid)):
                liked = 1
        return render(request, 'users/sports.html',
                      {"main_sports": main_sports, "side_news": side_news, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details, "comments": comments,"main_news_id":pid,"news_likes":len(news_likes)+1,"liked":liked})
    return render(request, 'users/sports.html',
                  {"main_sports":main_sports,"side_news":side_news,"lang_list":lang_list,"comments":comments,"main_news_id":pid,"news_likes":len(news_likes)+1})


def sports(request,sport_id):
    id=sport_id
    main_news=news.objects.filter(id=sport_id)
    flash_news=news.objects.filter(news_type='1' ).exclude(id=sport_id).order_by('news_date')[::-1]
    lang_list=googletrans.LANGUAGES
    comments = users.objects.raw(
        'select u.id, n.comment,n.id as commentid,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [sport_id])
    news_likes = likes.objects.filter(news_id=sport_id)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == sport_id  and l.userid == int(uid)):
                liked=1
        return render(request, 'users/sports.html',
                      {"main_news": main_news, "flash_news": flash_news, "sport_id": sport_id, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details,
                       "comments": comments,"main_news_id":sport_id,"news_likes":len(news_likes)+1,"liked":liked})

    return render(request, 'users/sports.html',
                  {"main_news":main_news,"flash_news":flash_news,"sport_id":sport_id,"lang_list":lang_list,"comments":comments,"main_news_id":sport_id,"news_likes":len(news_likes)+1})


def technologyNews(request):

    # news.objects.filter(news_type='1').exclude(id=sport_id).order_by('news_date')[::-1]

    sports_count = news.objects.filter(news_type='2').order_by('news_date').count()
    main_sports = news.objects.filter(news_type='2').order_by('news_date')[sports_count-1:sports_count]
    pid = main_sports[0].id
    side_news = news.objects.filter(news_type='2').exclude(id=pid).order_by('created_date')[::-1]

    # side_news=news.objects.filter(news_type='1').order_by('news_date')[::-1]
    lang_list = googletrans.LANGUAGES
    #comments = newscomments.objects.filter(news_id=pid).order_by('created_date')[::-1]
    comments = users.objects.raw(
        'select u.id,n.id as commentid, n.comment,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [pid])
    news_likes = likes.objects.filter(news_id=pid)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == pid and l.userid == int(uid)):
                liked = 1
        return render(request, 'users/technology.html',
                      {"main_sports": main_sports, "side_news": side_news, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details, "comments": comments,"main_news_id":pid,"news_likes":len(news_likes)+1,"liked":liked})
    return render(request, 'users/technology.html',
                  {"main_sports":main_sports,"side_news":side_news,"lang_list":lang_list,"comments":comments,"main_news_id":pid,"news_likes":len(news_likes)+1})



def technology(request,technology_id):
    id=technology_id
    main_news=news.objects.filter(id=technology_id)
    flash_news=news.objects.filter(news_type='2' ).exclude(id=technology_id).order_by('news_date')[::-1]
    lang_list=googletrans.LANGUAGES
    comments = users.objects.raw(
        'select u.id, n.comment,n.id as commentid,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [technology_id])
    news_likes = likes.objects.filter(news_id=technology_id)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == technology_id  and l.userid == int(uid)):
                liked=1
        return render(request, 'users/technology.html',
                      {"main_news": main_news, "flash_news": flash_news, "sport_id": technology_id, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details,
                       "comments": comments,"main_news_id":technology_id,"news_likes":len(news_likes)+1,"liked":liked})

    return render(request, 'users/technology.html',
                  {"main_news":main_news,"flash_news":flash_news,"sport_id":technology_id,"lang_list":lang_list,"comments":comments,"main_news_id":sport_id,"news_likes":len(news_likes)+1})


def businessNews(request):

    # news.objects.filter(news_type='1').exclude(id=sport_id).order_by('news_date')[::-1]

    sports_count = news.objects.filter(news_type='3').order_by('news_date').count()
    main_sports = news.objects.filter(news_type='3').order_by('news_date')[sports_count-1:sports_count]
    pid = main_sports[0].id
    side_news = news.objects.filter(news_type='3').exclude(id=pid).order_by('created_date')[::-1]

    # side_news=news.objects.filter(news_type='1').order_by('news_date')[::-1]
    lang_list = googletrans.LANGUAGES
    #comments = newscomments.objects.filter(news_id=pid).order_by('created_date')[::-1]
    comments = users.objects.raw(
        'select u.id,n.id as commentid, n.comment,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [pid])
    news_likes = likes.objects.filter(news_id=pid)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == pid and l.userid == int(uid)):
                liked = 1
        return render(request, 'users/business.html',
                      {"main_sports": main_sports, "side_news": side_news, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details, "comments": comments,"main_news_id":pid,"news_likes":len(news_likes)+1,"liked":liked})
    return render(request, 'users/business.html',
                  {"main_sports":main_sports,"side_news":side_news,"lang_list":lang_list,"comments":comments,"main_news_id":pid,"news_likes":len(news_likes)+1})


def business(request,business_id):
    id=business_id
    main_news=news.objects.filter(id=business_id)
    flash_news=news.objects.filter(news_type='3').exclude(id=business_id).order_by('news_date')[::-1]
    lang_list=googletrans.LANGUAGES
    comments = users.objects.raw(
        'select u.id, n.comment,n.id as commentid,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [business_id])
    news_likes = likes.objects.filter(news_id=business_id)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == business_id  and l.userid == int(uid)):
                liked=1
        return render(request, 'users/business.html',
                      {"main_news": main_news, "flash_news": flash_news, "sport_id": business_id, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details,
                       "comments": comments,"main_news_id":business_id,"news_likes":len(news_likes)+1,"liked":liked})

    return render(request, 'users/business.html',
                  {"main_news":main_news,"flash_news":flash_news,"sport_id":business_id,"lang_list":lang_list,"comments":comments,"main_news_id":business_id,"news_likes":len(news_likes)+1})


def educationNews(request):

    # news.objects.filter(news_type='1').exclude(id=sport_id).order_by('news_date')[::-1]

    sports_count = news.objects.filter(news_type='4').order_by('news_date').count()
    main_sports = news.objects.filter(news_type='4').order_by('news_date')[sports_count-1:sports_count]
    pid = main_sports[0].id
    side_news = news.objects.filter(news_type='4').exclude(id=pid).order_by('created_date')[::-1]

    # side_news=news.objects.filter(news_type='1').order_by('news_date')[::-1]
    lang_list = googletrans.LANGUAGES
    #comments = newscomments.objects.filter(news_id=pid).order_by('created_date')[::-1]
    comments = users.objects.raw(
        'select u.id,n.id as commentid, n.comment,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [pid])
    news_likes = likes.objects.filter(news_id=pid)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == pid and l.userid == int(uid)):
                liked = 1
        return render(request, 'users/education.html',
                      {"main_sports": main_sports, "side_news": side_news, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details, "comments": comments,"main_news_id":pid,"news_likes":len(news_likes)+1,"liked":liked})
    return render(request, 'users/education.html',
                  {"main_sports":main_sports,"side_news":side_news,"lang_list":lang_list,"comments":comments,"main_news_id":pid,"news_likes":len(news_likes)+1})


def education(request,education_id):
    id=education_id
    main_news=news.objects.filter(id=education_id)
    flash_news=news.objects.filter(news_type='4' ).exclude(id=education_id).order_by('news_date')[::-1]
    lang_list=googletrans.LANGUAGES
    comments = users.objects.raw(
        'select u.id, n.comment,n.id as commentid,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [education_id])
    news_likes = likes.objects.filter(news_id=education_id)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == education_id  and l.userid == int(uid)):
                liked=1
        return render(request, 'users/education.html',
                      {"main_news": main_news, "flash_news": flash_news, "sport_id": education_id, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details,
                       "comments": comments,"main_news_id":education_id,"news_likes":len(news_likes)+1,"liked":liked})

    return render(request, 'users/education.html',
                  {"main_news":main_news,"flash_news":flash_news,"sport_id":education_id,"lang_list":lang_list,"comments":comments,"main_news_id":education_id,"news_likes":len(news_likes)+1})


def entertainmentNews(request):

    # news.objects.filter(news_type='1').exclude(id=sport_id).order_by('news_date')[::-1]

    sports_count = news.objects.filter(news_type='5').order_by('news_date').count()
    main_sports = news.objects.filter(news_type='5').order_by('news_date')[sports_count-1:sports_count]
    pid = main_sports[0].id
    side_news = news.objects.filter(news_type='5').exclude(id=pid).order_by('created_date')[::-1]

    # side_news=news.objects.filter(news_type='1').order_by('news_date')[::-1]
    lang_list = googletrans.LANGUAGES
    #comments = newscomments.objects.filter(news_id=pid).order_by('created_date')[::-1]
    comments = users.objects.raw(
        'select u.id,n.id as commentid, n.comment,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [pid])
    news_likes = likes.objects.filter(news_id=pid)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == pid and l.userid == int(uid)):
                liked = 1
        return render(request, 'users/entertainment.html',
                      {"main_sports": main_sports, "side_news": side_news, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details, "comments": comments,"main_news_id":pid,"news_likes":len(news_likes)+1,"liked":liked})
    return render(request, 'users/entertainment.html',
                  {"main_sports":main_sports,"side_news":side_news,"lang_list":lang_list,"comments":comments,"main_news_id":pid,"news_likes":len(news_likes)+1})


def entertainment(request,entertainment_id):
    id=entertainment_id
    main_news=news.objects.filter(id=entertainment_id)
    flash_news=news.objects.filter(news_type='5' ).exclude(id=entertainment_id).order_by('news_date')[::-1]
    lang_list=googletrans.LANGUAGES
    comments = users.objects.raw(
        'select u.id, n.comment,n.id as commentid,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [entertainment_id])
    news_likes = likes.objects.filter(news_id=entertainment_id)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == entertainment_id  and l.userid == int(uid)):
                liked=1
        return render(request, 'users/entertainment.html',
                      {"main_news": main_news, "flash_news": flash_news, "sport_id": entertainment_id, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details,
                       "comments": comments,"main_news_id":entertainment_id,"news_likes":len(news_likes)+1,"liked":liked})

    return render(request, 'users/education.html',
                  {"main_news":main_news,"flash_news":flash_news,"sport_id":entertainment_id,"lang_list":lang_list,"comments":comments,"main_news_id":entertainment_id,"news_likes":len(news_likes)+1})


def socialproblemsNews(request):
    sp_count = posts.objects.all().order_by('created_date').count()
    main_sports = posts.objects.all().order_by('created_date')[sp_count-1:sp_count]
    pid=main_sports[0].id
    # news.objects.filter(news_type='1').exclude(id=sport_id).order_by('news_date')[::-1]
    side_news = posts.objects.all().exclude(id=pid).order_by('created_date')[::-1]
    lang_list = googletrans.LANGUAGES

    comments = users.objects.raw(
        'select u.id, n.comment,n.id as commentid,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [pid])
    news_likes = likes.objects.filter(news_id=pid)
    news_support=supportnews.objects.filter(news_id=pid)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == pid and l.userid == int(uid)):
                liked = 1
        supported = 0
        for l in news_support:
            if (l.news_id == pid and l.userid == int(uid)):
                supported = 1
        return render(request, 'users/socialproblems.html',
                      {"main_sports": main_sports, "side_news": side_news, "lang_list": lang_list,
                       "users_details": users_details,
                       "notifications": notifications_details,"main_news_id":pid, "comments": comments,"news_likes":len(news_likes)+1,"liked":liked,"news_support":len(news_support)+1,"supported":supported})
    return render(request, 'users/socialproblems.html',
                  {"main_sports": main_sports, "side_news": side_news, "lang_list": lang_list, "main_news_id":pid,"comments": comments,"news_likes":len(news_likes)+1,"news_support":len(news_support)+1})


def socialproblems(request,sp_id):
    sp_id=sp_id
    posts.objects.all().order_by('created_date')
    main_news=posts.objects.filter(id=sp_id)
    flash_news=posts.objects.all().exclude(id=sp_id).order_by('created_date')[::-1]
    lang_list=googletrans.LANGUAGES
    comments = comments = users.objects.raw(
        'select u.id, n.comment,n.id as commentid,u.username,n.created_date from users_newscomments as n  inner join users_users as u on u.id=n.userid and n.news_id=%s order by n.created_date desc',
        [sp_id])
    news_likes = likes.objects.filter(news_id=sp_id)
    news_support = supportnews.objects.filter(news_id=sp_id)
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)
        notifications_details = notifications.objects.filter(userid=uid, status=0)
        liked = 0
        for l in news_likes:
            if (l.news_id == sp_id and l.userid == int(uid)):
                liked = 1
        supported = 0
        for l in news_support:
            if (l.news_id == sp_id and l.userid == int(uid)):
                supported = 1
        return render(request, 'users/socialproblems.html',
                      {"main_news": main_news, "flash_news": flash_news, "sport_id": sp_id, "lang_list": lang_list,"users_details": users_details,
                       "notifications": notifications_details,"main_news_id":sp_id,
                       "comments": comments,"news_likes":len(news_likes)+1,"liked":liked,"news_support":len(news_support)+1,"supported":supported})

    return render(request, 'users/socialproblems.html',
                  {"main_news":main_news,"flash_news":flash_news,"sport_id":sp_id,"lang_list":lang_list,"main_news_id":sp_id,"comments":comments,"news_likes":len(news_likes)+1,"news_support":len(news_support)+1})

def checksessionid(request):
    if (request.session.get('id')):
        uid = request.session['id']
        return HttpResponse(1)
    return  HttpResponse(0)


def navbar(request):
    return render(request, 'users/navbar.html',{"mainpage":1,"flash_news":[1,2,3,4],"sports":[1,2,3,4,5],"business":[1,2,4]})



def addpost(request):
    if request.session.get('id'):
        return render(request, 'users/addpost.html')
    else:
        return redirect('login')

def confirmfaceverification(request):
    if request.session.get('id'):
        userid = request.session['id']
        title = request.POST['title']
        location = request.POST['location']
        issuedate = request.POST['issuedate']
        image = request.FILES['profileimage']
        description=request.POST['description']
        submit_details = demoposts(userid=userid, title=title, location=location, photo=image,news_date=issuedate,description=description)
        submit_details.save()
        posts_id = submit_details.id
        request.session['posts_id']=str(posts_id)
        print("confirm",posts_id)

        return render(request, 'users/confirm_faceverification.html')


def confirmProfile(request):
    img_data = request.POST['imageprev']
    userid=request.session['id']

    # posts_id="a4be2afd-5c74-4e0b-af54-522b34d738de"
    posts_id=str(request.session['posts_id'])
    print('cp',posts_id)
    # image = data.replace("data:image/png;base64,", "")
    format, imgstr = img_data.split(';base64,')
    ext = format.split('/')[-1]
    data = ContentFile(base64.b64decode(imgstr), name='demoposts.' + ext)
    demoposts_table = demoposts.objects.get(id=posts_id)
    demoposts_table.webcampic = data
    demoposts_table.save()
    return HttpResponse("1")

def analyse_user_posts(request):
    print("Anlyxing face")
    userid = request.session['id']
    posts_id = request.session['posts_id']
    demoposts_table = demoposts.objects.get(id=posts_id)
    users_table=users.objects.get(id=userid)

    profile_image=users_table.photo.url
    print(profile_image)
    fp="C:/Users/HP/PycharmProjects/Bulletin/mysite/users"+profile_image
    baseimg = face_recognition.load_image_file(fp)
    baseimg=cv2.cvtColor(baseimg,cv2.COLOR_BGR2RGB)
    try:
        myface=face_recognition.face_locations(baseimg)[0]
        encodemyface=face_recognition.face_encodings(baseimg)[0]
    except IndexError as e:
        return render(request, 'users/login.html', {"message": "Please Upload Straight and Clear Image"})


    cv2.rectangle(baseimg,(myface[3],myface[0]),(myface[1],myface[2]),(255,0,255),2)

    webcampicture=demoposts_table.webcampic.url
    print('webcampicture',webcampicture)
    wp = "C:/Users/HP/PycharmProjects/Bulletin/mysite/users" + webcampicture

    sampleimg = face_recognition.load_image_file(wp)
    sampleimg = cv2.cvtColor(sampleimg, cv2.COLOR_BGR2RGB)


    try:
        samplefacetest = face_recognition.face_locations(sampleimg)[0]
        encodesamplefacetest = face_recognition.face_encodings(sampleimg)[0]
    except IndexError as e:
        return render(request, 'users/confirm_faceverification.html', {"message": "No Face Detected Please Recapture the photo"})
        print("index error . Authrntication Faled")
        sys.exit()
    result=face_recognition.compare_faces([encodemyface],encodesamplefacetest)
    print(result)
    output=str(result)
    if output=="[True]":
        title = demoposts_table.title
        location = demoposts_table.location
        issuedate = demoposts_table.news_date
        temp_image = demoposts_table.photo
        photo = temp_image
        description = demoposts_table.description
        submit_details = posts(userid=userid, title=title, location=location, photo=photo ,news_date=issuedate,
                                   description=description)
        submit_details.save()

        # file_name=str(temp_image).split('/')[-1]
        # print(file_name)
        # # data = ContentFile(temp_image, name=file_name)
        # # print("data",data)

        return render(request, 'users/ProfileVerification.html', {"posts_data": demoposts_table,"users_data":users_table,"message":["go to social problems"]})
        print("user authenticate")
    else:
        webcam_image = demoposts_table.webcampic
        submit_notifications=notifications(userid=userid,webcamimage=webcam_image)
        submit_notifications.save()

        return render(request, 'users/ProfileVerificationUnsucess.html', {"posts_data": demoposts_table,"users_data":users_table,"message":["wrong person"]})
        print("authentication failed")



def profileVerification(request):


    name = request.POST['name']
    email = request.POST['email']
    password = request.POST['password']
    profile = request.FILES['profile']
    #prof_img=str(profile.read())

    submit_details = users(username=name, email=email, password=password, photo=profile)
    submit_details.save()
    users_data=users.objects.get(email=email)
    prof_img=users_data.photo.url
    print(prof_img)
    capture = cv2.VideoCapture(0)

    while (True):
        ret, frame = capture.read(0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('static/cam/cam_image.png', frame)
            capture.release()
            cv2.destroyAllWindows()

            break
    if(name=="pavan"):
        faces=[]
        # faces=[extract_face('C:/Users/HP/OneDrive/Pictures/MainProject/users/pavankumar.jpg'),extract_face('C:/Users/HP/OneDrive/Pictures/MainProject/users/pavank.jpg')]

    else:
        faces=[]
        # faces=[extract_face('C:/Users/HP/OneDrive/Pictures/MainProject/users/pavankumar.jpg'),extract_face('C:/Users/HP/OneDrive/Pictures/MainProject/users/raviteja.jpg')]




    # x = get_similarity(faces)
    x=0.1
    if x<=0.2:
        return render(request, 'users/ProfileVerification.html',{"users_data":users_data})
    return render(request, 'users/ProfileVerificationUnsucess.html', {"users_data": users_data})

def ProfileVerificationUnsucess(request):
    return render(request, 'users/ProfileVerificationUnsucess.html')

def logout(request):
    if (request.session.get('id')):
        del request.session['id']
        return redirect('login')
def user_notifications(request):
    if (request.session.get('id')):
        uid = request.session['id']
        users_details = users.objects.get(id=uid)

        notifications_details = notifications.objects.filter(userid=uid).order_by('created_date')[::-1]


        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        for i in notifications_details:
            i.status=1
            i.save()
        return render(request,'users/notifications.html',{"notifiers":notifications_details,"users_details": users_details,"hostname":hostname,"ipaddr":IPAddr})
    else:
        return HttpResponse("no notifications")
def toxiccomment(request):
    return render(request,'users/toxiccomment.html')

def checktoxiccomment(request):
    if (request.session.get('id')):
        uid=request.session['id']
        comment=request.POST['c']
        cnd=request.POST['cnd']
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        corpus3 = []
        a = set(stopwords.words('english'))

        review = re.sub('[^a-zA-Z]', ' ', comment)
        review = review.lower()
        corpus3.append(review)

        corpus4 = []
        for ele in corpus3:
            l = list(ele.split())
            review = [ps.stem(word) for word in l if not word in a]
            corpus4.append(' '.join(review))

        # ================================

        loaded_vectorizer = pickle.load(open('C:/Users/HP/PycharmProjects/Bulletin/vectorizer.pickle', 'rb'))

        # load the model
        loaded_model = pickle.load(open('C:/Users/HP/PycharmProjects/Bulletin/classification.model', 'rb'))

        # make a prediction
        # print(loaded_model.predict(loaded_vectorizer.transform(['bitch please'])))

        res = loaded_model.predict(loaded_vectorizer.transform(corpus4))
        s = []
        cmt=comment.replace("n't"," not")

        comment = comment.replace("n't", " not")
        c_split = comment.split(" ")
        if "not" in c_split or "no" in c_split:
            return HttpResponse("Text contain Negative Information.")
        if 1 in res[0]:
            if res[0][0] == 1:
                s.append("Toxic")
            if res[0][1] == 1:
                s.append("Severe_Toxic")
            if res[0][2] == 1:
                s.append("onscene")
            if res[0][3] == 1:
                s.append("Threat")
            if res[0][4] == 1:
                s.append("Insult")
            if res[0][5] == 1:
                s.append("Identity Hate")
            error_str=", ".join(s)
            err_str="Text  contains "+ error_str+" information"
            return  HttpResponse(err_str)

        else:
            addcomment=newscomments(news_id=cnd,comment=comment,userid=uid)
            addcomment.save()
            return HttpResponse("")
def clicklike(request):
    newsid=request.POST['newsid']
    userid=request.POST['userid']
    likes_table=likes.objects.all()
    for i in likes_table:
        if(i.news_id==newsid and i.userid==int(userid)):
            s=likes.objects.filter(news_id=newsid,userid=userid)
            s.delete()
            return HttpResponse(0)
    s = likes(userid=userid, news_id=newsid)
    s.save()
    return HttpResponse(1)
def deletecomment(request):
    newsid = request.POST['newsid']
    userid = request.POST['userid']
    commentid=int(request.POST['comment_id'])
    s=newscomments.objects.filter(id=commentid,news_id=newsid,userid=userid)
    s.delete()
    return HttpResponse(1)

def supportpost(request):
    newsid=request.POST['newsid']
    userid=request.POST['userid']
    support_table=supportnews.objects.all()
    for i in support_table:
        if(i.news_id==newsid and i.userid==int(userid)):
            s=supportnews.objects.filter(news_id=newsid,userid=userid)
            s.delete()
            return HttpResponse(0)
    s = supportnews(userid=userid, news_id=newsid)
    s.save()
    return HttpResponse(1)







