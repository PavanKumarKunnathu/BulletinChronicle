import spacy
from spacy.lang.en.stop_words import  STOP_WORDS
from string import  punctuation
from googletrans import Translator

translator = Translator()

punctuation=punctuation+"\n"
from heapq import nlargest

# def textsummary(text):
#     stopwords = list(STOP_WORDS)
#     nlp = spacy.load('en_core_web_sm')
#
#     # model intilization
#     doc = nlp(text)
#
#     tokens = [token.text for token in doc]
#
#     # step1:text cleaning: removing stop words and punctuations and count worf frequencies
#
#     word_frequenies = {}
#     for word in doc:
#         if word.text.lower() not in stopwords:
#             if word.text.lower() not in punctuation:
#                 if word.text not in word_frequenies.keys():
#                     word_frequenies[word.text] = 1
#                 else:
#                     word_frequenies[word.text] += 1
#     max_frequency = max(word_frequenies.values())
#
#     for word in word_frequenies.keys():
#         word_frequenies[word] = word_frequenies[word] / max_frequency
#
#     # sentence tokenisation
#     sentence_tokens = [sent for sent in doc.sents]
#
#     # To calculate sentence score
#     sentence_scores = {}
#     for sent in sentence_tokens:
#         for word in sent:
#             if word.text.lower() in word_frequenies.keys():
#                 if sent not in sentence_scores.keys():
#                     sentence_scores[sent] = word_frequenies[word.text.lower()]
#                 else:
#                     sentence_scores[sent] += word_frequenies[word.text.lower()]
#     # to get 30% of the score is maximizing score
#
#     select_length = int(len(sentence_tokens) * 0.3)
#
#     summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
#
#     final_summary = [word.text for word in summary]
#     return " ".join(final_summary)
text="Ratan Tata, in full Ratan Naval Tata, (born December 28, 1937, Bombay [now Mumbai], India), Indian businessman who became chairman (1991–2012 and 2016–17) of the Tata Group, a Mumbai-based conglomerate. A member of a prominent family of Indian industrialists and philanthropists (see Tata family), he was educated at Cornell University, Ithaca, New York, where he earned a B.S. (1962) in architecture before returning to work in India. He gained experience in a number of Tata Group businesses and was named director in charge (1971) of one of them, the National Radio and Electronics Co. He became chairman of Tata Industries a decade later and in 1991 succeeded his uncle, J.R.D. Tata, as chairman of the Tata Group. Upon assuming leadership of the conglomerate, Tata aggressively sought to expand it, and increasingly he focused on globalizing its businesses. In 2000 the group acquired London-based Tetley Tea for $431.3 million, and in 2004 it purchased the truck-manufacturing operations of South Korea’s Daewoo Motors for $102 million. In 2007 Tata Steel completed the biggest corporate takeover by an Indian company when it acquired the giant Anglo-Dutch steel manufacturer Corus Group for $11.3 billion. In 2008 Tata oversaw Tata Motors’ purchase of the elite British car brands Jaguar and Land Rover from the Ford Motor Company. The $2.3 billion deal marked the largest-ever acquisition by an Indian automotive firm. The following year the company launched the Tata Nano, a tiny rear-engined, pod-shaped vehicle with a starting price of approximately 100,000 Indian rupees, or about $2,000. Although only slightly more than 10 feet (3 metres) long and about 5 feet (1.5 metres) wide, the highly touted “People’s Car” could seat up to five adults and, in Tata’s words, would provide a “safe, affordable, all-weather form of transport” to millions of middle- and lower-income consumers both in India and abroad. In December 2012 Tata retired as chairman of the Tata Group. He briefly served as interim chairman beginning in October 2016 following the ouster of his successor, Cyrus Mistry. Tata returned to retirement in January 2017 when Natarajan Chandrasekaran was appointed chairman of the Tata Group."
# ts=textsummary(text)

# translation = translator.translate(ts, dest='hi')
# print(translation.text)

import socket
import os

import wmi

c = wmi.WMI()
my_system = c.Win32_ComputerSystem()[0]
print(my_system.Manufacturer)
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
print(socket.getfqdn(IPAddr))
print(hostname)
print(IPAddr)


