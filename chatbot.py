# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 05:04:09 2018

@author: plerzundi
"""

#Importación de librerias
import numpy as np
import tensorflow as tf
import re
import time


#######  PROCESAMIENTO DE DATOS #######
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')


## crear un diccionario para cada mapeo de las ids ##
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
## crear una lista de todas las conversiones de id ##
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

## hacer la primera limpieza en el texto ##

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"\wont't","will not",text)
    text = re.sub(r"\cant't","cannot",text)
    text = re.sub(r"[-()\"#/@:;<>{}+=~|.?,]","",text)
    return text

## limpuendo las preguntas ##
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

## limpiando las respuestas ##
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

## crear un diccionario ##
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1
        
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1

tokens = ['<PAD>', '<EOS>','<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

## diccionario inverso ##
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}