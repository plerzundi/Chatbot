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
    _line =line.split('+++$+++')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]