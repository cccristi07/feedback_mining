from collections import defaultdict
import pandas as pd
import numpy as np
from numpy import nan
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import RomanianStemmer

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import csv
import spacy
import gensim


csv_path = 'opinion_data.csv'

df = pd.read_csv(csv_path)
no_studs = len(df)
specs = ["B1", "B2", "B3"]

rom_sw = set(stopwords.words("romanian"))

print("{} gave their opinions on this form")

# no of students from each directions
studs_spec = {}
for spec in specs:
    studs_spec[spec] = len(df[df.Dir == spec])
    print("{} studenti de la {} au participat".format(studs_spec[spec], spec))


# how many students from each dir worked during the sem
no_work_studs = {}
for spec, no_studs in studs_spec.items():
    no_work_studs[spec] = len(df[df.Dir == spec][df.Work == 'Da'])
    proc = no_work_studs[spec]/studs_spec[spec] * 100
    print("De la specializarea {0} au lucrat {1:.2f}%".format(spec, proc))


# attendance for each direction
attendance = {}
for spec in specs:
    att_data = list(filter(lambda x: x is not nan, df[df.Dir == spec].Attendance))
    att_data = [op for op in att_data if 'nu' not in op.lower() or 'dar' in op.lower() or 'str' in op.lower() or 'da' in op.lower()]
    attendance[spec] = att_data
    proc = len(att_data) / studs_spec[spec] * 100
    print("{0:.2f}% din studentii de la {1} mentioneaza prezente obligatorii".format(proc, spec))


# homework for each direction
homeworks = {}
for spec in specs:
    hw_data = list(filter(lambda x: x is not nan, df[df.Dir == spec].HW))
    hw_data = [op for op in hw_data if op.lower() == 'da']
    homeworks[spec] = hw_data
    proc = len(hw_data) / studs_spec[spec] * 100
    print("{0:.2f}% din studentii de la {1} se plang de teme".format(proc, spec))

# heavy load classes
classes = {}
for spec in specs:
    cls_data = list(df[df.Dir == spec].Class)
    tokenizer = RegexpTokenizer(r'\w+')
    cls_words = tokenizer.tokenize(" ".join(cls_data))
    cls_words = [w for w in cls_words if w not in rom_sw]
    cv = CountVectorizer(min_df=2)
    scor = cv.fit_transform(cls_words).toarray()
    scor = scor.sum(axis=0) # obtinem cele mai importante coloane
    scor = list(zip(scor, cv.get_feature_names()))
    scor = sorted(scor, key=lambda x: x[0], reverse=True)
    print(scor)

# abuse for each direction


# bad profs for each direction

# what workload did the directions consider

# what were the top words / topics for each direction


# global analysis for all directions
