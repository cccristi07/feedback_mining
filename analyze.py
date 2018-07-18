from collections import defaultdict
import pandas as pd
import numpy as np
from numpy import nan
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import RomanianStemmer
from nltk import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap, LocallyLinearEmbedding
from sklearn.cluster import MiniBatchKMeans, DBSCAN

import csv
import spacy
import gensim


csv_path = 'opinion_data.csv'

df = pd.read_csv(csv_path)
df = df.replace(np.nan, '', regex=True)
no_studs = len(df)
specs = ["B1", "B2", "B3"]

rom_sw = set(stopwords.words("romanian"))

rom_sw.add("și")

print("{} studenti au participat la formular".format(no_studs))

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
    cls_words = []
    for c in cls_data:
        if c is "":
            continue
        tokens = tokenizer.tokenize(c)
        tokens = " ".join(list(filter(lambda x: x not in rom_sw, tokens)))
        cls_words.append(tokens)

    cls_words = [w for w in cls_words if w not in rom_sw]
    cv = CountVectorizer(min_df=1, ngram_range=(1, 3))
    scor = cv.fit_transform(cls_words).toarray()
    scor = scor.sum(axis=0) # obtinem cele mai importante coloane
    scor = list(zip(scor, cv.get_feature_names()))
    scor = sorted(scor, key=lambda x: x[0], reverse=True)
    classes[spec] = [cls for (_, cls) in scor[:3]]
    print("La specializarea {} materiile cu probleme au fost {}".format(spec, classes[spec]))

# abuse for each direction
abuse = {}
for spec in specs:
    abuse_data = list(df[df.Dir == spec].Abuse)
    abuse_data = [ad for ad in abuse_data if 'nu' not in ad.lower()]
    abuse[spec] = len(abuse_data)
    proc = abuse[spec] / studs_spec[spec] * 100
    print("{0:.2f}% dintre studentii de la {1} raporteaza abuzuri din partea profesorilor".format(proc, spec))

# bad profs for each direction
profs = {}
for spec in specs:
    prof_data = list(df[df.Dir == spec].Prof)
    tokenizer = RegexpTokenizer(r'\w+')
    prof_words = []
    for p in prof_data:
        if p is "":
            continue
        tokens = tokenizer.tokenize(p)
        tokens = " ".join(list(filter(lambda x: x not in rom_sw, tokens)))
        prof_words.append(tokens)

    cv = CountVectorizer(min_df=1, ngram_range=(1, 3))
    scor = cv.fit_transform(prof_words).toarray()
    scor = scor.sum(axis=0) # obtinem cele mai importante coloane
    scor = list(zip(scor, cv.get_feature_names()))
    scor = sorted(scor, key=lambda x: x[0], reverse=True)
    profs[spec] = [prof for _, prof in scor[:5]]
    print("Studentii de la {} reclama probleme de la {}".format(spec,
                                                                profs[spec]))

# what workload did the directions consider
work_load = {}
for spec in specs:
    wl_data = list(df[df.Dir == spec].Difficulty)
    tokenizer = RegexpTokenizer(r'\w+')
    wl_words = []
    for w in wl_data:
        if w is "":
            continue
        tokens = tokenizer.tokenize(w)
        tokens = " ".join(list(filter(lambda x: x not in rom_sw, tokens)))
        wl_words.append(tokens)

    cv = CountVectorizer(min_df=2, ngram_range=(1, 3))
    scor = cv.fit_transform(wl_words).toarray()
    scor = scor.sum(axis=0)
    scor = list(zip(scor, cv.get_feature_names()))
    scor = sorted(scor,key=lambda x: x[0], reverse=True)
    work_load[spec] = [wl for(_, wl) in scor[:5]]
    proc = scor[0][0] / studs_spec[spec] * 100
    print("pentru {0} încărcarea a fost {1} in {2:.2f}% din stud".format(spec, scor[0][1], proc))



# what were the top words / topics for each direction
top_words = {}
opinions = []
opinionV = []
spec_labels = []
for spec in specs:
    # preprocess data and turn into a wordcloud
    ops = list(df[df.Dir == spec].Opinion)
    ops = [op for op in ops if op is not ""]
    opinionV.extend(ops)
    label = int(spec[1])
    spec_labels.extend([label]*len(ops))
    txt = " ".join(ops)
    tokens = tokenizer.tokenize(txt)
    tokens = [t for t in tokens if t.lower() not in rom_sw]
    txt = " ".join(tokens)
    opinions.append(txt.lower())
    wc = WordCloud(width=1280, height=1024).generate(txt)
    top_words[spec] = txt
    plt.imshow(wc)
    plt.savefig("{}top_words.png".format(spec))



# global analysis for all directions
opinions = " ".join(opinions)
wc = WordCloud(width=1280, height=1024).generate(opinions)
plt.imshow(wc)
plt.savefig("all_specs.png")

plt.close()

vect = TfidfVectorizer()
X = vect.fit_transform(opinionV).toarray()

pc = PCA(n_components=2,)

X_ = pc.fit_transform(X)


for spec in specs:
    label = int(spec[1])
    Xl = [x for ind, x in enumerate(X_) if spec_labels[ind] == label]
    Xl = np.array(Xl)
    plt.scatter(Xl[:, 0], Xl[:, 1], label=spec)

plt.legend()
plt.show()
