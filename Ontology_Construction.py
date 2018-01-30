import nltk
import gensim
from os import listdir
from os.path import isfile, join
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from nltk import word_tokenize

from nltk.corpus import wordnet as wn

training_files_range = range(0, 301)
seminar_files_range = range(301, 485)

mChem = ['analytical', 'physical', 'organic', 'inorganic',
         'biochemistry']
mCs = ['graphics', 'algorithms', 'robotics', 'databases', 'hardware']
mPhsc = ['mechanics', 'electronics', 'thermodynamics', 'nuclear',
         'optics']
mMath = ['combinatorics', 'logic', 'calculus', 'geometry', 'algebra']
mTopic = ['physics', 'mathematics', 'chemistry', 'computer science']

training_root = 'training/{}.txt'
untaggedRe = 'untagged/{}.txt'
untgL = []


def mapTpcs():
    mTopics = {}
    mTopics['chemistry'] = mChem
    mTopics['computer science'] = mCs
    mTopics['physics'] = mPhsc
    mTopics['mathematics'] = mMath
    mTopics['topic'] = mTopic

    return mTopics


def calcFrqnc(inputt):
    frqnc = {}

    for i in inputt:
        frqnc[i] = 0
    for i in inputt:
        frqnc[i] += 1
    frqncRate = []
    for f in frqnc:
        if frqnc[f] <= 2 and wn.synsets(f) != []:
            frqncRate = frqncRate + [f.lower()]
    return frqncRate


gglC = \
    Word2Vec.load_word2vec_format('/home/google-news-corpus/GoogleNews-vectors-negative300.bin'
                                  , binary=True)

def getLst(rgx, rangee):
    filesG = [open(rgx.format(i), 'r') for i in rangee]
    filesS = [file.read().strip() for file in filesG]
    return filesS


untgL = getLst(untaggedRe, seminar_files_range)


def cleanTbl(labelT):
    result = {}
    for t in labelT:
        if labelT[t] != []:
            result[t] = labelT[t]
    return result


def compareL(tokens, branch):
    freq = 0
    for t in tokens:
        try:
            freq = freq + gglC.compareL(t, branch)
        except:
            pass
    return freq


def runM():
    mapTbl = mapTpcs()
    nameTbl = mapTbl['topic']
    labelTable = {}
    topicSbjct = mTopic
    lst = mChem + mPhsc + mCs + mMath
    for tsb in topicSbjct:
        for br in lst:
            labelTable[tsb + ': ' + br] = []
    for lUn in untgL:
        cap = 0
        tag = ''
        wordTkns = word_tokenize(lUn)
        for nameSbj in nameTbl:
            sumx = 0
            for branch in mapTbl[nameSbj]:
                try:
                    sumx = sumx + compareL(wordTkns, branch)
                except ValueError:
                    sumx = sumx + 0
                if sumx > cap:
                    cap = sumx
                    tag = nameSbj + ': ' + branch
        if len(tag) > 0:
            labelTable[tag] = labelTable[tag] + [lUn]
    return cleanTbl(labelTable)