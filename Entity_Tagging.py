import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

filesTnr = range(0, 301)
filesSnr = range(301, 485)
filesT = [open('training/{}.txt'.format(i), 'r') for i in filesTnr]

# rgxs
# split header and body
hb_rgx = re.compile(r'([\s\S]+(?:\b.+\b:.+\n\n|\bAbstract\b:))([\s\S]*)')
# find paragraphs
par_rgx = re.compile(r'(?<=\n\n)(?:(?:\s*\b.+\b:(?:.|\s)+?)|(\s{0,4}[A-Za-z0-9](?:.|\n)+?\s*))(?=\n\n)')
# find times
tim_rgx = r'\b([0-9]{1,2}(?::[0-9]{2}\s?(?:AM|PM|am|pm|a\.m|p\.m)|:[0-9]{2}|\s?(?:AM|PM|am|pm|a\.m|p\.m)))\b'
# find stimes
stim_rgx = r'(?:\bTime\b:\s*){0}(?:\s*(?:-|until)?\s*){0}'.format(tim_rgx)
# find etimes
etim_rgx = r'(?:\bTime\b:\s*){0}'.format(tim_rgx)
# find locations
loc_rgx = re.compile(r'(?:\b(?:Place|Location|Where)\b:\s*)(.*)', re.IGNORECASE)
# find speakers
spkr_rgx = re.compile(r'(?:\b(?:Speaker|Name|Who)\b:\s*)(.*?,|.*)')
# pos regex
posT_rgx = re.compile('((?:\w+?///NNP\s{0,3}){1,5})')


def tag(seminar):
    header, body = hb_rgx.search(seminar).groups()
    header = header.strip()
    body = '\n\n' + body.strip() + '\n\n'
    #Tried doing pos tagging and wrote a posT rgx ^
    tokenized_body = word_tokenize(body)
    pos_taggs = nltk.pos_tag(tokenized_body)
    pos_tagged_body = ''
    for pos_tag in pos_taggs:
        pos_tagged_body += '{}///{} '.format(pos_tag[0], pos_tag[1])

    # paragraphs and sentences tagging
    paragraphs = [match.group(1) for match in par_rgx.finditer(body)]
    sentences = []
    for par in paragraphs:
        if par:
            body = body.replace(par, '<paragraph>{}</paragraph>'.format(par))
            sentences += sent_tokenize(par)

    for sent in sentences:
        body = body.replace(sent, '<sentence>{}</sentence>'.format(sent))

    seminar = header + body
    # time tagging
    stime = None
    etime = None
    setime_search = re.search(stim_rgx, header)
    stime_search = re.search(etim_rgx, header)
    if setime_search:
        stime, etime = setime_search.groups()
    elif stime_search:
        stime = stime_search.group(1)

    if stime:
        seminar = seminar.replace(stime, '<stime>{}</stime>'.format(stime))

    if etime:
        seminar = seminar.replace(etime, '<etime>{}</etime>'.format(etime))
    # location tagging

    location_search = re.search(loc_rgx, header)
    if location_search:
        location = location_search.group(1)
        seminar = seminar.replace(location, '<location>{}</location>'.format(location))

    # speakers tagging
    speaker_search = re.search(spkr_rgx, header)
    if speaker_search:
        speaker = speaker_search.group(1)
        seminar = seminar.replace(speaker, '<speaker>{}</speaker>'.format(speaker))

    return seminar


def purgeTxt(badTxt):
    # removes tags and not alphanumerical chars.
    badTxt = re.sub(r"<.+?>", "", badTxt)
    return re.sub(r'\W', '', badTxt)


def getTxt(strr, tag):
    regex = re.compile(r'<{0}>[ \t]*((?:.|\s)+?)[ \t]*</{0}>'.format(tag), re.M)
    tags_content = regex.findall(strr)
    return {purgeTxt(content) for content in tags_content}

tags = ['stime', 'etime', 'paragraph', 'sentence', 'location', 'speaker']
fSem = [open('untagged/{}.txt'.format(i), 'r') for i in filesSnr]
seminars = [file.read().strip() for file in fSem]
tagged_seminars = [tag(sem) for sem in seminars]

tagSet = set()
for t in tags:
    for sem in tagged_seminars:
        tagSet |= getTxt(sem, t)

test_files = [open('test/{}.txt'.format(i)) for i in filesSnr]
test_seminars = [file.read().strip() for file in test_files]

tagTestSet = set()
for t in tags:
    for sem in test_seminars:
        tagTestSet |= getTxt(sem, t)


tp = 0
fp = 0
fn = 0

for t in tagSet:
    if t in tagTestSet:
        tp += 1
    else:
        fp += 1

for t in tagTestSet:
    if t not in tagSet:
        fn += 1
        
precision = tp / (tp + fp)
recall = tp / (tp + fn)

f_measure = 0
if precision + recall is not 0:
    f_measure = (precision * recall) / (precision + recall)

print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('f_measure: ' + str(f_measure))