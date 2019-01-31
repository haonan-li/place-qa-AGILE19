import re
import json
import sys, time
import sklearn
from tqdm import tqdm
from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

PRONOUN = dict({'where':'1','what':'2','which':'3','when':'4','how':'5','whom':'6','whose':'6','why':'7'})
SP_RELATION = set(['near','top','front','south','west','north','east','southwest','southeast','northwest','northeast','part','seat','center'])
POUNC = set(['.',',','\'','?','!','@','#','$','%','^','*','(',')','[',']',';','"',])
NOUN_TAG = set(['NN','NNS','NNP','NNPS'])
ADJ_TAG = set(['JJ','JJR','JJS'])
NO = set (['a','an','the','and','do','did','does','be','am','is','was','are','can','could','may','might','must','shall','should','will','would'])
VERB_TAG = set(['VB','VBD','VBG','VBN','VBP','VBZ'])
PREP_TAG = set(['IN','TO'])

# load gazetteer in dictionary
def load_gazetteer(fga):
    fga = open(fga,'r')
    gazetteer = dict()
    for line in fga.readlines():
        line = line.strip()
        gazetteer[line.lower()] = line
    fga.close()
    return gazetteer

# load word
def load_word(fword):
    words = set()
    fword = open(fword,'r')
    for line in fword.readlines():
        word = line.strip()
        words.add(word)
    fword.close()
    return words

# load place type
def load_pt(fpt):
    pt_set = set()
    pt_dict = dict()
    fpt = open(fpt,'r')
    for line in fpt.readlines():
        word = line.strip().split()
        pt_set.add(word[1])
        pt_dict[word[0]] = word[1]
    fpt.close()
    return pt_set,pt_dict

def load_data(fdata):
    fdata = open(fdata,'r')
    data = json.load(fdata)
    return data

def load_abbr(fabbr):
    abbr = dict()
    fabbr = open(fabbr,'r')
    for line in fabbr.readlines():
        line  = line.strip().split()
        abbr[line[-1]] = ' '.join(line[:-1])
    return abbr

# filter from dict 'ga' delete set 'topn'
def filterr(ga, topn):
    for word in topn:
        if word in ga:
            ga.pop(word)
    return ga

def output(data,foutput):
    with open(foutput,'w') as f:
        json.dump(data,f,indent=4)

class Token():
    def __init__(self,token):
        token = token.split('\t')
        self.tid = token[0]
        # expand abbreviation
        self.text = abbr[token[1]] if (token[1] in abbr) else token[1]
        self.pos = token[3]
        self.depid = int(token[6])
        self.dep = token[7]
        self.code = '.'

    def replace_text(self,text):
        self.text = text
        return self

    def replace_code(self,code):
        self.code = code
        return self

class Sentence():
    def __init__(self, sent):
        self.sent = list(map(lambda x: Token(x), sent))
        self.raw_sent = list(map(lambda x: x.text,self.sent))
        self.emb = elmo(batch_to_ids(self.raw_sent))['elmo_representations'][0].detach().numpy()
        self.len = len(self.sent)
        self.ptype = []
        self.code = self.encode()

    def get_code(self):
        return self.code

    def get_analysis(self):
        analysis = {'placeName':self.place_names,
                    'placeType':self.place_types,
                    'sentence':' '.join(self.raw_sent),
                    'code': self.code}
        return analysis


    def encode(self):
        self.detect_placename()
        self.normolize_placetype()
        self.encode_noun()
        self.encode_verb()
        self.encode_preposition()
        self.encode_quality()
        code = re.sub('[\.,-]','',(''.join([x.code for x in self.sent])))
        return code


    def detect_placename(self):
        self.place_names = []
        i = 0
        while i < self.len:
            for j in range(self.len,i,-1):
                phrase = ' '.join(self.raw_sent[i:j])
                if phrase in ga:
                    self.place_names.append(phrase)
                    self.sent[i:j] = [t.replace_text(tr) for t,tr in zip(self.sent[i:j],ga[phrase].split())]
                    self.sent[i].code = 'n'
                    self.sent[i+1:j] = [t.replace_code(tr) for t,tr in zip(self.sent[i+1:j],['-' for k in range(j-i-1)])]
                    i = j
                    break
            i += 1

    def normolize_placetype(self):
        self.place_types = []
        for i,token in enumerate(self.sent):
            # filter
            if token.text in NO:
                token.code = ','
            # normalize place type
            if token.text in pt_dict:
                token.text = pt_dict[token.text]
                self.place_types.append(token.text)

    def encode_noun(self):
        for i,token in enumerate(self.sent):
            if token.code == '.':
                # pronoun
                if (token.text in PRONOUN):
                    token.code = PRONOUN[token.text]
                # place type
                elif (token.text in pt_set):
                    token.code = 't'
                    self.ptype.append(token.text)
                # other object
                elif (token.pos in NOUN_TAG):
                    token.code = 'o'

    def encode_verb(self):
        for i,token in enumerate(self.sent):
            if token.code == '.' and token.pos in VERB_TAG:
                word_emb = self.emb[i]
                stav_similar = sklearn.metrics.pairwise.cosine_similarity(stav_emb.squeeze(),word_emb).max()
                actv_similar = sklearn.metrics.pairwise.cosine_similarity(actv_emb.squeeze(),word_emb).max()
                if stav_similar > max(actv_similar, 0.8):
                    token.code = 'a'
                elif actv_similar > max(stav_similar, 0.8):
                    token.code = 's'

    def encode_preposition(self):
        for i,token in enumerate(self.sent):
            if token.code == '.' and token.pos in PREP_TAG and token.dep in ['case','amod'] \
                and token.depid != 0 and self.sent[token.depid-1].code == 'n':
                token.code = 'r'

    def encode_quality(self):
        for i,token in enumerate(self.sent):
            # place quality
            if token.code == '.' and token.pos in ADJ_TAG and i+1<self.len and self.sent[i+1].code == 't':
                 token.code = 'q'


    def __str__(self):
        return ' '.join(self.raw_sent)


def main():

    fdata = 'data/raw_data/dataset-v2.1-location-queries.json'
    fga = 'data/gazetteer/gazetteer.txt'
    fabbr = 'data/gazetteer/abbr.txt'
    fcommon_word = 'data/common_word/top10000.txt'
    fpt = 'data/place_type/place_type.txt'
    factv = 'data/verb/action_verb.txt'
    fstav = 'data/verb/stative_verb.txt'
    foutput = 'data/result/result.json'
    fsp_prep = 'data/prep/prep.txt'

    # Load data
    pt_set,pt_dict = load_pt(fpt)
    actv = load_word(factv)
    stav = load_word(fstav)
    sp_prep = load_word(fsp_prep)
    data = load_data(fdata)
    abbr = load_abbr(fabbr)
    ga = load_gazetteer(fga)

    topn = load_word(fcommon_word)
    ga = filterr(ga,topn)

    # Verb Elmo representation
    actv_emb = elmo(batch_to_ids([[v] for v in actv]))['elmo_representations'][0].detach().numpy()
    stav_emb = elmo(batch_to_ids([[v] for v in stav]))['elmo_representations'][0].detach().numpy()


    with open('data/result/parse_result','r') as f:
        parse = f.read()
    parse = list(map(lambda x: x.split('\n'), parse.split('\n\n')))

    sents = []
    for x in tqdm(parse[:10]):
        sents.append(Sentence(x))

    result = [x.get_analysis() for x in sents]
    output(result,foutput)

main()
