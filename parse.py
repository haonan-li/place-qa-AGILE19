import os
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser

os.environ['STANFORD_PARSER'] = './stanford-parser/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = './stanford-parser/stanford-parser-3.9.1-models.jar'
parser = StanfordDependencyParser(model_path='./stanford-parser/englishPCFG.ser.gz')

fin = open('data/result/raw_query','r')
fout = open('data/result/parse_result','w')
i=0
for line in fin.readlines():
    print (i)
    i+=1
    sen, = parser.parse(line.split())
    fout.write(sen.to_conll(10))
    fout.write('\t')

