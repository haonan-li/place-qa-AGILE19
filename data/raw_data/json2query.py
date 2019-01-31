import json
f = open('dataset-v2.1-location-queries.json','r')
out_q = open('query.txt','w')
out_a = open('answer.txt','w')
data = json.load(f)
for piece in data:
    out_q.write(str(piece['identifier']) + '\t' + piece['query']+'\n')
    if 'answers' in piece:
        for sent in piece['answers']:
            sent = sent.replace('\n','')
            out_a.write(str(piece['identifier']) + '\t' + sent+'\n')
    else:
        out_a.write('.\n')
