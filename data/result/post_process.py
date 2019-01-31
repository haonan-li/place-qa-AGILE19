import json

fin = open('result.json','r')
data = json.load(fin)

foutq = open('query_noname.txt','w')
fouta = open('answer_noname.txt','w')

def replace_name(piece):
    sen = piece['sentence']
    for pn in piece['placeName']:
        sen = sen.replace(pn,'Name')
    return sen

for item in data:
    idd = item['identifier']
    query = item['queryAnalyze']
    sen = replace_name(query)
    foutq.write(str(idd)+'\t'+sen+'\n')
    for answer in item['answersAnalyze']:
        sen = replace_name(answer)
        fouta.write(str(idd)+'\t'+sen+'\n')
