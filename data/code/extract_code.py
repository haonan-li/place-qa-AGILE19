import json

fin = open('result.json','r')
data = json.load(fin)

foutq = open('query_code.txt','w')
fouta = open('answer_code.txt','w')
foutc = open('concat_code.txt','w')

for item in data:
    idd = item['identifier']
    code = item['queryAnalyze']['code']
    foutq.write(str(idd)+'\t'+code+'\n')
    for answer in item['answersAnalyze']:
        code1 = answer['code']
        fouta.write(str(idd)+'\t'+code1+'\n')
        concat_code = code+'-'+code1
        foutc.write(str(idd)+'\t'+concat_code+'\n')
