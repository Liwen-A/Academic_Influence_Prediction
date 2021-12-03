import time 
import requests 
import pandas as pd 

def getCitations(paperID,num_year):
    query = 'https://api.semanticscholar.org/graph/v1/paper/{}?fields=year,citations.year'.format(paperID)
    paper = requests.get(query)
    if not 'citations' in paper.json():
        return None
    citations = paper.json()['citations']
    year = int(paper.json()['year'])
    citation_count = [0]*num_year
    for i in range(num_year):
        citation_year = year+i
        for p in citations:
            if p['year'] is not None and citation_year == p['year']:
                citation_count[i] += 1 
    return citation_count 

#get nobel price wining paper data 
data1 = pd.read_csv("Chemistry publication record.csv",encoding= "ISO-8859-1")
data2 = pd.read_csv("Physics publication record.csv",encoding= "ISO-8859-1")
data3 = pd.read_csv("Medicine publication record.csv",encoding= "ISO-8859-1")
data = pd.concat([data1,data2,data3])
data = data.sample(frac=1)
P = []
iter = 0
for q in data['Title']:
    iter += 1
    if iter % 99 == 0:
        print("iter {} finished, {} data collected".format(iter,len(P)))
        pd.DataFrame(P).to_csv("Nobeldata.csv",index = False)
        time.sleep(301)
    try:
        query = 'https://api.semanticscholar.org/graph/v1/paper/search?&query={}&fields=title,abstract,year,referenceCount,citationCount,influentialCitationCount&offest={}&limit=1'.format(q,0)
        response = requests.get(query)
        paper = response.json()['data'][0]
        paperId = paper['paperId']
        citation_by_year = getCitations(paperId,10)
        for i,c in enumerate(citation_by_year):
            paper['year{}_citation_count'.format(i)] = c
        P.append(paper)
    except:
        continue