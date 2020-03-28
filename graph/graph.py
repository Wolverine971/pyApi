#https://py2neo.org/v4/
# from py2neo import Database, Graph, Node, Relationship
    
from nltk.tokenize import sent_tokenize
# import logging
# import json
# import nltk
# import nltk.corpus
# from nltk.util import bigrams, trigrams, ngrams
from elasticsearch import Elasticsearch
from allennlp.predictors.predictor import Predictor
import re

# g = Graph(host="http://localhost:7474")
# g = Graph(auth=('neo4j', 'poop'))

# default_db = Database()
# default_db

sop = []

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
rgx = "/ISTJ|ISFJ|INFJ|INTJ|ISTP|ISFP|INFP|INTP|ESTP|ESFP|ENFP|ENTP|ESTJ|ESFJ|ENFJ|ENTJ+(?=\'?s|S)*"

corefPredictor = Predictor.from_path("C:\\Users\\djway\\Desktop\\ML-notebook\\coref-model-2018.02.05")
openPredictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")

class Tripple:
  def __init__(self, sub, pred, obj):
    self.sub = sub
    self.pred = pred
    self.obj = obj

#https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/


def connect_elasticsearch():
    es = None
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if es.ping():
        print('Yay Connect')
    else:
        print('Awww it could not connect!')
    return es
# if __name__ == '__main__':
#   logging.basicConfig(level=logging.ERROR)

#es.get(index="my-index", doc_type="test-type", id=42)['_source']

def search(es_object, index_name, *search):
    res = es_object.search(index=index_name, body=search)

def ESget(indexx):
    res  = es.search(index=indexx)
    hits = res["hits"]["hits"]
    allHits = []
    for hit in hits:
        allHits.append(hit["_source"]['section'])
        
    return allHits

#"C:\\Users\\djway\\Desktop\\ML-notebook\\coref-model-2018.02.05"
#C:\Users\djway\Desktop\ML-notebook\bidaf\bidaf-elmo-model-2018.11.30-Ccharpad.tar
#"https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz"
##ENTJs are analytical and objective, and like bringing order to the world around them. When there are flaws in a system, the ENTJ sees them, and enjoys the process of discovering and implementing a better way. ENTJs are assertive and enjoy taking charge; they see their role as that of leader and manager, organizing people and processes to achieve their goals.",


def findCoReference(text):
    pred = corefPredictor.predict(
      document=text,
    )
    # print('clusters: ' + pred["clusters"])
    # print('document: ' + pred["document"])
    return pred

#MBTItype = ''
# find word in sentence
def getWord(clusters, doc):
    MBTItype = ''
    try:
        for cluster in clusters:
            for word in cluster:
                #print(word)
                wordGroup = ''

                for w in word:
                    # wordGroup = wordGroup + ' ' + doc[w]
                    wordGroup = ' '.join([wordGroup, doc[w]])

                found = re.findall(rgx, wordGroup, re.IGNORECASE)
                #print(pred["document"][word[0]])
                if found != None and len(found) > 0:
                    #print(found)
                    MBTItype = found[0].lower()
                    #print(MBTItype)
                    print(MBTItype)
                    break

            return MBTItype
    except:
        print("********errrrrorrrr*********")


    #if(word[0] != word[1]):
        
    #print(pred["document"][word[0]])
    #if pred["document"][word]


def getWordTest():
    tDoc = [' ', 'ENTJs', 'are', 'strategic', 'leaders', ',', 'motivated', 'to','organize', 'change', '.', 'They', 'are', 
            'quick', 'to', 'see', 'inefficiency', 'and', 'conceptualize', 'new', 'solutions',',', 'and', 'enjoy', 'developing',
            'long', '-', 'range', 'plans', 'to', 'accomplish', 'their', 'vision', '.', 'They', 'excel','at', 'logical', 
            'reasoning', 'and', 'are', 'usually', 'articulate', 'and', 'quick', '-', 'witted', '.', 'ENTJs', 'are', 'analytical',
            'and', 'objective', ',', 'and', 'like', 'bringing', 'order', 'to', 'the', 'world', 'around', 'them', '.', 'When', 'there', 
            'are', 'flaws', 'in', 'a', 'system', ',', 'the', 'ENTJ', 'sees', 'them', ',', 'and', 'enjoys', 'the', 'process', 'of', 
            'discovering', 'and', 'implementing','a', 'better', 'way', '.', 'ENTJs', 'are', 'assertive', 'and', 'enjoy', 'taking',
            'charge', ';', 'they', 'see', 'their', 'role', 'as', 'that', 'of', 'leader', 'and', 'manager', ',', 'organizing', 'people',
            'and', 'processes', 'to', 'achieve', 'their','goals', '.']
    tClusters = [[[1, 1],
  [11, 11],
  [31, 31],
  [34, 34],
  [48, 48],
  [62, 62],
  [72, 73],
  [75, 75],
  [89, 89],
  [97, 97],
  [99, 99],
  [114, 114]]]
    print(getWord(tClusters, tDoc))
#getWordTest()

# replace all groups with the MBTItype
def replaceWords(word, doc, clusters):
    #print(doc)
    #print(word)
    #print(doc)
    replacedDoc = doc
    count = 0
    #print("************needs to return array of docs******************")
    #print(clusters)
    index = 0
    for swap in clusters[0]:
        try:
            #print(swap)
            #print(swap[count])

            ########here######
            for w in swap:





            if swap[0] != swap[1]:
               # print(swap[count])
                #print("swap")
                #print(swap)
                index = swap[0]
                index = index - count
                if(replacedDoc[index] == None):
                    print("errror")
                    print(replacedDoc)
                    print('doc len')
                    print(len(replacedDoc))
                    print("index")
                    print(index)
                replacedDoc[index] = word
                for i in range(swap[1] - swap[0]):
                    iToPop = index + 1
                    count += 1
                    #print("iToPop " + str(iToPop))
                    replacedDoc.pop(iToPop)
                   # print(replacedDoc)

            swapper = swap[0] - count
            #print('swapper' + replacedDoc[swapper])
            #print(swapper)
            replacedDoc[swapper] = word
        except:
            print("errror3")
            print(clusters)
            print(word)
            print(replacedDoc)
            print('doc len')
            print(len(replacedDoc))
            print("index")
            print(index)
            
    #print(replacedDoc)
    return replacedDoc


def testReplaceWords():
    tdoc = [' ', 'ENFP', 'is', 'a', 'moderately', 'common', 'personality', 'type', ',', 'and', 'is', 'the', 'fifth', 'most', 'common', 'among', 'women', '.', 'ENFPs', 'make', 'up', ':', '8', '%', 'of', 'the', 'general', 'population', '10', '%', 'of', 'women', '6', '%', 'of', 'men', 'Famous', 'ENFPs', 'Famous', 'ENFPs', 'include', 'Bill', 'Clinton', ',', 'Phil', 'Donahue', ',', 'Mark', 'Twain', ',', 'Edith', 'Wharton', ',', 'Will', 'Rogers', ',', 'Carol', 'Burnett', ',', 'Dr.', 'Seuss', ',', 'Robin', 'Williams', ',', 'Drew', 'Barrymore', ',', 'Julie', 'Andrews', ',', 'Alicia', 'Silverstone', ',', 'Joan', 'Baez', ',', 'and', 'Regis', 'Philbin', '.']
    tword = "enfp"
    tclusters = [[[50, 51], [65, 66], [71, 72], [74, 75], [78, 79]]]
    print(replaceWords(tword, tdoc, tclusters))  
#testReplaceWords()


def cleanVerbs(desc):
    cleanedObjects = []
    cleanedSubject = ''
    subjects = re.findall('\[ARG0(.*?)\]', desc)
    #print(subjects)
    objects = re.findall('\[ARG[1-9](.*?)\]', desc)
    for section in subjects:
        cleanedSub = section.split(":")[1]
        isMBTI = re.findall(rgx, cleanedSub, re.IGNORECASE)
        if isMBTI != None:
            cleanedSubject = re.sub(r']', '', cleanedSub).strip()
            break
            
    for section in objects:
        cleanedobj = section.split(":")[1] 
        cleanedObjects.append(re.sub(r']', '', cleanedobj).strip())
        
    return {'subject': cleanedSubject, 'objects': cleanedObjects}

#https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz
#"C:\\Users\\djway\\Desktop\\ML-notebook\\openie-model.2018-08-20"

def getPreds(text):
    predList = []
    for sent in text:
        pred = openPredictor.predict(sentence=sent)
        predList.append(pred)
    return predList


def predTest():
    texx = ['  entj are strategic leaders , motivated to organize change .', 'entj are quick to see inefficiency and conceptualize new solutions , and enjoy developing long - range plans to accomplish entj vision .', 'entj excel at logical reasoning and are usually articulate and quick - witted .', 'entj are analytical and objective , and like bringing order to the world around entj .', 'When there are flaws in a system entj entj sees entj , and enjoys the process of discovering and implementing a better way .', 'entj are assertive and enjoy taking charge ; entj see entj role as that of leader and manager , organizing people and processes to achieve entj goals .']
    print(getPreds(texx))
#predTest()

def createEntity(entity):
    for obj in entity["objects"]:


        # entity = {
        #     'subject': entity["subject"],
        #     'predicate': entity["verb"],
        #     'object': obj
        # }

        sop.append(Tripple(entity["subject"], entity["verb"], obj))
        # sop.append(entity)



        # mbti = Node("MBTI", name=entity["subject"])
        # gObj = Node("Object", name=obj)
        
        # #rel = Relationship(mbti, entity["verb"], gObj)
        # rel = Relationship.type(entity["verb"])
        # g.merge(rel(mbti, gObj), "MBTI", "name")
        
        
        
        # #a = Node("Person", name="Alice", age=33)
        # #b = Node("Person", name="Bob", age=44)
        # #KNOWS = Relationship.type("KNOWS")
        # #g.merge(KNOWS(a, b), "Person", "name")
        
        
        
        #rVerb = Relationship.type(entity["verb"])
        #a = Node("Person", name="Alice", age=33)
        #b = Node("Person", name="Bob", age=44)
        
        #gVerb = Relationship.type(entity["verb"])
        #g.merge(gVerb(mbti, gObj), "MBTI", "name")
        
        
        
        
        #mbti[entity["verb"]] = obj
        #rel = Relationship(mbti, entity["verb"], gObj)
        #g.merge(mbti,"MBTI","name") #node,label,pirmary key
        #g.merge(gObj,"Object","name") 
        #g.merge(rel)
        #g.commit()
        #g.merge(rVerb(a, b), "MBTI", "name")


def storePreds(preds):
    for sent in preds:
        #print(sent)
        for verb in sent["verbs"]:
            #print(verb)
            gVerb = verb["verb"]
            words = re.findall('\[(.*?)\]', verb["description"])
            cleanedObj = cleanVerbs(verb["description"])
            #print("cleanedOBJ -----------------")
            #print(cleanObj)
            cleanedObj["verb"] = gVerb
            #print(cleanObj)
            createEntity(cleanedObj)
            
            #for obj in cleanedObj["objects"]:
             #   def createEntity()
              #  a = Node("MBTI", name=cleanedObj[subject])
               # b = Node("Object", name=obj)
                #rVerb = Relationship.type(gVerb)
                #g.merge(rVerb(a, b), "MBTI", "name")
            

def runPipeLine(pred):
    #print(pred)
    doc = pred["document"].copy()
    clusters = pred["clusters"].copy()
    #print(doc)
    if(clusters != '' and len(clusters) > 0 and doc != None ):
        word = getWord(clusters, doc)
        if word != '':
            cleanDoc = replaceWords(word, doc, clusters)
            #print(cleanDoc)

            s = " ".join(filter(None, cleanDoc))
            sent_text = sent_tokenize(s) 
            #print("*************************")
            #print(sent_text)
            #print("-----------------------")
            listofPreds = getPreds(sent_text)
            #print(listofPreds[0])
            storePreds(listofPreds)


def createGraph(index):
    results = ESget(index)
    #print(entj[0])
    #references = findCoReference(results[1])
    #runPipeLine(references)
    for hit in results:
        references = findCoReference(hit)
        runPipeLine(references)
    print('done')


#https://py2neo.org/v4/data.html#node-and-relationship-objects
#tx = graph.begin()

createGraph("enfp")