# TODO: subject must be index passed in
# TODO: run all sections through
# TODO: try to optimize- parallel processes??


# https://py2neo.org/v4/
# from py2neo import Database, Graph, Node, Relationship
# from pandas.io.json import json_normalize
import torch
import re
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import allennlp_models.syntax.srl
import os
import requests
import nltk.corpus
import nltk
from nltk.tokenize import sent_tokenize
import pprint
import pandas as pd
import multiprocessing as mp
import time
import asyncio

import nltk
nltk.download('punkt')
pp = pprint.PrettyPrinter(
    indent=2, width=80, depth=None, stream=None, compact=False)

# nltk.download('punkt')
# import multiprocessing as mp
# import logging
# import json
# from nltk.util import bigrams, trigrams, ngrams
# import allennlp


# g = Graph(host="http://localhost:7474")
# g = Graph(auth=('neo4j', 'poop'))

# default_db = Database()
# default_db

sop = []


from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': '192.168.1.251', 'port': 9200}])
# es = requests

rgx = "/([E|I]+[N|S][T|F]+[P|J])+(?=\'?s|S)"
torch.cuda.empty_cache()

# corefPredictor = Predictor.from_path(
#     "C:\\Users\\djway\\Desktop\\ML-notebook\\coref-model-2018.02.05")

# openPredictor = Predictor.from_path(
# "https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")

# windows
# corefPredictor = Predictor.from_path(
#     "C:\\Users\\djway\\Desktop\\ML-notebook\\coref-model-2018.02.05")

# dirname = os.path.dirname(__file__)
# corefPath = os.path.join(dirname, '..\coref-model-2018.02.05')


dirname = os.path.dirname(__file__)
# corefPath = os.path.join(dirname, '..\coref-spanbert-large-2020.02.27')

# corefPath = "C:\\Users\\djway\Desktop\\pyApi\\coref-spanbert-large-2020.02.27"

# corefPath = "C:\\Users\\djway\\Downloads\\coref-spanbert-large-2020.02.27.tar"
# corefPath = "C:\\Users\\djway\Desktop\\pyApi\\coref-model-2018.02.05\\config.json"


print("cuda available")
print(torch.cuda.is_available())
# linux
corefPredictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
# archive_path=corefPath, predictor_name='coref', cuda_device=0, dataset_reader_to_load="validation")
# corefPredictor = Predictor.from_path(corefPath)
# Predictor.from_path(
#     # "C:\\Users\\djway\\Desktop\\ML-notebook\\coref-model-2018.02.05")
#     "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
# "/mnt/c/Users/djway/Desktop/ML-notebook/coref-model-2018.02.05")
# corefPredictor._model = corefPredictor._model.cuda()

# coreArchive = load_archive("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz", cuda_device=0)
# corefPredictor = Predictor.from_archive(coreArchive)


# import allennlp_models.syntax.srl
# semanticRoleLabelPredict = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz", cuda_device=0)
# semanticRoleLabelPredict._model = semanticRoleLabelPredict._model.cuda()





# openPredictor = Predictor.from_path(
#     "https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
import allennlp_models.syntax.srl
openPredictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
# openPredictor._model = openPredictor._model.cuda()

# openArchive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz", cuda_device=0)
# openPredictor = Predictor.from_archive(openArchive)

# def corefPredictor(word):
#     req = "echo '{'passage': %}' | allennlp predict 'C:\\Users\\djway\\Desktop\\ML-notebook\\coref-model-2018.02.05 -" % word
#     stream = os.popen(req)
#     output = stream.read()
#     return output


# def openPredictor(word):
#     req = "echo '{'passage': %}' | allennlp predict 'https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz -" % word
#     stream = os.popen(req)
#     output = stream.read()
#     return output


class Tripple:
    def __init__(self, sub, pred, obj):
        self.sub = sub
        self.pred = pred
        self.obj = obj

# https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/


def connect_elasticsearch():
    es = None
    # es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
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
    res = es.search(index=indexx)
    # res = requests.get("http://localhost:9200/" + indexx + "/_search")
    hits = res["hits"]["hits"]
    # allHits = []
    # for hit in hits:
    #     if 'section' in hit["_source"]:
    #         # if key['section']:
    #         allHits.append(hit["_source"]['section'])

    # return allHits
    return hits

# "C:\\Users\\djway\\Desktop\\ML-notebook\\coref-model-2018.02.05"
# C:\Users\djway\Desktop\ML-notebook\bidaf\bidaf-elmo-model-2018.11.30-Ccharpad.tar
# "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz"
# ENTJs are analytical and objective, and like bringing order to the world around them. When there are flaws in a system, the ENTJ sees them, and enjoys the process of discovering and implementing a better way. ENTJs are assertive and enjoy taking charge; they see their role as that of leader and manager, organizing people and processes to achieve their goals.",


def findCoReference(text):
    # make api call
    # /coRef
    # pred = corefPredictor(text)
    try:
        pred = corefPredictor.predict(document=text)
        # pred = semanticRoleLabelPredict.predict(sentence=text)


        
        return pred

    except RuntimeError as e:
        print('error')
        print(e)
        torch.cuda.empty_cache()
        try:
            pred = corefPredictor.predict(document=text)
            return pred
        except RuntimeError as e:
            print('failed twice on corefPredictor')

#MBTItype = ''
# find word in sentence


def getWord(clusters, doc, index):
    MBTItype = ''
    indexRgx = re.compile(index, re.IGNORECASE)  # "/" + index + "+(?=\'?s|S)*"
    # indexRgx = "/" + index + "+(?=\'?s|S)*"
    try:
        for cluster in clusters:
            for word in cluster:
                # print(word)
                wordGroup = ''

                for w in word:
                    # wordGroup = wordGroup + ' ' + doc[w]
                    wordGroup = ' '.join([wordGroup, doc[w]])

                #found = re.findall(rgx, wordGroup, re.IGNORECASE)
                found = re.findall(indexRgx, wordGroup)
                # print(pred["document"][word[0]])
                if found != None and len(found) > 0:
                    # print(found)
                    MBTItype = found[0].lower()
                    # print(MBTItype)
                    print(MBTItype)
                    break

            return MBTItype
    except RuntimeError as e:
        print("********errrrrorrrr*********")
        print(e)
    # if(word[0] != word[1]):

    # print(pred["document"][word[0]])
    # if pred["document"][word]


def getWordTest():
    tDoc = [' ', 'ENTJs', 'are', 'strategic', 'leaders', ',', 'motivated', 'to', 'organize', 'change', '.', 'They', 'are',
            'quick', 'to', 'see', 'inefficiency', 'and', 'conceptualize', 'new', 'solutions', ',', 'and', 'enjoy', 'developing',
            'long', '-', 'range', 'plans', 'to', 'accomplish', 'their', 'vision', '.', 'They', 'excel', 'at', 'logical',
            'reasoning', 'and', 'are', 'usually', 'articulate', 'and', 'quick', '-', 'witted', '.', 'ENTJs', 'are', 'analytical',
            'and', 'objective', ',', 'and', 'like', 'bringing', 'order', 'to', 'the', 'world', 'around', 'them', '.', 'When', 'there',
            'are', 'flaws', 'in', 'a', 'system', ',', 'the', 'ENTJ', 'sees', 'them', ',', 'and', 'enjoys', 'the', 'process', 'of',
            'discovering', 'and', 'implementing', 'a', 'better', 'way', '.', 'ENTJs', 'are', 'assertive', 'and', 'enjoy', 'taking',
            'charge', ';', 'they', 'see', 'their', 'role', 'as', 'that', 'of', 'leader', 'and', 'manager', ',', 'organizing', 'people',
            'and', 'processes', 'to', 'achieve', 'their', 'goals', '.']
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
    print(getWord(tClusters, tDoc, 'entj'))
# getWordTest()

# replace all groups with the MBTItype


def replaceWords(word, doc, clusters):
    # print(doc)
    # print(word)
    # print(doc)
    replacedDoc = doc
    count = 0
    #print("************needs to return array of docs******************")
    # print(clusters)
    index = 0
    for swap in clusters[0]:
        try:
            # print(swap)
            # print(swap[count])

            ########here######
            for w in swap:
                if swap[0] != swap[1]:
                    # print(swap[count])
                    # print("swap")
                    # print(swap)
                    index = swap[0]
                    # index = index - count

                    if(replacedDoc[index] == None):
                        print("errror")
                        print(replacedDoc)
                        print('doc len')
                        print(len(replacedDoc))
                        print("index")
                        print(index)
                        replacedDoc[index] = word
                    for i in range(swap[1] - swap[0]):
                        iToPop = index + i
                        count += 1
                        #print("iToPop " + str(iToPop))
                        replacedDoc.pop(iToPop)
                        # print(replacedDoc)
            else:
                swapper = swap[0] - count
                #print('swapper' + replacedDoc[swapper])
                # print(swapper)
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

    # print(replacedDoc)
    return replacedDoc


def testReplaceWords():
    tdoc = [' ', 'ENFP', 'is', 'a', 'moderately', 'common', 'personality', 'type', ',', 'and', 'is', 'the', 'fifth', 'most', 'common', 'among', 'women', '.', 'ENFPs', 'make', 'up', ':', '8', '%', 'of', 'the', 'general', 'population', '10', '%', 'of', 'women', '6', '%', 'of', 'men', 'Famous', 'ENFPs', 'Famous', 'ENFPs', 'include',
            'Bill', 'Clinton', ',', 'Phil', 'Donahue', ',', 'Mark', 'Twain', ',', 'Edith', 'Wharton', ',', 'Will', 'Rogers', ',', 'Carol', 'Burnett', ',', 'Dr.', 'Seuss', ',', 'Robin', 'Williams', ',', 'Drew', 'Barrymore', ',', 'Julie', 'Andrews', ',', 'Alicia', 'Silverstone', ',', 'Joan', 'Baez', ',', 'and', 'Regis', 'Philbin', '.']
    tword = "enfp"
    tclusters = [[[50, 51], [65, 66], [71, 72], [74, 75], [78, 79]]]
    print(replaceWords(tword, tdoc, tclusters))
# testReplaceWords()


def cleanVerbs(verbObj, words, index):
    desc = verbObj["description"]
    tags = verbObj['tags']
    subjects = re.findall('\[ARG0(.*?)\]', desc)
    objects = re.findall('\[ARG[1-9](.*?)\]', desc)

    allSplitWords = words

    tempDesc = desc
    cleanedSubject = ''
    verbs = ''

    verbGroup = []
    verbStart = 0
    verbEnd = 0
    for i, pos in enumerate(tags):
        if pos == 'B-V':
            if verbStart == 0:
                verbStart = i
            else:
                verbEnd = i
            verbGroup.append('V')
        elif pos == 'B-BV':
            if verbStart == 0:
                verbStart = i
            else:
                verbEnd = i
            verbGroup.append('BV')

    if verbEnd == 0:
        verbEnd = verbStart
        verbEnd += 1

    sub = ' '.join(words[:verbStart])
    pred = ' '.join(words[verbEnd:])
    countVerb = words[verbStart:]
    verbGroup = []
    for j in range(verbEnd - verbStart):
        verbGroup.append(countVerb[j])

    verbs = ' '.join(verbGroup)

    if sub and pred and verbs:
        entity = {'subject': sub,
                  'objects': pred, 'verb': verbs}
        # createEntity(entity)
        df = pd.json_normalize(entity)
        df.to_csv(index + '.csv', mode='a', header=False)
        print('stored one doc')

    # cleanedVerbGroup = ''
    # for pos in verbGroup:
    #     posRgx = re.compile('\[' + pos + '(.*?)\]', re.IGNORECASE)

    #     verbs = re.findall(posRgx, tempDesc)
    #     if len(verbs) > 0:
    #         cleanedVerb = verbs[0].split(":")[1]
    #         cleanedVerb = re.sub(r']', '', cleanedVerb).strip()
    #         cleanedVerbGroup = ' '.join([cleanedVerbGroup, cleanedVerb])

    # verbs = cleanedVerbGroup.strip()

    # if subjects == None or len(subjects) == 0:
    #     return
    #     stoppingIndex = 0
    #     # get all words before verb to make that the subject
    #     for i, pos in enumerate(tags):
    #         if pos == 'B-V' or pos == 'B-BV':
    #             stoppingIndex = i
    #             break

    #     cleanedSubject = ' '.join(allSplitWords[:stoppingIndex])
    #     cleanedSubject = cleanedSubject.strip()
    # else:
    #     for subject in subjects:
    #         cleanedSub = subject.split(":")[1]
    #         isMBTI = re.findall(rgx, cleanedSub, re.IGNORECASE)
    #         if isMBTI != None:
    #             cleanedSubject = re.sub(r']', '', cleanedSub).strip()
    #             break

    # cleanedObjects = []
    # for obj in objects:
    #     splitSection = obj.split(":")[1]
    #     # cleanedObjects.append(re.sub(r']', '', splitSection).strip())
    #     cleanedObjects = re.sub(r']', '', splitSection).strip()

    # if cleanedSubject and cleanedObjects and verbs:
    #     entity = {'subject': cleanedSubject,
    #               'objects': cleanedObjects, 'verb': verbs}
    #     # createEntity(entity)
    #     df = pd.json_normalize(entity)
    #     df.to_csv('trip.csv', mode='a', header=False)

# https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz
# "C:\\Users\\djway\\Desktop\\ML-notebook\\openie-model.2018-08-20"


def infoExtract_fromArr(sentenceArr):
    # make api call
    # /preds
    predList = []
    for sent in sentenceArr:
        # pred = openPredictor(sent)
        try:
            pred = openPredictor.predict(sentence=sent)
            predList.append(pred)

        except RuntimeError as e:
            print('error')
            print(e)
            torch.cuda.empty_cache()
            try:
                pred = openPredictor.predict(sentence=sent)
                predList.append(pred)
            except RuntimeError as e:
                print('failed twice on openPredictor')

    return predList




async def infoExtract(sentence):
    try:
        preds = openPredictor.predict(sentence=sentence)

    except RuntimeError as e:
        print('error')
        print(e)
        torch.cuda.empty_cache()
        try:
            pred = openPredictor.predict(sentence=sentence)
        except RuntimeError as e:
            print('failed twice on openPredictor')

    return preds

def predTest():
    texx = ['  entj are strategic leaders , motivated to organize change .', 'entj are quick to see inefficiency and conceptualize new solutions , and enjoy developing long - range plans to accomplish entj vision .', 'entj excel at logical reasoning and are usually articulate and quick - witted .',
            'entj are analytical and objective , and like bringing order to the world around entj .', 'When there are flaws in a system entj entj sees entj , and enjoys the process of discovering and implementing a better way .', 'entj are assertive and enjoy taking charge ; entj see entj role as that of leader and manager , organizing people and processes to achieve entj goals .']
    print(infoExtract_fromArr(texx))
# predTest()


def createEntity(entity):
    for obj in entity["objects"]:

        # entity = {
        #     'subject': entity["subject"],
        #     'predicate': entity["verb"],
        #     'object': obj
        # }
        t = {
            "subject": entity["subject"],
            "verb": entity["verb"],
            "pred": obj
        }
        # t = Tripple(entity["subject"], entity["verb"], obj)
        sop.append(t)

        df = pd.json_normalize(t)
        df.to_csv('trip.csv', mode='a', header=False)
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
        # g.merge(mbti,"MBTI","name") #node,label,pirmary key
        # g.merge(gObj,"Object","name")
        # g.merge(rel)
        # g.commit()
        #g.merge(rVerb(a, b), "MBTI", "name")


def storePreds(preds, index):
    for sent in preds:
        if len(sent["verbs"]) > 0:
            cleanVerbs(sent["verbs"][0], sent["words"], index )
        # print(sent)
        # for verb in sent["verbs"]:
        #     # # print(verb)
        #     # # gVerb = verb["verb"]
        #     # # words = re.findall('\[(.*?)\]', verb["description"])
        #     # cleanedObj = cleanVerbs(verb)
        #     # #print("cleanedOBJ -----------------")
        #     # # print(cleanObj)
        #     # # cleanedObj["verb"] = gVerb
        #     # # print(cleanObj)
        #     # createEntity(cleanedObj)

        #     cleanVerbs(verb, sent["words"] )

        return

        # for obj in cleanedObj["objects"]:
        #   def createEntity()
        #  a = Node("MBTI", name=cleanedObj[subject])
        # b = Node("Object", name=obj)
        #rVerb = Relationship.type(gVerb)
        #g.merge(rVerb(a, b), "MBTI", "name")


def runIndexPipeLine(references, index, originalDoc):
    # print(pred)
    doc = None
    clusters = None
    if(references and references["document"]):
        doc = references["document"].copy()
    if(references and references["clusters"]):
        clusters = references["clusters"].copy()
    # print(doc)
    if(clusters != None and clusters != '' and len(clusters) > 0 and doc != None):
        # word = getWord(clusters, doc, index)
        # if word != '':
        # cleanDoc = replaceWords(word, doc, clusters)
        # # print(cleanDoc)

        # docString = " ".join(filter(None, cleanDoc))
        # sent_text = sent_tokenize(docString)
        # # print("*************************")
        # # print(sent_text)
        # # print("-----------------------")
        # docString = " ".join(filter(None, doc))
        sent_text = sent_tokenize(originalDoc)
        listofPreds = infoExtract_fromArr(sent_text)
        # print(listofPreds[0])
        return storePreds(listofPreds, index)
        # else:
        #     return
    else:
        return


def runDocPipeLine(pred, index, originalDoc):
    # print(pred)
    doc = None
    clusters = None
    if(pred and pred["document"]):
        doc = pred["document"].copy()
    if(pred and pred["clusters"]):
        clusters = pred["clusters"].copy()
    # print(doc)
    if(clusters != None and clusters != '' and len(clusters) > 0 and doc != None):
        # word = getWord(clusters, doc, index)
        # if word != '':
        # cleanDoc = replaceWords(word, doc, clusters)
        # # print(cleanDoc)

        # docString = " ".join(filter(None, cleanDoc))
        # sent_text = sent_tokenize(docString)
        # # print("*************************")
        # # print(sent_text)
        # # print("-----------------------")
        # docString = " ".join(filter(None, doc))
        sent_text = sent_tokenize(originalDoc)
        listofPreds = infoExtract_fromArr(sent_text)

        
        # print(listofPreds[0])
        return storePreds(listofPreds, index)
        # else:
        #     return
    else:
        return


def parallelProcess(index, hit):
    references = findCoReference(hit)
    return runIndexPipeLine(references, index, hit)


def createGraph(index):
    results = ESget(index)
    # count = mp.cpu_count()
    # pool = mp.Pool(count)
    # results = pool.starmap(parallelProcess, [(hit) for hit in results])
    # pool.close()

    # print(entj[0])
    #references = findCoReference(results[1])
    # runIndexPipeLine(references)
    print('running pipeline')
    print(len(results))
    pool = mp.Pool(processes=4)
    start_time = time.time()
    # t.start()

    for result in results:
        if 'section' in result["_source"] and result["_source"]['section'].strip() != '':
            hit = result["_source"]['section']
            references = findCoReference(hit)
            runIndexPipeLine(references, index, hit)
            # pool.apply(parallelProcess, args=(index, hit))
        else:
            print('delete doc')
            res = es.delete(index=result['_index'],doc_type=result['_type'],id=result['_id'])
            if res['result'] == 'deleted':
                print('delete success')
            else:
                print('failed to delete')


    # pp.pprint(sop)
    pp.pprint('done')
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    # f = open('tripples.txt', 'w')

    # df = json_normalize(sop)
    # df.to_csv('tripples.csv')

    # f.write(json_normalize(sop))
    return sop




async def annotateDoc(doc):
    print('running pipeline for single doc')
    pool = mp.Pool(processes=4)
    start_time = time.time()
    section = doc['section']
    section.strip()

    if section != '':
        docToStore = {}
        docToStore["title"] = doc["title"].strip()
        docToStore["index"] = doc["index"].strip()
        docToStore["section"] = section
        docToStore["url"] = doc["url"]
        docToStore["rootNode"] = doc["rootNode"]
        docToStore["sentences"] = []




        hit = doc['section']
        index = doc['index']

        references = findCoReference(hit)
        # runDocPipeLine(references, index, hit)
        # pool.apply(parallelProcess, args=(index, hit))

        coRefdoc = None
        coRefClusters = None
        if(references and references["document"] and references["clusters"]):
            coRefdoc = references["document"].copy()
            coRefClusters = references["clusters"].copy()
            
        if(coRefClusters != None and coRefClusters != '' and len(coRefClusters) > 0 and coRefdoc != None):
            # word = getWord(clusters, doc, index)
            # if word != '':
            # cleanDoc = replaceWords(word, doc, clusters)
            # # print(cleanDoc)

            # docString = " ".join(filter(None, cleanDoc))
            # sent_text = sent_tokenize(docString)
            sent_Arr = sent_tokenize(hit)
            

            for sent in sent_Arr:
                sent_toStore = {}
                info = await infoExtract(sent)
                sent_toStore['sentence'] = sent
                sent_toStore['info'] = info
                docToStore["sentences"].append(sent_toStore)

            
            return docToStore

        else:
            return docToStore




    else:
        print('no content do nothing')
        return 'no content do nothing'
        # res = es.delete(index=result['_index'],doc_type=result['_type'],id=result['_id'])
        # if res['result'] == 'deleted':
        #     print('delete success')
        # else:
        #     print('failed to delete')


    # pp.pprint(sop)
    pp.pprint('done')
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    # f = open('tripples.txt', 'w')

    # df = json_normalize(sop)
    # df.to_csv('tripples.csv')

    # f.write(json_normalize(sop))
    return sop

# https://py2neo.org/v4/data.html#node-and-relationship-objects
#tx = graph.begin()

# createGraph("entj")
