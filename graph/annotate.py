import time

import nltk.corpus
import nltk
from nltk.tokenize import sent_tokenize

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

# Stanford NLP Tripplet Extract Subject Verb Predicate
# from extractTrip import triplet_extraction

import torch
print("gpu count")
torch.cuda.empty_cache()
print("cuda available")
print(torch.cuda.is_available())
torch.cuda.empty_cache()


openPredictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", cuda_device=torch.cuda.current_device())

corefPredictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz", cuda_device=torch.cuda.current_device())



import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import en_core_web_lg
#nlp = en_core_web_lg.load()
nlp = spacy.load('en_core_web_lg')

#nlp = English()
ruler = EntityRuler(nlp, overwrite_ents=True)
#patterns = [{"label": "PERSON", "pattern": [{"LOWER": "enfj"}]}]
patterns = [{"label": "PERSON", "pattern": [{"LOWER": {"REGEX": "[e|i]+[n|s]+[t|f]+[j+p]"}},{'LEMMA': 'be', 'OP': '?'}]}]
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)

async def annotateDoc(doc):
    torch.cuda.empty_cache()
    print('running pipeline for single doc')
    # pool = mp.Pool(processes=4)
    # start_time = time.time()
    section = doc['section'] # a paragraph of text
    section.strip()

    if section != '':
        docToStore = {}
        docToStore["title"] = doc["title"].strip()
        docToStore["index"] = doc["index"].strip()
        docToStore["section"] = section
        # used for elasticsearch to index documents scraped
        docToStore["url"] = doc["url"]
        docToStore["rootNode"] = doc["rootNode"]
        docToStore["sentences"] = []

        docToStore["sectionSub"] = findCoReference(section)

        if docToStore["sectionSub"] != None and "document" in docToStore["sectionSub"]:
            subsect = docToStore["sectionSub"]
            # print(subsect)
            swapWord = docToStore['index'].upper()  # docToStore["index"]
            try:
                docToStore["corefSwap"] = corefSwap(subsect, swapWord)

            except RuntimeError as e:
                print('corefSwap index error')
                print(docu[fr])
                print(docu[to])

        # interface Sectionsub {
        #   antecedent_indices: number[][];
        #   clusters: number[][][];
        #   document: string[];
        #   predicted_antecedents: number[];
        #   top_spans: number[][];
        # }

        hit = doc['section']
        sent_Arr = sent_tokenize(hit)

        for sent in sent_Arr:
            sent_toStore = {}
            info = await infoExtract(sent)
            sent_toStore['sentence'] = sent
            sent_toStore['info'] = info

            # Stanford NLP Tripplet Extract Subject Verb Predicate
            # sent_toStore['tripple'] = triplet_extraction(sent)
            spacy = nlp(sent)
            sent_toStore['spacy'] = {}

            tokens = []
            for token in spacy:
                tokens.append({'text': token.text, 'lemma': token.lemma_, 'pos': token.pos_, 'tag': token.tag_, 'dep' :token.dep_})
            
            sent_toStore['spacy']['tokens'] = tokens
            entities = []
            for ent in spacy.ents:
                entities.append({'text': ent.text, 'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_})

            sent_toStore['spacy']['entities'] = entities

            docToStore["sentences"].append(sent_toStore)

        return docToStore

    pp.pprint('done')
    return 'error processing Document'


def corefSwap(subsect, swapWord):
    docu = subsect["document"]
    totalLength = len(docu)
    for clusterGroup in subsect["clusters"]:
        for lastCluster in reversed(clusterGroup):
            # for word in cluster:
            fr = lastCluster[0]
            to = lastCluster[1]
            if fr < totalLength:
                if(fr != to):
                    to += 1
                    if fr == 0:
                        docu = [swapWord] + docu[to:]
                    else:
                        docu = docu[:fr] + [swapWord] + docu[to:]
                elif fr < len(docu):
                    docu[fr] = swapWord

    return ' '.join(docu)



# "C:\\Users\\djway\\Desktop\\ML-notebook\\coref-model-2018.02.05"
# C:\Users\djway\Desktop\ML-notebook\bidaf\bidaf-elmo-model-2018.11.30-Ccharpad.tar
# "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz"
# ENTJs are analytical and objective, and like bringing order to the world around them. When there are flaws in a system, the ENTJ sees them, and enjoys the process of discovering and implementing a better way. ENTJs are assertive and enjoy taking charge; they see their role as that of leader and manager, organizing people and processes to achieve their goals.",

def findCoReference(text):
    # make api call
    # /coRef
    # pred = corefPredictor(text)
    try:
        # pred = corefPredictor.predict(document=text)
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            pred = corefPredictor.predict(document=text)
            # pred = semanticRoleLabelPredict.predict(sentence=text)
            return pred
        else:
            time.sleep(5)
            pred = corefPredictor.predict(document=text)
            # pred = semanticRoleLabelPredict.predict(sentence=text)
            return pred

    except RuntimeError as e:
        print('error')
        time.sleep(5)
        print(e)
        torch.cuda.empty_cache()
        try:
            pred = corefPredictor.predict(document=text)
            return pred
        except RuntimeError as e:
            print('failed twice on corefPredictor')
            return findCoReference(text)

#MBTItype = ''
# find word in sentence
async def infoExtract(sentence):
    try:
        preds = openPredictor.predict(sentence=sentence)
        return preds

    except RuntimeError as e:
        print('error')
        print(e)
        torch.cuda.empty_cache()
        time.sleep(5)
        try:
            pred = openPredictor.predict(sentence=sentence)
            return preds
        except RuntimeError as e:
            print('failed twice on openPredictor')

    return preds
