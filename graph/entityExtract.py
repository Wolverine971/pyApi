from spacy.tokens import Span
from spacy.matcher import Matcher
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy

import neuralcoref
import logging
logging.basicConfig(level=logging.INFO)


import en_core_web_lg
nlp = en_core_web_lg.load()
# nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)
spacy.prefer_gpu()

# pd.set_option('display.max_colwidth', 200)

# doc = nlp("the drawdown process is governed by astm standard d823")

# for tok in doc:
#   print(tok.text, "...", tok.dep_)


def get_entities(sent):
    # print('test')
    # chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################
    print(nlp(sent))
    for tok in nlp(sent):
        # chunk 2
        # if token is a punctuation mark then move on to the next token
        print(tok)
        if tok.dep_ != "punct":
                # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            # chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            # chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            # chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
        #############################################################

        return [ent1.strip(), ent2.strip()]


# print(get_entities("he is a good boy"))









from spacy.lang.en import English
from spacy.pipeline import EntityRuler

# ruler = EntityRuler(nlp)
# # patterns = [{"label": "ORG", "pattern": "Apple"},
# #             {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}]
# patterns = [{'pattern': 'enfj', 'label': 'nsubj'}]
# ruler.add_patterns(patterns)
# nlp.add_pipe(ruler)

# doc = nlp("Apple is opening its first big office in San Francisco.")
# print([(ent.text, ent.label_) for ent in doc.ents])





def entity_pairs(text, coref=True):
    text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
    text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
    text = nlp(text)
    print(text)
    if coref:
        text = nlp(text._.coref_resolved)  # resolve coreference clusters
    sentences = [sent.string.strip() for sent in text.sents]  # split text into sentences
    ent_pairs = list()
    for sent in sentences:
        pattern = [{},
           {'LEMMA': 'tool', 'POS': 'NOUN'},
           {}]
        sent = nlp(sent)
        spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
        spans = spacy.util.filter_spans(spans)
        with sent.retokenize() as retokenizer:
            [retokenizer.merge(span) for span in spans]
        dep = [token.dep_ for token in sent]
        if (dep.count('obj')+dep.count('dobj'))==1 \
                and (dep.count('subj')+dep.count('nsubj'))==1:
            for token in sent:
                if token.dep_ in ('obj', 'dobj'):  # identify object nodes
                    subject = [w for w in token.head.lefts if w.dep_
                               in ('subj', 'nsubj')]  # identify subject nodes
                    if subject:
                        subject = subject[0]
                        # identify relationship by root dependency
                        relation = [w for w in token.ancestors if w.dep_ == 'ROOT']  
                        if relation:
                            relation = relation[0]
                            # add adposition or particle to relationship
                            if relation.nbor(1).pos_ in ('ADP', 'PART'):  
                                relation = ' '.join((str(relation),
                                        str(relation.nbor(1))))
                        else:
                            relation = 'unknown'
                        subject, subject_type = refine_ent(subject, sent)
                        token, object_type = refine_ent(token, sent)
                        ent_pairs.append([str(subject), str(relation), str(token),
                                str(subject_type), str(object_type)])
    filtered_ent_pairs = [sublist for sublist in ent_pairs
                          if not any(str(x) == '' for x in sublist)]
    pairs = pd.DataFrame(filtered_ent_pairs, columns=['subject',
                         'relation', 'object', 'subject_type',
                         'object_type'])
    print('Entity pairs extracted:', str(len(filtered_ent_pairs)))
    return pairs


def refine_ent(ent, sent):
    unwanted_tokens = (
        'PRON',  # pronouns
        'PART',  # particle
        'DET',  # determiner
        'SCONJ',  # subordinating conjunction
        'PUNCT',  # punctuation
        'SYM',  # symbol
        'X',  # other
        )
    ent_type = ent.ent_type_  # get entity type
    if ent_type == '':
        ent_type = 'NOUN_CHUNK'
        ent = ' '.join(str(t.text) for t in
                nlp(str(ent)) if t.pos_
                not in unwanted_tokens and t.is_stop == False)
    elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
        t = ''
        for i in range(len(sent) - ent.i):
            if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                t += ' ' + str(ent.nbor(i))
            else:
                ent = t.strip()
                break
    return ent, ent_type


import wikipediaapi
import pandas as pd

def wiki_page(page_name):
    wiki_api = wikipediaapi.Wikipedia(language='en',
                                      extract_format=wikipediaapi.ExtractFormat.WIKI)
    page_name = wiki_api.page(page_name)
    if not page_name.exists():
        print('page does not exist')
        return
    page_data = {'page': page_name, 'text': page_name.text, 'link': page_name.fullurl,
                 'categories': [[y[9:] for y in list(page_name.categories.keys())]]}
    page_data_df = pd.DataFrame(page_data)
    return page_data_df

# wiki_data = wiki_page('dj')

# print(wiki_data.loc[0,'text'])

enfj = """ The ENFJ Preferences are Extraversion , Intuition , Feeling and Judging .These determine the ENFJ Personality Type . 
Extraverted (E) Extraversion is characterized by a preference to focus on the world outside the self. ENFJs are energized by social gatherings, parties 
and group activities. Extraverts are usually enthusiastic, gregarious and animated. Their communication style is verbal and assertive. As Extraverts, ENFJ often 
need to talk. They enjoy the limelight. Feeling (F) As Feeling people, ENFJs are subjective. They make decisions based on their principles and values. They are ruled 
by their heart instead of their head. ENFJs judge situations and others based on their feelings and extenuating circumstances. They seek to please others and want to 
be appreciated. They value harmony and empathy. Intuitive (N) People with Intuition live in the future. They are immersed in the world of possibilities. They process 
information through patterns and impressions. As Intuitives, ENFJ value inspiration and imagination. They gather knowledge by reading between the lines. Their abstract 
nature attracts them toward deep ideas, concepts and metaphors. Judging (J) As Judging people, ENFJs think sequentially. They value order and organization. Their lives 
are scheduled and structured. ENFJs seek closure and enjoy completing tasks. They take deadlines seriously. They work then they play. The Judging preference does not mean
 judgmental . Judging refers to how a day-to-day activities at dealt with. ENFJ is cool."""

# print(entity_pairs(enfj))



from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
matched_sents = []

def collect_sents(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]  # Matched span
    sent = span.sent  # Sentence containing matched span
    # Append mock entity for match in displaCy style to matched_sents
    # get the match span by ofsetting the start and end of the span with the
    # start and end of the sentence in the doc
    match_ents = [{
        "start": span.start_char - sent.start_char,
        "end": span.end_char - sent.start_char,
        "label": "MATCH",
    }]
    matched_sents.append({"text": sent.text, "ents": match_ents})

pattern = [{"LOWER": "enfj"}, {"LEMMA": "be"}, {"POS": "ADV", "OP": "*"},
           {"POS": "ADJ"}]
matcher.add("enfj", collect_sents, pattern)  # add pattern
doc = nlp(enfj)
matches = matcher(doc)

print(matches)
displacy.render(matched_sents, style="ent", manual=True)


# d = {'text': ['He is a good boy. Sometimes he does no do his homework']}
# df = pd.DataFrame(data=d)
# print(entity_pairs(wiki_data.loc[0,'text']))
