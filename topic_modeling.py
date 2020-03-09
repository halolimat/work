'''

Gensim tutorial here: https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py

CLIA IDs: https://tempuslabs.atlassian.net/wiki/spaces/DATA/pages/798328269/Data+Explration

'''

import warnings
warnings.filterwarnings('ignore')

import re
import json
from collections import defaultdict

from tqdm import tqdm

# # Download from here: https://drive.google.com/uc?id=1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD&export=download
# # Read: https://huggingface.co/transformers/quickstart.html
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('biobert_v1.1_pubmed')

import spacy
nlp = spacy.load("en_core_web_lg")
nlp.vocab.add_flag(lambda s: s in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

from gensim.models import LdaModel
from gensim.corpora import Dictionary

import ES_wrapper

##########

# Source: https://gist.github.com/dideler/5219706#gistcomment-2222286
emails_pattern = re.compile(r"([a-z0-9!#$%&*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
                    "{|}~-]+)*(@)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.))+[a-z0-9]"
                    "(?:[a-z0-9-]*[a-z0-9])?)|([a-z0-9!#$%&*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
                    "{|}~-]+)*(\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\sdot\s))+[a-z0-9]"
                    "(?:[a-z0-9-]*[a-z0-9])?)",re.S)

phone_numbers_pattern = re.compile(r"(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?")

url_pattern = re.compile(r'''(?i)\b((?:[a-z][\w-]+:(?:\/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''')

#########

def get_texts(CLIA_ID):
    docs = []
    for hit in ES_wrapper.search(CLIA_ID):
        for page in hit["_source"]["doc_pages"]:
            docs.append(page["page_contents"])
    return docs

def featurize(docs, size=None):

    if size:
        docs=docs[:size]

    # Featurize and preprocess
    for idx in tqdm(range(len(docs))):
        docs[idx] = docs[idx].replace("\n", ". ").lower()

        doc_features=[]

        # (1) Extract phone numbers:
        emails=[e[0] for e in re.findall(emails_pattern, docs[idx])]
        doc_features+=emails

        # (2) phone numbers
        phone_numbers=["".join(e) for e in re.findall(phone_numbers_pattern, docs[idx])]
        doc_features+=phone_numbers

        # (3) URLs
        urls=["".join(e) for e in re.findall(url_pattern, docs[idx])]
        doc_features+=urls

        # (4) chunks, entities, and tokens
        doc=nlp(docs[idx])

        doc_features+=[chunk.text.replace(" ", "_") for chunk in doc.noun_chunks]
        doc_features+=[ent.text.replace(" ", "_") for ent in doc.ents]
        doc_features+=[token.text for token in doc if not token.is_stop]

        # remove single character tokens
        docs[idx]=[f for f in doc_features if len(f)>1]

    return docs

def topic_model(docs):
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Set training parameters.
    num_topics = 10
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    print("Training LDA Model ...")
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    return model.top_topics(corpus)

if __name__ == "__main__":
    CLIA_ID="34D2044309"
    docs=get_texts(CLIA_ID)
    docs=featurize(docs, 100)
    top_topics=topic_model(docs)

    terms_rank=defaultdict(int)
    for topic in top_topics:
        for term in topic[0]:
            terms_rank[term[1]]+=1

    for term in sorted(terms_rank, key=terms_rank.get, reverse=True):
        print(term, terms_rank[term])
