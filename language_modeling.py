from functools import partial
from nltk.util import everygrams
from nltk.lm import KneserNeyInterpolated, Vocabulary
from nltk.lm.preprocessing import flatten, pad_both_ends, padded_everygram_pipeline

import os
import pickle
from tqdm import tqdm

import spacy

from pathlib import Path

from .ES_wrapper import QueryES
from .s3_util_HA import S3Util


################################################################################
################################################################################

CLIA_dict={

    # Orgs below are the ones from the report_scraper project
    'TumorNext':   ['05D0981414'],
    'Oncotype DX': ['05D1018272'],
    'MammaPrint': ['99D1030869', '05D1089250'],
    'Counsyl': ['05D1102604'],
    'Afirma': ['05D2014120', '45D2037953', '4602052137'],

    # Orgs below are the ones annotated in N and CLQ
    'PGD':                                 [],
    'Broad':                               [],
    'MLabs':                               ['23D0366712'],
    'Caris':                               ['03D1019490', '45D0975010'],
    'Tempus':                              ['14D2114007'],
    'Invitae':                             [],
    'Guardant':                            ['05D2070300'],

    # Patients annotated with this had reports from Foundation One and Ambry Genetics Corporation
    'NCI MATCH':                           [],

    'UPenn CPD':                           [],
    'eMERGE-Seq':                          [],
    'CANCERPLEX':                          ['22D2060722'],
    'Foundation':                          ['22D2027531', '34D2044309'],
    'MSK-IMPACT':                          [],
    'MD dictated':                         [],
    'Neogenomics':                         ['05D1021650'],
    'MD Anderson (409)':                   [],
    'Hospital Molecular Pathology':        ['39D2059110'],
    'OHSU Knight Diagnostic Laboratories': ['38D0881787']
}

class LanguageModel:

    def __init__(self, cache_local=False, push_s3=False, save_dir="/opt/ml/model/", run_here=False, model_type="ngram"):
        self.cache_local=cache_local
        self.push_s3=push_s3
        self.save_dir=save_dir

        spacy.cli.download('en_core_web_md')
        self.nlp = spacy.load("en_core_web_md")
        self.ocr_elasticsearch=QueryES()
        self.s3_util=S3Util()
        self.model_type=model_type # neural or ngram

        if run_here:
            self.save_dir=""

    def get_data_s3(self, CLIDA_ID):
        print("Reading ES hits from S3")
        hits=self.s3_util.read_pickled_file(f'projects/MRFT_HA/ES-HITS/{CLIDA_ID}_ES_hits.pkl')

        print("tokenizing text")
        sents=[]
        for hit in tqdm(hits):
            for page in hit["_source"]["doc_pages"]:
                doc=self.nlp(page["page_contents"])
                for sent in doc.sents:
                    sents.append(tuple([token.text if token.text !="\n" else "." for token in sent]))
        return sents

    def get_data_es(self, CLIA_ID):
        print("Reading data from ES")
        print("tokenizing text")
        sents=[]
        for hit in tqdm(self.ocr_elasticsearch.search(CLIA_ID)):
            for page in hit["_source"]["doc_pages"]:
                doc=self.nlp(page["page_contents"])
                for sent in doc.sents:
                    sents.append(tuple([token.text if token.text !="\n" else "." for token in sent]))
        return sents

    def save_model(self, model, ORG, model_file_path):
        if model:
            if self.cache_local:
                print(f"Caching the Language model for {ORG}")
                Path(self.save_dir+"LanguageModels").mkdir(parents=True, exist_ok=True)

                with open(model_file_path, "wb") as outf:
                    pickle.dump(model, outf)

                if self.push_s3:
                    self.s3_util.upload_file(project_name="MRFT_HA", f_abs_path=model_file_path)
        else:
            print(f"No sentences were found to build a LM from for {ORG}")

    def build_ngram_lm(self, train):
        if not train: return None

        n=5 # up to 5 gram language model
        train, vocab = padded_everygram_pipeline(n, train)
        model = KneserNeyInterpolated(n)
        model.fit(train, vocab)
        return model

    def build_neural_lm(self, train):
        if not train: return None

        ####
        return None

    def build_one(self, ORG):
        print(f"Building a Language model for {ORG}")

        if self.model_type=="neural":
            model_file_path=self.save_dir+"LanguageModels/"+ORG.replace(" ", "_")+"_neural_lm.pkl"
        else:
            model_file_path=self.save_dir+"LanguageModels/"+ORG.replace(" ", "_")+"_lm.pkl"

        # Skip the ones we already have a LM for
        if os.path.isfile(model_file_path):
            print("We already have a model for this Org ...")
            return

        sents=[]
        for CLIA_ID in CLIA_dict[ORG]:
            if self.model_type=="neural":

                print("Not ready to train !!!")
                exit()

                sents+=self.get_data_s3(CLIA_ID)
            else:
                if self.data_from=="s3":
                    sents+=self.get_data_s3(CLIA_ID)
                else:
                    sents+=self.get_data_es(CLIA_ID)

        if self.model_type=="neural":
            model=self.build_neural_lm(sents)
        else:
            model=self.build_ngram_lm(sents)

        self.save_model(model, ORG, model_file_path)

    def build_one_ES(self, name):
        print(f"Building a Language model for {name}")

        if self.model_type=="neural":
            model_file_path=self.save_dir+"LanguageModels/"+name.replace(" ", "_")+"_neural_lm.pkl"
        else:
            model_file_path=self.save_dir+"LanguageModels/"+name.replace(" ", "_")+"_lm.pkl"

        # Skip the ones we already have a LM for
        if os.path.isfile(model_file_path):
            print("We already have a model for this Org ...")
            return

        sents=[]
        if self.model_type=="neural":
            print("Not ready to train !!!")
            exit()

            sents+=self.get_data_s3(name)
        else:
            sents+=self.get_data_s3(name)

        if self.model_type=="neural":
            model=self.build_neural_lm(sents)
        else:
            model=self.build_ngram_lm(sents)

        self.save_model(model, name, model_file_path)

    def build_all(self):
        for ORG in CLIA_dict:
            if not CLIA_dict[ORG] or ORG=="Neogenomics": continue
            self.build_one(ORG)
