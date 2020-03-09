from .s3_util_HA import S3Util
import spacy
from collections import defaultdict
from .ES_wrapper import QueryES

from nltk.lm.preprocessing import flatten, pad_both_ends, padded_everygram_pipeline

import os
import pickle
from tqdm import tqdm

import timeit

class LMScoring:

    def get_models(self):
        count=3
        self.models=defaultdict()
        for mname in tqdm(self.model_names):
            if "Tempus" in mname:
                continue

            if "Oncotype" not in mname:
                continue

            count-=1
            if count<0:
                break
            # mname='projects/MRFT_HA/Caris_lm.pkl'
            m=mname.split("/")[-1].split("_")[0]
            if os.path.isfile("LMs/"+m+".lm"):
                # print("reading from disk")
                with open("LMs/"+m+".lm", "rb") as f:
                    model=pickle.load(f)
            else:
                # print("Reading models from S3")
                model=self.s3_util.read_pickled_file(mname)
                with open("LMs/"+m+".lm", "wb") as outf:
                    pickle.dump(model, outf)
            self.models[m]=model
            # break


    def __init__(self, get_lms=False):
        if get_lms:
            self.s3_util=S3Util()
            s3_dir_path="projects/MRFT_HA"
            self.model_names=[fname for fname in self.s3_util.list_files_in_dir(s3_dir_path) if "lm.pkl" in fname]
            self.get_models()

        spacy.cli.download('en_core_web_md')
        self.nlp = spacy.load("en_core_web_md")

        self.es=QueryES()

    def score(self, patient_id):

        # txt="Cancer refers to any one of a large number of diseases characterized by the development of abnormal cells that divide uncontrollably and have the ability to infiltrate and destroy normal body tissue. Cancer often has the ability to spread throughout your body. Cancer is the second-leading cause of death in the world."

        # ===================================
        import json
        if False:
            hits=self.es.search_list("patient_id", [patient_id])
            with open("txt", "w") as f:
                json.dump(hits, f)
        else:
            with open("txt") as f:
                hits=json.load(f)

        seq=[]
        for hit in hits:
            for page in hit["_source"]["doc_pages"]:
                doc=self.nlp(page["page_contents"])
                seq+=[tuple(token.text for token in sent) for sent in doc.sents]

        m_names=list(self.models)
        scores=[]

        test_data, _ = padded_everygram_pipeline(2, seq)

        seq=[]
        for i, test in enumerate(test_data):
            seq.append(tuple([x for x in test]))

        for mname in self.models:
            # print(mname)
            # int_scores=[]
            # for i, test in enumerate(test_data):
            #     pp=self.models[mname].perplexity(test)
            #     scores.append(pp)
            #     int_scores.append(pp)
            #
            # print(min(int_scores))
            # scores.append(min(int_scores))

            start = timeit.default_timer()

            # the winning model is the model with the lowest perplexity
            pp=self.models[mname].perplexity(seq)
            scores.append(pp)
            print(mname)
            print(pp)
            print("----")
            print('Time: ', timeit.default_timer() - start)

        m=min(scores)

        pred_class=[]
        for idx in range(len(scores)):
            if scores[idx]==m:
                pred_class.append(m_names[idx])

        return pred_class
