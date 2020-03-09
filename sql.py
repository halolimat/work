#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, json
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import pandas.io.sql as sqlio

# datascience tools to connect to dbs
sys.path.append("/Users/hussein.alolimat/work/codes/tempus-machine-learning/")
from tools.tempuslabs.db_connector import db_connector as dbc

# Read the database connection info from your secrets file:
with open("/Users/hussein.alolimat/.ssh/tempus_secrets.json", 'rb') as file:
    secrets = json.load(file)


# In[2]:


# connect to clq
db_clq = dbc.AutoConnector('clq', secrets['db']['clq'])
conn_clq = db_clq.auto_connect()

# connect to n
db_n = dbc.AutoConnector('n', secrets['db']['n'])
conn_n = db_n.auto_connect()


# In[3]:


qry = """
SELECT b.patient_tempus_id AS q_uuid 
       , c.attachment_id 
       , a.tempus_document_type 
FROM   vault.clinical_document_gateway_meta_sat AS a 
       JOIN vault.patient_clinical_document_link AS b 
         ON a.clinical_document_hkey = b.clinical_document_hkey 
       JOIN vault.patient_clinical_document_sat AS c 
         ON b.patient_clinical_document_link_hkey = c.patient_clinical_document_link_hkey
"""

clq = sqlio.read_sql_query(qry, conn_clq)


# In[ ]:





# In[ ]:


n


# In[ ]:


# What are the doc types that are in the sequencing view that are not in the remaining patients records


# In[ ]:


# what are the doc types that has clia hits and the ones from the same patient but with no clia hits?


# In[ ]:


qry = """
SELECT b.patient_tempus_id AS q_uuid 
       , c.attachment_id 
       , a.mime_type 
       , a.tempus_document_type 
       , a.partner_document_created_date 
FROM   vault.clinical_document_gateway_meta_sat AS a 
       JOIN vault.patient_clinical_document_link AS b 
         ON a.clinical_document_hkey = b.clinical_document_hkey 
       JOIN vault.patient_clinical_document_sat AS c 
         ON b.patient_clinical_document_link_hkey = c.patient_clinical_document_link_hkey
"""

clq = sqlio.read_sql_query(qry, conn_clq)


# In[5]:


import pandas as pd


# In[7]:


data=pd.read_csv("src/05D1018272_pat_att.csv")


# In[ ]:





# In[9]:


new_df = pd.merge(data, clq,  how='left', left_on=['patient_id','attachment_id'], right_on = ['q_uuid','attachment_id'])


# In[ ]:





# In[11]:


# df.groupby([new_df.index.tempus_document_type, 'action']).count().plot(kind='bar')
new_df['tempus_document_type'].value_counts().plot(kind='barh')


# In[12]:


clq['tempus_document_type'].value_counts().plot(kind='barh')


# In[15]:


clq[~clq['tempus_document_type'].isin(new_df['tempus_document_type'])]['tempus_document_type'].value_counts().plot(kind='barh')


# In[16]:


treatment_attachments=clq.loc[clq['tempus_document_type'] == "Treatments"]


# In[22]:


pd.Series(treatment_attachments["attachment_id"].unique())[:50000].to_csv("att_ids.csv")


# In[ ]:




