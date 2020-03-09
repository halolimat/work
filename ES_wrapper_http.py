
import json
import requests

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

querystring = {"pretty":"true"}

headers = {'Content-Type': "application/json",
           'cache-control': "no-cache" }

prod=True
if prod: index_name="ocr_min3_max20_prd"
else: index_name="ocr"

################################################################################

def get_ES_info(query):
    if query == "mapping":
        url='https://localhost:9200/'+index_name+'/_mapping' # mappings
        response=requests.request("GET", url, params=querystring, verify=False)
    else:
        url='https://localhost:9200' # Health and info
        response=requests.request("GET", url, verify=False)
    return response.text

################################################################################

def search_field(field_name, string_query):
    """Elasticsearch Exact Match Query"""
    query = json.dumps({
        "query": {
            "match": {
                field_name: string_query
            }
        }
    })
    uri="https://localhost:9200/"+index_name+"/_search"
    response = requests.get(uri, data=query, headers=headers, params=querystring, verify=False)
    try:
        return response.json()["hits"]["hits"]
    except Exception as e:
        return e

def search(string_query):
    query = json.dumps({
        "query": {
            "bool": {
              "must": [
                {
                  "nested": {
                    "path": "doc_pages",
                    "query": {
                      "query_string":{
                        "fields": ["doc_pages.page_contents"],
                        "query": f"{string_query}",
                        "phrase_slop": 0
                      }
                    }
                  }
                }
              ]
            }
      },
      "size": 10000
      })

    uri="https://localhost:9200/"+index_name+"/_search"

    # Verify = False is needed because we are using SSH but without a certificate.
    response = requests.get(uri, data=query, headers=headers, params=querystring, verify=False)
    try:
        return response.json()["hits"]["hits"]
    except Exception as e:
        return e

################################################################################

def search_filter(string_query, filter):
    query = json.dumps({
        "query": {
            "bool": {
              "must": [
                {
                  "nested": {
                    "path": "doc_pages",
                    "query": {
                      "query_string":{
                        "fields": ["doc_pages.page_contents"],
                        "query": f"{string_query}",
                        "phrase_slop": 0
                      }
                    }
                  }
                }
              ],
              "filter": [
                {"term": {"doc_pages.patient_id": filter}}
              ]
            }
      }
      })

    uri="https://localhost:9200/"+index_name+"/_search"

    # Verify = False is needed because we are using SSH but without a certificate.
    response = requests.get(uri, data=query, headers=headers, params=querystring, verify=False)
    try:
        return response.json()["hits"]["hits"]
    except Exception as e:
        return e

def match_list(field, l):

    query = json.dumps({
      "query": {
        "bool": {
          "must": {
            "match_all": {}
          },
          "filter": {
            "terms": {
              field: l
            }
          }
        }
      }
    })

    uri="https://localhost:9200/"+index_name+"/_search"

    # Verify = False is needed because we are using SSH but without a certificate.
    response = requests.get(uri, data=query, headers=headers, params=querystring, verify=False)
    try:
        return response.json()["hits"]["hits"]
    except Exception as e:
        return e


def search_fuzzy(string_query):
    query = json.dumps({
        "query": {
            "bool": {
              "must": [
                {
                  "nested": {
                    "path": "doc_pages",
                    "query": {
                      "query_string":{
                        "fields": ["doc_pages.page_contents"],
                        "query": f"{string_query}",
                        "phrase_slop": 0,
                        "fuzziness": "AUTO"
                      }
                    }
                  }
                }
              ]
        }
      }
    })

    uri="https://localhost:9200/"+index_name+"/_search"

    # Verify = False is needed because we are using SSH but without a certificate.
    response = requests.get(uri, data=query, headers=headers, params=querystring, verify=False)
    try:
        return response.json()
    except Exception as e:
        return e

################################################################################
################################################################################
################################################################################

if __name__ == "__main__":

    pass
