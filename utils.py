def search_kwd_emb(es, kwd, index):
    body = {
        "query": {
            "match_phrase": {
                "keyword": kwd
            }
        }
    }

    response = es.search(index=index, body=body, request_timeout=120)
    try:
        doc = response['hits']['hits'][0]["_source"]
        return (doc["keyword"], doc["embedding"])
    except:
        return None


def search_by_embedding(es, emb, index, size=20):
    body = {
        "size": size + 1,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.emb, 'embedding') + 1.0",
                    "params": {"emb": emb}
                }
            }
        },
        "_source": ["keyword"]
    }

    response = es.search(index=index, body=body, request_timeout=120)
    kwds = []
    for doc in response['hits']['hits']:
        kwds.append((doc["_source"]["keyword"], doc["_score"]))

    return kwds


def get_suggestions(es, query, emb, index, n):
    suggs = search_by_embedding(es, emb, index, n)
    return [item[0] for item in suggs if item[0].lower().strip() != query.strip().lower()]


def get_suggestions_fastkeywords(es, query, index, n):
    res = search_kwd_emb(es, query, index)

    if res:
        _, emb = res
    else:
        return []

    suggs = search_by_embedding(es, emb, index, n)
    return [item[0] for item in suggs if item[0].strip() != query.strip()]


def _clean_query(query):
    return query.replace(":", "").replace(
        "/", " ").replace(",", " ").replace("(", "").replace(")", "").replace("'", "").replace("!", "")


def get_docs_ranking_pyterrier(query, model, size=1000):
    query = _clean_query(query)

    res = model.search(query, size)

    ranking = list()
    for _, row in res.iterrows():
        ranking.append(row["docno"])

    return ranking
