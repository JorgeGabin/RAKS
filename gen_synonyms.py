import numpy as np
import pickle as pkl
import ollama as llama
import openai
import pyterrier as pt

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModel
)

from utils import (
    search_by_embedding,
    get_suggestions_fastkeywords,
    get_suggestions
)
from elasticsearch import Elasticsearch

DATASET = "trec_fair_2021"
# DATASET = "trec_7_8"

openai.organization = "your_org_key"
OPENAI_API_KEY = "your_key"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

QUERIES = f"{DATASET}_topics_eval.jsonl"

HOST = "host_name"  # Elasticsearch host
PORT = 9200  # Elasticsearch port

INDEX_NAME = f"{DATASET}_clean"
CONTEXT_SIZE = 10

SBERT_INDEX = f"{DATASET}-kwds-sbert"
sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

FK_INDEX = "fastkwds_mag"

KBIR_INDEX = f"{DATASET}-kwds-kbir"
kbir_tok = AutoTokenizer.from_pretrained("bloomberg/KBIR")
kbir = AutoModel.from_pretrained("bloomberg/KBIR").to("cuda")

PHRASEBERT_INDEX = f"{DATASET}-kwds-phrasebert"
phrasebert = SentenceTransformer("whaleloops/phrase-bert")

E5_INDEX = f"{DATASET}-kwds-e5"
e5 = SentenceTransformer("intfloat/e5-base-v2")

es = Elasticsearch([{'host': HOST, 'port': PORT}])

pt.init()

index_ref = pt.IndexRef.of(f"resources/indices/{DATASET}")
bm25 = pt.BatchRetrieve(index_ref, wmodel='BM25')


def _score_rr(reform_ranking, relevant_docs):
    for rank, doc_id in enumerate(reform_ranking, 1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    return 0.0


def _score_recall(reform_ranking, relevant_docs):
    hits = 0
    for doc_id in reform_ranking:
        if doc_id in relevant_docs:
            hits += 1

    return hits / len(relevant_docs)


def _score_ap(reform_ranking, relevant_docs):
    hits = 0
    r = []
    for i, doc_id in enumerate(reform_ranking):
        if doc_id in relevant_docs:
            hits += 1
            r.append(hits / (i + 1))
    return sum(r) / len(relevant_docs)


def _score(reform_ranking, relevant_docs):
    return {
        "recall": _score_recall(reform_ranking[:10], relevant_docs),
        "ap": _score_ap(reform_ranking[:1000], relevant_docs),
        "rr": _score_rr(reform_ranking[:10], relevant_docs)
    }


def _get_prompt_with_context(query: str, context):

    messages = [
        {
            "role": "system",
            "content": "You are an expert on doing query expansion for keyword-format queries."
        },
        {
            "role": "user",
            "content": """Given an input query, generate a list of 15 expansion keywords that enhance and broaden the query's scope. Follow the following steps:
1. Include relevant and widely used acronyms as standalone keywords, not in parentheses (e.g., suggest "NLP" for "natural language processing").
2. If the input query covers more than one concept, generate expansions for each one, including their specific subtopics (hyponyms).
3. Suggest keywords for important topics in the field of the input query.
4. Only suggest expansion keywords with more than three words if they are highly relevant.
5. Do not suggest many keywords that include words from the input query.
6. Avoid suggesting broader terms (hypernyms).
7. Sort the suggested keywords by their relevance and importance.
8. We provide some candidate expansions to aid the expansions generation.
9. Candidate expansions may be included as expansion keywords.

Input Format:

    Provide expansions for:
    <InputQuery>{{input query}}</InputQuery>
    <CandidateExpansions>{{candidate expansions}}</CandidateExpansions>

Output Format:

    <ExpansionKeywords>{{keyword 1}}, {{keyword 2}}, {{keyword 3}}, {{keyword 4}}, {{keyword 5}}</ExpansionKeywords>

Provide expansions for:
<InputQuery>natural language processing</InputQuery>
<CandidateExpansions>computational linguistics, parsing of natural language, natural language understanding, word-processing, text-mining, natural language generation, parsing algorithms, intelligent text analysis, syntactic analysis, syntactic parsing</CandidateExpansions>
"""
        }, {
            "role": "assistant",
            "content": "<ExpansionKeywords>NLP, computational linguistics, text-mining, natural language understanding, natural language generation, NLU, NLG</ExpansionKeywords>"
        }, {
            "role": "user",
            "content": f"Provide expansions for:\n<InputQuery>information retrieval</InputQuery>\n<CandidateExpansions>text retrieval, information classification, information retrieval query language, search engine, document retrieval, information mining, retrieval model, memory retrieval, information extraction, text and data mining</CandidateExpansions>"
        },
        {
            "role": "assistant",
            "content": "<ExpansionKeywords>IR, text retrieval, document retrieval, search engine, document ranking, retrieval model, web search, ranked retrieval, semantic search, information extraction</ExpansionKeywords>"
        }, {
            "role": "user",
            "content": f"Provide expansions for:\n<InputQuery>mussel</InputQuery>\n<CandidateExpansions>mulletfish, mulisp, bivalve molusc, mund, mytilus edulis, palaeoheterodonta, mytilus trossellus, muscardinidae, mulac, mytilidae</CandidateExpansions>"
        },
        {
            "role": "assistant",
            "content": "<ExpansionKeywords>mytilidae, bivalve mollusc, bathymodiolus, pteriomorphia, palaeoheterodonta, heterodonta, dreissena polymorpha, mytilus edulis, mytilus galloprovincialis, mytilus trossellus, perna canaliculus, filter-feeder</ExpansionKeywords>"
        },
        {
            "role": "user",
            "content": f"Provide expansions for:\n<InputQuery>{query}</InputQuery>\n<CandidateExpansions>{context}</CandidateExpansions>"
        }
    ]
    return messages


def _get_prompt(query: str):

    messages = [
        {
            "role": "system",
            "content": "You are an expert on doing query expansion for keyword-format queries."
        },
        {
            "role": "user",
            "content": """Given an input query, generate a list of 15 expansion keywords that enhance and broaden the query's scope. Follow the following steps:
1. Include relevant and widely used acronyms as standalone keywords, not in parentheses (e.g., suggest "NLP" for "natural language processing").
2. If the input query covers more than one concept, generate expansions for each one, including their specific subtopics (hyponyms).
3. Suggest keywords for important topics in the field of the input query.
4. Only suggest expansion keywords with more than three words if they are highly relevant.
5. Do not suggest many keywords that include words from the input query.
6. Avoid suggesting broader terms (hypernyms).
7. Sort the suggested keywords by their relevance and importance.

Input Format:

    Provide expansions for:
    <InputQuery>{{input query}}</InputQuery>

Output Format:

    <ExpansionKeywords>{{keyword 1}}, {{keyword 2}}, {{keyword 3}}, {{keyword 4}}, {{keyword 5}}</ExpansionKeywords>

Provide expansions for:
<InputQuery>natural language processing</InputQuery>
"""
        }, {
            "role": "assistant",
            "content": "<ExpansionKeywords>NLP, computational linguistics, natural language understanding, natural language generation, NLU, NLG</ExpansionKeywords>"
        }, {
            "role": "user",
            "content": f"Provide expansions for:\n<InputQuery>information retrieval</InputQuery>"
        },
        {
            "role": "assistant",
            "content": "<ExpansionKeywords>IR, text retrieval, document retrieval, search engine, document ranking, retrieval model, web search, ranked retrieval, semantic search, information extraction</ExpansionKeywords>"
        }, {
            "role": "user",
            "content": f"Provide expansions for:\n<InputQuery>mussel</InputQuery>"
        },
        {
            "role": "assistant",
            "content": "<ExpansionKeywords>mytilidae, bivalve mollusc, bathymodiolus, pteriomorphia, palaeoheterodonta, heterodonta, dreissena polymorpha, mytilus edulis, mytilus galloprovincialis, mytilus trossellus, perna canaliculus, filter-feeder</ExpansionKeywords>"
        },
        {
            "role": "user",
            "content": f"Provide expansions for:\n<InputQuery>{query}</InputQuery>"
        },
    ]
    return messages


def get_synonyms_gpt(query: str, context: str):

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=_get_prompt_with_context(
            query, context) if context else _get_prompt(query),
        temperature=0.0,
        user="advisor",
        seed=10
    )
    return response.choices[0].message.content


def get_synonyms_mistral(query: str, context: str):

    response = llama.chat(
        model="mistral:7b-instruct-v0.3-q8_0",
        messages=_get_prompt_with_context(
            query, context) if context else _get_prompt(query),
        options={
            "seed": 10,
            "temperature": 0.0
        }
    )
    return response["message"]["content"]


def get_synonyms_sentence_transformers(query: str, index: str, model, num_suggs: int = 21):
    query = query.lower().strip()
    query_emb = model.encode(query)
    synonyms = get_suggestions(es, query, query_emb, index, num_suggs)
    return ", ".join(synonyms)


def get_synonyms_bert(query: str, index: str, tokenizer, model, num_suggs: int = 21):
    query = query.lower().strip()
    tok_query = tokenizer(query, return_tensors="pt").to("cuda")
    output = model(**tok_query, output_hidden_states=True)
    embs = []
    for k_tensor in output.last_hidden_state.tolist()[0]:
        embs.append(k_tensor)
    query_emb = np.mean(embs, axis=0)
    synonyms = get_suggestions(
        es, query, query_emb, index, num_suggs)
    return ", ".join(synonyms)


def get_synonyms_fastkwds(query: str, num_suggs: int = 21):
    query = query.lower().strip()
    synonyms = get_suggestions_fastkeywords(es, query, FK_INDEX, num_suggs)
    return ", ".join(synonyms)


def extract_strings_recursive(text, tag):
    start_idx = text.find("<" + tag + ">")
    if start_idx == -1:
        return []

    end_idx = text.find("</" + tag + ">", start_idx)
    res = [text[start_idx+len(tag)+2:end_idx]]
    res += extract_strings_recursive(text[end_idx+len(tag)+3:], tag)

    return res


def get_expanded_queries(model, use_rag=False):
    print(f"Getting expanded queries for {model}")
    with open(f"{DATASET}_queries.txt", "r") as f:
        contents = f.readlines()
        expanded_queries = {}
        for query in contents:
            if use_rag:
                query_emb = sbert.encode(query)
                context_kwds = search_by_embedding(
                    es, query_emb, SBERT_INDEX)[:CONTEXT_SIZE]
                context = ", ".join([i[0] for i in context_kwds])
            else:
                context = ""

            match model:
                case "gpt":
                    expansion = extract_strings_recursive(
                        get_synonyms_gpt(query, context), "ExpansionKeywords")[0]
                case "mistral":
                    expansion = extract_strings_recursive(
                        get_synonyms_mistral(query, context), "ExpansionKeywords")[0]
                case "fastkwds":
                    expansion = get_synonyms_fastkwds(query)
                case "kbir":
                    expansion = get_synonyms_bert(
                        query, KBIR_INDEX, kbir_tok, kbir)
                case "sbert":
                    expansion = get_synonyms_sentence_transformers(
                        query, SBERT_INDEX, sbert)
                case "phrasebert":
                    expansion = get_synonyms_sentence_transformers(
                        query, PHRASEBERT_INDEX, phrasebert)
                case "e5":
                    expansion = get_synonyms_sentence_transformers(
                        query, E5_INDEX, e5)
                case _:
                    raise Exception(f"Model {model} not supported")

            expanded_queries[query] = expansion.lower()

    return list(expanded_queries.values())


def create_synonyms(model_name, rag_enabled=False, context_index=None, context_size=None):
    file_name = f"resources/synonyms/{DATASET}/{model_name}_synonyms"
    if rag_enabled:
        file_name += f"_rag_{context_index}_{context_size}"
    file_name += ".pkl"

    queries = get_expanded_queries(model_name, rag_enabled)
    pkl.dump(queries, open(file_name, "wb"))

    return queries


create_synonyms("gpt")
create_synonyms("mistral")
create_synonyms("gpt", True, SBERT_INDEX, CONTEXT_SIZE)
create_synonyms("mistral", True, SBERT_INDEX, CONTEXT_SIZE)
create_synonyms("fastkwds")
create_synonyms("kbir")
create_synonyms("sbert")
create_synonyms("phrasebert")
create_synonyms("e5")
