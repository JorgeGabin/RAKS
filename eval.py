import json
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pyterrier as pt

from collections import defaultdict

from utils import (
    get_docs_ranking_pyterrier,
)

DATASET = "trec_fair_2021"
# DATASET = "trec_7_8" # Uncomment if you want to reproduce the TREC 7_8 experiments

QUERIES = f"{DATASET}_topics_eval.jsonl"

SBERT_INDEX = f"{DATASET}-kwds-sbert"
CONTEXT_SIZE = 10


pt.init()

index_ref = pt.IndexRef.of(f"./resources/indices/{DATASET}")
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

def load_synonyms(model_name, rag_enabled=False, context_index=None, context_size=None):
    file_name = f"./resources/synonyms/{DATASET}/{model_name}_synonyms"
    if rag_enabled:
        file_name += f"_rag_{context_index}_{context_size}"
    file_name += ".pkl"

    queries = pkl.load(open(file_name, "rb"))

    return queries


def prepare_query_data(query, suggestions, num_sugg):
    suggestions = suggestions.split(", ")
    query_expanded = f"{query} {' '.join(suggestions[:num_sugg])}"

    return query_expanded


def compute_scores(suggestions, num_docs, rel_docs, scores):
    for name, expanded_query in suggestions.items():
        ranking = get_docs_ranking_pyterrier(expanded_query, bm25, num_docs)
        score = _score(ranking, rel_docs)

        for key, value in score.items():
            scores[name][key].append(value)

    return scores


def plot_line_charts(results, systems, markers, colors, lines, x_axis, metric, metric_display_name):
    for system, marker, color, line in zip(systems, markers, colors, lines):
        plt.plot(x_axis, results[system][metric], label=system,
                 marker=marker, color=color, linestyle=line)

    plt.xlabel("Number of synonyms")
    plt.ylabel(metric_display_name)
    plt.xticks(num_suggs)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
               loc='lower left', ncols=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"./resources/figures/{DATASET}/{metric}.svg")
    plt.close()


num_suggs = list(range(1, 16))
num_docs = 1000

files = {
    "gpt": load_synonyms("gpt"),
    "mistral": load_synonyms("mistral"),
    "rag_gpt": load_synonyms("gpt", True, SBERT_INDEX, CONTEXT_SIZE),
    "rag_mistral": load_synonyms("mistral", True, SBERT_INDEX, CONTEXT_SIZE),
    "fastkwds": load_synonyms("fastkwds"),
    "kbir": load_synonyms("kbir"),
    "sbert": load_synonyms("sbert"),
    "phrasebert": load_synonyms("phrasebert"),
    "e5": load_synonyms("e5"),
}

results = defaultdict(lambda: defaultdict(list))
systems = ["Default", "Manual", "GPT4", "Mistral", "RAG GPT4",
           "RAG Mistral", "FastKeywords", "KBIR", "SBERT", "PhraseBERT", "E5", "RM3"]
markers = ["o", ".", "v", "^", "s", "p", "*", "x", "d", "<", ">", "+", "D"]
lines = ["-", "--", "-.", ":"]*3
colors = [
    (1.0, 0.31, 0.35, 0.7),   # Bold Red - #FF4F5A
    (1.0, 0.55, 0.42, 0.7),   # Vibrant Peach - #FF8C6A
    (0.49, 0.86, 0.34, 0.7),  # Bright Green - #7ED957
    (1.0, 0.70, 0.0, 0.7),    # Bright Yellow-Orange - #FFB400
    (0.35, 0.36, 0.85, 0.7),  # Bold Blue-Violet - #5A5CD9
    (1.0, 0.17, 0.57, 0.7),   # Hot Pink - #FF2B92
    (0.24, 0.69, 0.62, 0.7),  # Rich Teal - #3EB1A0
    (0.77, 0.42, 0.42, 0.7),  # Muted Dark Pink - #C56B6B
    (1.0, 0.43, 0.57, 0.7),   # Vivid Pink - #FF6F92
    (0.18, 0.64, 0.89, 0.7),  # Bright Cyan - #2DA3E3
    (0.65, 0.43, 0.83, 0.7),  # Vivid Purple - #A46DD3
    (0.91, 0.19, 0.55, 0.7)   # Vivid Magenta - #E82F8D
]

scores = defaultdict(lambda: defaultdict(list))
for num_sugg in num_suggs:
    rm3 = bm25 >> pt.rewrite.RM3(index_ref, fb_terms=num_sugg) >> bm25

    with open(QUERIES, "r") as f:
        for i, line in enumerate(f):
            query_data = json.loads(line)
            query = query_data["title"].lower().strip()
            rel_docs = query_data["rel_docs"]
            keywords = ", ".join(query_data["keywords"])

            suggestions = {
                "Default": query,
                "Manual": prepare_query_data(query, keywords, num_sugg),
                "GPT4": prepare_query_data(query, files["gpt"][i], num_sugg),
                "Mistral": prepare_query_data(query, files["mistral"][i], num_sugg),
                "RAG GPT4": prepare_query_data(query, files["rag_gpt"][i], num_sugg),
                "RAG Mistral": prepare_query_data(query, files["rag_mistral"][i], num_sugg),
                "FastKeywords": prepare_query_data(query, files["fastkwds"][i], num_sugg),
                "KBIR": prepare_query_data(query, files["kbir"][i], num_sugg),
                "SBERT": prepare_query_data(query, files["sbert"][i], num_sugg),
                "PhraseBERT": prepare_query_data(query, files["phrasebert"][i], num_sugg),
                "E5": prepare_query_data(query, files["e5"][i], num_sugg),
                "RM3": query
            }

            compute_scores(suggestions, num_docs, rel_docs, scores)

        for name in systems:
            for metric in ["recall", "rr", "ap"]:
                metric_score = np.mean([scores[name][metric]])
                results[name][metric].append(metric_score)

        for name in systems:
            recall_zero = len([s for s in scores[name]["recall"]
                              if not s]) / len(scores[name]["recall"])
            results[name]["recall_zero"].append(recall_zero)


plot_line_charts(results, systems, markers, colors, lines,
                 num_suggs, "recall", "R@10")
plot_line_charts(results, systems, markers, colors,
                 lines, num_suggs, "rr", "MRR@10")
plot_line_charts(results, systems, markers, colors,
                 lines, num_suggs, "ap", "MAP@1000")
plot_line_charts(results, systems, markers, colors, lines,
                 num_suggs, "recall_zero", "R$_0$@10")
