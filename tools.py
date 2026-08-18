import math
import collections
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r'\w+', text.lower())

def compute_tf(tokens: List[str]) -> dict:
    """Compute Term Frequency for a list of tokens."""
    tf = collections.Counter(tokens)
    length = len(tokens)
    return {word: count / length for word, count in tf.items()}

def compute_idf(corpus: List[List[str]]) -> dict:
    """Compute Inverse Document Frequency across a corpus."""
    n_docs = len(corpus)
    idf = {}
    all_words = set(word for doc in corpus for word in doc)
    for word in all_words:
        doc_count = sum(1 for doc in corpus if word in doc)
        idf[word] = math.log(n_docs / (1 + doc_count))
    return idf

def retrieve_information(query: str, db_path: str = "knowledge_base.txt", top_k: int = 4) -> List[str]:
    """
    Retrieve top_k relevant chunks from a text file using a custom TF-IDF implementation.
    """
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error("Knowledge base file not found.")
        return []

    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    if not chunks:
        return []

    corpus_tokens = [tokenize(chunk) for chunk in chunks]
    idf = compute_idf(corpus_tokens)

    query_tokens = tokenize(query)
    query_tf = compute_tf(query_tokens)
    query_vec = {word: query_tf[word] * idf.get(word, 0) for word in query_tokens}

    scores = []
    for i, doc_tokens in enumerate(corpus_tokens):
        doc_tf = compute_tf(doc_tokens)
        doc_vec = {word: doc_tf[word] * idf.get(word, 0) for word in set(doc_tokens)}

        dot_product = sum(query_vec.get(w, 0) * doc_vec.get(w, 0) for w in query_vec)
        mag_query = math.sqrt(sum(v**2 for v in query_vec.values()))
        mag_doc = math.sqrt(sum(v**2 for v in doc_vec.values()))

        score = dot_product / (mag_query * mag_doc) if mag_query and mag_doc else 0.0
        scores.append((score, chunks[i]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scores[:top_k] if score > 0]
