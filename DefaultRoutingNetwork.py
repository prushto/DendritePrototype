import nltk
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from unit_interfaces import RoutingNetwork


class DefaultRoutingNetwork(RoutingNetwork):
    """
    Simple RoutingUnit implementation that tokenizes document corpus and queries
    using NLTK's punkt tokenizer models.
    """

    def process(self, corpus_file_path, query_file_path):
        pid_list, vocab, token_passage_matrix = self.process_corpus(corpus_file_path)
        queries = self.process_queries(query_file_path, vocab)
        return pid_list, vocab, token_passage_matrix, queries


    def process_corpus(self, file_path):
        try:
            # Check if 'punkt' tokenizer models are already downloaded
            nltk.data.find('tokenizers/punkt')
            print("'punkt' tokenizer models are already downloaded.")
        except LookupError:
            # If not, then download them
            print("Downloading 'punkt' tokenizer models...")
            nltk.download('punkt')
            print("'punkt' tokenizer models downloaded.")
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        pid_list = [] # to store Passage IDs
        all_tokens = set()

        for line in lines:
            pid, passage = line.split('\t', 1)
            pid_list.append(pid)
            tokens = word_tokenize(passage)
            all_tokens.update(tokens)

        # vocab is a dict where key is token and value is the index position of the token
        vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}

        token_passage_matrix = np.zeros((len(vocab), len(lines)), dtype=int)
        # Populate a matrix such that row_num is idx of token in vocab and col is a document
        # Sets value to 1 if the token is present in that passage
        for col_idx, line in tqdm(enumerate(lines), desc="Corpus memorization"):
            _, passage = line.split('\t', 1)
            tokens = set(word_tokenize(passage))
            for token in tokens:
                row_idx = vocab[token]
                token_passage_matrix[row_idx][col_idx] = 1

        return pid_list, vocab, token_passage_matrix

    def process_queries(self, file_path, vocab):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        queries = {}
        for line in lines:
            qid, query = line.split('\t', 1)
            tokens = set(word_tokenize(query))
            query_vector = np.zeros(len(vocab), dtype=int)
            for token in tokens:
                if token in vocab:
                    query_vector[vocab[token]] = 1
            queries[qid] = query_vector
        return queries
