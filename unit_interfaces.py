from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from numpy import ndarray

"""
This file specifies the signatures for the abstract functions 
required by the RoutingNetwork, ScoringUnit and RankingNetwork classes, 
which in turn are required by the DendritePrototype object.
Adjust the config file to specify which classes you want to use.
"""


class RoutingNetwork(ABC):
    @abstractmethod
    def process(self, corpus_file_path, query_file_path) -> Tuple[List, Dict, ndarray, Dict]:
        """
        Argument:   file_path to document corpus.
                    file_path to queries.
        Returns:    passage IDs -> List of passage IDs in order of corpus.
                    vocab -> Dict(token: idx) results of tokenization of corpus.
                    token_passage_matrix -> ndarray(rows: token_idx, cols: passages in order of corpus)
                    where 0 means token not present in document, 1 means present.
                    queries -> Dict(query_id: list[token presence/absence]), where the list_idx corresponds
                    to the vocab, and, as in  the token_passage_matrix above 0 means token not present in
                    document, 1 means present.
        """
        pass

class ScoringUnit(ABC):
    @abstractmethod
    def score(self, queries, token_passage_matrix) -> Dict:
        """
        Argument:   queries -> Dict(query_id: list[token presence/absence]), where the list_idx corresponds
                    to the vocab, and, as in  the token_passage_matrix above 0 means token not present in
                    document, 1 means present.
                    token_passage_matrix -> ndarray(rows: token_idx, cols: passages in order of corpus)
                    where 0 means token not present in document, 1 means present.
        Returns:    scores -> Dict(query_id: list[scores by doc in order of corpus])
        """
        pass


class RankingNetwork(ABC):
    @abstractmethod
    def ranker(self, scores, pid_list, R=1000) -> List:
        """
        Argument:   scores -> Dict(query_id: list[scores by doc in order of corpus])
                    passage IDs -> List of passage IDs in order of corpus
                    R -> ranking threshold hyperparameter
        Returns:    top documents - > Dict(query_id: list[passage IDs in descending order of relevance, with length of R])
        """
        pass
