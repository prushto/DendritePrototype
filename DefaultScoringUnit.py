from scipy.sparse import csr_matrix
from tqdm import tqdm
from unit_interfaces import ScoringUnit


class DefaultScoringUnit(ScoringUnit):
    """
    Simple RoutingNetwork implementation that uses scipy sparse vectors
    to compute dot product between the embedding of every query and document
    in the corpus.
    """
    def find_most_relevant_documents(self, queries, token_passage_matrix):
        if not isinstance(token_passage_matrix, csr_matrix):
            token_passage_matrix = csr_matrix(token_passage_matrix)

        scores_for_all_docs = {}
        for qid, query_vector in tqdm(queries.items(), desc="Scoring"):
            if not isinstance(query_vector, csr_matrix):
                query_vector = csr_matrix(query_vector)
            scores = token_passage_matrix.T.dot(query_vector.T)
            scores_for_all_docs[qid] = scores.toarray()

        return scores_for_all_docs
