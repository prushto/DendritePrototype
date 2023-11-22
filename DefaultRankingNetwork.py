from unit_interfaces import RankingNetwork
import heapq
from tqdm import tqdm


class DefaultRankingNetwork(RankingNetwork):
    """
        Simple RankingNetwork implementation that uses a binary heap to
        sort the documents by relevance score for each query and
        returns a list of the R-most relevant documents for each query.
        (as a dict where key is query_id).
        """
    def ranker(self, scores_for_all_docs, pid_vector, R=1000):
        ranked_documents = {}
        for qid, scores in tqdm(scores_for_all_docs.items(), desc="Ranking"):
            score_pid_pairs = [(-score, pid) for score, pid in zip(scores, pid_vector)]
            heapq.heapify(score_pid_pairs)
            top_documents = []
            for _ in range(min(R, len(score_pid_pairs))):
                _, pid = heapq.heappop(score_pid_pairs)
                top_documents.append(pid)
            ranked_documents[qid] = top_documents
        return ranked_documents
