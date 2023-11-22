from typing import Optional
import json
from DefaultRoutingNetwork import DefaultRoutingNetwork
from DefaultScoringUnit import DefaultScoringUnit
from DefaultRankingNetwork import DefaultRankingNetwork
import unit_interfaces
import importlib

"""
See readme.md for details on usage.
"""
class DendritePrototype:
    def __init__(self, corpus_filepath, query_filepath, rank_num, evaluate,
                 labels_filepath: Optional = None,
                 output_filepath: Optional = None,
                 routing_network: Optional["unit_interfaces.RoutingNetwork"] = None,
                 scoring_unit: Optional["unit_interfaces.ScoringUnit"] = None,
                 ranking_network: Optional["unit_interfaces.RankingNetwork"] = None):
        self.corpus = corpus_filepath
        self.queries = query_filepath
        self.rank_num = rank_num
        self.evaluate = evaluate
        self.labels = labels_filepath
        self.output = output_filepath
        self.vocab = None
        self.token_passage_matrix = None
        self.pid_vector = None
        self.routing_network = routing_network
        self.scoring_unit = scoring_unit
        self.ranking_network = ranking_network

    @staticmethod
    def evaluate_predictions(ranked_documents, labels, log_file_path, R):
        total_relevant = 0
        total_retrieved_relevant = 0

        with open(log_file_path, 'w') as log_file:
            for qid, ranked_pids in ranked_documents.items():
                actual_pids = set(labels.get(qid, {}).keys())
                total_relevant += len(actual_pids)
                retrieved_relevant = set(ranked_pids[:R]).intersection(actual_pids)
                total_retrieved_relevant += len(retrieved_relevant)
                log_file.write(f"Query ID: {qid}\n")
                log_file.write(f"Top {R} Ranked Document IDs: {ranked_pids[:R]}\n")
                log_file.write(f"Actual Relevant Document IDs: {list(actual_pids)}\n")
                retrieval_success = "Success" if len(retrieved_relevant) > 0 else "Failure"
                log_file.write(f"Retrieval Status: {retrieval_success}\n\n")
        print("Total Correct Retrievals: ", total_retrieved_relevant)
        print("Total Relevant Labels:", total_relevant)
        recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
        return recall

    def run(self):
        self.pid_vector, self.vocab, self.token_passage_matrix, queries_enc = self.routing_network.process(self.corpus, self.queries)
        print("Total Number of Documents:", len(self.pid_vector))
        print("Vocabulary Size:", len(self.vocab))
        scores_for_all_docs = self.scoring_unit.score(queries_enc, self.token_passage_matrix)
        ranked_documents = self.ranking_network.ranker(scores_for_all_docs, self.pid_vector, self.rank_num)

        # Evaluation if applicable
        if self.evaluate:
            with open(self.labels, 'r', encoding='utf-8') as file:
                labels = json.load(file)
            recall = self.evaluate_predictions(ranked_documents, labels, self.output, self.rank_num)
            print(f"Recall: {recall}")


def load_class_from_module(module_path, class_name):
    try:
        if module_path and class_name:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)()
        else:
            raise ValueError("Module path and class name must be provided")
    except ImportError:
        print(f"Error: The module '{module_path}' could not be found. Using default instead.")
    except AttributeError:
        print(f"Error: The class '{class_name}' could not be found in the module '{module_path}'. Using default instead.")
    except ValueError as ve:
        print(f"Configuration Error: {ve}. Using default instead.")
    return None


def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    routing_network = load_class_from_module(
        config.get('routing_network', {}).get('module'),
        config.get('routing_network', {}).get('class')) or DefaultRoutingNetwork()

    scoring_unit = load_class_from_module(
        config.get('scoring_unit', {}).get('module'),
        config.get('scoring_unit', {}).get('class')) or DefaultScoringUnit()

    ranking_network = load_class_from_module(
        config.get('ranking_network', {}).get('module'),
        config.get('ranking_network', {}).get('class')) or DefaultRankingNetwork()

    corpus_filepath = config.get('corpus_filepath')
    query_filepath = config.get('query_filepath')
    rank_num = config.get('rank_num')
    evaluate = 'labels_filepath' in config is not None
    labels_filepath = config.get('labels_filepath')
    output_filepath = config.get('output_filepath')
    print(f"Corpus filepath: {corpus_filepath}")
    print(f"Query filepath: {query_filepath}")
    print(f"Ranking threshold: {rank_num}")
    print(f"Routing network: {routing_network}")
    print(f"Scoring unit: {scoring_unit}")
    print(f"Ranking network: {ranking_network}")
    if evaluate is not None:
        print(f"In evaluation mode using labels: {labels_filepath}")
        if output_filepath is None:
            print(f"No output filepath for evaluation results.")
        else:
            print(f"Evaluation results output to: {output_filepath}")


    dp = DendritePrototype(
        config['corpus_filepath'],
        config['query_filepath'],
        config['rank_num'],
        evaluate,
        labels_filepath=config.get('labels_filepath'),
        output_filepath=config.get('output_filepath'),
        routing_network=routing_network,
        scoring_unit=scoring_unit,
        ranking_network=ranking_network
    )
    dp.run()

if __name__ == "__main__":
    main()