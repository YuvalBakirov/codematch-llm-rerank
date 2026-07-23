import os
import config
import pandas as pd
from datetime import datetime
from collections import defaultdict
from core.utils import generate_code_embedding_generic

from tqdm import tqdm

from qdrant_client.http.models import Filter, FieldCondition, MatchAny

import logging


class Metric:
    def __init__(self, name):
        self.name = name
        self.overall = None
        self.by_clone_type = defaultdict(lambda: None)
        self.within_type = defaultdict(lambda: defaultdict(lambda: None))

        self.by_domain = defaultdict(lambda: None)

    def update_overall(self, value):
        self.overall = value

    def update_clone_type(self, clone_type, value):
        self.by_clone_type[clone_type] = value

    def update_within_type(self, clone_type, subtype, value):
        self.within_type[clone_type][subtype] = value

    def update_domain(self, domain, value):
        self.by_domain[domain] = value

    def to_dict(self):
        return {
            "Overall": self.overall,
            "By Clone Type": dict(self.by_clone_type),
            "Within Type": {k: dict(v) for k, v in self.within_type.items()},

            "By Domain": dict(self.by_domain)
        }

class MetricsCollection:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, metric_name):
        self.metrics[metric_name] = Metric(metric_name)

    def update_metric(self, metric_name, overall=None, clone_type=None, subtype=None, domain=None, value=None):

        if metric_name not in self.metrics:
            self.add_metric(metric_name)

        if overall is not None:
            self.metrics[metric_name].update_overall(value)

        if clone_type is not None and subtype is None:
            self.metrics[metric_name].update_clone_type(clone_type, value)

        if clone_type is not None and subtype is not None:
            self.metrics[metric_name].update_within_type(clone_type, subtype, value)

        if domain is not None and subtype is None:
            self.metrics[metric_name].update_domain(domain, value)

    def to_dict(self):
        return {metric_name: metric.to_dict() for metric_name, metric in self.metrics.items()}

    def to_dataframe_old(self):
        rows = []
        for metric_name, metric in self.metrics.items():
            data_dict = metric.to_dict()
            rows.append({"Metric": metric_name, "Type": "Overall", "Subtype": None, "Value": data_dict["Overall"]})
            for clone_type, value in data_dict["By Clone Type"].items():
                rows.append({"Metric": metric_name, "Type": clone_type, "Subtype": None, "Value": value})
            for clone_type, subtypes in data_dict["Within Type"].items():
                for subtype, value in subtypes.items():
                    rows.append({"Metric": metric_name, "Type": clone_type, "Subtype": subtype, "Value": value})
        return pd.DataFrame(rows)
    

    def to_dataframe(self):
        # Initialize a nested dictionary to hold the data
        data = {}

        # Collect Overall metrics
        for metric_name, metric in self.metrics.items():
            if metric.overall is not None:
                data.setdefault('Overall', {})[metric_name] = metric.overall

        # Collect By Clone Type metrics
        for metric_name, metric in self.metrics.items():
            for clone_type, value in metric.by_clone_type.items():
                if value is not None:
                    data.setdefault(clone_type, {})[metric_name] = value

        # Collect Within Type metrics
        for metric_name, metric in self.metrics.items():
            for clone_type, subtypes in metric.within_type.items():
                for subtype, value in subtypes.items():
                    if value is not None:
                        group_name = f"{clone_type} - {subtype}"
                        data.setdefault(group_name, {})[metric_name] = value

        # Collect By Domain metrics
        for metric_name, metric in self.metrics.items():
            for domain, value in metric.by_domain.items():
                if value is not None:
                    group_name = f"Domain - {domain}"
                    data.setdefault(group_name, {})[metric_name] = value

        # Convert the nested dictionary to a DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = 'Group'
        df.reset_index(inplace=True)
        return df


def calculate_tp_tn_fp_fn(filtered_df, similarity_threshold):
    tp = len(filtered_df[(filtered_df['similarity_score'] >= similarity_threshold) & (filtered_df['clone_type'] != 'Non-Clone')])
    fp = len(filtered_df[(filtered_df['similarity_score'] >= similarity_threshold) & (filtered_df['clone_type'] == 'Non-Clone')])
    tn = len(filtered_df[(filtered_df['similarity_score'] < similarity_threshold) & (filtered_df['clone_type'] == 'Non-Clone')])
    fn = len(filtered_df[(filtered_df['similarity_score'] < similarity_threshold) & (filtered_df['clone_type'] != 'Non-Clone')])
    return tp, tn, fp, fn


def calculate_metrics(filtered_df, metrics_collection, similarity_threshold, overall=None, clone_type=None, subtype=None, domain=None):
    """
    Calculates various performance metrics based on the filtered subset of data and updates the metrics collection.

    Args:
        filtered_df (pd.DataFrame): The subset of the DataFrame for which metrics are being calculated, 
                                    filtered by overall data, clone type, sub-type, or domain.
        metrics_collection (MetricsCollection): The collection object where metrics are stored and organized.
        similarity_threshold (float): The threshold value for considering a similarity score as a match (true positive).
        overall (str, optional): The category label for overall metrics, typically 'Overall'.
        clone_type (str, optional): The specific clone type label for metrics calculation, if applicable.
        subtype (str, optional): The specific clone sub-type label within a clone type for metrics calculation, if applicable.
        domain (str, optional): The specific domain label for metrics calculation, if applicable.
    """

    tp, tn, fp, fn = calculate_tp_tn_fp_fn(filtered_df, similarity_threshold)

    num_instances = tp + tn + fp + fn
    total_passed = tp
    total_failed = fp + fn
    avg_similarity = filtered_df['similarity_score'].mean() if num_instances > 0 else 0
    median_similarity = filtered_df['similarity_score'].median() if num_instances > 0 else 0
    accuracy = (tp + tn) / num_instances if num_instances > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = calculate_f1(precision, recall)

    # Update metrics
    metrics_collection.update_metric("Number of Instances", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=num_instances)
    metrics_collection.update_metric("Total Passed", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=total_passed)
    metrics_collection.update_metric("Total Failed", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=total_failed)
    metrics_collection.update_metric("Average similarity_score", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=avg_similarity)
    metrics_collection.update_metric("Median similarity_score", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=median_similarity)
    metrics_collection.update_metric("Accuracy", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=accuracy)
    metrics_collection.update_metric("Precision", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=precision)
    metrics_collection.update_metric("Recall", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=recall)
    metrics_collection.update_metric("F1-Score", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=f1_score)
    metrics_collection.update_metric("True Positives (TP)", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=tp)
    metrics_collection.update_metric("True Negatives (TN)", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=tn)
    metrics_collection.update_metric("False Positives (FP)", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=fp)
    metrics_collection.update_metric("False Negatives (FN)", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=fn)


def calculate_precision(filtered_df, similarity_threshold):
    true_positives = len(filtered_df[filtered_df['similarity_score'] >= similarity_threshold])
    false_positives = len(filtered_df[filtered_df['similarity_score'] < similarity_threshold])
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

def calculate_recall(filtered_df, similarity_threshold):
    true_positives = len(filtered_df[filtered_df['similarity_score'] >= similarity_threshold])
    false_negatives = len(filtered_df[filtered_df['similarity_score'] < similarity_threshold])
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def evaluate_direct_comparison_results(direct_comparison_logger, results_df, model_name, similarity_threshold):
    """
    Evaluates and calculates various metrics based on similarity scores between clones and base codes,
    categorizing the results by overall, clone type, clone sub-type, and domain. The results are saved to a CSV file.

    Args:
        results_df (pd.DataFrame): The DataFrame containing test results with columns including 
                                   'similarity_score', 'clone_type', 'clone_sub_type', and 'domain'.
        model_name (str): The name of the model used, which will be used in the output file path.
        similarity_threshold (float): The threshold value used to determine whether a similarity score qualifies as a "pass."
    """

    metrics_collection = MetricsCollection()

    # Add metrics to the collection
    metrics_collection.add_metric("Number of Instances")
    metrics_collection.add_metric("Total Passed")
    metrics_collection.add_metric("Total Failed")
    metrics_collection.add_metric("Average similarity_score")
    metrics_collection.add_metric("Median similarity_score")
    metrics_collection.add_metric("Accuracy")
    metrics_collection.add_metric("Precision")
    metrics_collection.add_metric("Recall")
    metrics_collection.add_metric("F1-Score")
    metrics_collection.add_metric("True Positives (TP)")
    metrics_collection.add_metric("True Negatives (TN)")
    metrics_collection.add_metric("False Positives (FP)")
    metrics_collection.add_metric("False Negatives (FN)")

    # Step 1: Overall metrics
    calculate_metrics(results_df, metrics_collection, similarity_threshold, overall='Overall')

    # Step 2: By Clone Type
    if 'clone_type' in results_df.columns:
        clone_types = results_df['clone_type'].unique()
        for clone_type in clone_types:
            filtered_df = results_df[results_df['clone_type'] == clone_type]
            calculate_metrics(filtered_df, metrics_collection, similarity_threshold, clone_type=clone_type)

        # Step 3: Within Clone Type
        if 'clone_sub_type' in results_df.columns:
            for clone_type in clone_types:
                sub_types = results_df[results_df['clone_type'] == clone_type]['clone_sub_type'].unique()
                for sub_type in sub_types:
                    filtered_df = results_df[(results_df['clone_type'] == clone_type) & (results_df['clone_sub_type'] == sub_type)]
                    calculate_metrics(filtered_df, metrics_collection, similarity_threshold, clone_type=clone_type, subtype=sub_type)

    # Step 4: By Domain
    if 'domain' in results_df.columns:
        domains = results_df['domain'].unique()
        for domain in domains:
            filtered_df = results_df[results_df['domain'] == domain]
            calculate_metrics(filtered_df, metrics_collection, similarity_threshold, domain=domain)

    # Convert metrics to DataFrame
    metrics_df = metrics_collection.to_dataframe()

    # Ensure the Output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.join(output_dir, 'direct-clone')
    os.makedirs(output_dir, exist_ok=True)

    # Get the current datetime for the filename
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_time = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

    # Save the DataFrame to a CSV file in the Output directory
    output_filename = os.path.join(output_dir, f"{model_name}_direct_clone_comparison_evaluation_{current_time}.csv")
    metrics_df.to_csv(output_filename, index=False)
    direct_comparison_logger.info(f"Evaluation saved to: {output_filename}")
    
    return metrics_df


def direct_clone_comparison_test(client, test_code_csv, model, tokenizer, model_name, similarity_threshold=0.5):
    """
    Run tests to compare embeddings against expected results and save results to a CSV file.

    Args:
        client (QdrantClient): The client that allows the connection to Qdrant Database
        test_code_csv (str): Path to the CSV file containing test cases.
        model: The pre-loaded model to use for embedding creation.
        tokenizer: The tokenizer associated with the model.
        model_name (str): The model name.
        similarity_threshold (float): The threshold for determining a "pass" on similarity.
    """
    try:
        direct_comparison_logger = logging.getLogger("benchmark.direct_comparison")
        tests_df = pd.read_csv(test_code_csv)

        results = []

        for _, row in tqdm(tests_df.iterrows(), desc="Running Direct Clone Comparison...", total=len(tests_df)):
            clone_code_id = row['clone_code_id']
            base_code_id = row["base_code_id"]
            task = row['task']
            domain = row['domain'] if 'domain' in row else "Unknown"
            subdomain = row['subdomain'] if 'subdomain' in row else "Unknown"
            clone_language = row['clone_language']
            clone_type = row["clone_type"] if "clone_type" in row else "Unknown"
            clone_sub_type = row["clone_sub_type"] if "clone_sub_type" in row else "Unknown"
            code = row['code']

            
            # embedding = generate_code_embedding(code, model, tokenizer)
            embedding = generate_code_embedding_generic(direct_comparison_logger, code, model, tokenizer)

            if embedding is not None:
                
                if base_code_id:

                    direct_code_result = client.search(
                        collection_name=config.BENCHMARK_COLLECTION_NAME,
                        query_vector=embedding,
                        limit=1,
                        query_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="base_code_id",
                                    match=MatchAny(any=[base_code_id])
                                )
                            ]
                        )
                    )

                    if direct_code_result:
                        result = direct_code_result[0]
                        similarity_score = result.score

                        result_payload = result.payload
                        # result_domain = result_payload.get("domain", "Unknown")
                        # result_subdomain = result_payload.get("subdomain", "Unknown")
                        

                    else:
                        similarity_score = "Not found"
                        # result_domain = "Unknown"
                        # result_subdomain = "Unknown"

                else:
                    similarity_score = "No Index Given"
                    # result_domain = "Unknown"
                    # result_subdomain = "Unknown"


                direct_comparison_logger.info(f"Clone Code ID: {clone_code_id}, Base Code ID: {base_code_id}, "
                            f"Clone Type: {clone_type}, Clone Sub-Type: {clone_sub_type}, "
                            f"similarity_score: {similarity_score}")
                

                results.append({
                    "clone_code_id": clone_code_id,
                    "base_code_id": base_code_id,
                    "task": task,
                    "clone_language": clone_language,
                    "similarity_score": similarity_score,
                    "domain": domain,
                    "subdomain": subdomain,
                    "clone_type": clone_type,
                    "clone_sub_type": clone_sub_type
                })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Remove columns where all values are "Unknown"
        results_df = results_df.loc[:, (results_df != 'Unknown').any(axis=0)]

        # Get the current datetime for the filename
        current_time = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

        # Ensure the Output directory exists
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)

        output_dir = os.path.join(output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        output_dir = os.path.join(output_dir, 'direct-clone')
        os.makedirs(output_dir, exist_ok=True)

        # Save results to a CSV file in the Output directory
        output_filename = os.path.join(output_dir, f"{model_name}_direct_clone_comparison_scores_{current_time}.csv")
        results_df.to_csv(output_filename, index=False)
        direct_comparison_logger.info(f"Results saved to: {output_filename}")

        # Evaluate results and update metrics
        evaluate_direct_comparison_results(direct_comparison_logger, results_df, model_name, similarity_threshold)

    except Exception as e:
        exp = f"Error running tests: {e}"
        direct_comparison_logger.error(exp)
        raise Exception(exp)
    


def global_clone_search(client, test_code_csv, model, tokenizer, model_name, similarity_threshold=0.5):
    """
    Run tests to compare embeddings against expected results and save results to a CSV file.

    Args:
        client (QdrantClient): The client that allows the connection to Qdrant Database
        test_code_csv (str): Path to the CSV file containing test cases.
        model: The pre-loaded model to use for embedding creation.
        tokenizer: The tokenizer associated with the model.
        model_name (str): The model name.
        similarity_threshold (float): The threshold for determining a "pass" on similarity.
    """
    try:
        global_search_logger = logging.getLogger("benchmark.global_search")
        tests_df = pd.read_csv(test_code_csv)

        results = []

        for _, row in tqdm(tests_df.iterrows(), desc="Running Global Clone Search...", total=len(tests_df)):
            clone_code_id = row['clone_code_id']
            base_code_id = row["base_code_id"]  # Desired base_code_id
            task = row['task']
            domain = row['domain'] if 'domain' in row else "Unknown"
            subdomain = row['subdomain'] if 'subdomain' in row else "Unknown"
            clone_language = row['clone_language']
            clone_type = row["clone_type"] if "clone_type" in row else "Unknown"
            clone_sub_type = row["clone_sub_type"] if "clone_sub_type" in row else "Unknown"
            code = row['code']

            # embedding = generate_code_embedding(code, model, tokenizer)
            embedding = generate_code_embedding_generic(global_search_logger, code, model, tokenizer)

            if embedding is not None:

                global_search_result = client.search(
                    collection_name=config.BENCHMARK_COLLECTION_NAME,
                    query_vector=embedding,
                    limit=5
                )

                if global_search_result:

                    for index, result in enumerate(global_search_result):
                        result_payload = result.payload

                        test_id = f"{clone_code_id}_{index+1}"

                        result_id = result_payload.get("base_code_id")
                        result_language = result_payload.get("language")
                        result_task = result_payload.get("task")
                        result_domain = result_payload.get("domain", "Unknown")
                        result_subdomain = result_payload.get("subdomain", "Unknown")

                        similarity_score = result.score


                        global_search_logger.info(f"Clone Code ID: {clone_code_id}, Result ID: {result_id}, "
                                  f"Clone Type: {clone_type}, Clone Sub-Type: {clone_sub_type}, "
                                  f"similarity_score: {similarity_score}")

                        results.append({
                            "test_id": test_id,
                            "clone_code_id": clone_code_id,
                            "desired_base_code_id": base_code_id,  # Include desired base_code_id
                            "clone_language": clone_language,
                            "clone_task": task,
                            "clone_domain": domain,
                            "clone_subdomain": subdomain,
                            "clone_type": clone_type,
                            "clone_sub_type": clone_sub_type,
                            "similarity_score": similarity_score,
                            "base_code_id": result_id,
                            "base_language": result_language,
                            "base_task": result_task,
                            "base_domain": result_domain,
                            "base_subdomain": result_subdomain
                        })

                else:
                    global_search_logger.info(f"No results found for Clone Code ID: {clone_code_id}")

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Remove columns where all values are "Unknown"
        results_df = results_df.loc[:, (results_df != 'Unknown').any(axis=0)]

        # Get the current datetime for the filename
        current_time = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

        # Ensure the Output directory exists
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)

        output_dir = os.path.join(output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        output_dir = os.path.join(output_dir, 'global-clone')
        os.makedirs(output_dir, exist_ok=True)

        # Save results to a CSV file in the Output directory
        output_filename = os.path.join(output_dir, f"{model_name}_global_clone_search_scores_{current_time}.csv")
        results_df.to_csv(output_filename, index=False)
        global_search_logger.info(f"Results saved to: {output_filename}")

        # Evaluate results and update metrics
        evaluate_global_search_results(global_search_logger, results_df, model_name)

    except Exception as e:
        exp = f"Error running tests: {e}"
        global_search_logger.error(exp)
        raise Exception(exp)


def evaluate_global_search_results(global_search_logger, results_df, model_name):
    """
    Evaluate the results of global clone search and update metrics.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results of the global clone search.
        model_name (str): Name of the model used.
    """
    metrics_collection = MetricsCollection()
    metrics_collection.add_metric('Number of Tests')
    metrics_collection.add_metric('Number of Top 5 Hits')
    metrics_collection.add_metric('Top 5 Hit Rate')
    metrics_collection.add_metric('Number of 1st in Top 5')
    metrics_collection.add_metric('First Hit Rate')


     # Step 1: Overall metrics
    calculate_global_metrics(results_df, metrics_collection, overall='Overall')

    # Step 2: By Clone Type
    if 'clone_type' in results_df.columns:
        clone_types = results_df['clone_type'].unique()
        for clone_type in clone_types:
            filtered_df = results_df[results_df['clone_type'] == clone_type]
            calculate_global_metrics(filtered_df, metrics_collection, clone_type=clone_type)

        # Step 3: Within Clone Type
        if 'clone_sub_type' in results_df.columns:
            for clone_type in clone_types:
                sub_types = results_df[results_df['clone_type'] == clone_type]['clone_sub_type'].unique()
                for sub_type in sub_types:
                    filtered_df = results_df[(results_df['clone_type'] == clone_type) & (results_df['clone_sub_type'] == sub_type)]
                    calculate_global_metrics(filtered_df, metrics_collection, clone_type=clone_type, subtype=sub_type)

    # Step 4: By Domain
    if 'clone_domain' in results_df.columns:
        domains = results_df['clone_domain'].unique()
        for domain in domains:
            filtered_df = results_df[results_df['clone_domain'] == domain]
            calculate_global_metrics(filtered_df, metrics_collection, domain=domain)

    # Convert metrics to DataFrame
    metrics_df = metrics_collection.to_dataframe()
    # metrics_df = pd.DataFrame(metrics_collection)

    # Ensure the Output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.join(output_dir, 'global-clone')
    os.makedirs(output_dir, exist_ok=True)

    # Get the current datetime for the filename
    current_time = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

    # Save the DataFrame to a CSV file in the Output directory
    output_filename = os.path.join(output_dir, f"{model_name}_global_clone_search_evaluation_{current_time}.csv")
    metrics_df.to_csv(output_filename, index=False)
    global_search_logger.info(f"Evaluation saved to: {output_filename}")

    return metrics_df


def calculate_global_metrics(filtered_df, metrics_collection, overall=None, clone_type=None, subtype=None, domain=None):
    """
    Calculate global search metrics for a filtered DataFrame and update the metrics collection.

    Args:
        filtered_df (pd.DataFrame): The filtered DataFrame to calculate metrics on.
        metrics_collection (MetricsCollection): The metrics collection to update.
        overall (str): Whether the given dataframe is of all the data.
        clone_type (str): The clone type.
        subtype (str): The clone subtype.
        domain (str): The domain.
    """
    # Number of Tests
    number_of_tests = filtered_df[['clone_code_id', 'clone_language']].drop_duplicates().shape[0]

    # Initialize counters
    number_of_top5_hits = 0
    number_of_first_hits = 0

    # Group by clone_code_id
    # if subtype == 'Different Language':
    #     grouped = filtered_df.groupby('clone_language')
    # else:
    #     grouped = filtered_df.groupby('clone_code_id')

    grouped = filtered_df.groupby(['clone_language', 'clone_code_id'])

    for clone_code_id, group in grouped:
        desired_base_code_id = group['desired_base_code_id'].iloc[0]

        # Get the top 5 base_code_id values
        top_base_code_ids = group['base_code_id'].values

        # Check if desired_base_code_id is in top 5 base_code_id
        if desired_base_code_id in top_base_code_ids:
            number_of_top5_hits += 1

            # Check if desired_base_code_id is in the first position
            if top_base_code_ids[0] == desired_base_code_id:
                number_of_first_hits += 1

    # Calculate rates
    top5_hit_rate = number_of_top5_hits / number_of_tests if number_of_tests > 0 else 0
    first_hit_rate = number_of_first_hits / number_of_tests if number_of_tests > 0 else 0

    # Update metrics
    metrics_collection.update_metric("Number of Tests", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=number_of_tests)
    metrics_collection.update_metric("Number of Top 5 Hits", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=number_of_top5_hits)
    metrics_collection.update_metric("Top 5 Hit Rate", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=top5_hit_rate)
    metrics_collection.update_metric("Number of 1st in Top 5", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=number_of_first_hits)
    metrics_collection.update_metric("First Hit Rate", overall=overall, clone_type=clone_type, subtype=subtype, domain=domain, value=first_hit_rate)
