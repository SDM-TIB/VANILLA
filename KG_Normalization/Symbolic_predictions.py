"""
Unified script for handling symbolic predictions with and without constants
"""
import json
import pandas as pd
from rdflib.plugins.sparql.processor import SPARQLResult
from pandasql import sqldf
from rdflib import Graph, URIRef
import re
import os
import time
from Validation import travshacl
from Normalization_transform import transform
import logging

def load_graph(file):
    """Load RDF graph from file"""
    g1 = Graph()
    with open(file, "r", encoding="utf-8") as rdf_file:
        lines = rdf_file.readlines()
        for line_number, line in enumerate(lines, start=1):
            try:
                g1.parse(data=line, format="nt")
            except Exception as e:
                print(f"Error parsing line {line_number}: {e}")
    return g1


def detect_rule_type(rules_df):
    """
    Detect if rules contain constants by analyzing the Body and Head columns
    Returns: 'constant' or 'variable'
    """
    # Sample a few rules to check for patterns
    sample_size = min(10, len(rules_df))
    sample_rules = rules_df.sample(n=sample_size) if len(rules_df) > sample_size else rules_df

    # Check if any rule contains a word that's not a variable (doesn't start with ?)
    # and is not a predicate (not in the middle of a triple)
    has_constants = False
    for _, row in sample_rules.iterrows():
        body_parts = row['Body'].split()
        head_parts = row['Head'].split()

        # Check every third word in body (object position in triples)
        for i in range(2, len(body_parts), 3):
            if i < len(body_parts) and not body_parts[i].startswith('?'):
                has_constants = True
                break

        # Check object position in head
        if len(head_parts) >= 3 and not head_parts[2].startswith('?'):
            has_constants = True
            break

    return 'constant' if has_constants else 'variable'


def rdflib_query_with_constants(rule_df, prefix_query, rdf_data, head_val, predictions_folder):
    """Handle rules with constants"""
    all_results = []

    for _, rule in rule_df.iterrows():
        fun_var = rule['Functional_variable']
        body = rule['Body']
        head = rule['Head']

        # Process body and head
        words = body.split()
        head_split = head.split()
        prefix = 'ex:'
        pattern = re.compile(r'^\w+$')

        # Modify terms with prefix
        modified_list = [prefix + item if pattern.match(item) else item for item in words]
        modified_head = [prefix + item if pattern.match(item) else item for item in head_split]

        # Create SPARQL query components
        body_parts = [modified_list[i:i + 3] for i in range(0, len(modified_list), 3)]
        body_str = ' .\n'.join(' '.join(part) for part in body_parts) + ' .'
        new_head = ' '.join(modified_head)

        # Generate appropriate SPARQL query based on functional variable
        if fun_var == '?a':
            query = f"""
                PREFIX ex: <{prefix_query}>
                SELECT DISTINCT ?a WHERE {{
                    {body_str}
                    FILTER(!EXISTS {{{new_head}}})
                }}"""
        else:
            h = new_head.replace("?a", "?a1")
            query = f"""
                PREFIX ex: <{prefix_query}>
                SELECT DISTINCT ?a WHERE {{
                    {body_str}
                    {h} .
                    FILTER(?a1 != ?a)
                    FILTER(!EXISTS {{{new_head}}})
                }}"""

        print(f"Executing query:\n{query}")
        logger.info(f"Executing query:\n{query}")

        # Execute query
        file_triple = load_graph(file=rdf_data)
        qres = file_triple.query(query)

        # Process results for this rule
        if qres:
            for row in qres:
                subject = str(row[0]).replace(prefix_query, '')
                all_results.append({
                    'subject': subject,
                    'predicate': head_val,
                    'object': head_split[2]
                })

    # Create final DataFrame
    if all_results:
        result_df = pd.DataFrame(all_results)
        return result_df
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['subject', 'predicate', 'object'])


def rdflib_query_without_constants(rule_df, prefix_query, rdf_data, head_val, predictions_folder):
    """Handle rules without constants"""
    all_results = []

    for _, rule in rule_df.iterrows():
        fun_var = rule['Functional_variable']
        body = rule['Body']
        head = rule['Head']

        # Process body and head
        words = body.split()
        head_split = head.split()
        prefix = 'ex:'
        pattern = re.compile(r'^\w+$')

        # Modify terms with prefix
        modified_list = [prefix + item if pattern.match(item) else item for item in words]
        modified_head = [prefix + item if pattern.match(item) else item for item in head_split]

        new_head = ' '.join(modified_head)

        # Extract variables from head
        head_vars = re.findall(r'\?[a-z]', head)
        subject_var = head_vars[0].replace('?', '')
        object_var = head_vars[1].replace('?', '')

        # Create SPARQL query components
        body_parts = [modified_list[i:i + 3] for i in range(0, len(modified_list), 3)]
        body_str = ' .\n'.join(' '.join(part) for part in body_parts) + ' .'

        # Generate appropriate SPARQL query based on functional variable
        if fun_var == '?a':
            query = f"""
                PREFIX ex: <{prefix_query}>
                SELECT DISTINCT ?{subject_var} ?{object_var} WHERE {{
                    {body_str}
                    FILTER(!EXISTS {{{new_head}}})
                }}"""
        else:
            h = new_head.replace("?a", "?a1")
            query = f"""
                PREFIX ex: <{prefix_query}>
                SELECT DISTINCT ?{subject_var} ?{object_var} WHERE {{
                    {body_str}
                    {h} .
                    FILTER(?a1 != ?{subject_var})
                    FILTER(!EXISTS {{{new_head}}})
                }}"""

        print(f"Executing query:\n{query}")
        logger.info(f"Executing query:\n{query}")

        # Execute query
        file_triple = load_graph(file=rdf_data)
        qres = file_triple.query(query)

        # Process results
        for row in qres:
            subject = str(row[0]).replace(prefix_query, '')
            object_val = str(row[1]).replace(prefix_query, '')
            all_results.append({
                'subject': subject,
                'predicate': head_val,
                'object': object_val
            })

    # Create final DataFrame
    if all_results:
        result_df = pd.DataFrame(all_results)
        return result_df
    else:
        return pd.DataFrame(columns=['subject', 'predicate', 'object'])


# def process_rules(file, prefix, rdf_data, predictions_folder, kg, pca_threshold):
#     """Process rules and generate predictions based on rule type"""
#     print(f"Reading rules from {file}")
#     rules = pd.read_csv(file)
#
#     # Verify required columns exist
#     required_columns = ['Body', 'Head', 'PCA_Confidence', 'Pca_Confidence', 'Pca Confidence',
#                         'Standard_Confidence', 'Std_Confidence', 'Standard Confidence']
#     found_columns = [col for col in required_columns if col in rules.columns]
#     if not any(col in rules.columns for col in ['PCA_Confidence', 'Pca_Confidence', 'Pca Confidence']):
#         raise ValueError("Neither 'PCA_Confidence' nor 'Pca_Confidence' column found in rules file")
#     if not any(col in rules.columns for col in ['Standard_Confidence', 'Std_Confidence', 'Standard Confidence']):
#         raise ValueError("Neither 'Standard_Confidence' nor 'Std_Confidence' column found in rules file")
#
#     print(f"Found columns: {found_columns}")
#     rule_type = detect_rule_type(rules)
#     print(f"Detected rule type: {rule_type}")
#
#     # Identify confidence columns
#     confidence_col = 'Standard_Confidence' if 'Standard_Confidence' in rules.columns else 'Std_Confidence'
#     pca_col = 'PCA_Confidence' if 'PCA_Confidence' in rules.columns else 'Pca_Confidence'
#
#     # First filter rules that meet PCA confidence threshold
#     q_filter = f"""SELECT DISTINCT Head, COUNT(*) AS num FROM rules WHERE {pca_col} < 1 AND {pca_col} > {pca_threshold} GROUP BY Head ORDER BY num DESC"""
#     head_df = sqldf(q_filter, locals())
#
#     if head_df.empty:
#         print("No rules found meeting the PCA confidence threshold criteria.")
#         return pd.DataFrame(), Graph()
#
#     final_result_df = pd.DataFrame()
#
#     # Initialize RDF graph
#     g = Graph()
#     g.parse(rdf_data, format='nt')
#
#     for _, val in head_df.iterrows():
#         head = val['Head']
#         if head and isinstance(head, str):  # Check if head is valid
#             head_val = head.split()[1]
#             print(f"\nProcessing rules for predicate: {head_val}")
#
#             # Select rules for current head predicate with PCA confidence threshold
#             q2 = f"""
#                 SELECT * FROM rules
#                 WHERE Head LIKE '%{head}%'
#                 AND {pca_col} < 1 AND {pca_col} > {pca_threshold}
#                 ORDER BY {confidence_col} DESC
#             """
#             rule_subset = sqldf(q2, locals())
#
#         print(f"Found {len(rule_subset)} rules for predicate {head_val}")
#         try:
#             if logger:
#                 logger.info(f"Found {len(rule_subset)} rules for predicate {head_val}")
#         except NameError:
#             # If logger is not defined, just continue without logging
#             pass
#
#         # Process rules based on type
#         if rule_type == 'constant':
#             result_df = rdflib_query_with_constants(rule_subset, prefix, rdf_data, head_val, predictions_folder)
#         else:
#             result_df = rdflib_query_without_constants(rule_subset, prefix, rdf_data, head_val, predictions_folder)
#
#         if not result_df.empty:
#             print(f"Generated {len(result_df)} predictions for predicate {head_val}")
#
#             # Add predictions to graph
#             for _, row in result_df.iterrows():
#                 subject = URIRef(prefix + row['subject'])
#                 predicate = URIRef(prefix + row['predicate'])
#                 object = URIRef(prefix + row['object'])
#                 g.add((subject, predicate, object))
#
#             final_result_df = pd.concat([final_result_df, result_df], ignore_index=True)
#
#             # Save individual predicate results
#             if not os.path.exists(predictions_folder):
#                 os.makedirs(predictions_folder)
#             result_df.to_csv(f"{predictions_folder}/{head_val}.tsv", sep='\t', index=False, header=None)
#         else:
#             print(f"No predictions generated for predicate {head_val}")
#
#     # Print total number of predictions
#     total_predictions = len(final_result_df)
#     print(f"\n=== SUMMARY ===")
#     print(f"Total number of predictions generated: {total_predictions}")
#
#     # Only log if logger is properly initialized
#     try:
#         if logger:
#             logger.info(f"Total number of predictions generated: {total_predictions}")
#     except NameError:
#         # If logger is not defined, just continue without logging
#         pass
#
#     # Save enriched knowledge graph
#     enriched_kg_path = os.path.join(os.path.dirname(predictions_folder), f"{kg}_EnrichedKG", f"{kg}_Enriched_KG.nt")
#     os.makedirs(os.path.dirname(enriched_kg_path), exist_ok=True)
#     g.serialize(destination=enriched_kg_path, format='nt')
#     print(f"Enriched knowledge graph saved to: {enriched_kg_path}")
#     try:
#         if logger:
#             logger.info(f"Enriched KG successfully generated")
#             logger.info(f"Enriched KG saved to: {enriched_kg_path}")
#     except NameError:
#         # If logger is not defined, just continue without logging
#         pass
#
#     return final_result_df, g

def process_rules(file, prefix, rdf_data, predictions_folder, kg, pca_threshold):
    """Process rules and generate predictions based on rule type"""
    print(f"Reading rules from {file}")
    rules = pd.read_csv(file)

    # Verify required columns exist
    required_columns = ['Body', 'Head', 'PCA_Confidence', 'Pca_Confidence', 'Pca Confidence',
                        'Standard_Confidence', 'Std_Confidence', 'Standard Confidence']
    found_columns = [col for col in required_columns if col in rules.columns]
    if not any(col in rules.columns for col in ['PCA_Confidence', 'Pca_Confidence', 'Pca Confidence']):
        raise ValueError("Neither 'PCA_Confidence' nor 'Pca_Confidence' column found in rules file")
    if not any(col in rules.columns for col in ['Standard_Confidence', 'Std_Confidence', 'Standard Confidence']):
        raise ValueError("Neither 'Standard_Confidence' nor 'Std_Confidence' column found in rules file")

    print(f"Found columns: {found_columns}")
    rule_type = detect_rule_type(rules)
    print(f"Detected rule type: {rule_type}")

    # Identify confidence columns
    confidence_col = 'Standard_Confidence' if 'Standard_Confidence' in rules.columns else 'Std_Confidence'
    pca_col = 'PCA_Confidence' if 'PCA_Confidence' in rules.columns else 'Pca_Confidence'

    # First filter rules that meet PCA confidence threshold
    q_filter = f"""SELECT DISTINCT Head, COUNT(*) AS num FROM rules WHERE {pca_col} < 1 AND {pca_col} > {pca_threshold} GROUP BY Head ORDER BY num DESC"""
    head_df = sqldf(q_filter, locals())

    if head_df.empty:
        print("No rules found meeting the PCA confidence threshold criteria.")
        return pd.DataFrame(), Graph()

    final_result_df = pd.DataFrame()
    total_utilized_rules = 0  # Counter for total rules utilized
    utilized_rules_per_predicate = {}  # Keep track of rules per predicate

    # Initialize RDF graph
    g = Graph()
    g.parse(rdf_data, format='nt')

    for _, val in head_df.iterrows():
        head = val['Head']
        if head and isinstance(head, str):  # Check if head is valid
            head_val = head.split()[1]
            print(f"\nProcessing rules for predicate: {head_val}")

            # Select rules for current head predicate with PCA confidence threshold
            q2 = f"""
                SELECT * FROM rules 
                WHERE Head LIKE '%{head}%' 
                AND {pca_col} < 1 AND {pca_col} > {pca_threshold} 
                ORDER BY {confidence_col} DESC
            """
            rule_subset = sqldf(q2, locals())

        print(f"Found {len(rule_subset)} rules for predicate {head_val}")
        total_utilized_rules += len(rule_subset)  # Add rules to total count
        utilized_rules_per_predicate[head_val] = len(rule_subset)  # Track per predicate

        try:
            if logger:
                logger.info(f"Found {len(rule_subset)} rules for predicate {head_val}")
        except NameError:
            # If logger is not defined, just continue without logging
            pass

        # Process rules based on type
        if rule_type == 'constant':
            result_df = rdflib_query_with_constants(rule_subset, prefix, rdf_data, head_val, predictions_folder)
        else:
            result_df = rdflib_query_without_constants(rule_subset, prefix, rdf_data, head_val, predictions_folder)

        if not result_df.empty:
            print(f"Generated {len(result_df)} predictions for predicate {head_val}")

            # Add predictions to graph
            for _, row in result_df.iterrows():
                subject = URIRef(prefix + row['subject'])
                predicate = URIRef(prefix + row['predicate'])
                object = URIRef(prefix + row['object'])
                g.add((subject, predicate, object))

            final_result_df = pd.concat([final_result_df, result_df], ignore_index=True)

            # Save individual predicate results
            if not os.path.exists(predictions_folder):
                os.makedirs(predictions_folder)
            result_df.to_csv(f"{predictions_folder}/{head_val}.tsv", sep='\t', index=False, header=None)
        else:
            print(f"No predictions generated for predicate {head_val}")

    # Print total number of predictions and rules
    total_predictions = len(final_result_df)
    print(f"\n=== SUMMARY ===")
    print(f"Total number of rules utilized: {total_utilized_rules}")
    print(f"Total number of predictions generated: {total_predictions}")
    print(
        f"Prediction efficiency (predictions per rule): {total_predictions / total_utilized_rules if total_utilized_rules > 0 else 0:.2f}")

    # Print breakdown by predicate
    print("\n=== RULES AND PREDICTIONS BY PREDICATE ===")
    for predicate, rule_count in utilized_rules_per_predicate.items():
        # Count predictions for this predicate
        predicate_predictions = len(
            final_result_df[final_result_df['predicate'] == predicate]) if not final_result_df.empty else 0
        print(f"{predicate}: {rule_count} rules, {predicate_predictions} predictions")

    # Only log if logger is properly initialized
    try:
        if logger:
            logger.info(f"Total number of rules utilized: {total_utilized_rules}")
            logger.info(f"Total number of predictions generated: {total_predictions}")
            logger.info(
                f"Prediction efficiency: {total_predictions / total_utilized_rules if total_utilized_rules > 0 else 0:.2f} predictions per rule")
    except NameError:
        # If logger is not defined, just continue without logging
        pass

    # Save enriched knowledge graph
    enriched_kg_path = os.path.join(os.path.dirname(predictions_folder), f"{kg}_EnrichedKG", f"{kg}_Enriched_KG.nt")
    os.makedirs(os.path.dirname(enriched_kg_path), exist_ok=True)
    g.serialize(destination=enriched_kg_path, format='nt')
    print(f"Enriched knowledge graph saved to: {enriched_kg_path}")
    try:
        if logger:
            logger.info(f"Enriched KG successfully generated")
            logger.info(f"Enriched KG saved to: {enriched_kg_path}")
    except NameError:
        # If logger is not defined, just continue without logging
        pass

    return final_result_df, g

def initialize(input_config):
    """Initialize configuration from input file"""
    print(f"Reading configuration from {input_config}")
    with open(input_config, "r") as input_file_descriptor:
        input_data = json.load(input_file_descriptor)

    prefix = input_data['prefix']
    kg = input_data['KG']
    path = os.path.join('KG', input_data['KG'])
    rules = os.path.join('Rules', input_data['rules_file'])
    rdf = os.path.join(path, input_data['rdf_file'])
    predictions_folder = os.path.join('Predictions', input_data['KG'] + "_predictions")
    constraints = os.path.join('Constraints',input_data['constraints_folder'])
    pca_threshold = input_data['pca_threshold']

    print(f"Configuration loaded:\n"
          f"- Prefix: {prefix}\n"
          f"- Rules file: {rules}\n"
          f"- RDF file: {rdf}\n"
          f"- Predictions folder: {predictions_folder}\n"
          f"- Constraints folder: {constraints}\n"
          f"- PCA Threshold: {pca_threshold}")

    logger.info(f"Configuration loaded:\n "
          f"- Prefix: {prefix}\n"
          f"- Rules file: {rules}\n"
          f"- RDF file: {rdf}\n"
          f"- Predictions folder: {predictions_folder}\n"
          f"- Constraints folder: {constraints}\n"
          f"- PCA Threshold: {pca_threshold}")

    return prefix, rules, rdf, path, predictions_folder, constraints, kg, pca_threshold


if __name__ == '__main__':
    try:
        start_time = time.time()
        print("Starting symbolic prediction generation...")

        # Initialize configuration
        input_config = 'input.json'

        # Create logs directory if it doesn't exist
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Set up logging with file output and rotation
        log_level = 'INFO'  # Default to INFO level
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_file = os.path.join(logs_dir, f'symbolic_predictions_{timestamp}.log')

        # Configure logging to write to both console and file
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # This sends output to console too
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Starting symbolic prediction process with config: {input_config}")

        prefix, rulesfile, rdf_data, path, predictions_folder, constraints, kg, pca_threshold = initialize(input_config)

        # Process rules and generate predictions
        print("\nProcessing rules and generating predictions...")
        logger.info(f"Processing rules from {rulesfile} and generating predictions")
        result_df, enriched_kg = process_rules(rulesfile, prefix, rdf_data, predictions_folder, kg, pca_threshold)

        # Validate results
        # print("\nValidating results...")
        # val_results = travshacl(enriched_kg, constraints, kg)
        #
        # # Transform results
        # print("\nTransforming results...")
        # transform(enriched_kg, kg)

        # Print execution time
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        print("Process completed successfully!")

    except Exception as e:
        error_msg = f"Error occurred during execution: {str(e)}"
        print(f"\n{error_msg}")
        if 'logger' in locals():
            logger.error(error_msg, exc_info=True)  # Logs the full traceback
        raise