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
from validation import travshacl
#from Transformation_updated import transform
from transform_new import transform

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


def process_rules(file, prefix, rdf_data, predictions_folder, kg):
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
    q_filter = f"""SELECT DISTINCT Head, COUNT(*) AS num FROM rules WHERE {pca_col} < 1 AND {pca_col} > 0.50 GROUP BY Head ORDER BY num DESC"""
    head_df = sqldf(q_filter, locals())

    if head_df.empty:
        print("No rules found meeting the PCA confidence threshold criteria.")
        return pd.DataFrame(), Graph()

    final_result_df = pd.DataFrame()

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
                AND {pca_col} < 1 AND {pca_col} > 0.50 
                ORDER BY {confidence_col} DESC
            """
            rule_subset = sqldf(q2, locals())

        print(f"Found {len(rule_subset)} rules for predicate {head_val}")

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

    # Save enriched knowledge graph
    enriched_kg_path = os.path.join(os.path.dirname(predictions_folder), f"{kg}_EnrichedKG", f"{kg}_Enriched_KG.nt")
    os.makedirs(os.path.dirname(enriched_kg_path), exist_ok=True)
    g.serialize(destination=enriched_kg_path, format='nt')
    print(f"\nEnriched knowledge graph saved to: {enriched_kg_path}")

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

    print(f"Configuration loaded:\n"
          f"- Prefix: {prefix}\n"
          f"- Rules file: {rules}\n"
          f"- RDF file: {rdf}\n"
          f"- Predictions folder: {predictions_folder}\n"
          f"- Constraints folder: {constraints}")

    return prefix, rules, rdf, path, predictions_folder, constraints, kg


if __name__ == '__main__':
    try:
        start_time = time.time()
        print("Starting symbolic prediction generation...")

        # Initialize configuration
        input_config = 'input.json'
        prefix, rulesfile, rdf_data, path, predictions_folder, constraints, kg = initialize(input_config)

        # Process rules and generate predictions
        print("\nProcessing rules and generating predictions...")
        result_df, enriched_kg = process_rules(rulesfile, prefix, rdf_data, predictions_folder, kg)

        # Validate results
        print("\nValidating results...")
        val_results = travshacl(enriched_kg, constraints, kg)

        # Transform results
        print("\nTransforming results...")
        transform(enriched_kg, kg)

        # Print execution time
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        print("Process completed successfully!")

    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        raise