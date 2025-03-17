from TravSHACL import parse_heuristics, GraphTraversal, ShapeSchema

def travshacl(enrichedKG, constraints, kg):
    prio_target = 'TARGET'  # shapes with target definition are preferred, alternative value: ''
    prio_degree = 'IN'  # shapes with a higher in-degree are prioritized, alternative value 'OUT'
    prio_number = 'BIG'  # shapes with many constraints are evaluated first, alternative value 'SMALL'

    output_path = constraints + '/result_' +kg

    shape_schema = ShapeSchema(
        schema_dir=constraints,
        endpoint=enrichedKG,
        endpoint_user=None,  # username if validating a private endpoint
        endpoint_password=None,  # password if validating a private endpoint
        graph_traversal=GraphTraversal.DFS,
        heuristics=parse_heuristics(prio_target + ' ' + prio_degree + ' ' + prio_number),
        use_selective_queries=True,
        max_split_size=256,
        # output_dir='./Constraints/result/',  # directory where the output files will be stored
        output_dir= output_path,
        order_by_in_queries=False,
        # sort the results of SPARQL queries in order to ensure the same order across several runs
        save_outputs=True  # save outputs to output_dir, alternative value: False
    )

    result = shape_schema.validate()  # validate the SHACL shape schema
    print(f"Constraint Validation Result saved to {output_path}")
    return result
