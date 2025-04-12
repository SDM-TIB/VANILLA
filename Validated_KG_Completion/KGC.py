import numpy as np
import pandas as pd
import torch
import json
import logging
import os
import gc
from typing import List, Dict
from matplotlib import pyplot as plt
from pykeen.pipeline import pipeline, plot_losses
from pykeen import predict
from pykeen.triples import TriplesFactory

def main():
    # Set default configuration file path
    config_file = 'input_KGC.json'
    
    # Load configuration from JSON file
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file {config_file} not found!")
        return
    except json.JSONDecodeError:
        print(f"Error: Configuration file {config_file} contains invalid JSON!")
        return
    
    # Set up logging
    log_level = config.get('log_level', 'INFO')
    logging.basicConfig(level=getattr(logging, log_level))
    logger = logging.getLogger(__name__)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Extract configuration parameters
    kg_path = config.get('kg_path')
    results_path = config.get('results_path')
    models = config.get('models', ['TransE', 'TransH', 'TransD', 'ComplEx', 'RotatE', 'TuckER'])
    num_epochs = config.get('num_epochs', 100)
    embedding_dim = config.get('embedding_dim', 50)
    batch_size = config.get('batch_size', 1024)
    random_seed = config.get('random_seed', 1235)
    create_inverse_triples = config.get('create_inverse_triples', False)
    filtered_negative_sampling = config.get('filtered_negative_sampling', True)
    save_splits = config.get('save_splits', True)
    
    if not kg_path:
        logger.error("Knowledge graph path is required in the configuration file!")
        return
    
    if not results_path:
        logger.error("Results path is required in the configuration file!")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading knowledge graph from {kg_path}")
    try:
        tf, triple_data, entity_label, relation_label = load_dataset(kg_path, create_inverse_triples)
        
        # Split into train and test
        training, testing = tf.split(random_state=random_seed)
        
        # Save splits if requested
        if save_splits:
            training_triples = pd.DataFrame(training.triples)
            training_triples.to_csv(os.path.join(results_path, 'train'), index=False, sep='\t', header=False)
            logger.info(f"Saved training split to {os.path.join(results_path, 'train')}")
            
            testing_triples = pd.DataFrame(testing.triples)
            testing_triples.to_csv(os.path.join(results_path, 'test'), index=False, sep='\t', header=False)
            logger.info(f"Saved testing split to {os.path.join(results_path, 'test')}")
        
        # Train and evaluate models
        logger.info(f"Training the following models: {', '.join(models)}")
        for m in models:
            logger.info(f"Training {m} model")
            model_results_path = os.path.join(results_path, m)
            
            # Create model directory if it doesn't exist
            os.makedirs(model_results_path, exist_ok=True)
            
            # Train model
            model, result = create_model(
                tf_training=training, 
                tf_testing=testing, 
                embedding=m, 
                n_epoch=num_epochs, 
                path=results_path,
                embedding_dim=embedding_dim,
                batch_size=batch_size,
                random_seed=random_seed,
                filtered_negative_sampling=filtered_negative_sampling
            )
            
            # Create loss plot
            plotting(result, m, results_path)
            
            logger.info(f"Finished training {m} model")
        
        logger.info("All models trained successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def load_dataset(name, create_inverse_triples=False):
    """
    Load and preprocess the knowledge graph from a TSV/NT file.

    Args:
        name (str): Path to the dataset file
        create_inverse_triples (bool): Whether to create inverse triples

    Returns:
        tuple: (TriplesFactory, raw_data, entity_labels, relation_labels)
    """
    logger = logging.getLogger(__name__)
    try:
        triple_data = open(name, encoding='utf-8').read().strip()
        data = np.array([triple.split('\t') for triple in triple_data.split('\n')])
        tf_data = TriplesFactory.from_labeled_triples(triples=data, create_inverse_triples=create_inverse_triples)
        entity_label = tf_data.entity_to_id.keys()
        relation_label = tf_data.relation_to_id.keys()

        logger.info(f"Loaded dataset with {len(entity_label)} entities and {len(relation_label)} relations")
        return tf_data, triple_data, entity_label, relation_label

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def create_model(tf_training, tf_testing, embedding, n_epoch, path, 
                embedding_dim=50, batch_size=1024, random_seed=1235,
                filtered_negative_sampling=True):
    """
    Train KGE models with required hyperparameters
    
    Args:
        tf_training: Training triples factory
        tf_testing: Testing triples factory
        embedding: Model name
        n_epoch: Number of training epochs
        path: Path to save results
        embedding_dim: Dimension of embeddings
        batch_size: Batch size for training
        random_seed: Random seed for reproducibility
        filtered_negative_sampling: Whether to use filtered negative sampling
        
    Returns:
        tuple: (trained_model, results)
    """
    logger = logging.getLogger(__name__)
    try:
        results = pipeline(
            training=tf_training,
            testing=tf_testing,
            model=embedding,
            training_loop='sLCWA',
            model_kwargs=dict(embedding_dim=embedding_dim),
            negative_sampler_kwargs=dict(filtered=filtered_negative_sampling),
            # Training configuration
            training_kwargs=dict(
                num_epochs=n_epoch,
                use_tqdm_batch=True,
                drop_last=False,
                batch_size=batch_size
            ),
            # Runtime configuration
            random_seed=random_seed,
        )
        model = results.model
        results.save_to_directory(os.path.join(path, embedding))  # save results to the directory
        logger.info(f"Model {embedding} trained and saved successfully")
        return model, results
    except Exception as e:
        logger.error(f"Error creating model {embedding}: {str(e)}")
        raise

def plotting(result, m, results_path):
    """
    Plotting observed losses per KGE model
    
    Args:
        result: Pipeline result
        m: Model name
        results_path: Path to save plot
    """
    logger = logging.getLogger(__name__)
    try:
        plot_losses(result)
        plt.savefig(os.path.join(results_path, m, "loss_plot.png"), dpi=300)
        plt.close()  # Close the plot to free memory
        logger.info(f"Loss plot saved for model {m}")
    except Exception as e:
        logger.error(f"Error creating loss plot for model {m}: {str(e)}")

def tail_prediction(model, head, relation, training):
    """
    Predict tail entity
    
    Args:
        model: Trained model
        head: Head entity
        relation: Relation entity
        training: Training triples factory
        
    Returns:
        DataFrame: Prediction results
    """
    pred = predict.predict_target(model=model, head=head, relation=relation, triples_factory=training).df
    return pred

if __name__ == "__main__":
    main()