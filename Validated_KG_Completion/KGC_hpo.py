import json
import numpy as np
import pandas as pd
import torch
import logging
import gc
from typing import List, Dict
import os
import os.path
from matplotlib import pyplot as plt
from pykeen.pipeline import pipeline, plot_losses
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory

def main():
    # Set default configuration file path
    config_file = 'input_KGC_hpo.json'
    
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
    
    # Get configuration parameters
    dataset_path = config.get('dataset_path')
    output_dir = config.get('output_dir')
    models = config.get('models', ['TransE'])
    n_trials = config.get('n_trials', 30)
    train_ratio = config.get('train_ratio', 0.8)
    test_ratio = config.get('test_ratio', 0.1)
    val_ratio = config.get('val_ratio', 0.1)
    random_state = config.get('random_state', 1234)
    num_epochs = config.get('num_epochs', 100)
    
    if not dataset_path:
        logger.error("Dataset path is required in the configuration file!")
        return
    
    if not output_dir:
        logger.error("Output directory is required in the configuration file!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split dataset
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        tf, triple_data, entity_label, relation_label = load_dataset(dataset_path)
        
        # Calculate split ratios
        split_ratios = [train_ratio, test_ratio, val_ratio]
        
        # Ensure ratios sum to 1
        if sum(split_ratios) != 1.0:
            logger.warning(f"Split ratios sum to {sum(split_ratios)}, normalizing to 1.0")
            split_ratios = [r/sum(split_ratios) for r in split_ratios]
        
        training_tf, testing_tf, validation_tf = tf.split(split_ratios, random_state=random_state)
        
        # Run HPO experiments
        logger.info(f"Running HPO experiments for models: {', '.join(models)}")
        run_hpo_experiments(
            training_tf=training_tf,
            testing_tf=testing_tf,
            validation_tf=validation_tf,
            models=models,
            output_dir=output_dir,
            n_trials=n_trials,
            num_epochs=num_epochs
        )
        
        logger.info("HPO experiments completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def load_dataset(name: str):
    """
    Load and preprocess the dataset from a TSV file.

    Args:
        name (str): Path to the dataset file

    Returns:
        tuple: (TriplesFactory, raw_data, entity_labels, relation_labels)
    """
    logger = logging.getLogger(__name__)
    try:
        triple_data = open(name, encoding='utf-8').read().strip()
        data = np.array([triple.split('\t') for triple in triple_data.split('\n')])
        tf_data = TriplesFactory.from_labeled_triples(triples=data, create_inverse_triples=True)
        entity_label = tf_data.entity_to_id.keys()
        relation_label = tf_data.relation_to_id.keys()

        logger.info(f"Loaded dataset with {len(entity_label)} entities and {len(relation_label)} relations")
        return tf_data, triple_data, entity_label, relation_label

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def get_model_specific_params(model: str) -> Dict:
    """
    Get model-specific hyperparameter configurations.
    """
    # Common parameters for all models
    common_params = {
        'embedding_dim': {
            'type': 'int',
            'low': 50,
            'high': 200,
            'q': 50
        }
    }

    # Model-specific parameters
    model_params = {
        'TransE': {
            **common_params,
            'scoring_fct_norm': {
                'type': 'int',
                'low': 1,
                'high': 2,
            }
        },
        'ComplEx': {
            **common_params,
            'regularizer': {
                'type': 'categorical',
                'choices': [None, 'LP']
            }
        },
        'RotatE': {
            **common_params
        },
        'DistMult': {
            **common_params
        },
        'CompGCN':{
            **common_params,
        },
        'ConvE': {
            **common_params,
            'input_channels': {
                'type': 'int',
                'low': 1,
                'high': 3,
            },
            'output_channels': {
                'type': 'int',
                'low': 32,
                'high': 128,
                'q': 32,
            },
            'kernel_height': {
                'type': 'int',
                'low': 2,
                'high': 5,
            },
            'kernel_width': {
                'type': 'int',
                'low': 2,
                'high': 5,
            },
            'embedding_height': {
                'type': 'int',
                'low': 5,
                'high': 20,
                'q': 5,
            },
            'embedding_width': {
                'type': 'int',
                'low': 5,
                'high': 20,
                'q': 5,
            },
            'input_dropout': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5,
            },
            'feature_map_dropout': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5,
            },
            'output_dropout': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5,
            }
        },
        'TuckER': {
            'embedding_dim': {
                'type': 'int',
                'low': 50,
                'high': 200,
                'q': 50
            },
            'relation_dim': {
                'type': 'int',
                'low': 50,
                'high': 200,
                'q': 50
            },
            'dropout_0': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5,
            },
            'dropout_1': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5,
            },
            'dropout_2': {
                'type': 'float',
                'low': 0.0,
                'high': 0.5,
            },
            'apply_batch_normalization': {
                'type': 'categorical',
                'choices': [True, False]
            }
        }
    }

    return model_params.get(model, common_params)

def run_hpo_experiments(training_tf: TriplesFactory,
                       testing_tf: TriplesFactory,
                      validation_tf: TriplesFactory,
                       models: List[str],
                       output_dir: str,
                       n_trials: int = 30,
                       num_epochs: int = 100):
    """
    Run hyperparameter optimization experiments for different models.
    """
    logger = logging.getLogger(__name__)
    
    for model in models:
        logger.info(f"Starting HPO for {model}")

        try:
            results = hpo_pipeline(
                training=training_tf,
                testing=testing_tf,
                validation=validation_tf,
                model=model,
                training_loop='sLCWA',

                # Model hyperparameter ranges - model specific
                model_kwargs_ranges=get_model_specific_params(model),

                # Training hyperparameter ranges
                training_kwargs_ranges={
                    'batch_size': {
                        'type': 'int',
                        'low': 128,
                        'high': 512,
                        'q': 128,
                    }
                },

                # Negative sampling configuration
                negative_sampler_kwargs={
                    'filtered': True,
                },

                negative_sampler_kwargs_ranges={
                    'num_negs_per_pos': {
                        'type': 'int',
                        'low': 1,
                        'high': 10,
                    },
                },

                # Training configuration
                training_kwargs=dict(
                    num_epochs=num_epochs,
                    use_tqdm=True,
                ),

                # HPO configuration
                n_trials=n_trials,
                metric='hits@1',
                direction='maximize',
            )

            # Save results
            save_path = os.path.join(output_dir, f"{model}")
            results.save_to_directory(save_path)

            # Log best trial results
            best_trial = results.study.best_trial
            logger.info(f"Best trial value for {model}: {best_trial.value}")
            logger.info("Best hyperparameters:")
            for key, value in best_trial.params.items():
                logger.info(f"{key}: {value}")

        except Exception as e:
            logger.error(f"Error in HPO for {model}: {str(e)}")
            continue

if __name__ == "__main__":
    main()