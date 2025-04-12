## VANILLA: Validated KG Completion

![Graphical Abstract](images/DesignPattern(b)VANILLA.png)
## 🔍 Overview

The design pattern components of VANILLA for Validated KG Completion process. <br>
VANILLA utilizes the Normalized and Validate KG to show the impact of KG normalization on the
downstream task of KG completion using link prediction. <br>

## 🚀 Running the Pipeline of Validates KG Completion

1. **Configure input**
   Modify `input.json` to select the benchmark KG and rule/constraint files.
```json
{
  "kg_path": "path_to_your_dataset/TransformedKG_YAGO3-10.tsv",
  "results_path": "path_to_your_dataset/TransformedKG",
  "models": ["TuckER"],
  "num_epochs": 100,
  "embedding_dim": 50,
  "batch_size": 32,
  "random_seed": 1235,
  "create_inverse_triples": false,
  "filtered_negative_sampling": true,
  "save_splits": true,
  "log_level": "INFO",
  "create_inverse_triples": false
}
```
2. **Executing KG Normalization**

```python
python KGC.py
```

## 🚀 Running the Pipeline of Validates KG Completion with Hyperparameter Optimization

1. **Configure input**
   Modify `input.json` to select the benchmark KG and rule/constraint files.
```json
{
  "dataset_path": "path_to_your_dataset/DB100K.tsv",
  "output_dir": "path_to_your_dataset/CompGCN-HPO",
  "models": ["CompGCN"],
  "n_trials": 10,
  "train_ratio": 0.8,
  "test_ratio": 0.1,
  "val_ratio": 0.1,
  "random_state": 1234,
  "num_epochs": 100,
  "log_level": "INFO"
}
```
2. **Executing KG Normalization**

```python
python KGC_hpo.py
```

## 📈 Evaluation Metrics

We evaluate KG completion using embedding models:
- **TransE**, **TransH**, **TransD**
- **RotatE**, **ComplEx**, **TuckER**
- **CompGCN**

Metrics reported:
- Hits@1, Hits@3, Hits@5, Hits@10
- Mean Reciprocal Rank (MRR)

---

## 🧠 Graphical Summary

The VANILLA framework integrates **symbolic rules**, **domain constraints**, and **neural embeddings** for high-quality knowledge graph completion. It identifies valid and invalid triples using evolving logical constraints and employs numerical models to infer missing links, ensuring semantic consistency and logical soundness in the normalized KG.

---

## 📄 License

This project is licensed under the terms of the [LICENSE.txt](LICENSE.txt).

## Authors
VANILLA has been developed by members of the Scientific Data Management Group at TIB, as an ongoing research effort.
The development is co-ordinated and supervised by Maria-Esther Vidal.
We strongly encourage you to report any issues you have with VANILLA.
Please, use the GitHub issue tracker to do so.
VANILLA has been implemented in joint work by Disha Purohit, and Yashrajsinh Chudasama.
