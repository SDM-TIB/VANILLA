[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# VANILLA: Validated Knowledge Graph Completion - A Normalization-based Framework for Integrity, Link Prediction, and Logical Accuracy

![Graphical Abstract](images/DesignPattern(b)-VANILLA.png)
## 🔍 Overview


## ⚙️ Setup Instructions

3. **Configure input**
   Modify `input.json` to select the benchmark KG and rule/constraint files.

---

## 🚀 Running the Pipeline

### 1. Symbolic Predictions & Constraint Validation
Run the script to generate predictions and validate them:
```bash
python Symbolic_predictions.py
```
This will:
- Generate inferred predictions using logical rules.
- Validate them against SHACL constraints.
- Output:
  - Transformed KGs
  - Constraint validation reports in `Constraints/`
  - Predictions in `Predictions/`
---

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
