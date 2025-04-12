## VANILLA - KG Normalization

![KG Normalization](https://raw.githubusercontent.com/SDM-TIB/VANILLA/main/images/DesignPattern(a)VANILLA.png "KG Normalization")
## 🔍 Overview

The design pattern components of VANILLA for KG Normalization process. <br>
VANILLA demonstrates the use of symbolic rules in conjunction with symbolic constraint validation to
normalize the KGs and eliminate anomalies. <br>
Normalized KGs improve the predictive performance of numerical inductive learning approaches.

## 🚀 Running the Pipeline of KG Normalization

1. **Configure input**
   Modify `input.json` to select the benchmark KG and rule/constraint files.
```json
{
  "KG": "FrenchRoyalty",
  "prefix": "http://FrenchRoaylty.org/",
  "rules_file": "french_royalty.csv",
  "rdf_file": "french_royalty.nt",
  "constraints_folder": "FrenchRoaylty",
  "log_level": "INFO",
  "pca_threshold": 0.75
}
```
2. **Executing KG Normalization**

```python
python Symbolic_predictions.py
```

This will:
- Generate inferred predictions using symbolic rules.
- Validate them against SHACL constraints.
- Output:
  - Transformed KGs
  - Constraint validation reports in `Constraints/`
  - Predictions in `Predictions/`

## 🧠 Graphical Summary

The VANILLA framework integrates **symbolic rules**, **domain constraints** for knowledge graph normalization. <br>

---

## 📄 License

This project is licensed under the terms of the [LICENSE.txt](LICENSE.txt).

## Authors
VANILLA has been developed by members of the Scientific Data Management Group at TIB, as an ongoing research effort.
The development is co-ordinated and supervised by Maria-Esther Vidal.
We strongly encourage you to report any issues you have with VANILLA.
Please, use the GitHub issue tracker to do so.
VANILLA has been implemented in joint work by Disha Purohit, and Yashrajsinh Chudasama.
