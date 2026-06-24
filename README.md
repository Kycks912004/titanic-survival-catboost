# 🚢 Titanic Survival Prediction — CatBoost

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-ML-FFCC00?style=for-the-badge&logo=yandex&logoColor=black)](https://catboost.ai/)
[![Accuracy](https://img.shields.io/badge/Accuracy-80.68%25-success?style=for-the-badge)](https://www.kaggle.com/c/titanic)
[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/titanic)

> Prédiction de survie des passagers du Titanic avec **CatBoost** et feature engineering avancé.  
> **Précision obtenue : 80.68 %**

---

## 🏆 Résultats

| Métrique | Valeur |
|----------|--------|
| **Précision (validation)** | **80.68 %** |
| Algorithme | CatBoostClassifier |
| Split train/val | 84 % / 16 % |
| Random seed | 42 |

---

## 🧠 Approche

### Feature Engineering

| Feature créée | Description |
|--------------|-------------|
| `Titre` | Extrait du nom (`Mr`, `Miss`, `Master`, `Rare`…) |
| `Est_enfant` | `True` si Titre ∈ {Master, Miss} **et** Age < 18 |
| `Nb_mots_nom` | Nombre de mots dans le nom complet |
| `Cabin` | Première lettre de la cabine (`X` si manquant) |

### Gestion des valeurs manquantes

| Colonne | Stratégie |
|---------|-----------|
| `Age` | Médiane |
| `Fare` | Médiane |
| `Cabin` | `"X"` |
| `Embarked` | `"X"` |

### Modèle

- **CatBoostClassifier** avec gestion native des variables catégorielles
- Pas d'encodage manuel nécessaire (`cat_features` passé directement)
- `use_best_model=True` pour éviter l'overfitting

---

## 📁 Structure

```
titanic-survival-catboost/
├── solution.py          # Script principal (CatBoost)
├── requirements.txt     # Dépendances
├── titanic/
│   ├── train.csv        # Données d'entraînement (Kaggle)
│   ├── test.csv         # Données de test (Kaggle)
│   └── gender_submission.csv
└── soumission_finale.csv  # Fichier de soumission généré
```

---

## 🚀 Lancer le projet

```bash
# 1. Cloner le repo
git clone https://github.com/Kycks912004/titanic-survival-catboost.git
cd titanic-survival-catboost

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Télécharger les données Kaggle et les placer dans titanic/
#    https://www.kaggle.com/c/titanic/data

# 4. Lancer le script
python solution.py
```

---

## 📊 Variables catégorielles utilisées

```python
COLS_CATEGORIELLES = [
    "Pclass", "SibSp", "Parch", "Ticket", "Cabin",
    "Embarked", "Titre", "Nb_mots_nom", "Est_enfant", "Sex"
]
```

---

## 👤 Auteur

**Kylian Pinto** — M1 Ingénierie Data & IA, ECE Paris  
[![GitHub](https://img.shields.io/badge/GitHub-Kycks912004-181717?style=flat&logo=github)](https://github.com/Kycks912004)
