# Modélisation et détection des ruptures de la chaîne du froid alimentaire

Projet réalisé dans le cadre du module EDP – CY Tech (2025–2026).

Ce projet vise à modéliser l’évolution de la température d’un produit alimentaire pendant le transport et à détecter automatiquement les ruptures de la chaîne du froid à partir de données issues de capteurs IoT.

---

## Objectifs du projet

- Étudier l’évolution de la température interne d’un produit alimentaire
- Modéliser cette évolution à l’aide de l’équation de la chaleur
- Mettre en place un pipeline de traitement de données
- Détecter automatiquement les ruptures de la chaîne du froid
- Comparer deux méthodes de classification : Random Forest et KNN

---

## Données utilisées

Le projet s’appuie sur un jeu de données issu de capteurs IoT, disponible sur la plateforme Kaggle :

Cold Supply Chain Data  
https://www.kaggle.com/datasets/syedalihaidernaqvi/cold-supply-chain-data

Les données contiennent notamment :
- la température ambiante,
- la température interne du produit,
- l’humidité,
- l’état du système de refroidissement.

---

## Structure du projet

.
├── dataset_preparation.py  
├── heat_equation.py  
├── feature_extraction.py  
├── feature_engineering.py  
├── classification.py  
├── dataset_preprocessed.csv  
├── dataset_features_ml.csv  
├── sample_temperature-ELM_v1.0_optimizationComp_data.csv  
├── requirements.txt  
└── README.md  

---

## Pipeline de traitement

### 1. Prétraitement des données

Script : dataset_preparation.py

- nettoyage du dataset d’origine,
- suppression des variables redondantes,
- renommage des colonnes,
- création d’un repère temporel (1 lecture = 1 minute),
- sauvegarde du dataset prétraité.

---

### 2. Modélisation par l’équation de la chaleur

Script : heat_equation.py

- résolution numérique de l’équation de la chaleur,
- utilisation de la température ambiante comme condition aux limites,
- comparaison température simulée / température mesurée,
- calcul de l’erreur absolue moyenne (MAE).

---

### 3. Extraction de caractéristiques

Script : feature_extraction.py

- température maximale et minimale,
- durée au-dessus d’un seuil critique,
- pente maximale de température,
- écart moyen entre température interne et externe.

---

### 4. Feature engineering et labellisation

Script : feature_engineering.py

- calcul de la pente de la température ambiante,
- calcul de la moyenne glissante,
- création des labels :
  - 0 : OK
  - 1 : Rupture courte
  - 2 : Rupture critique
- génération du dataset final pour le machine learning.

---

### 5. Classification

Script : classification.py

- séparation des données en ensembles d’apprentissage et de test,
- apprentissage des modèles Random Forest et KNN,
- évaluation à l’aide de matrices de confusion et métriques de performance.

---

## Installation et exécution

### Clonage du projet

```
git clone https://github.com/asmayzd/Modelisation-et-detection-des-ruptures-de-la-chaine-du-froid-alimentaire.git
cd Modelisation-et-detection-des-ruptures-de-la-chaine-du-froid-alimentaire
```

### Installation des dépendances

```
pip install -r requirements.txt
```

### Exécution des scripts

```
python dataset_preparation.py
python heat_equation.py
python feature_extraction.py
python feature_engineering.py
python classification.py
```

## Résultats
La modélisation physique met en évidence une évolution progressive de la température interne, cohérente avec l’inertie thermique du produit.

L’erreur MAE obtenue est d’environ 4,8 °C.

Les modèles Random Forest et KNN atteignent une accuracy d’environ 0,996.

## Auteurs
Projet réalisé par :

Eish Aicha

Dos Santos Emma

Ouadah Shaima

Yazidi Asma

ING2 GMA – CY Tech
