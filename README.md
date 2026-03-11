# Détection d'Anomalies Astrophysiques par Deep Learning (VAE)

Ce projet de Data Science / IA vise à identifier des étoiles aux comportements atypiques (anomalies cinématiques ou photométriques) au sein de la Voie Lactée, en utilisant les données massives du catalogue **Gaia DR3** de l'Agence Spatiale Européenne (ESA). 

Face à des données non labellisées et multidimensionnelles, l'approche retenue est un apprentissage non supervisé basé sur un **Autoencodeur Variationnel (VAE)** développé avec **PyTorch**.

## Enjeux et Valeur Métier
* **Traitement de données massives (Big Data) :** Requêtage spatial via ADQL et optimisation du stockage (format `.parquet`).
* **Ingénierie des incertitudes :** Prise en compte des marges d'erreur matérielles des télescopes pour éviter les faux positifs.
* **Apprentissage Non Supervisé :** Détection d'anomalies sans connaissance préalable des classes, une problématique courante dans l'industrie (détection de fraude, maintenance prédictive, cybersécurité).

---

## Architecture de l'Intelligence Artificielle



Le modèle repose sur un réseau de neurones génératif (VAE) conçu pour apprendre la distribution sous-jacente des étoiles "normales". 

1. **Encodeur :** Compresse les 10 caractéristiques physiques de l'étoile (position, mouvement propre, parallaxe, luminosité, erreurs) vers un espace latent probabiliste de 4 dimensions ($\mu$ et $\sigma$).
2. **Reparameterization Trick :** Permet la rétropropagation du gradient tout en échantillonnant depuis la distribution latente.
3. **Décodeur :** Tente de reconstruire les caractéristiques originales de l'étoile.

### Fonction de Perte (Loss Function)
L'entraînement du modèle minimise l'Evidence Lower Bound (ELBO), composée de l'erreur de reconstruction (MSE) et de la divergence de Kullback-Leibler :

$$\mathcal{L} = \text{MSE}(x, \hat{x}) + D_{KL}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, 1))$$

* **Le Scoring :** Une fois le modèle entraîné sur la norme, toute étoile présentant une erreur de reconstruction (MSE) anormalement élevée est classifiée comme anomalie.

---

## Résultats et Visualisation

Le modèle a scanné un échantillon représentatif de 100 000 étoiles. Le fichier `top_100_anomalies_gaia.csv` contient les identifiants (`source_id`) des objets les plus aberrants trouvés par l'IA (notamment des étoiles à grande vitesse - *High Proper Motion Stars*).

Pour comprendre l'espace latent du modèle en 4 dimensions, une réduction de dimensionnalité via **UMAP** a été appliquée :
* **Zone centrale dense :** Le comportement galactique standard.
* **Points isolés périphériques :** Les anomalies détectées par le VAE.

---

## Stack Technique
* **Langage :** Python 3
* **Deep Learning :** PyTorch (Entraînement accéléré via CUDA/GPU)
* **Data manipulation :** Pandas, Scikit-Learn (StandardScaler)
* **Collecte de données :** Astroquery (API de l'ESA)
* **Visualisation :** Matplotlib, Seaborn, UMAP

---

## Reproductibilité (Installation Locale)

Pour exécuter ce projet sur votre machine avec prise en charge du GPU :

1. Clonez ce dépôt :
   ```bash
   git clone [https://github.com/VOTRE_PSEUDO/gaia-anomaly-detection-ia.git](https://github.com/VOTRE_PSEUDO/gaia-anomaly-detection-ia.git)
   cd gaia-anomaly-detection-ia

2. Créez un environnement virtuel et installez les dépendances :
   ```bash
   conda create -n gaia_ai python=3.10
   conda activate gaia_ai
   pip install torch pandas matplotlib seaborn astroquery umap-learn pyarrow fastparquet


3. Lancez le notebook :
   ```bash
   jupyter lab

_______________________________________________________________________________________________________________________________

Projet réalisé par Philippe COMBOT dans le cadre du cursus Ingénieur en Intelligence Artificielle (OpenClassrooms).

✉️ Me contacter par emaiL : mail-pro@philippecombot.com
   
