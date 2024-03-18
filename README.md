# Bacterial_clustering_AMR
This repository contains the Python scripts used to cluster pathogenic bacterial strains based on their genomic sequence similarity and the prediction model based on the clustering output

Requirements:
A multifasta file with bacterial genomic sequences and 
Python 3.10

The uploaded scripts should be used in the below mentioned sequence:
Preprocessing:
The sequence_kmers.py script will frangment the bacterial genomic sequences into kmers of length 10.
The remove_non_bases.py script will filter out the 10-mers with bases other than 'A', 'T', 'G', 'C'.
The species_tenmers.R script will generate a count matrix indicating the presence and absence of 10-mers in each of the bacterial strain.
The make_binary.py script will then convert the count matrix into a binary matrix.

Clustering:
afinity_propogation_clustering.py script can then be used to calculate the Jaccard distances between the bacterial strains and segregate them into groups of strains sharing sequential similarity.

Classification model - Multi-class classification:
The algorithm_selection_and_feature_selection.py script will help in deciding the suitable algorithm capable of predicting the cluster label obtained in the previous step. Also the high scoring 10-mers can be selected usig the same script.
In our cases Random forest algorithm worked better than the other algorithms. Thus, further classification_model_with_tuning.py script was used to generate a random forest model. The script also includes hyperparameter tuning.

Multi-label perceptron model:
The calculate_seq_features.py script was used to calculate the nucleotide features such as GC content and mononucleotide counts of each bacterial strain.
multilabel_classification_model.py script was then used to predict the antibacterial resistance of each bacterial strain.



