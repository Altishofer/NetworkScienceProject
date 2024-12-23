## Network Science - Trade Network Analysis
Repository for the 2024 edition of Network Science (03SM22MI0019) at the University of Zurich
[(Course Information)](https://studentservices.uzh.ch/uzh/anonym/vvz/index.html#/details/2024/003/SM/51110722/50000003/Wirtschaftswissenschaftliche%2520Fakult%25C3%25A4t/51085510/Master%2520of%2520Science%2520UZH%2520in%2520Informatik%2520(RVO22)/51087487/Data%2520Science)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Introduction](#introduction)
3. [Repository Structure](#repository-structure)
4. [Contributors](#contributors)

---

### Project Overview

This project focuses on analyzing global trade networks, with a specific emphasis on lithium. We leverage several datasets to explore trade flows and their reaction to trade wars, tariffs and sanctions. 
The primary datasets used in this project include:

1. **BACI CEPII dataset (2002–2022)**  
   <https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37>  
   This dataset provides bilateral trade data covering a wide range of products. In total the dataset contains more than 200 Mio entries for various products.

2. **CEPII Gravity dataset (2002–2022)**  
   <https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8>  
   This dataset contains country-pair information on various factors (such as distance, GDP, common language, and more) that can be used as control factors.

---

## Introduction

Global trade, a cornerstone of globalization, has repeatedly been put to the test by recent geopolitical and economic disturbances. The COVID-19 pandemic disrupted supply chains, prompting countries to reassess their reliance on distant markets. Simultaneously, exogenous incidents such as the blockage of the Suez Canal and the Houthi rebels’ attacks in Yemen intensified vulnerabilities within critical maritime routes. The ongoing trade wars and the recent Russian invasion of Ukraine have further highlighted how deeply interconnected and yet fragile these international exchanges can be. In this evolving landscape, governments and businesses alike face unprecedented challenges in ensuring resilience and stability within the global trading system.

In response, our study aims to investigate the patterns of global trade decoupling brought forth by these disruptive events. Leveraging graph machine learning techniques, we seek to detect anomalies and identify network topologies that exhibit unusual trading behaviors. Through use of analytics, we examine structural shifts in the global trade network, providing an understanding of how these shocks reconfigure international commerce.

---

## Repository Structure

## dataloader.py
This module provides a high-level class `DataLoader` that streamlines the process of loading, merging, and cleaning trade data from the BACI dataset, country and product codes, and CEPII’s Gravity variables. It prepares a unified dataset for further steps such as network construction and modeling.

## 1.1_lithium_export_import.ipynb
This notebook explores the export and import trends of lithium across different countries and regions. The aim is to provide insights into global lithium trade patterns, including major exporters, importers, and the overall trade balance.

### Table of Contents
1. [Introduction](#Introduction)  
2. [Data Loading and Preparation](#Data-Loading-and-Preparation)  
3. [Top Trading Countries](#Top-Trading-Countries)  
4. [Ranked Trading Countries](#Ranked-Trading-Countries)  
5. [Map Visualizations](#Map-Visualizations)

---

## 1.2_lithium_communities.ipynb
This notebook explores various community detection algorithms applied to graphs. The goal is to compare the effectiveness of different methods, analyze the structure of detected communities, and gain insights into graph-based clustering techniques.

### Table of Contents
1. [Introduction](#introduction)  
2. [Data Loading and Preparation](#data-loading-and-preparation)  
3. [LeidenAlg Temporal Community Detection](#leidenalg-temporal-community-detection)  
4. [GCN Community Detection](#gcn-community-detection)  
5. [Greedy Modularity Communities](#greedy-modularity-communities)  
6. [Louvain Communities](#louvain-communities)  
7. [GCNConv Communities](#gcnconv-communities)  
8. [Blockmodel Communities](#blockmodel-communities)

---

## 1.3_lithium_robustness.ipynb
This notebook investigates the robustness of lithium-related networks through simulations and centrality comparisons. The objective is to analyze how these networks respond to various attacks and explore structural metrics like centralities.

### Table of Contents
1. [Introduction](#introduction)  
2. [Data Loading and Preparation](#data-loading-and-preparation)  
3. [Attack Simulation](#attack-simulation)  
4. [Compare Centralities of 2020](#compare-centralities-of-2020)

---

## 1.4_lithium_key_players.ipynb
This notebook investigates the export evolution and centrality metrics for the countries with the largest known lithium reserves as of 2022.

### Table of Contents
1. [Introduction](#introduction)
2. [Data Loading and Preparation](##data-loading-and-preparation)
3. [Export Key Player Analysis](#export-analysis)
4. [Import Key Player Analaysis](#import-analysis)
---

## 1.5_lithium_metrics_temporal.ipynb
This notebook computes key structural metrics for trade networks over time, providing insights into both their topology and node-level properties. It also examines macroeconomic factors that might influence the evolution of the network.

### Table of Contents
1. [Introduction](#introduction)  
2. [Data Loading and Preparation](#data-loading-and-preparation)  
3. [Temporal Graph Metrics](#temporal-graph-metrics)  
4. [Macro Economic Metrics](#macro-economic-metrics)

---

## 1.7_lithium_network_model.ipynb
This notebook provides a comprehensive analysis of degree distributions within temporal trade networks. By leveraging both empirical data and mathematical models, we aim to uncover the structural characteristics of trade networks and explore the fit of various distribution models to the observed data.

### Table of Contents
1. [Introduction](#introduction)  
2. [Data Loading and Preparation](#data-loading-and-preparation)  
3. [Fit Graph to Model](#fit-graph-to-model)

---

## 1.8_lithium_centrality.ipynb
This notebook explores centrality metrics within material trade networks, focusing on individual materials and their cumulative impact across the entire network. By visualizing centrality trends, it highlights key players and their roles in global trade dynamics.

### Table of Contents
1. [Introduction](#introduction)  
2. [Data Loading and Preparation](#data-loading-and-preparation)  
3. [Material Trade Centralities](#material-trade-centralities)  
4. [Summed Trade Centralities Over All Materials](#summed-trade-centralities-over-all-materials)

---

## 00_agnostic_full_pipeline_value.ipynb

This notebook explores the export and import trends of the aggregated product trades across different countries and regions. The aim is to provide insights into global trade patterns, including major exporters, importers, and the overall trade balance.

## Table of Contents

1. [Introduction](#Introduction)
2. [Data Loading and Preparation](#Data-Loading-and-Preparation)
3. [Load and Preprocess Data](#Load-and-Preprocess-Data)
4. [Countries Ranked by Global Trade Volume](#Countries-Ranked-by-Global-Trade-Volume)
5. [Temporal Trade Network Properties](#Temporal-Trade-Network-Properties)
6. [Map Visualization of Country-Specific Trade](#Map-Visualization-of-Country-Specific-Trade)
7. [Theoretical Graph Properties](#Theoretical-Graph-Properties)
8. [Community Detection](#Community-Detection)
9. [Temporal Macroeconomic Metrics](#Temporal-Macroeconomic-Metrics)
10. [Centrality Analysis](#Centrality-Analysis)

---

## 10_outlier_detection_transformer.ipynb
In this notebook, we engineer edge-level and node-level features, and build a Transformer-enhanced Graph Autoencoder (GAE) to detect anomalies in global trade patterns based on reconstruction errors.

---

## 11_outlier_detection_cnn.ipynb
In this notebook, we extend our trade anomaly detection to a Convolutional Neural Network (CNN)-inspired Graph Autoencoder.

---

## Contributors
| Github User    | Github User Profile                                | Name               | StudentId   |
|----------------|----------------------------------------------------|--------------------|-------------|
| altishofer     | [@altishofer](https://github.com/Altishofer)       | Sandrin Hunkeler   | 18‑253‑815  |
| aezsmgfjf      | [@aezsmgfjf](https://github.com/aezsmgfjf)         | Marco Meyer        | 19‑210‑145  |
| random9nessss  | [@random9nessss](https://github.com/random9nessss) | Kevin Bründler     | 19‑919‑273  |

---
