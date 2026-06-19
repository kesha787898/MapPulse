# MapPulse: Predicting Urban Perception from Digital Maps

## 📌 Overview

Human perception of urban environments strongly influences behavior, health, and economic outcomes. Most existing deep learning approaches estimate urban perception using street-level imagery, but these methods suffer from outdated images and potential dataset biases.

To address these limitations, we propose **MapPulse**, a framework that predicts human psychological perceptions of cities directly from **digital map data (OpenStreetMap)** instead of street-level images.

Maps are:
- universally accessible  
- lightweight and compact  
- frequently updated  

This makes them a promising alternative data source for urban perception modeling.

---

## 🧠 Method

We introduce a two-stage training pipeline:

### 1. Pretraining Stage
- Backbone: **Siamese ResNet-18**
- Initialization: satellite-domain pretrained weights
- Dataset: 111,290 map patches from 113 cities
- Task: multi-task prediction of quality-of-life indicators

### 2. Ranking Stage
- Lightweight ranking head trained on **MapPulse dataset**
- Uses **strict city-level train/test split**
- Ensures generalization and prevents spatial leakage

---

## 📊 Dataset

The **MapPulse dataset** combines:
- Geolocated human judgment annotations from **Place Pulse 2.0**
- Corresponding rasterized map tiles from **OpenStreetMap**

Perceptual attributes include:
- Safety  
- Liveliness  
- Beauty  
- Wealth  
- Depression  
- Boredom  

---

## 📈 Results

Our model achieves:

- **Mean Accuracy:** 61.46%  
- **Mean AUROC:** 0.6324  

across six perceptual attributes.

Key findings:
- Socioeconomic attributes (e.g., *wealth*) are predicted most accurately
- Map-based models outperform ImageNet and Sentinel-2 baselines
- Street-level imagery still provides upper-bound performance, but maps are competitive

---

## 🚀 Getting Started

To explore the pipeline:
Pipeline.md

## 🚀 Project structure

1) data_processing - everything nessesary for image rendering and data split. For some scripts it is neccecary to have a insalled java.
2) pretraining - pretraining code
3) training - code for training model
4) paths.py - configuration of all paths used in project.
5) tools - some functiions used in the whole project.
6) 