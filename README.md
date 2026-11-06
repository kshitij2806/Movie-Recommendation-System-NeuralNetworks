# 🎬 CineMatch — Movie Recommendation System

A neural network-powered movie recommendation engine built with **Restricted Boltzmann Machines (RBM)** and served through an elegant web interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)

## Overview

CineMatch uses a Restricted Boltzmann Machine — a generative stochastic neural network — to learn latent user preferences from movie rating patterns. It can then predict ratings for unseen movies and generate personalized recommendations.

### Architecture

```
┌─────────────────────────────────────────┐
│             Hidden Layer (128 units)     │
│         Learned latent features          │
│   ○  ○  ○  ○  ○  ○  ...  ○  ○  ○       │
│   │╲ │╲ │╲ │╲ │╲ │╲      │╲ │╲ │╲      │
│   │ ╲│ ╲│ ╲│ ╲│ ╲│ ╲     │ ╲│ ╲│ ╲     │
│   ○  ○  ○  ○  ○  ○  ...  ○  ○  ○       │
│          Visible Layer (1682 units)      │
│           Movie ratings per user         │
└─────────────────────────────────────────┘
```

## Features

- 🧠 **RBM-based collaborative filtering** trained on MovieLens 100K
- 🎯 **Personalized recommendations** for any of 943 users
- 🔗 **Similar movie discovery** via learned weight embeddings
- 🔥 **Trending movies** ranked by Bayesian average
- ✨ **Elegant glassmorphism UI** with smooth animations
- ⚡ **Lightweight & fast** — trains in ~30 seconds on CPU

## Quick Start

### 1. Setup Environment

```bash
# Create & activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py
```

This will:
- Download the MovieLens 100K dataset automatically
- Preprocess and binarize the ratings
- Train the RBM for 50 epochs (~30s on CPU)
- Save the model to `models/rbm_trained.pt`

### 3. Launch the App

```bash
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

## Project Structure

```
Movie-Recommendation-System/
├── data/ml-100k/              # MovieLens 100K dataset (auto-downloaded)
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing.py  # Data loading & transformation
│   ├── rbm_model.py           # RBM neural network (PyTorch)
│   ├── train.py               # Training pipeline
│   └── recommend.py           # Recommendation engine
├── models/
│   └── rbm_trained.pt         # Trained model weights
├── static/                    # CSS & JavaScript
├── templates/                 # HTML templates
├── app.py                     # Flask application
├── requirements.txt           # Dependencies
└── README.md
```

## How It Works

### Restricted Boltzmann Machine (RBM)

An RBM is a two-layer generative neural network:
- **Visible layer**: Movie ratings (one neuron per movie)
- **Hidden layer**: Latent features (user taste patterns)

**Training** uses Contrastive Divergence (CD-k):
1. Clamp user ratings to the visible layer
2. Propagate to hidden layer → sample hidden activations
3. Reconstruct visible layer from hidden → sample visible
4. Update weights to minimize reconstruction error

**Inference**: Given a user's partial ratings, the trained RBM reconstructs the full visible layer, predicting ratings for unseen movies.

## Dataset

**MovieLens 100K** by GroupLens Research:
- 100,000 ratings (1-5 stars)
- 943 users, 1,682 movies
- Collected between Sep 1997 – Apr 1998

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Neural Network | PyTorch (RBM) |
| Data Processing | Pandas, NumPy, scikit-learn |
| Web Server | Flask |
| Frontend | HTML5, CSS3, Vanilla JS |
| Visualization | Matplotlib, Seaborn |

## License

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) which is provided for research purposes by GroupLens Research.
