# Airline Tweet Sentiment Analysis

A comprehensive tweet sentiment classification system using GloVe embeddings and neural networks to analyze customer sentiment towards U.S. airlines in real-time.

##  Project Overview

This project implements a production-ready sentiment analysis pipeline that:
- Classifies tweets as **positive**, **neutral**, or **negative**
- Uses 100-dimensional GloVe word embeddings for text representation
- Employs advanced preprocessing techniques including hashtag splitting and emoji processing
- Achieves **77.32% accuracy** on airline sentiment data (continuously improving)
- Includes comprehensive OOV (Out-of-Vocabulary) analysis and data exploration

##  Features

### Core Features
- **Custom Tweet Tokenizer**: Optimized for social media text processing
- **GloVe Integration**: Efficient loading and utilization of pre-trained embeddings
- **Neural Network Classifier**: Lightweight single-layer architecture for CPU-friendly deployment
- **Advanced Preprocessing**: Contraction expansion, punctuation normalization, and GDPR-compliant data masking

### Enhanced Features
- **Emoji2Vec Support**: Integration with emoji embeddings for better social media understanding
- **Hashtag Splitting**: Intelligent decomposition of hashtags into component words
- **Ensemble Classification**: Multiple specialized models for improved neutral detection
- **Data Deduplication**: Advanced duplicate detection with similarity thresholds
- **OOV Analysis**: Comprehensive out-of-vocabulary word analysis and reporting

##  Requirements

### Core Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.9.0
nltk>=3.6.0
```

### Optional Dependencies (for enhanced features)
```
emoji>=2.0.0
wordninja>=2.0
symspellpy>=6.7.0
```

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/shaunraj46/Airline-Tweet-Sentiment-Analysis.git
cd Airline-Tweet-Sentiment-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required Data Files

**GloVe Embeddings:**
- Download `glove.6B.100d.txt` from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
- Place in the project root directory

**Dataset Files:**
- `tweet_sentiment.train.jsonl` (~11k samples)
- `tweet_sentiment.test.jsonl` (~3k samples)
- Place both files in the project root directory

**Optional - Emoji2Vec Embeddings:**
- Download `emoji2vec.txt` or the system will create demo embeddings automatically
- Note: Current emoji2vec file may need format adjustment (0 embeddings loaded in recent run)

### 4. Download NLTK Data
```python
import nltk
nltk.download('punkt')
```

##  Usage

### Run the Complete Pipeline
```bash
python neural_classifier.py
```

### Basic Usage (Programmatic)
```python
from neural_classifier import main

# Run the complete pipeline
analyzer = main()

# Predict sentiment for a single tweet
sentiment = analyzer.predict_single("Great flight experience! @airline")
print(f"Predicted sentiment: {sentiment}")
```

### Enhanced Features
```python
from neural_classifier import main_with_emoji_support

# Run with emoji support and advanced features
analyzer = main_with_emoji_support()
```

### Configuration Options
You can enable various enhancements by modifying the configuration in the main functions:

```python
# Enhancement options
USE_ENSEMBLE = False          # Enable ensemble classification
USE_DEDUPLICATION = False     # Enable data deduplication
USE_EMOJI_EMBEDDINGS = True   # Enable emoji2vec integration
```

##  Expected Output

### Model Performance
```
Overall Accuracy: 0.7732
OOV Rate: 1.4%

Classification Report:
              precision    recall  f1-score   support
    negative       0.82      0.90      0.86      1835
     neutral       0.62      0.54      0.58       620
    positive       0.75      0.58      0.66       473

    accuracy                           0.77      2928
   macro avg       0.73      0.67      0.70      2928
weighted avg       0.77      0.77      0.77      2928
```

### Enhancement Results
Based on the latest run:
```
Enhancements applied:
  Hashtag splitting: 2,674 hashtags processed
  Emoji integration: 0 emoji embeddings loaded (file format issue)
  Custom tweet tokenizer: enabled
  OOV analysis: enabled (1.4% OOV rate achieved)
```

### Confusion Matrix
![Confusion Matrix](https://github.com/shaunraj46/Airline-Tweet-Sentiment-Analysis/blob/main/confusion_matrix.png)

```
        negative   neutral  positive
negative:     1660       125        50  (Total: 1835)
 neutral:      252       328        40  (Total: 620)
positive:      129        63       281  (Total: 473)
```

### Misclassification Analysis
- **Total misclassifications**: 659 out of 2,928 (22.5%)
- **Key insights from the matrix**:
  - **Negative sentiment**: Excellent detection (90.5% accuracy, 1660/1835)
  - **Neutral sentiment**: Most challenging class (52.9% accuracy, 328/620)  
  - **Positive sentiment**: Good detection (59.4% accuracy, 281/473)
- **Common error patterns**: 
  - Neutral tweets often misclassified as negative (252 cases, 8.6% of total)
  - Positive tweets sometimes seen as negative (129 cases, 4.4% of total)
- **Example challenges**:
  - *"@JetBlue would you say a delay is more likely? Thanks so much."* (True: positive → Predicted: negative)
  - *"@USAirways Well I did miss it. But gate agents had rebooked boarding pass waiting when I landed..."* (True: positive → Predicted: negative)

The system generates detailed error analysis with misclassification examples for continuous improvement.

### Data Exploration
```
Dataset sizes:
  Training samples: 11,712
  Test samples: 2,928
  Total samples: 14,640

Class Distribution Analysis:
Training set:
  Negative: 1,835 (62.7%)
  Neutral: 620 (21.2%)
  Positive: 473 (16.1%)
```

##  Project Structure

```
├── neural_classifier.py              # Main sentiment analysis pipeline
├── requirements.txt                   # Project dependencies
├── README.md                         # Project documentation
├── confusion_matrix.png              # Generated confusion matrix visualization
├── glove.6B.100d.txt                 # GloVe embeddings (download required)
├── tweet_sentiment.train.jsonl       # Training dataset (download required)
├── tweet_sentiment.test.jsonl        # Test dataset (download required)
└── emoji2vec.txt                     # Optional emoji embeddings
```

##  Key Components

### TweetSentimentAnalyzer
Main class that orchestrates the entire pipeline:
- Data loading and exploration
- Model training and evaluation
- Prediction and analysis

### TextPreprocessor
Advanced text preprocessing with:
- Contraction expansion
- Hashtag splitting
- Emoji processing
- GDPR-compliant data masking

### EmbeddingLoader / EnhancedEmbeddingLoader
Efficient loading of GloVe embeddings with optional emoji2vec integration

### FeatureExtractor
Comprehensive feature engineering including:
- Sentiment indicators
- Linguistic patterns
- Text statistics
- Punctuation analysis

##  Model Architecture

**Single-Layer Neural Network:**
- Input: GloVe embeddings + engineered features
- Architecture: Linear layer (input_dim → 3 classes)
- Optimizer: Adam with learning rate 0.001
- Loss function: CrossEntropyLoss
- Training: 30 epochs with batch size 64

**Enhanced Ensemble Mode:**
- General sentiment classifier
- Specialized neutral vs non-neutral detector
- Polarity classifier for positive vs negative
- Weighted combination for final predictions

##  Performance Analysis

### Strengths
- **Outstanding negative sentiment detection** (90.5% accuracy, critical for customer service monitoring)
- **Excellent recall for negative class** (1660/1835 correctly identified)
- Very low OOV rate of 1.4% indicating excellent vocabulary coverage
- Efficient CPU-friendly architecture
- Comprehensive preprocessing pipeline with 2,674 hashtags processed
- Robust handling of social media text patterns
- Professional confusion matrix visualization with percentage breakdowns

# Airline-Tweet-Sentiment-Analysis
