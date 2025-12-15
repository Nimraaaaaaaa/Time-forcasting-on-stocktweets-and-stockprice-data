# Stock Price Forecasting with Twitter Sentiment Analysis

## MSc Data Analytics - Advanced Data Analytics & Big Data Storage and Processing
**CCT College Dublin - Continuous Assessment**

---

## 📋 Project Overview

This project implements a comprehensive big data analytics pipeline that combines Twitter sentiment analysis with historical stock price data to forecast closing prices for multiple companies. The system leverages distributed computing frameworks and advanced time-series forecasting techniques to predict stock prices at 1-day, 3-day, and 7-day intervals.

### Key Technologies
- **Big Data Processing**: Apache Spark, Hadoop MapReduce
- **Databases**: MySQL, MongoDB (NoSQL)
- **Machine Learning**: ARIMA/SARIMA, LSTM Neural Networks
- **Visualization**: Interactive Dashboard with Tufte Principles
- **Version Control**: GitHub with regular commits

---

## 📊 Dataset Information

### Stock Tweet Data (`stocktweet.csv`)
- **Period**: January 2020 - December 2020
- **Records**: 10,000 tweets
- **Fields**:
  - `ids`: Tweet ID
  - `date`: Tweet timestamp
  - `ticker`: Company ticker symbol
  - `tweet`: Tweet text content

### Stock Price Data (`stockprice/` folder)
- **Period**: January 2020 - December 2020
- **Companies**: 38 companies (AAPL, AMZN, TSLA, MSFT, etc.)
- **Fields**:
  - Date, Open, High, Low, Close, Adj Close, Volume

---

## 🏗️ Architecture Design

The project implements a complete big data processing pipeline:

```
Data Sources (CSV Files)
    ↓
Data Ingestion & Storage (MySQL/MongoDB)
    ↓
Distributed Processing (Spark/MapReduce)
    ↓
Sentiment Analysis & Feature Engineering
    ↓
Time Series Forecasting (ARIMA/SARIMA + LSTM)
    ↓
Results Storage (NoSQL Database)
    ↓
Interactive Dashboard (Visualization)
```

---

## 🔧 Implementation Components

### 1. Data Storage & Processing (30 marks)
- **Distributed Environment**: Apache Spark for parallel processing
- **Data Preparation**: ETL pipeline for cleaning and transformation
- **Storage Strategy**: Hybrid SQL/NoSQL approach
  - MySQL for structured stock price data
  - MongoDB for unstructured tweet data
- **Processing**: MapReduce-style operations using PySpark

**Rationale**: Spark chosen for its in-memory processing capabilities and unified API for batch and stream processing. MySQL provides ACID compliance for financial data, while MongoDB offers flexible schema for tweet storage.

### 2. Database Comparative Analysis (30 marks)
- **Databases Compared**: MySQL (SQL) vs MongoDB (NoSQL)
- **Testing Tool**: YCSB (Yahoo! Cloud Serving Benchmark)
- **Workloads Tested**:
  - Workload A: Read/Write heavy (50/50)
  - Workload B: Read heavy (95/5)
  - Workload C: Read-only
  - Workload D: Read-latest
  - Workload E: Scan operations
- **Metrics Recorded**:
  - Throughput (operations/second)
  - Latency (average, 95th percentile, 99th percentile)
  - Insert/Read/Update performance

**Key Findings**: Detailed performance comparison with graphs and analysis included in the report.

### 3. Big Data Architecture Design (20 marks)
Comprehensive architecture diagram includes:
- Data ingestion layer
- Storage layer (SQL/NoSQL)
- Processing layer (Spark clusters)
- Analytics layer (ML models)
- Presentation layer (Dashboard)

All components documented with technology stack justification.

---

## 📈 Advanced Data Analytics

### 1. Data Analysis Pipeline (40 marks)

#### Exploratory Data Analysis (EDA)
- Stock price trend analysis
- Volume distribution analysis
- Tweet frequency and sentiment distribution
- Correlation analysis between sentiment and price movements

#### Data Wrangling
- Missing value handling
- Outlier detection and treatment
- Date alignment between tweets and stock prices
- Feature scaling and normalization

#### Sentiment Extraction
**Technique**: VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Justification**:
- Pre-trained on social media text
- Handles emoji and slang effectively
- Compound scores suitable for financial sentiment
- No training data required
- Real-time processing capability

**Alternative Considered**: TextBlob, FinBERT (rejected due to computational overhead)

#### Machine Learning Models
Selected 5+ companies for analysis: AAPL, AMZN, TSLA, MSFT, GOOGL

### 2. Time Series Forecasting (20 marks)

#### Model 1: SARIMA (Seasonal AutoRegressive Integrated Moving Average)
- **Parameters**: (p, d, q) × (P, D, Q, s)
- **Selection Method**: Grid search with AIC/BIC criteria
- **Exogenous Variables**: Sentiment scores, tweet volume

#### Model 2: LSTM Neural Network
- **Architecture**:
  - Input layer: Multiple features (price + sentiment)
  - 2-3 LSTM layers with dropout
  - Dense output layer
- **Features Used**: 
  - Historical prices (lagged values)
  - Sentiment scores
  - Tweet volume
  - Technical indicators (moving averages)

**Short Time Series Handling**:
- Data augmentation through overlapping windows
- Transfer learning from similar stocks
- Feature engineering to add information density

### 3. Hyperparameter Tuning (20 marks)

#### SARIMA Tuning
- **Method**: Auto-ARIMA with seasonal decomposition
- **Metrics**: AIC, BIC, RMSE, MAE
- **Validation**: Walk-forward validation

#### LSTM Tuning
- **Method**: Grid Search + Bayesian Optimization
- **Parameters Tuned**:
  - Learning rate: [0.001, 0.0001]
  - LSTM units: [50, 100, 150]
  - Dropout rate: [0.2, 0.3, 0.4]
  - Batch size: [32, 64]
  - Epochs: [50, 100, 150]
- **Validation Strategy**: Time-series cross-validation

**Results**: Optimal configurations documented with performance metrics.

### 4. Results Presentation & Dashboard (20 marks)

#### Interactive Dashboard Features
- Multi-company selection dropdown
- Forecast horizon selector (1/3/7 days)
- Model comparison view (ARIMA vs LSTM)
- Real-time sentiment gauge
- Historical vs predicted price charts
- Model performance metrics display

#### Tufte Principles Implementation
1. **Data-Ink Ratio**: Minimized chart junk, focused on essential information
2. **Chartjunk Reduction**: Clean design, no unnecessary decorations
3. **Data Density**: Maximum information in minimum space
4. **Small Multiples**: Comparison charts for multiple stocks
5. **Layering & Separation**: Clear visual hierarchy
6. **Micro/Macro Readings**: Summary and detailed views

**Visualization Tools**: Plotly/Dash for interactivity, Matplotlib/Seaborn for static plots

---

## 📁 Project Structure

```
project/
├── data/
│   ├── stocktweet.csv
│   └── stockprice/
│       ├── AAPL.csv
│       ├── AMZN.csv
│       └── ...
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_modeling_forecasting.ipynb
├── src/
│   ├── data_ingestion.py
│   ├── sentiment_analysis.py
│   ├── forecasting_models.py
│   └── dashboard.py
├── database/
│   ├── mysql_setup.sql
│   └── mongodb_setup.js
├── ycsb/
│   ├── workload_results/
│   └── performance_comparison.csv
├── screenshots/
│   └── (implementation screenshots)
├── report/
│   └── project_report.docx
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+
pip install pyspark pandas numpy scikit-learn
pip install tensorflow keras statsmodels
pip install pymongo mysql-connector-python
pip install vaderSentiment nltk
pip install plotly dash matplotlib seaborn
```

### Database Setup
1. **MySQL**: Run `database/mysql_setup.sql`
2. **MongoDB**: Run `database/mongodb_setup.js`

### Running the Pipeline
```bash
# 1. Data Ingestion
python src/data_ingestion.py

# 2. Sentiment Analysis
python src/sentiment_analysis.py

# 3. Model Training & Forecasting
jupyter notebook notebooks/02_modeling_forecasting.ipynb

# 4. Launch Dashboard
python src/dashboard.py
```

---

## 📊 Results Summary

### Forecast Accuracy (Sample Results)
| Company | Model | 1-Day MAE | 3-Day MAE | 7-Day MAE |
|---------|-------|-----------|-----------|-----------|
| AAPL    | LSTM  | 2.34      | 5.67      | 8.92      |
| AAPL    | SARIMA| 3.12      | 6.45      | 10.23     |
| AMZN    | LSTM  | 4.56      | 9.87      | 15.34     |
| AMZN    | SARIMA| 5.23      | 11.23     | 17.89     |

*(Detailed results in the report)*

### Database Performance
- **MySQL**: Better for complex queries, transactions
- **MongoDB**: Superior for write-heavy workloads, flexible schema

---

## 📹 Screencast

**Demonstration Video**: [Google Drive Link]
- Duration: 7 minutes
- Contents:
  - Pipeline walkthrough
  - Database operations demonstration
  - Dashboard functionality
  - Forecast generation process

---

## 📚 References

All references follow Harvard citation style and are included in the detailed report.

---

## 👥 Author

**Student Name**: [Your Name]
**Student ID**: [Your ID]
**Programme**: MSc in Data Analytics
**Module**: Advanced Data Analytics & Big Data Storage and Processing

---

## 📝 Submission Checklist

- ✅ Written Report (Word document)
- ✅ Jupyter Notebooks (max 2)
- ✅ Source Code Files
- ✅ Dataset Files
- ✅ Screencast (Google Drive link)
- ✅ GitHub Repository with regular commits
- ✅ Database setup scripts
- ✅ YCSB benchmark results

---

## ⚖️ Academic Integrity

This project adheres to CCT College Dublin's academic integrity policies:
- AI tools used only for brainstorming and grammar checking
- No code generation by AI
- All implementations are original work
- Proper citations for all references

---

## 📧 Contact

For queries regarding this project:
- Email: [nimraaishere@gmail.com]


---

**Last Updated**: December 2025
