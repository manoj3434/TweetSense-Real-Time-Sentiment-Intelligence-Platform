# 🧠 TweetSense: Real-Time Sentiment Intelligence Platform

TweetSense is a real-time sentiment analysis platform that leverages **Natural Language Processing (NLP)** and **Deep Learning** to analyze social media posts and classify them into sentiment categories. Built with Streamlit and powered by Hugging Face Transformers, the application provides instant sentiment insights for user-defined keywords, hashtags, and phrases.

---

## 🚀 Features

* Real-time sentiment analysis of social media content
* Search using keywords, hashtags, or phrases
* Automated text preprocessing and cleaning
* Deep Learning-based sentiment classification
* Interactive dashboard built with Streamlit
* Confidence score generation for predictions
* Sentiment distribution visualization
* Tabular display of analyzed results

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning & NLP

* Hugging Face Transformers
* PyTorch
* NLP Techniques
* Sentiment Classification

### Data Processing

* Pandas
* NumPy

### Web Framework

* Streamlit

### APIs

* Twitter API
* Tweepy

---

## 📊 Dataset

The project utilizes Twitter data and sentiment-labeled social media datasets for model evaluation and testing.

Dataset Size:

* **74,000+ social media posts**

Sentiment Categories:

* Positive
* Negative
* Neutral
* Irrelevant

---

## ⚙️ Workflow

1. User enters a keyword, hashtag, or phrase.
2. Tweets are fetched using the Twitter API.
3. Text preprocessing removes:

   * URLs
   * Mentions
   * Special characters
   * Unwanted whitespace
4. Cleaned text is passed to the sentiment model.
5. The model predicts sentiment labels and confidence scores.
6. Results are visualized through interactive charts and tables.

---

## 🧠 Model

TweetSense uses the following transformer model:

```text
cardiffnlp/twitter-roberta-base-sentiment
```

The model is specifically trained for Twitter sentiment classification and is optimized for social media language.

---

## 📂 Project Structure

```bash
TweetSense-Real-Time-Sentiment-Intelligence-Platform/
│
├── app.py
├── requirements.txt
├── twitter_validation.csv
├── README.md
└── assets/
```

---

## ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/singhankiitt/TweetSense-Real-Time-Sentiment-Intelligence-Platform.git
```

Navigate to the project directory:

```bash
cd TweetSense-Real-Time-Sentiment-Intelligence-Platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

Set your Twitter API Bearer Token:

### Windows

```bash
set TWITTER_BEARER_TOKEN=YOUR_TOKEN_HERE
```

### Linux / macOS

```bash
export TWITTER_BEARER_TOKEN=YOUR_TOKEN_HERE
```

Alternatively, add the token to:

```toml
.streamlit/secrets.toml
```

```toml
[TWITTER]
bearer_token = "YOUR_TOKEN_HERE"
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📈 Sample Output

The application provides:

* Sentiment Labels
* Confidence Scores
* Sentiment Distribution Charts
* Interactive Data Tables
* Real-Time Social Media Insights

---

## 🔮 Future Enhancements

* Multi-language sentiment analysis
* Real-time trend monitoring
* Emotion detection
* Topic modeling
* Cloud deployment on AWS/GCP
* Historical sentiment tracking dashboard

---

## 👨‍💻 Author

**Ankit Kumar Singh**

GitHub: https://github.com/singhankiitt

LinkedIn: https://www.linkedin.com/in/ankit-singh-4014bb296

---

## ⭐ If you found this project useful, consider giving it a star!
