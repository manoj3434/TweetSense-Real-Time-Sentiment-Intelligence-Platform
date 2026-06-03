import os
import re
from typing import List, Dict, Any

import pandas as pd
import plotly.express as px
import streamlit as st
import tweepy
from transformers import pipeline


def load_twitter_client(source: str = "Environment variable", bearer_token: str | None = None) -> tweepy.Client:
    if source == "Environment variable":
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    elif source == "Streamlit secrets":
        bearer_token = st.secrets.get("TWITTER", {}).get("bearer_token") if "TWITTER" in st.secrets else None
    elif source == "Paste token":
        bearer_token = bearer_token or None

    if not bearer_token:
        st.error(
            f"Twitter API bearer token is missing from {source}. "
            "Provide it via the selected source and then rerun the analysis."
        )
        st.stop()

    return tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)


@st.cache_resource
def load_sentiment_model() -> Any:
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
        device=-1,
    )


def clean_tweet_text(text: str) -> str:
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"@[A-Za-z0-9_]+", "", cleaned)
    cleaned = re.sub(r"#([A-Za-z0-9_]+)", r"\1", cleaned)
    cleaned = re.sub(r"&amp;", "&", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9\s\.,!?\'\"-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@st.cache_data
def load_demo_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv("twitter_validation.csv", header=None, names=["id", "entity", "sentiment", "tweet_text"])
        # Map sentiment labels
        sentiment_map = {
            'Positive': 'Positive',
            'Negative': 'Negative',
            'Neutral': 'Neutral',
            'Irrelevant': 'Neutral'  # Treat Irrelevant as Neutral
        }
        df['sentiment'] = df['sentiment'].map(sentiment_map)
        df['confidence'] = 0.85  # Set mock confidence for demo
        df['tweets'] = df['tweet_text'].apply(clean_tweet_text)
        return df[['tweet_text', 'tweets', 'sentiment', 'confidence']]
    except Exception as e:
        st.error(f"Failed to load demo dataset: {e}")
        # Fallback to sample data
        return pd.DataFrame({
            'tweet_text': [
                "I love this new feature! It's amazing! #streamlit",
                "This is okay, nothing special. Just average.",
                "Terrible experience, very disappointed. Won't recommend.",
                "Absolutely fantastic! Best thing ever! 😍",
                "Meh, it's alright but could be better.",
                "Hate it! Worst decision ever. 😡",
                "So excited about this! Can't wait to use it more!",
                "Not impressed at all. Very boring.",
                "This is incredible! Game changer! 🚀",
                "Disappointing. Expected more from this."
            ],
            'cleaned_text': [
                "I love this new feature! It's amazing! streamlit",
                "This is okay, nothing special. Just average.",
                "Terrible experience, very disappointed. Won't recommend.",
                "Absolutely fantastic! Best thing ever!",
                "Meh, it's alright but could be better.",
                "Hate it! Worst decision ever.",
                "So excited about this! Can't wait to use it more!",
                "Not impressed at all. Very boring.",
                "This is incredible! Game changer!",
                "Disappointing. Expected more from this."
            ],
            'sentiment': ['Positive', 'Neutral', 'Negative', 'Positive', 'Neutral', 'Negative', 'Positive', 'Negative', 'Positive', 'Negative'],
            'confidence': [0.9, 0.7, 0.85, 0.95, 0.6, 0.88, 0.92, 0.75, 0.98, 0.8]
        })


def fetch_recent_tweets(client: tweepy.Client, query: str, max_results: int = 50, demo_mode: bool = False) -> List[str]:
    if demo_mode:
        dataset = load_demo_dataset()
        # Filter tweets containing the query (case insensitive)
        filtered = dataset[dataset['tweet_text'].str.contains(query, case=False, na=False)]
        if len(filtered) < max(20, max_results // 2):  # Ensure at least 20 or half of max_results
            # Sample additional tweets to reach the minimum
            additional_needed = max(20, max_results) - len(filtered)
            remaining = dataset[~dataset.index.isin(filtered.index)]
            if len(remaining) > 0:
                sampled = remaining.sample(min(additional_needed, len(remaining)))
                filtered = pd.concat([filtered, sampled])
        tweets = filtered['tweet_text'].tolist()[:max_results]
        return tweets

    search_query = f"{query} -is:retweet lang:en"
    response = client.search_recent_tweets(
        query=search_query,
        tweet_fields=["created_at", "public_metrics", "lang"],
        max_results=max_results,
    )

    if not response.data:
        return []

    return [tweet.text for tweet in response.data]


def analyze_tweets(sentiment_model: Any, tweets: List[str], demo_mode: bool = False, dataset: pd.DataFrame = None) -> pd.DataFrame:
    if demo_mode and dataset is not None:
        # For demo, use the dataset sentiments directly
        filtered = dataset[dataset['tweet_text'].isin(tweets)].copy()
        filtered['tweets'] = filtered['tweet_text'].apply(clean_tweet_text)
        return filtered[['tweet_text', 'tweets', 'sentiment', 'confidence']].copy()

    cleaned_tweets = [clean_tweet_text(tweet) for tweet in tweets]
    raw_results = sentiment_model(cleaned_tweets, truncation=True)
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
    }

    records = []
    progress_bar = st.progress(0)
    for i, (original, cleaned, prediction) in enumerate(zip(tweets, cleaned_tweets, raw_results)):
        sentiment_label = label_map.get(prediction["label"], prediction["label"])
        records.append(
            {
                "tweet_text": original,
                "tweets": cleaned,
                "sentiment": sentiment_label,
                "confidence": round(float(prediction["score"]), 5),
            }
        )
        progress_bar.progress((i + 1) / len(tweets))

    progress_bar.empty()
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="confidence", ascending=False).reset_index(drop=True)
    return df


def render_dashboard(df: pd.DataFrame, query: str) -> None:
    st.subheader(f"Sentiment results for: {query}")
    st.markdown("**Most recent tweets analyzed:**")
    # Make dataframe scrollable horizontally
    st.dataframe(
        df[["tweets", "sentiment", "confidence"]], 
        use_container_width=True,
        height=400
    )

    sentiment_counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    st.markdown("### Sentiment distribution")
    chart_data = sentiment_counts.reset_index()
    chart_data.columns = ["Sentiment", "Count"]
    fig = px.pie(chart_data, names="Sentiment", values="Count", title="Sentiment Breakdown", color="Sentiment", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Summary statistics")
    st.write(
        {
            "Total tweets analyzed": len(df),
            "Positive": int(sentiment_counts["Positive"]),
            "Neutral": int(sentiment_counts["Neutral"]),
            "Negative": int(sentiment_counts["Negative"]),
        }
    )

    if not df.empty:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("#### Top positive tweets")
            for _, row in df[df["sentiment"] == "Positive"].head(5).iterrows():
                with st.expander(f"Confidence: {row['confidence']:.5f}"):
                    st.write(row["tweets"])
        with cols[1]:
            st.markdown("#### Top negative tweets")
            for _, row in df[df["sentiment"] == "Negative"].head(5).iterrows():
                with st.expander(f"Confidence: {row['confidence']:.5f}"):
                    st.write(row["tweets"])


def main() -> None:
    st.set_page_config(
        page_title="Real-Time Twitter Sentiment Analysis",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Real-Time Twitter Sentiment Analysis")
    st.write(
        "Enter a keyword, hashtag, or phrase to fetch recent tweets and analyze sentiment with a pre-trained Hugging Face model."
    )

    with st.sidebar:
        st.header("Configuration")
        query = st.text_input("Search keyword or hashtag", value="#streamlit")
        max_tweets = st.slider("Number of tweets to fetch", min_value=10, max_value=100, value=50, step=10)
        demo_mode = st.checkbox("Demo mode (use sample tweets)", value=True, help="Use sample tweets for testing without API credits")
        credential_source = st.radio(
            "Twitter bearer token source",
            ["Environment variable", "Streamlit secrets", "Paste token"],
            index=2,  # Default to "Paste token"
        )

        bearer_token_input = None
        if credential_source == "Paste token" and not demo_mode:
            bearer_token_input = st.text_input(
                "Paste bearer token",
                type="password",
                help="Enter your Twitter API bearer token for this session.",
            )
            if not bearer_token_input:
                st.warning("Please paste your Twitter bearer token to proceed.")

    if not query:
        st.warning("Please enter a search term to begin analysis.")
        return

    # Disable button if token is not provided for paste option and not demo mode
    button_disabled = not demo_mode and credential_source == "Paste token" and not bearer_token_input
    analyze_button = st.button("Fetch & Analyze", disabled=button_disabled)
    if analyze_button:
        with st.spinner("Fetching tweets and analyzing sentiment..."):
            if demo_mode:
                dataset = load_demo_dataset()
                tweets = fetch_recent_tweets(None, query, max_results=max_tweets, demo_mode=True)
                df = analyze_tweets(None, tweets, demo_mode=True, dataset=dataset)
            else:
                with st.spinner("Connecting to Twitter API..."):
                    client = load_twitter_client(source=credential_source, bearer_token=bearer_token_input)

                try:
                    tweets = fetch_recent_tweets(client, query, max_results=max_tweets)
                except tweepy.TweepyException as exc:
                    st.error(f"Twitter API error: {exc}")
                    return

                if not tweets:
                    st.warning("No tweets were found for that search term. Try a different keyword or expand your query.")
                    return

                try:
                    with st.spinner("Loading sentiment analysis model..."):
                        sentiment_model = load_sentiment_model()
                    df = analyze_tweets(sentiment_model, tweets)
                except Exception as exc:
                    st.error(f"Sentiment analysis failed: {exc}")
                    return

            render_dashboard(df, query)

    st.sidebar.markdown(
        "---\n"
        "#### Required environment variable\n"
        "`TWITTER_BEARER_TOKEN`\n\n"
        "#### Streamlit secrets example:\n"
        "```toml\n[TWITTER]\nbearer_token = \"YOUR_TOKEN_HERE\"\n```"
    )


if __name__ == "__main__":
    main()
