# MIT License

# Copyright (c) [2022] [VVCB]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# EXCEPTION
# ----------
# The sentiment models developed by Imperial College London may be covered by other
# licenses. Please contact the project team for access to these models.

# Production code
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
import os
import pickle
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC

__version__ = "0.1.0"

# Set paths
DIR_MODELS = Path(os.getenv("DIR_MODELS", "../models"))

# Regex for missing comments. Many ways to do this.
MISSING_COMMENTS = [
    re.compile(f"^{x}$", flags=re.IGNORECASE)
    for x in ["no reply", "none", "no comment", "[0-9]+"]
]


if DIR_MODELS.exists() == False:
    raise IOError("Ensure models directory path is correct.")


def clean_comments(comments: Iterable) -> pd.Series:
    """Cleans comment test

    args:
        comments[Iterable]: Comment texts as an iterable (eg. list) or pandas series

    returns:
        Transformed/cleaned comments as pandas series.
    """
    comments = pd.Series(comments)
    # transform comment_text - strip and replace missing comments with np.nan
    return (
        comments.str.lower()
        .str.strip()
        .replace(MISSING_COMMENTS, np.nan, regex=True)
        .fillna("missing data")
    )


class ThemedSentimentProcessor:
    def __init__(self, org_name: str, dir_models: Path):

        org_name = org_name.lower()

        # If the model files are not found, this section will raise an unhandled exception

        self.feature_sentiment: dict = pickle.load(
            dir_models.joinpath(f"{org_name}_feature_sentiment.pkl").open("rb")
        )
        self.feature_theme: dict = pickle.load(
            dir_models.joinpath(f"{org_name}_feature_theme.pkl").open("rb")
        )

        self.tfidf_transformer_sentiment: TfidfTransformer = pickle.load(
            dir_models.joinpath(f"{org_name}_tfidftransformer_sentiment.pkl").open("rb")
        )
        self.tfidf_transformer_theme: TfidfTransformer = pickle.load(
            dir_models.joinpath(f"{org_name}_tfidftransformer_theme.pkl").open("rb")
        )

        self.classifier_sentiment: SVC = pickle.load(
            dir_models.joinpath(f"{org_name}_classifier_sentiment.pkl").open("rb")
        )
        self.classifier_theme: SVC = pickle.load(
            dir_models.joinpath(f"{org_name}_classifier_theme.pkl").open("rb")
        )

    def predict_sentiment(self, comment_text: Iterable) -> Iterable:
        """Returns a sentiment value between -1 (negative) and 1 (positive) for each comment"""

        countvect = CountVectorizer(
            decode_error="replace", vocabulary=self.feature_sentiment
        ).transform(comment_text)
        tfidf = self.tfidf_transformer_sentiment.transform(countvect)
        y_pred = self.classifier_sentiment.predict(tfidf)

        return y_pred

    def predict_theme(self, comment_text: Iterable) -> Iterable:
        """Returns the theme of each comment"""

        countvect = CountVectorizer(
            decode_error="replace", vocabulary=self.feature_theme
        ).transform(comment_text)
        tfidf = self.tfidf_transformer_theme.transform(countvect)
        y_pred = self.classifier_theme.predict(tfidf)

        return y_pred


# =============================================================================
# Start of FastAPI code
# =============================================================================

app = FastAPI(
    title="FFT Sentiment Analysis",
    description="""
    API for Friends and Family Test Sentiment Analysis
    For testing only.
    Models developed by Imperial College London
    and API developed by Lancashire Teaching Hospitals NHS Trust.
    """,
    root_path="",
)


class FFTInputCommentsModel(BaseModel):
    comment_id: str
    comment_text: str


class FFTInputModel(BaseModel):
    org_code: str
    comments: List[FFTInputCommentsModel]


class FFTOutputModel(FFTInputCommentsModel):
    comment_text_processed: str
    pred_sentiment: str  # =Field(description='One of Positive, Negative or Neutral')
    pred_theme: int  # = Field(description='One of several themes that relate to ')


@app.get("/")
def root():
    return {
        "status": "ok",
        "info": "Visit /docs for API documentation.",
        "api_version": __version__,
    }


@app.post("/analyse_sentiment", response_model=List[FFTOutputModel])
def analyse_sentiment_json(data: FFTInputModel):
    """Predict sentiments and themes of feedback received through Friends and Family Test surveys.

    Args:
        data (FFTInputModel): JSON data with ONS organisation code and list of comments.

    Returns:
        JSON: JSON array containing submitted data along with predicted theme and sentiment for each comment.
    """
    tsp = ThemedSentimentProcessor(org_name=data.org_code, dir_models=DIR_MODELS)

    # Load the data here.

    df = pd.DataFrame(i.__dict__ for i in data.comments)
    df["comment_text_processed"] = clean_comments(df.comment_text)

    df["pred_sentiment"] = tsp.predict_sentiment(df.comment_text_processed.values)
    df["pred_theme"] = tsp.predict_theme(df.comment_text_processed)

    # Assign "" to predicted sentiment and theme if original comment was 'missing data'
    df.loc[
        df.comment_text_processed == "missing data", ["pred_sentiment", "pred_theme"]
    ] = ""

    df.pred_sentiment = df.pred_sentiment.replace(
        {1: "Positive", 0: "Neutral", -1: "Negative"}
    )

    return df.to_dict(orient="records")
