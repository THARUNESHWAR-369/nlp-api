from fastapi import FastAPI, Query


from typing import List

app = FastAPI()

from modules.sentiment_analysis import SENTIMENT_ANALYSIS

@app.get("/")
def home():
    return 503

@app.get("/api/v1")
def api():
    return 503

@app.get("/api/v1/sentiment-analysis/")
async def sentiment(text: str):
    sentiment_obj = SENTIMENT_ANALYSIS(text=text)
    return sentiment_obj.getSentiment()
    
@app.get("/api/v1/sentiment-analysis/multi-text/")
async def sentiment(text: List[str] = Query(None)):
    sentiment_obj = SENTIMENT_ANALYSIS(textList=text)
    print(sentiment_obj.getMultipleSentiment())
    return sentiment_obj.getMultipleSentiment()
