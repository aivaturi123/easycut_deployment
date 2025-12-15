from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from newspaper import Article
import re
import nltk
from datetime import datetime
from bs4 import BeautifulSoup  # for fallback raw html parsing

nltk.download('punkt')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CardRequest(BaseModel):
    url: str
    idea: str

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def get_full_text_with_fallback(article: Article) -> str:
    MIN_LENGTH = 300  # threshold, adjust if needed

    if article.text and len(article.text) > MIN_LENGTH:
        return article.text

    # fallback parsing raw html for paragraphs
    soup = BeautifulSoup(article.html, 'html.parser')
    paragraphs = soup.find_all('p')
    full_text = ' '.join(p.get_text() for p in paragraphs)
    return full_text

def highlight_sentences_with_boxes(text: str, tag_keywords: list[str], max_highlights: int = 4) -> str:
    sentences = nltk.sent_tokenize(text)
    scored_sentences = []

    warrant_keywords = {'because', 'therefore', 'proves', 'demonstrates', 'results in', 'leads to', 'shows that', 'causes'}

    for sent in sentences:
        score = 0
        lower_sent = sent.lower()

        # Score presence of tag keywords
        score += sum(1 for kw in tag_keywords if kw.lower() in lower_sent)

        # Score warrant phrases higher
        score += sum(2 for wk in warrant_keywords if wk in lower_sent)

        # Penalize very long sentences
        if len(sent.split()) > 40:
            score -= 1

        scored_sentences.append((score, sent.strip()))

    scored_sentences.sort(reverse=True)
    top_sentences = set(sent for _, sent in scored_sentences[:max_highlights])

    rebuilt = []
    for sent in sentences:
        clean_sent = sent.strip()
        if clean_sent in top_sentences:
            highlighted_sent = clean_sent
            for kw in sorted(tag_keywords, key=len, reverse=True):
                pattern = re.compile(rf'\b({re.escape(kw)})\b', re.IGNORECASE)
                highlighted_sent = pattern.sub(r'<b class="highlight-box">\1</b>', highlighted_sent)
            rebuilt.append(f'<span>{highlighted_sent}</span>')
        else:
            rebuilt.append(clean_sent)

    return ' '.join(rebuilt)

def make_citation(article: Article, url: str, tag: str) -> str:
    authors = article.authors if article.authors else ["Unknown Author"]
    year = article.publish_date.year if article.publish_date else datetime.now().year
    month_day = ''
    if article.publish_date:
        month_day = article.publish_date.strftime("%B %d")

    bracket_parts = []
    names = ', '.join(authors)
    bracket_parts.append(f"{names}")
    if month_day:
        bracket_parts.append(f"{month_day}")
    bracket_parts.append("No qualifications available")
    bracket_parts.append(f"{article.source_url or 'Unknown Source'}, \"{article.title}\", {url}")

    cite = f"{authors[0].split()[-1]} '{year} [{'; '.join(bracket_parts)}]"
    return cite

@app.post("/generate_card")
def generate_card(req: CardRequest):
    url = req.url
    idea = req.idea.strip()

    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

    text = get_full_text_with_fallback(article)
    text = clean_text(text)

    idea_words = idea.split()

    highlighted_excerpt = highlight_sentences_with_boxes(text, idea_words)

    citation = make_citation(article, url, idea)

    return {
        "tag": idea,
        "citation": citation,
        "excerpt": highlighted_excerpt
    }
