from langchain.tools import tool
from bs4 import BeautifulSoup
from typing import TypedDict, List
import requests
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class SearchResponse(TypedDict):
    url: str
    title: str
    content: str

@tool
def web_search(query: str) -> List[SearchResponse]:
    """
    Search the web for recent and reliable information on a topic.
    Returns: Title, URLs and Snippets
    """
    results = tavily.search(
        query=query,
        max_results=os.getenv("TAVILY_MAX_RESULTS", 5)
    )
    response = []
    for res in results['results']:
        response.append(
            SearchResponse(url=res['url'], title=res['title'], content=res['content'][:100])
        )
    return response

@tool
def scrape_url(url: str) -> str:
    """Scrap and return clean text content from a given URL for deeper reading"""
    try:
        response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
    
