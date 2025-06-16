import json
import uuid
import requests
import tiktoken

from bs4 import BeautifulSoup
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from django.conf import settings

# CREDENTIALS
open_ai_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    organization=settings.OPENAI_ORG_ID)
pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)


# CHECK AND CREATE INDEX THEN CONNECT
if settings.PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=3072,  # text-embedding-3-large
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
pinecone_index = pinecone_client.Index(settings.PINECONE_INDEX_NAME)


# MAIN FUNCTION FOR /SCRAPE ENDPOINT
def scrape_and_store_api():
    pages_to_scrape = settings.SCRAPED_PAGES_URL_MAP
    for label, url in pages_to_scrape.items():
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            chunked_article = chunk_article(response.text, label, url)
            embed_and_store_chunked_article(chunked_article)
        except Exception as e:
            print(f"Failed to scrape and store {label}: {e}")

    return {
        "status_code": 200,
        "message": "Scrape and Store Successful"
    }


# CHUNK ARTICLE FROM URL BY PARAGRAPH SECTIONS AND LISTS INTO OPTIMAL TOKEN SIZE
def chunk_article(html: str,  label: str, source_url: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    div_id = 'cruises-text-container'
    article_div = soup.find("div", id=div_id)

    if not article_div:
        raise ValueError(f"No div with id='{div_id}' found")

    # REMOVE SITE NOISES
    ignored_classes = [".btn-share-wrapper", ".st-ad-default", ".ads-container"]
    for class_to_ignore in ignored_classes:
        for el in article_div.select(class_to_ignore):
            el.decompose()

    article_chunks = []

    # EXTRACT PARAGRAPH SECTION CHUNKS FROM DOM
    sections = article_div.find_all('article')
    for section in sections:
        section_title_el = section.find(
            lambda tag: tag.has_attr("class") and any(cls.endswith("__title") for cls in tag["class"])
        )
        content_el = section.find(
            lambda tag: tag.has_attr("class") and any(cls.endswith("__content") for cls in tag["class"])
        )
        if not section_title_el and not content_el:
            continue

        section_title = section_title_el.get_text(separator="\n", strip=True) if section_title_el else ""
        section_content = content_el.get_text(separator="\n", strip=True).split("\n") if content_el else ""
        section_text = "\n".join(section_content)

        chunks = chunk_text_tokenwise(section_text)
        for chunk in chunks:
            article_chunks.append({
                "section": section_title,
                "text": chunk,
                "label": label,
                "source_url": source_url
            })

    # EXTRACT LIST SECTION CHUNKS FROM REACT DATA
    react_classes = {
        "OtherPorts": "https://www.shermanstravel.com/ports_from_cruise_destination?cruiseDestinationId={}",
        "OtherItems": "https://www.shermanstravel.com/ships_from_cruise_destination?cruiseDestinationId={}"
    }

    react_divs = soup.find_all('div', attrs={'data-react-class': True})
    for div in react_divs:
        react_class = div.get('data-react-class')
        if react_class in react_classes:
            props_json = div.get('data-react-props')
            if not props_json:
                continue
            props = json.loads(props_json)
            title = props.get("title") or props.get("destinationName") + f" ({react_class})"
            cruiseDestinationId = props.get('cruiseDestinationId')

            api_url = react_classes[react_class].format(cruiseDestinationId)
            try:
                response = requests.get(api_url)
                loaded_items = response.json()
                details = ', '.join(
                    f"{item['title']} ({item['subtitle']})" if item.get('subtitle') else f"{item['title']}"
                    for item in loaded_items
                )

                chunks = chunk_text_tokenwise(details)
                for chunk in chunks:
                    article_chunks.append({
                        "section": title,
                        "text": chunk,
                        "label": label,
                        "source_url": source_url
                    })

            except Exception as e:
                raise ValueError(f"Error fetching {api_url}: {e}")

    return article_chunks


# HELPER FUNCTION TO CHUNK LONG TEXTS TO OPTIMAL TOKEN SIZE
def chunk_text_tokenwise(text, max_tokens=300, overlap=50):
    enc = tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap  # OVERLAP TO PRESERVE CONTEXT

    return chunks


# EMBED CHUNKS USING OPENAI AND STORE IN PINECONE INDEX BY BATCH
def embed_and_store_chunked_article(chunked_article):
    BATCH_SIZE = 10

    for i in tqdm(range(0, len(chunked_article), BATCH_SIZE)):
        chunk_batch = chunked_article[i:i + BATCH_SIZE]
        texts_batch = [cb["text"] for cb in chunk_batch]
        metadata = [{
            "label": cb["label"],
            "section": cb["section"],
            "text": cb["text"],
            "source_url": cb["source_url"]
        } for cb in chunk_batch]

        # EMBED CHUNKS
        response = open_ai_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts_batch
        )
        vectors = response.data

        # FORMAT VECTORS AND UPSERT IN INDEX
        pinecone_vectors = [
            {
                "id": str(uuid.uuid4()),
                "values": vector.embedding,
                "metadata": metadata[idx]
            }
            for idx, vector in enumerate(vectors)
        ]
        pinecone_index.upsert(vectors=pinecone_vectors)
