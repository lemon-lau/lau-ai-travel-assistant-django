from datetime import datetime
from openai import OpenAI
from pinecone import Pinecone
from supabase import create_client

from django.conf import settings

# CREDENTIALS
open_ai_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    organization=settings.OPENAI_ORG_ID)
pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_API_KEY)

# CONNECT PINECONE INDEX
pinecone_index = pinecone_client.Index(settings.PINECONE_INDEX_NAME)


def chat_api(query: str):
    try:
        query_embed = open_ai_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=query
        ).data[0].embedding

        results = pinecone_index.query(vector=query_embed, top_k=5, include_metadata=True)

        sources = list(set([match['metadata']['source_url'] for match in results['matches']]))
        context = "\n".join([match['metadata']['text'] for match in results['matches']])

        response = open_ai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content":
                "You are a seasoned travel guide speaking directly to the traveler. \
                Respond in a confident and engaging tone, as if the knowledge you share comes from your own experience. \
                Use only the information provided in the context.\
                If the information is not available, simply say you don't have that information yet."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )

        answer = response.choices[0].message.content
        log_query_to_db(query, answer, sources)
    except Exception as e:
        print("User querying failed:", e)
        raise

    return {"answer": answer, "sources": sources}

def log_query_to_db(question, answer, sources):
    payload = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "sources": sources
    }

    try:
        supabase_client.table("query_logs").insert(payload).execute()
    except Exception as e:
        print("Supabase insert failed:", e)
        raise

def history_api():
    try:
        response = (
            supabase_client.table("query_logs")
            .select("*")
            .order("id", desc=True)
            .limit(50)
            .execute()
        )
        return response.data
    except Exception as e:
        print(e)
        return {"error": "Supabase query failed"}