from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .api_functions.scrape_api import scrape_and_store_api
from .api_functions.chat_api import chat_api

@csrf_exempt
def scrape_view(request):
    if request.method == "POST":
        scrape_and_store_api()
        return JsonResponse({"status": "scrape complete"})
    return JsonResponse({"error": "Only POST allowed"}, status=405)

@csrf_exempt
def chat_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("query", "")
        if not query:
            return JsonResponse({"error": "No query provided"}, status=400)
        result = chat_api(query)
        return JsonResponse(result)
    return JsonResponse({"error": "Only POST allowed"}, status=405)
