from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .api_functions.scrape_api import scrape_and_store_api
from .api_functions.chat_api import chat_api, history_api

@csrf_exempt
def scrape_view(request):
    if request.method == "PATCH":
        scrape_and_store_api()
        return JsonResponse({"status": "scrape complete"})
    return JsonResponse({"error": "Only PATCH allowed"}, status=405)

@csrf_exempt
def chat_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        question = data.get("question", "")
        if not question:
            return JsonResponse({"error": "No question provided"}, status=400)
        result = chat_api(question)
        return JsonResponse(result)
    return JsonResponse({"error": "Only POST allowed"}, status=405)

@csrf_exempt
def history_view(request):
    if request.method == "GET":
        result = history_api()
        return JsonResponse(result, safe=False)
    return JsonResponse({"error": "Only GET allowed"}, status=405)

def health_check_view(request):
    return HttpResponse("OK")