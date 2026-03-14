import json
import requests

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST

def index(request):
    return render(request, "Comparer/index.html")

def predict(request):
    if request.method == "POST":
        url = "http://127.0.0.1:8080/full_predict"
        
        post_data = json.loads(request.body)
        prompt = post_data.get('prompt')
        response_a = post_data.get('response_a')
        response_b = post_data.get('response_b')
        
        if not(prompt and response_a and response_b):
            return JsonResponse({"error": "Empty value"}, status=400)
        if len(prompt) != len(response_a) or len(response_a) != len(response_b):
            return JsonResponse({"error": "Mismatch of prompt and response sizes"}, status=400)
        if len(prompt) == 0: #If prompt is 0 length then all are 0 length because we asserted equal length in previous condition
            return JsonResponse({"error": "Empty values"}, status=400)
        
        FastAPI_data = {
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b
        }
        try:
            response = requests.post(url, json=FastAPI_data)
            response.raise_for_status()
        except requests.RequestException as e:
            return JsonResponse({"error": "Failed to send data", "details": str(e)}, status=500)
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({"error": "Failed to send data", "status_code": response.status_code})
    return HttpResponse("Invalid Request")