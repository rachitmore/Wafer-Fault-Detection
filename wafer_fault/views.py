from django.shortcuts import render

def home(request):
    return render(request, 'prediction_templates/index.html')
