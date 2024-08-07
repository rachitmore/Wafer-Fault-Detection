from django.shortcuts import render, redirect
from django.core.mail import send_mail
from .forms import ContactForm

def home(request):
    return render(request, 'home/index.html')

def about(request):
    return render(request, 'home/about.html')

def resume(request):
    return render(request, 'home/resume.html')

def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Send email or handle the form submission
            send_mail(
                f"Message from {form.cleaned_data['name']}",
                form.cleaned_data['message'],
                form.cleaned_data['email'],
                ['rachitmore3@gmail.com'],  # Replace with your email
            )
            return redirect('success')
    else:
        form = ContactForm()
    
    return render(request, 'home/contact.html', {'form': form})

def success(request):
    return render(request, 'home/success.html')

def projects(request):
    return render(request, 'home/projects.html')