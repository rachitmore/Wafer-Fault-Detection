from django.shortcuts import render, redirect
from django.core.mail import send_mail
from .forms import ContactForm
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent


def home(request):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = BASE_DIR / "artifacts" / "logs" / "django" / "home_app"/f"{current_time}.txt"
    logging = App_Logger(log_path)
    try:
        logging.log("INFO", "Home page accessed.")
        return render(request, 'home/index.html')
    
    except Exception as e:
        logging.log("ERROR", f"Error rendering home page: {str(e)}")
        raise AppException("An error occurred while rendering the home page.", str(e))

def about(request):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = BASE_DIR / "artifacts" / "logs" / "django" / "home_app"/f"{current_time}.txt"
    logging = App_Logger(log_path)
    try:
        logging.log("INFO", "About page accessed.")
        return render(request, 'home/about.html')
    
    except Exception as e:
        logging.log("ERROR", f"Error rendering about page: {str(e)}")
        raise AppException("An error occurred while rendering the about page.", str(e))

def resume(request):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = BASE_DIR / "artifacts" / "logs" / "django" / "home_app"/f"{current_time}.txt"
    logging = App_Logger(log_path)
    try:
        logging.log("INFO", "Resume page accessed.")
        return render(request, 'home/resume.html')
    
    except Exception as e:
        logging.log("ERROR", f"Error rendering resume page: {str(e)}")
        raise AppException("An error occurred while rendering the resume page.", str(e))

def contact(request):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = BASE_DIR / "artifacts" / "logs" / "django" / "home_app"/f"{current_time}.txt"
    logging = App_Logger(log_path)    
    try:
        if request.method == 'POST':
            logging.log("INFO", "Contact form submission initiated.")
            form = ContactForm(request.POST)
            
            if form.is_valid():
                try:
                    logging.log("INFO", f"Contact form valid. Sending email to recipient.")
                    send_mail(
                        f"Message from {form.cleaned_data['name']}",
                        form.cleaned_data['message'],
                        form.cleaned_data['email'],
                        ['rachitmore3@gmail.com'],  # Replace with your email
                    )
                    logging.log("INFO", "Email sent successfully.")
                    return redirect('success')
                except Exception as email_error:
                    logging.log("ERROR", f"Failed to send email: {str(email_error)}")
                    raise AppException("An error occurred while sending the email.", str(email_error))
            else:
                logging.log("WARNING", "Contact form submission failed due to invalid data.")
        else:
            logging.log("INFO", "Contact page accessed.")
            form = ContactForm()
        
        return render(request, 'home/contact.html', {'form': form})
    except Exception as e:
        logging.log("ERROR", f"Error processing contact page: {str(e)}")
        raise AppException("An error occurred on the contact page.", str(e))

def success(request):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = BASE_DIR / "artifacts" / "logs" / "django" / "home_app"/f"{current_time}.txt"
    logging = App_Logger(log_path)
    try:
        logging.log("INFO", "Success page accessed.")
        return render(request, 'home/success.html')
    
    except Exception as e:
        logging.log("ERROR", f"Error rendering success page: {str(e)}")
        raise AppException("An error occurred while rendering the success page.", str(e))

def projects(request):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = BASE_DIR / "artifacts" / "logs" / "django" / "home_app"/f"{current_time}.txt"
    logging = App_Logger(log_path)
    try:
        logging.log("INFO", "Projects page accessed.")
        return render(request, 'home/projects.html')
    
    except Exception as e:
        logging.log("ERROR", f"Error rendering projects page: {str(e)}")
        raise AppException("An error occurred while rendering the projects page.", str(e))
