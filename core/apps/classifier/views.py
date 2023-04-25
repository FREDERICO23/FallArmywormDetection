from django.shortcuts import render
from fastbook import *
from fastai.vision.all import *
import numpy as np

from base64 import b64encode
import pathlib
from django.conf import settings
from django.http import FileResponse, HttpResponse

from .forms import ImageUploadForm
from .models import UploadedImage
from .utils import generate_report

from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect
from django.contrib.auth import login,logout, authenticate
from django.contrib import messages
from .forms import SignUpForm, UserLoginForm
from .models import UploadedImage
from django.template.loader import get_template
from xhtml2pdf import pisa
from django.utils import timezone

User = get_user_model()

def home(request):
    return render(request, "home.html")

def register(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful." )
            return redirect("image-classifier")
        messages.error(request, "Unsuccessful registration. Invalid information.")
    form = SignUpForm()
    return render (request,"signup.html", {"register_form":form})

def logout_request(request):
    logout(request)
    messages.info(request, "You've successfully logged out")
    return redirect("home")

def login_request(request):
    if request.method == "POST":
        form = UserLoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                return redirect("image-classifier")
            else:
                messages.error(request,"Invalid username or password.")
        else:
            messages.error(request,"Invalid username or password.")
    form = UserLoginForm()
    return render(request, "login.html", {"login_form":form})


""" Model Inference"""

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
BASE_DIR = Path(__file__).resolve().parent.parent
filepath = os.path.join(BASE_DIR, 'classifier/resnet50_export.pkl')
model = load_learner(filepath)
classes = model.dls.vocab

classes = {
    0: 'healthy',
    1: 'fall armyworm attack',
}
severity_levels = {
    'healthy': ['low', 'medium', 'high'],
    'fall armyworm attack': ['low', 'medium', 'high'],

}

def estimate_severity(category, probs):
  
    if probs < 0.6:
        return severity_levels[category][0]  # Low severity
    elif 0.6 <= probs < 0.85:
        return severity_levels[category][1]  # Medium severity
    else:
        return severity_levels[category][2]  # High severity


def classify(img_file):
    img = PILImage.create(img_file)
    prediction = model.predict(img)
    probs_list = prediction[2].numpy()
    content = img_file.read()    
    encoded = b64encode(content)    
    img_file.seek(0)
    encoded = encoded.decode('ascii')
    mime = "image/jpg"
    image_uri = "data:%s;base64,%s" % (mime, encoded)

    category = classes[prediction[1].item()] # Use the updated classes dictionary
    probs = max(probs_list)
    severity = estimate_severity(category, probs)

    return {
        'image': image_uri,
        'category': category,
        'probs': "{:.2%}".format(probs),
        'severity': severity,
        'result': "It is {:.2%} {}. Severity level: {}!".format(probs, category, severity)
    }


def imageclassifier(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():  
            uploaded_image = form.save(commit=False) # don't commit to database yet
            # uploaded_image.image = request.FILES['image']
            media_path = os.path.join(settings.BASE_DIR, 'media', 'uploads')
            if not os.path.exists(media_path):
                os.makedirs(media_path)
            
            result = classify(request.FILES['image'])
            uploaded_image.prediction = result
              
            # Save the uploaded_image instance to the database
            
            uploaded_image._committed = True # set _committed attribute to True
            uploaded_image.save() # now save to database
            UploadedImage.objects.create(image=uploaded_image.image, result=result, date=timezone.now())

            context = {
                'form' : form,
                'result' : result,}
    
            return render(request, 'success.html', context) # Redirect to a success page

    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form})

def generate_report(request):
    if not request.user.is_authenticated:
        return redirect('login')
    
    preds = UploadedImage.objects.all()

    # Generate the PDF report
    template_path = 'report.html'
    context = {'preds': preds}
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'
    template = get_template(template_path)
    html = template.render(context)
    pisa_status = pisa.CreatePDF(html, dest=response)
    if pisa_status.err:
        return HttpResponse('An error occurred while generating the PDF.')
    return response


def download_report(request):
    report_file = generate_report()
    response = FileResponse(open(report_file, 'rb'), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(report_file)}"'
    return response