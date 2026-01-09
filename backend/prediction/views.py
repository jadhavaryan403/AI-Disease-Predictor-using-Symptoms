from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.paginator import Paginator
from django.db import transaction, IntegrityError

from .models import DoctorProfile, Patient, MedicalNote, DiseasePrediction
from .ml_service import get_predictor


# --------------------------------------------------
# AUTH
# --------------------------------------------------

def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'home.html')


def register_view(request):
    if request.method == 'POST':
        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=request.POST.get('username'),
                    password=request.POST.get('password'),
                    email=request.POST.get('email'),
                    first_name=request.POST.get('first_name'),
                    last_name=request.POST.get('last_name')
                )

                DoctorProfile.objects.update_or_create(
                    user=user,
                    defaults={
                        "specialization": request.POST.get('specialization'),
                        "phone": request.POST.get('phone')
                    }
                )

            messages.success(request, "Registration successful. Please login.")
            return redirect('login')

        except IntegrityError:
            messages.error(request, "User already exists.")
            return redirect('register')

    return render(request, 'register.html')



def login_view(request):
    if request.method == 'POST':
        user = authenticate(
            request,
            username=request.POST.get('username'),
            password=request.POST.get('password')
        )

        if user:
            login(request, user)
            return redirect('dashboard')

        messages.error(request, "Invalid credentials")

    return render(request, 'login.html')


@login_required
def logout_view(request):
    logout(request)
    return redirect('home')


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@login_required
def dashboard(request):
    doctor = request.user.doctor_profile

    total_patients = Patient.objects.filter(doctor=doctor).count()
    total_predictions = DiseasePrediction.objects.filter(
        medical_note__doctor=request.user
    ).count()

    context = {
        'total_patients': total_patients,
        'total_predictions': total_predictions,
        'recent_patients': Patient.objects.filter(doctor=doctor)[:5],
        'recent_predictions': DiseasePrediction.objects.filter(
            medical_note__doctor=request.user
        )[:5],
    }

    return render(request, 'dashboard.html', context)


# --------------------------------------------------
# PATIENTS
# --------------------------------------------------

@login_required
def patients_list(request):
    doctor = request.user.doctor_profile
    patients = Patient.objects.filter(doctor=doctor)

    paginator = Paginator(patients, 10)
    page = paginator.get_page(request.GET.get('page'))

    return render(request, 'patients_list.html', {'patients': page})


@login_required
def add_patient(request):
    if request.method == "POST":

        doctor_profile = get_object_or_404(
            DoctorProfile,
            user=request.user
        )

        Patient.objects.create(
            doctor=doctor_profile,   
            name=request.POST["name"],
            age=request.POST["age"],
            gender=request.POST["gender"],
            contact=request.POST["contact"],
            diagnosis=None           
        )

        return redirect("patients_list")

    return render(request, "add_patient.html")



@login_required
def patient_detail(request, patient_id):
    doctor = request.user.doctor_profile
    patient = get_object_or_404(Patient, id=patient_id, doctor=doctor)

    return render(request, 'patient_detail.html', {
        'patient': patient,
        'medical_notes': patient.medical_notes.all()
    })


@login_required
def edit_patient(request, patient_id):
    doctor_profile = get_object_or_404(
        DoctorProfile,
        user=request.user
    )

    patient = get_object_or_404(
        Patient,
        id=patient_id,
        doctor=doctor_profile   
    )

    if request.method == "POST":
        patient.name = request.POST.get("name")
        patient.age = request.POST.get("age")
        patient.gender = request.POST.get("gender")
        patient.contact = request.POST.get("contact")
        patient.diagnosis = request.POST.get("diagnosis") or None

        patient.save()

        return redirect("patient_detail", patient_id=patient.id)

    return render(request, "edit_patient.html", {"patient": patient})


@login_required
def delete_patient(request, patient_id):
    doctor_profile = get_object_or_404(
        DoctorProfile,
        user=request.user
    )

    patient = get_object_or_404(
        Patient,
        id=patient_id,
        doctor=doctor_profile  
    )

    if request.method == "POST":
        patient.delete()
        return redirect("patients_list")

    return render(request, "confirm_delete.html", {"patient": patient})


# --------------------------------------------------
# MEDICAL NOTES + ML
# --------------------------------------------------

@login_required
def add_medical_note(request, patient_id):
    doctor = request.user.doctor_profile
    patient = get_object_or_404(Patient, id=patient_id, doctor=doctor)

    if request.method == 'POST':
        note = MedicalNote.objects.create(
            patient=patient,
            doctor=request.user,
            note_text=request.POST.get('note_text'),
        )

        try:
            predictor = get_predictor()

            result = predictor.process_medical_note(
                note_text=note.note_text,
            )

            DiseasePrediction.objects.create(
                medical_note=note,
                predicted_diseases=result['top_predictions'],
                best_prediction=result['best_prediction'],
                confidence=result['best_confidence'],
                model_name="tiny-clinicalbert-ann"
            )

            messages.success(request, "Prediction completed")
            return redirect('prediction_result', note_id=note.id)

        except Exception as e:
            messages.error(request, f"Prediction failed: {e}")
            return redirect('patient_detail', patient_id=patient.id)

    return render(request, 'add_medical_note.html', {'patient': patient})


@login_required
def prediction_result(request, note_id):
    note = get_object_or_404(MedicalNote, id=note_id, doctor=request.user)

    if not hasattr(note, 'prediction'):
        messages.error(request, "Prediction not found")
        return redirect('patient_detail', patient_id=note.patient.id)

    return render(request, 'prediction_result.html', {
        'medical_note': note,
        'prediction': note.prediction,
        'patient': note.patient
    })


# --------------------------------------------------
# PROFILE
# --------------------------------------------------

@login_required
def profile_view(request):
    profile = request.user.doctor_profile

    if request.method == 'POST':
        request.user.first_name = request.POST.get('first_name')
        request.user.last_name = request.POST.get('last_name')
        request.user.email = request.POST.get('email')
        request.user.save()

        profile.specialization = request.POST.get('specialization')
        profile.phone = request.POST.get('phone')
        profile.save()

        messages.success(request, "Profile updated")
        return redirect('profile')

    return render(request, 'profile.html', {'doctor_profile': profile})
