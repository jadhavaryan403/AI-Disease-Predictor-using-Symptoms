from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class DoctorProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='doctor_profile')
    specialization = models.CharField(max_length=200)
    phone = models.CharField(max_length=15)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Dr. {self.user.get_full_name() or self.user.username}"


class Patient(models.Model):
    GENDER_CHOICES = [('M', 'Male'), ('F', 'Female'), ('O', 'Other')]

    doctor = models.ForeignKey(
        DoctorProfile,
        on_delete=models.CASCADE,
        related_name='patients'
    )
    name = models.CharField(max_length=200)
    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    contact = models.CharField(max_length=15)
    diagnosis = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class MedicalNote(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='medical_notes')
    doctor = models.ForeignKey(User, on_delete=models.CASCADE)
    note_text = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Note for {self.patient.name}"


class DiseasePrediction(models.Model):
    medical_note = models.OneToOneField(
        MedicalNote,
        on_delete=models.CASCADE,
        related_name='prediction'
    )
    predicted_diseases = models.JSONField()
    best_prediction = models.CharField(max_length=200)
    confidence = models.FloatField()
    model_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction for {self.medical_note.patient.name}"
