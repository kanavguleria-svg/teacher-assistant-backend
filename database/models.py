import uuid
from django.db import models
from django.contrib.auth.models import User

from django.contrib.auth.models import AbstractUser

class Role(models.Model):
    STUDENT = "STUDENT"
    TEACHER = "TEACHER"
    PRINCIPAL = "PRINCIPAL"
    DISTRICT_MANAGER = "DISTRICT_MANAGER"
    STATE_MANAGER = "STATE_MANAGER"

    ROLE_CHOICES = [
        (STUDENT, "Student"),
        (TEACHER, "Teacher"),
        (PRINCIPAL, "Principal"),
        (DISTRICT_MANAGER, "District Manager"),
        (STATE_MANAGER, "State Manager"),
    ]

    name = models.CharField(
        max_length=50,
        choices=ROLE_CHOICES,
        unique=True
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Soft disable role without deleting"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = "roles"
        ordering = ["name"]

    def __str__(self):
        return self.name
    
class UserRole(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="user_roles"
    )

    role = models.ForeignKey(
        Role,
        on_delete=models.CASCADE,
        related_name="user_roles"
    )

    assigned_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "user_roles"
        unique_together = ("user", "role")
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["role"]),
        ]

    def __str__(self):
        return f"{self.user.username} → {self.role.name}"
    
class State(models.Model):
    name = models.CharField(max_length=100)

    class Meta:
        db_table = "states"


class District(models.Model):
    name = models.CharField(max_length=100)
    state = models.ForeignKey(State, on_delete=models.CASCADE, related_name="districts")

    class Meta:
        db_table = "districts"


class School(models.Model):
    name = models.CharField(max_length=255)
    district = models.ForeignKey(District, on_delete=models.CASCADE, related_name="schools")

    class Meta:
        db_table = "schools"

class AcademicYear(models.Model):
    year_start = models.PositiveIntegerField()  # e.g. 2024
    year_end = models.PositiveIntegerField()    # e.g. 2025

    is_active = models.BooleanField(default=False)

    class Meta:
        db_table = "academic_years"
        unique_together = ("year_start", "year_end")

class Grade(models.Model):
    number = models.PositiveIntegerField()  # 1 to 12
    name = models.CharField(max_length=50)  # Class 10, Grade 5

    class Meta:
        db_table = "grades"
        unique_together = ("number",)


class StudentEnrollment(models.Model):
    PROMOTED = "PROMOTED"
    REPEAT = "REPEAT"
    DROPPED = "DROPPED"

    RESULT_CHOICES = [
        (PROMOTED, "Promoted"),
        (REPEAT, "Repeat"),
        (DROPPED, "Dropped"),
    ]

    student = models.ForeignKey(User, on_delete=models.CASCADE)
    school = models.ForeignKey(School, on_delete=models.CASCADE)
    grade = models.ForeignKey(Grade, on_delete=models.CASCADE)
    academic_year = models.ForeignKey(AcademicYear, on_delete=models.CASCADE)

    roll_number = models.CharField(max_length=20)

    result = models.CharField(
        max_length=20,
        choices=RESULT_CHOICES,
        null=True,
        blank=True,
        help_text="Set at end of academic year"
    )

    class Meta:
        unique_together = ("student", "academic_year")

class TeacherAssignment(models.Model):
    teacher = models.ForeignKey(User, on_delete=models.CASCADE)
    school = models.ForeignKey(School, on_delete=models.CASCADE)
    grade = models.ForeignKey(Grade, on_delete=models.CASCADE)
    academic_year = models.ForeignKey(AcademicYear, on_delete=models.CASCADE)

    subject = models.CharField(max_length=100)

    class Meta:
        db_table = "teacher_assignments"


class ManagementScope(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    state = models.ForeignKey(State, null=True, blank=True, on_delete=models.CASCADE)
    district = models.ForeignKey(District, null=True, blank=True, on_delete=models.CASCADE)
    school = models.ForeignKey(School, null=True, blank=True, on_delete=models.CASCADE)

    class Meta:
        db_table = "management_scopes"
