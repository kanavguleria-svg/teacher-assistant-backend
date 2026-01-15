import uuid
from django.db import models


class Book(models.Model):
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )

    title = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "books"

    def __str__(self):
        return self.title


class Chapter(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="chapters")

    chapter_number = models.IntegerField()
    title = models.CharField(max_length=255)
    content = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chapters"
        ordering = ["chapter_number"]
        unique_together = ("book", "chapter_number")

    def __str__(self):
        return f"{self.book.title} – Chapter {self.chapter_number}"
