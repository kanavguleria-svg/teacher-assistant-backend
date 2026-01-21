from pathlib import Path
from django.db.models.signals import post_delete
from django.dispatch import receiver

from .content import Chapter, Topic


# Local cache directory (matches pdfparser CACHE_DIR)
CACHE_DIR = Path(".cache")


def _cleanup_cache_for_keyword(keyword: str):
    """Remove any cache files that contain the given keyword (case-insensitive)."""
    if not CACHE_DIR.exists():
        return

    for p in CACHE_DIR.glob("*.json"):
        try:
            text = p.read_text()
            if keyword.lower() in text.lower():
                p.unlink()
        except Exception:
            # ignore read/unlink errors to avoid blocking the delete
            continue


@receiver(post_delete, sender=Chapter)
def chapter_post_delete(sender, instance: Chapter, **kwargs):
    """When a chapter is deleted, remove any cache entries that reference it.

    Topics are cascaded by the DB (Topic.chapter on_delete=models.CASCADE),
    and their post_delete handlers will run as well. Here we also remove any
    cached files that mention the chapter name or id so future processing will
    re-run cleanly.
    """
    try:
        _cleanup_cache_for_keyword(str(instance.id))
        _cleanup_cache_for_keyword(instance.chapter_name or "")
    except Exception:
        pass


@receiver(post_delete, sender=Topic)
def topic_post_delete(sender, instance: Topic, **kwargs):
    """When a topic is deleted, remove any cache entries that reference it."""
    try:
        _cleanup_cache_for_keyword(str(instance.id))
        _cleanup_cache_for_keyword(instance.topic_name or "")
    except Exception:
        pass
