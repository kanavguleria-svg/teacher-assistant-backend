from django.apps import AppConfig


class DatabaseConfig(AppConfig):
    name = "database"

    def ready(self):
        # Import signal handlers to register them. Importing here ensures the
        # app registry is fully populated and avoids "Apps aren't loaded yet"
        # errors that occur when importing models at module import time.
        try:
            from . import signals  # noqa: F401
        except Exception:
            # Don't let signal registration break startup in exotic environments
            pass
