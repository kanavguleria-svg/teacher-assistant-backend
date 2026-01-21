"""database app package

Keep this file minimal. Signal handlers are registered from the AppConfig
`database.apps.DatabaseConfig.ready()` to avoid importing models while the
app registry is being populated.
"""

# For older Django versions that rely on default_app_config, point to our AppConfig.
# Newer Django versions auto-discover AppConfig in apps.py but this is harmless.
default_app_config = "database.apps.DatabaseConfig"
