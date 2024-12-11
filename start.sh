#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Export Flask environment variables
export FLASK_APP=app.py
export FLASK_ENV=production

# Start Gunicorn
gunicorn --config gunicorn_config.py app:app
