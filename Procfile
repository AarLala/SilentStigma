web: gunicorn --bind 0.0.0.0:$PORT --workers 3 --threads 4 --timeout 300 --keep-alive 5 --max-requests 2000 --max-requests-jitter 100 --worker-class sync --preload --access-logfile - --error-logfile - src.dashboard.app:application

