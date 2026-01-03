web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 300 --keep-alive 5 --max-requests 1000 --max-requests-jitter 50 --access-logfile - --error-logfile - src.dashboard.app:app

