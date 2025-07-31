web: gunicorn dashboard_app:server
worker: python worker.py 
worker: rq worker default --url $REDIS_URL
