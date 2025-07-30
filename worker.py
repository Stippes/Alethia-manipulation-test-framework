import os
import redis
from rq import Worker, Queue
from scripts.judge_conversation import judge_conversation_llm

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = redis.from_url(REDIS_URL)
queue = Queue(connection=redis_conn)

def start_worker():
    worker = Worker([queue], connection=redis_conn)
    worker.work()

if __name__ == "__main__":
    start_worker()
