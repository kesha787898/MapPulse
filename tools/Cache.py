import json
import os
from threading import RLock


class Cache:
    def __init__(self, filename):
        self.filename = filename
        self.lock = RLock()
        self.cache = self._load()

    def _load(self):
        if not os.path.exists(self.filename):
            return {}

        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def get(self, key):
        with self.lock:
            return self.cache.get(key)

    def put(self, key, value):
        with self.lock:
            self.cache[key] = value
            self._save()

    def delete(self, key):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self._save()

    def clear(self):
        with self.lock:
            self.cache = {}
            self._save()