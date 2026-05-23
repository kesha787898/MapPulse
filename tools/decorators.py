from functools import wraps

def cache_result(cache):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = args[0]

            cached = cache.get(key)
            if cached:
                return cached

            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        return wrapper
    return decorator