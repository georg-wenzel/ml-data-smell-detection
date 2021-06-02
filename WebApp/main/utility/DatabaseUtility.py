# Utility functions for database calls

# returns an object of type model with given keyword arguments, or None if it does not exist
def safe_get(model, *args, **kwargs):
    try:
        return model.objects.get(*args, **kwargs)
    except model.DoesNotExist:
        return None