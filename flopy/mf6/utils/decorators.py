import functools
import inspect
import warnings

def deprecated(instructions):
    """
    Flags a method as deprecated.

    Parameters:
        instructions: A human-friendly string of instructions
    """
    def decorator(func):
        """
        This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being raised
        when the function is used.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = 'Call to deprecated function <{}>. {}'.format(
                func.__name__,
                instructions)

            frame = inspect.currentframe().f_back
            warnings.simplefilter('default', DeprecationWarning)
            warnings.warn_explicit(message, category=DeprecationWarning,
                                   filename=inspect.getfile(frame.f_code),
                                   lineno=frame.f_lineno)

            return func(*args, **kwargs)

        return wrapper

    return decorator
