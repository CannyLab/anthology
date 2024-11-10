import openai
import random
import time
import logging

from typing import Any, Callable, Collection, Generic, List, Optional, Type, TypeVar

Q = TypeVar("Q", bound=Callable[..., Any])


# define a retry decorator
def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    no_retry_on: Optional[Collection[Type[Exception]]] = None,
) -> Callable[[Q], Q]:
    """Retry a function with exponential backoff."""

    def decorator(func: Q) -> Q:
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            error = None

            # Loop until a successful response or max_retries is hit or an exception is raised
            while num_retries <= max_retries:
                try:
                    return func(*args, **kwargs)

                # Raise exceptions for any errors specified
                except Exception as e:
                    if no_retry_on is not None and type(e) in no_retry_on:
                        raise e

                    # Sleep for the delay
                    time.sleep(delay)

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Set the error to the last exception
                    error = e

                    # Increment retries
                    num_retries += 1

                    # logging.info(f"Retrying {func.__name__} after error: {e} (retry {num_retries} of {max_retries})")
                    print(
                        f"Retrying {func.__name__} after error: {e} (retry {num_retries} of {max_retries})"
                    )

            if error is not None:
                raise error

        return wrapper

    return decorator
