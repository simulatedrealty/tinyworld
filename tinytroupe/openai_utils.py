import os
import openai
from openai import OpenAI, AzureOpenAI
import time
import json
import pickle
import logging
import configparser
import tiktoken
from tinytroupe import utils

logger = logging.getLogger("tinytroupe")

# We'll use various configuration elements below
config = utils.read_config_file()

###########################################################################
# Default parameter values
###########################################################################

default_model = config["OpenAI"].get("TINYPERSON_MODEL", "gpt-4")
default_max_tokens = int(config["OpenAI"].get("TINYPERSON_MAX_TOKENS", "1024"))
default_temperature = float(config["OpenAI"].get("TINYPERSON_TEMPERATURE", "0.3"))
default_top_p = int(config["OpenAI"].get("TINYPERSON_TOP_P", "0"))
default_frequency_penalty = float(config["OpenAI"].get("TINYPERSON_FREQ_PENALTY", "0.0"))
default_presence_penalty = float(
    config["OpenAI"].get("TINYPERSON_PRESENCE_PENALTY", "0.0"))
default_timeout = float(config["OpenAI"].get("TINYPERSON_TIMEOUT", "30.0"))
default_max_attempts = float(config["OpenAI"].get("TINYPERSON_MAX_ATTEMPTS", "0.0"))
default_waiting_time = float(config["OpenAI"].get("TINYPERSON_WAITING_TIME", "0.5"))
default_exponential_backoff_factor = float(config["OpenAI"].get("TINYPERSON_EXPONENTIAL_BACKOFF_FACTOR", "5"))

default_cache_api_calls = config["OpenAI"].getboolean("TINYPERSON_CACHE_API_CALLS", False)
default_cache_file_name = config["OpenAI"].get("TINYPERSON_CACHE_FILE_NAME", "openai_api_cache.pickle")

###########################################################################
# Client class
###########################################################################

class OpenAIClient:
    """
    A utility class for interacting with the OpenAI API.
    """

    def __init__(self, cache_api_calls=default_cache_api_calls, cache_file_name=default_cache_file_name) -> None:
        logger.debug("Initializing OpenAIClient")

        # should we cache api calls and reuse them?
        self.set_api_cache(cache_api_calls, cache_file_name)
    
    def set_api_cache(self, cache_api_calls, cache_file_name=default_cache_file_name):
        """
        Enables or disables the caching of API calls.

        Args:
        cache_file_name (str): The name of the file to use for caching API calls.
        """
        self.cache_api_calls = cache_api_calls
        self.cache_file_name = cache_file_name
        if self.cache_api_calls:
            # load the cache, if any
            self.api_cache = self._load_cache()
    
    
    def _setup_from_config(self):
        """
        Sets up the OpenAI API configurations for this client.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def send_message(self,
                    current_messages,
                     model=default_model,
                     temperature=default_temperature,
                     max_tokens=default_max_tokens,
                     top_p=default_top_p,
                     frequency_penalty=default_frequency_penalty,
                     presence_penalty=default_presence_penalty,
                     stop=None,
                     timeout=default_timeout,
                     max_attempts=default_max_attempts,
                     waiting_time=default_waiting_time,
                     exponential_backoff_factor=default_exponential_backoff_factor,
                     n = 1,
                     echo=False):
        """
        Sends a message to the OpenAI API and returns the response.

        Args:
        current_messages (list): A list of dictionaries representing the conversation history.
        model (str): The ID of the model to use for generating the response.
        temperature (float): Controls the "creativity" of the response. Higher values result in more diverse responses.
        max_tokens (int): The maximum number of tokens (words or punctuation marks) to generate in the response.
        top_p (float): Controls the "quality" of the response. Higher values result in more coherent responses.
        frequency_penalty (float): Controls the "repetition" of the response. Higher values result in less repetition.
        presence_penalty (float): Controls the "diversity" of the response. Higher values result in more diverse responses.
        stop (str): A string that, if encountered in the generated response, will cause the generation to stop.
        max_attempts (int): The maximum number of attempts to make before giving up on generating a response.
        timeout (int): The maximum number of seconds to wait for a response from the API.

        Returns:
        A dictionary representing the generated response.
        """

        def aux_exponential_backoff():
            nonlocal waiting_time
            logger.info(f"Request failed. Waiting {waiting_time} seconds between requests...")
            time.sleep(waiting_time)

            # exponential backoff
            waiting_time = waiting_time * exponential_backoff_factor
        

        # setup the OpenAI configurations for this client.
        self._setup_from_config()
        
        # We need to adapt the parameters to the API type, so we create a dictionary with them first
        chat_api_params = {
            "messages": current_messages,
            "temperature": temperature,
            "max_tokens":max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "timeout": timeout,
            "stream": False,
            "n": n,
        }


        i = 0
        while i < max_attempts:
            try:
                i += 1

                try:
                    logger.debug(f"Sending messages to OpenAI API. Token count={self._count_tokens(current_messages, model)}.")
                except NotImplementedError:
                    logger.debug(f"Token count not implemented for model {model}.")
                    
                start_time = time.monotonic()
                logger.debug(f"Calling model with client class {self.__class__.__name__}.")

                ###############################################################
                # call the model, either from the cache or from the API
                ###############################################################
                cache_key = str((model, chat_api_params)) # need string to be hashable
                if self.cache_api_calls and (cache_key in self.api_cache):
                    response = self.api_cache[cache_key]
                else:
                    logger.info(f"Waiting {waiting_time} seconds before next API request (to avoid throttling)...")
                    time.sleep(waiting_time)
                    
                    response = self._raw_model_call(model, chat_api_params)
                    if self.cache_api_calls:
                        self.api_cache[cache_key] = response
                        self._save_cache()
                
                
                logger.debug(f"Got response from API: {response}")
                end_time = time.monotonic()
                logger.debug(
                    f"Got response in {end_time - start_time:.2f} seconds after {i + 1} attempts.")

                return self._raw_model_response_extractor(response)

            except InvalidRequestError as e:
                logger.error(f"[{i}] Invalid request error, won't retry: {e}")

                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None
            
            except openai.BadRequestError as e:
                logger.error(f"[{i}] Invalid request error, won't retry: {e}")
                
                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None
            
            except openai.RateLimitError:
                logger.warning(
                    f"[{i}] Rate limit error, waiting a bit and trying again.")
                aux_exponential_backoff()
            
            except NonTerminalError as e:
                logger.error(f"[{i}] Error: {e}")
                aux_exponential_backoff()
                
            except Exception as e:
                logger.error(f"[{i}] Error: {e}")

        logger.error(f"Failed to get response after {max_attempts} attempts.")
        return None
    
    def _raw_model_call(self, model, chat_api_params):
        """
        Calls the OpenAI API with the given parameters. Subclasses should
        override this method to implement their own API calls.
        """
        
        chat_api_params["model"] = model # OpenAI API uses this parameter name
        return self.client.chat.completions.create(
                    **chat_api_params
                )

    def _raw_model_response_extractor(self, response):
        """
        Extracts the response from the API response. Subclasses should
        override this method to implement their own response extraction.
        """
        return dict(response.choices[0].message)

    def _count_tokens(self, messages: list, model: str):
        """
        Count the number of OpenAI tokens in a list of messages using tiktoken.

        Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Args:
        messages (list): A list of dictionaries representing the conversation history.
        model (str): The name of the model to use for encoding the string.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.debug("Token count: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            logger.debug("Token count: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self._count_tokens(messages, model="gpt-3.5-turbo-0613")
        elif ("gpt-4" in model) or ("ppo" in model):
            logger.debug("Token count: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self._count_tokens(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _save_cache(self):
        """
        Saves the API cache to disk. We use pickle to do that because some obj
        are not JSON serializable.
        """
        # use pickle to save the cache
        pickle.dump(self.api_cache, open(self.cache_file_name, "wb"))

    
    def _load_cache(self):
        """
        Loads the API cache from disk.
        """
        # unpickle
        return pickle.load(open(self.cache_file_name, "rb")) if os.path.exists(self.cache_file_name) else {}


class AzureClient(OpenAIClient):

    def __init__(self) -> None:
        logger.debug("Initializing AzureClient")
    
    def _setup_from_config(self):
        """
        Sets up the Azure OpenAI Service API configurations for this client,
        including the API endpoint and key.
        """
        self.client = AzureOpenAI(azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
                                  api_version = config["OpenAI"]["AZURE_API_VERSION"],
                                  api_key = os.getenv("AZURE_OPENAI_KEY"))
    
    def _raw_model_call(self, model, chat_api_params):
        """
        Calls the Azue OpenAI Service API with the given parameters.
        """
        # Looks like Azue OpenAI Service API uses this parameter name,
        # which is different from what the OpenAI API uses ("model")
        # No idea why, but let's adapt it here.
        chat_api_params["engine"] = model 

        return openai.ChatCompletion.create(
                    **chat_api_params
                )


class InvalidRequestError(Exception):
    """
    Exception raised when the request to the OpenAI API is invalid.
    """
    pass

class NonTerminalError(Exception):
    """
    Exception raised when an unspecified error occurs but we know we can retry.
    """
    pass

###########################################################################
# Clients registry
#
# We can have potentially different clients, so we need a place to 
# register them and retrieve them when needed.
#
# We support both OpenAI and Azure OpenAI Service API by default.
# Thus, we need to set the API parameters based on the choice of the user.
# This is done within specialized classes.
#
# It is also possible to register custom clients, to access internal or
# otherwise non-conventional API endpoints.
###########################################################################
_api_type_to_client = {}
_api_type_override = None

def register_client(api_type, client):
    """
    Registers a client for the given API type.

    Args:
    api_type (str): The API type for which we want to register the client.
    client: The client to register.
    """
    _api_type_to_client[api_type] = client

def _get_client_for_api_type(api_type):
    """
    Returns the client for the given API type.

    Args:
    api_type (str): The API type for which we want to get the client.
    """
    try:
        return _api_type_to_client[api_type]
    except KeyError:
        raise ValueError(f"API type {api_type} is not supported. Please check the 'config.ini' file.")

def client():
    """
    Returns the client for the configured API type.
    """
    api_type = config["OpenAI"]["API_TYPE"] if _api_type_override is None else _api_type_override
    
    logger.debug(f"Using  API type {api_type}.")
    return _get_client_for_api_type(api_type)

def force_api_type(api_type):
    """
    Forces the use of the given API type, thus overriding any other configuration.

    Args:
    api_type (str): The API type to use.
    """
    global _api_type_override
    _api_type_override = api_type

def force_api_cache(cache_api_calls, cache_file_name=default_cache_file_name):
    """
    Forces the use of the given API cache configuration, thus overriding any other configuration.

    Args:
    cache_api_calls (bool): Whether to cache API calls.
    cache_file_name (str): The name of the file to use for caching API calls.
    """
    # set the cache parameters on all clients
    for client in _api_type_to_client.values():
        client.set_api_cache(cache_api_calls, cache_file_name)

# default client
register_client("openai", OpenAIClient())
register_client("azure", AzureClient())
    


