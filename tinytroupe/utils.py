"""
General utilities and convenience functions.
"""
import re
import json
import hashlib
import logging
from pathlib import Path
import configparser
from typing import Any, TypeVar, Union
AgentOrWorld = Union["TinyPerson", "TinyWorld"]

# logger
logger = logging.getLogger("tinytroupe")

def name_or_empty(named_entity: AgentOrWorld):
    """
    Returns the name of the specified agent or environment, or an empty string if the agent is None.
    """
    if named_entity is None:
        return ""
    else:
        return named_entity.name

################################################################################
# Model control utilities
################################################################################    

def repeat_on_error(retries:int, exceptions:list):
    """
    Decorator that repeats the specified function call if an exception among those specified occurs, 
    up to the specified number of retries. If that number of retries is exceeded, the
    exception is raised. If no exception occurs, the function returns normally.

    Args:
        retries (int): The number of retries to attempt.
        exceptions (list): The list of exception classes to catch.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    logger.debug(f"Exception occurred: {e}")
                    if i == retries - 1:
                        raise e
                    else:
                        logger.debug(f"Retrying ({i+1}/{retries})...")
                        continue
        return wrapper
    return decorator
    


################################################################################	
# Model output utilities
################################################################################
def extract_json(text: str) -> dict:
    """
    Extracts a JSON object from a string, ignoring: any text before the first 
    opening curly brace; and any Markdown opening (```json) or closing(```) tags.
    """
    try:
        # remove any text before the first opening curly or square braces, using regex. Leave the braces.
        text = re.sub(r'^.*?({|\[)', r'\1', text, flags=re.DOTALL)

        # remove any trailing text after the LAST closing curly or square braces, using regex. Leave the braces.
        text  =  re.sub(r'(}|\])(?!.*(\]|\})).*$', r'\1', text, flags=re.DOTALL)
        
        # remove invalid escape sequences, which show up sometimes
        # replace \' with just '
        text =  re.sub("\\'", "'", text) #re.sub(r'\\\'', r"'", text)

        # return the parsed JSON object
        return json.loads(text)
    
    except ValueError:
        return {}

################################################################################
# Rendering and markup 
################################################################################
def inject_html_css_style_prefix(html, style_prefix_attributes):
    """
    Injects a style prefix to all style attributes in the given HTML string.

    For example, if you want to add a style prefix to all style attributes in the HTML string
    ``<div style="color: red;">Hello</div>``, you can use this function as follows:
    inject_html_css_style_prefix('<div style="color: red;">Hello</div>', 'font-size: 20px;')
    """
    return html.replace('style="', f'style="{style_prefix_attributes};')

def break_text_at_length(text: str, max_length: int=None) -> str:
    """
    Breaks the text at the specified length, inserting a "(...)" string at the break point.
    If the maximum length is `None`, the content is returned as is.
    """
    if max_length is None or len(text) <= max_length:
        return text
    else:
        return text[:max_length] + " (...)"

################################################################################
# IO and startup utilities
################################################################################
_config = None

def read_config_file(use_cache=True, verbose=True) -> configparser.ConfigParser:
    global _config
    if use_cache and _config is not None:
        # if we have a cached config and accept that, return it
        return _config
    
    else:
        config = configparser.ConfigParser()

        # first, try the directory of the current main program
        config_file_path = Path.cwd() / "config.ini"
        if config_file_path.exists():
            
            config.read(config_file_path)
            _config = config
            return config
        else:
            if verbose:
                print(f"Failed to find custom config on: {config_file_path}")
                print("Now switching to default config file...")

        # if nothing there, use the default one in the module directory
        config_file_path = Path(__file__).parent.absolute() / 'config.ini'
        print(f"Looking for config on: {config_file_path}") if verbose else None
        if config_file_path.exists():
            config.read(config_file_path)
            _config = config
            return config
        else:
            raise ValueError("Could not find config.ini file anywhere")
    
def start_logger(config: configparser.ConfigParser):
    # create logger
    logger = logging.getLogger("tinytroupe")
    log_level = config['Logging'].get('LOGLEVEL', 'INFO').upper()
    logger.setLevel(level=log_level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

################################################################################
# Other
################################################################################
def custom_hash(obj):
    """
    Returns a hash for the specified object. The object is first converted
    to a string, to make it hashable. This method is deterministic,
    contrary to the built-in hash() function.
    """

    return hashlib.sha256(str(obj).encode()).hexdigest()
