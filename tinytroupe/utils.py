"""
General utilities and convenience functions.
"""
import re
import json
import os
import hashlib
import textwrap
import logging
import chevron
from datetime import datetime
from pathlib import Path
import configparser
from typing import Any, TypeVar, Union
AgentOrWorld = Union["TinyPerson", "TinyWorld"]

# logger
logger = logging.getLogger("tinytroupe")

from tinytroupe import config
rai_harmful_content_prevention = config["Simulation"].getboolean(
    "RAI_HARMFUL_CONTENT_PREVENTION", True 
)

rai_copyright_infringement_prevention = config["Simulation"].getboolean(
    "RAI_COPYRIGHT_INFRINGEMENT_PREVENTION", True
)

################################################################################
# Model input utilities
################################################################################

def compose_initial_LLM_messages_with_templates(system_template_name:str, user_template_name:str=None, rendering_configs:dict={}) -> list:
    """
    Composes the initial messages for the LLM model call, under the assumption that it always involves 
    a system (overall task description) and an optional user message (specific task description). 
    These messages are composed using the specified templates and rendering configurations.
    """

    system_prompt_template_path = os.path.join(os.path.dirname(__file__), f'prompts/{system_template_name}')
    user_prompt_template_path = os.path.join(os.path.dirname(__file__), f'prompts/{user_template_name}')

    messages = []

    messages.append({"role": "system", 
                         "content": chevron.render(
                             open(system_prompt_template_path).read(), 
                             rendering_configs)})
    
    # optionally add a user message
    if user_template_name is not None:
        messages.append({"role": "user", 
                            "content": chevron.render(
                                    open(user_prompt_template_path).read(), 
                                    rendering_configs)})
    return messages


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

def extract_code_block(text: str) -> str:
    """
    Extracts a code block from a string, ignoring any text before the first 
    opening triple backticks and any text after the closing triple backticks.
    """
    try:
        # remove any text before the first opening triple backticks, using regex. Leave the backticks.
        text = re.sub(r'^.*?(```)', r'\1', text, flags=re.DOTALL)

        # remove any trailing text after the LAST closing triple backticks, using regex. Leave the backticks.
        text  =  re.sub(r'(```)(?!.*```).*$', r'\1', text, flags=re.DOTALL)
        
        return text
    
    except ValueError:
        return ""

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
# Validation
################################################################################
def check_valid_fields(obj: dict, valid_fields: list) -> None:
    """
    Checks whether the fields in the specified dict are valid, according to the list of valid fields. If not, raises a ValueError.
    """
    for key in obj:
        if key not in valid_fields:
            raise ValueError(f"Invalid key {key} in dictionary. Valid keys are: {valid_fields}")
    
################################################################################
# Prompt engineering
################################################################################
def add_rai_template_variables_if_enabled(template_variables: dict) -> dict:
    """
    Adds the RAI template variables to the specified dictionary, if the RAI disclaimers are enabled.
    These can be configured in the config.ini file. If enabled, the variables will then load the RAI disclaimers from the 
    appropriate files in the prompts directory. Otherwise, the variables will be set to None.

    Args:
        template_variables (dict): The dictionary of template variables to add the RAI variables to.

    Returns:
        dict: The updated dictionary of template variables.
    """

    # Harmful content
    with open(os.path.join(os.path.dirname(__file__), "prompts/rai_harmful_content_prevention.md"), "r") as f:
        rai_harmful_content_prevention_content = f.read()

    template_variables['rai_harmful_content_prevention'] = rai_harmful_content_prevention_content if rai_harmful_content_prevention else None

    # Copyright infringement
    with open(os.path.join(os.path.dirname(__file__), "prompts/rai_copyright_infringement_prevention.md"), "r") as f:
        rai_copyright_infringement_prevention_content = f.read()

    template_variables['rai_copyright_infringement_prevention'] = rai_copyright_infringement_prevention_content if rai_copyright_infringement_prevention else None

    return template_variables

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

def break_text_at_length(text: Union[str, dict], max_length: int=None) -> str:
    """
    Breaks the text (or JSON) at the specified length, inserting a "(...)" string at the break point.
    If the maximum length is `None`, the content is returned as is.
    """
    if isinstance(text, dict):
        text = json.dumps(text, indent=4)

    if max_length is None or len(text) <= max_length:
        return text
    else:
        return text[:max_length] + " (...)"

def pretty_datetime(dt: datetime) -> str:
    """
    Returns a pretty string representation of the specified datetime object.
    """
    return dt.strftime("%Y-%m-%d %H:%M")

def dedent(text: str) -> str:
    """
    Dedents the specified text, removing any leading whitespace and identation.
    """
    return textwrap.dedent(text).strip()

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
def name_or_empty(named_entity: AgentOrWorld):
    """
    Returns the name of the specified agent or environment, or an empty string if the agent is None.
    """
    if named_entity is None:
        return ""
    else:
        return named_entity.name

def custom_hash(obj):
    """
    Returns a hash for the specified object. The object is first converted
    to a string, to make it hashable. This method is deterministic,
    contrary to the built-in hash() function.
    """

    return hashlib.sha256(str(obj).encode()).hexdigest()

_fresh_id_counter = 0
def fresh_id():
    """
    Returns a fresh ID for a new object. This is useful for generating unique IDs for objects.
    """
    global _fresh_id_counter
    _fresh_id_counter += 1
    return _fresh_id_counter
