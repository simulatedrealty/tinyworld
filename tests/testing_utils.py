"""
Testing utilities.
"""
import os
import sys
from time import sleep
sys.path.append('../../tinytroupe/')
sys.path.append('../../')
sys.path.append('..')

from tinytroupe.clients import client, force_api_cache
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld, TinySocialNetwork
import pytest
import importlib

# force caching, in order to save on API usage
force_api_cache(True, "tests_cache.pickle")

def contains_action_type(actions, action_type):
    """
    Checks if the given list of actions contains an action of the given type.
    """
    
    for action in actions:
        if action["action"]["type"] == action_type:
            return True
    
    return False

def contains_action_content(actions:list, action_content: str):
    """
    Checks if the given list of actions contains an action with the given content.
    """
    
    for action in actions:
        # checks whether the desired content is contained in the action content
        if action_content.lower() in action["action"]["content"].lower():
            return True
    
    return False

def contains_stimulus_type(stimuli, stimulus_type):
    """
    Checks if the given list of stimuli contains a stimulus of the given type.
    """
    
    for stimulus in stimuli:
        if stimulus["type"] == stimulus_type:
            return True
    
    return False

def contains_stimulus_content(stimuli, stimulus_content):
    """
    Checks if the given list of stimuli contains a stimulus with the given content.
    """
    
    for stimulus in stimuli:
        # checks whether the desired content is contained in the stimulus content
        if stimulus_content.lower() in stimulus["content"].lower():
            return True
    
    return False

def terminates_with_action_type(actions, action_type):
    """
    Checks if the given list of actions terminates with an action of the given type.
    """
    
    if len(actions) == 0:
        return False
    
    return actions[-1]["action"]["type"] == action_type


def proposition_holds(proposition: str) -> bool:
    """
    Checks if the given proposition is true according to an LLM call.
    This can be used to check for text properties that are hard to
    verify mechanically, such as "the text contains some ideas for a product".
    """

    system_prompt = f"""
    Check whether the following proposition is true or false. If it is
    true, write "true", otherwise write "false". Don't write anything else!
    """

    user_prompt = f"""
    Proposition: {proposition}
    """

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    
    # call the LLM
    next_message = client().send_message(messages)

    # check the result
    cleaned_message = only_alphanumeric(next_message["content"])
    if cleaned_message.lower().startswith("true"):
        return True
    elif cleaned_message.lower().startswith("false"):
        return False
    else:
        raise Exception(f"LLM returned unexpected result: {cleaned_message}")

def only_alphanumeric(string: str):
    """
    Returns a string containing only alphanumeric characters.
    """
    return ''.join(c for c in string if c.isalnum())

def create_test_system_user_message(user_prompt, system_prompt="You are a helpful AI assistant."):
    """
    Creates a list containing one system message and one user message. 
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})
    
    return messages

def agents_configs_are_equal(agent1, agent2, ignore_name=False):
    """
    Checks if the configurations of two agents are equal.
    """

    ignore_keys = []
    if ignore_name:
        ignore_keys.append("name")
    
    for key in agent1._configuration.keys():
        if key in ignore_keys:
            continue
        
        if agent1._configuration[key] != agent2._configuration[key]:
            return False
    
    return True
############################################################################################################
# I/O utilities
############################################################################################################

def remove_file_if_exists(file_path):
    """
    Removes the file at the given path if it exists.
    """
    
    if os.path.exists(file_path):
        os.remove(file_path)

def get_relative_to_test_path(path_suffix):
    """
    Returns the path to the test file with the given suffix.
    """
    
    return os.path.join(os.path.dirname(__file__), path_suffix)


############################################################################################################
# Fixtures
############################################################################################################

@pytest.fixture(scope="function")
def focus_group_world():
    import tinytroupe.examples as examples   
    
    world = TinyWorld("Focus group", [examples.create_lisa_the_data_scientist(), examples.create_oscar_the_architect(), examples.create_marcos_the_physician()])
    return world

@pytest.fixture(scope="function")
def setup():
    TinyPerson.clear_agents()
    TinyWorld.clear_environments()

    yield