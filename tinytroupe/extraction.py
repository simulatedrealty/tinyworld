import os
import json
import chevron
import logging
import pandas as pd

from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld
from tinytroupe.personfactory import TinyPersonFactory

from tinytroupe import openai_utils
import tinytroupe.utils as utils

class InteractionResultsExtractor:

    def __init__(self):
        self._extraction_prompt_template_path = os.path.join(os.path.dirname(__file__), 'prompts/interaction_results_extractor.mustache')
        self.logger = logging.getLogger("tinytroupe")

        # we'll cache the last extraction results for each type of extraction, so that we can use them to
        # generate reports or other additional outputs.
        self.agent_extraction = {}
        self.world_extraction = {}

    def extract_results_from_agent(self, 
                        tinyperson:TinyPerson, 
                        extraction_objective:str="The main points present in the agent's interactions history.", 
                        situation:str = "", 
                        fields:list=None,
                        verbose:bool=False):
        """
        Extracts results from a TinyPerson instance.

        Args:
            tinyperson (TinyPerson): The TinyPerson instance to extract results from.
            extraction_objective (str): The extraction objective.
            situation (str): The situation to consider.
            fields (list, optional): The fields to extract. If None, the extractor will decide what names to use. 
                Defaults to None.
            verbose (bool, optional): Whether to print debug messages. Defaults to False.
        """

        messages = []

        rendering_configs = {}
        if fields is not None:
            rendering_configs["fields"] = ", ".join(fields)
        
        messages.append({"role": "system", 
                         "content": chevron.render(
                             open(self._extraction_prompt_template_path).read(), 
                             rendering_configs)})


        interaction_history = tinyperson.pretty_current_interactions()

        extraction_request_prompt = \
f"""
## Extraction objective

{extraction_objective}

## Situation
You are considering a single agent, named {tinyperson.name}. Your objective thus refers to this agent specifically.
{situation}

## Agent Interactions History

You will consider an agent's history of interactions, which include stimuli it received as well as actions it 
performed.

{interaction_history}
"""
        messages.append({"role": "user", "content": extraction_request_prompt})

        next_message = openai_utils.client().send_message(messages, temperature=0.0)
        
        debug_msg = f"Extraction raw result message: {next_message}"
        self.logger.debug(debug_msg)
        if verbose:
            print(debug_msg)

        if next_message is not None:
            result = utils.extract_json(next_message["content"])
        else:
            result = None
        
        # cache the result
        self.agent_extraction[tinyperson.name] = result

        return result
    

    def extract_results_from_world(self, 
                                   tinyworld:TinyWorld, 
                                   extraction_objective:str="The main points that can be derived from the agents conversations and actions.", 
                                   situation:str="", 
                                   fields:list=None,
                                   verbose:bool=False):
        """
        Extracts results from a TinyWorld instance.

        Args:
            tinyworld (TinyWorld): The TinyWorld instance to extract results from.
            extraction_objective (str): The extraction objective.
            situation (str): The situation to consider.
            fields (list, optional): The fields to extract. If None, the extractor will decide what names to use. 
                Defaults to None.
            verbose (bool, optional): Whether to print debug messages. Defaults to False.
        """

        messages = []

        rendering_configs = {}
        if fields is not None:
            rendering_configs["fields"] = ", ".join(fields)
        
        messages.append({"role": "system", 
                         "content": chevron.render(
                             open(self._extraction_prompt_template_path).read(), 
                             rendering_configs)})

        # TODO: either summarize first or break up into multiple tasks
        interaction_history = tinyworld.pretty_current_interactions()

        extraction_request_prompt = \
f"""
## Extraction objective

{extraction_objective}

## Situation
You are considering various agents.
{situation}

## Agents Interactions History

You will consider the history of interactions from various agents that exist in an environment called {tinyworld.name}. 
Each interaction history includes stimuli the corresponding agent received as well as actions it performed.

{interaction_history}
"""
        messages.append({"role": "user", "content": extraction_request_prompt})

        next_message = openai_utils.client().send_message(messages, temperature=0.0)
        
        debug_msg = f"Extraction raw result message: {next_message}"
        self.logger.debug(debug_msg)
        if verbose:
            print(debug_msg)

        if next_message is not None:
            result = utils.extract_json(next_message["content"])
        else:
            result = None
        
        # cache the result
        self.world_extraction[tinyworld.name] = result

        return result
    
    def save_as_json(self, filename:str, verbose:bool=False):
        """
        Saves the last extraction results as JSON.

        Args:
            filename (str): The filename to save the JSON to.
            verbose (bool, optional): Whether to print debug messages. Defaults to False.
        """
        with open(filename, 'w') as f:
            json.dump({"agent_extractions": self.agent_extraction, 
                       "world_extraction": self.world_extraction}, f, indent=4)
        
        if verbose:
            print(f"Saved extraction results to {filename}")



import logging
class InteractionResultsReducer:

    def __init__(self):
        self.results = {}
        self.logger = logging.getLogger("tinytroupe")

        self.rules = {}
    
    def add_reduction_rule(self, trigger: str, func: callable):
        if trigger in self.rules:
            raise Exception(f"Rule for {trigger} already exists.")
        
        self.rules[trigger] = func
    
    def reduce_agent(self, agent: TinyPerson) -> list:
        reduction = []
        for message in agent.episodic_memory.retrieve_all():
            if message['role'] == 'system':
                continue # doing nothing for `system` role yet at least

            elif message['role'] == 'user':
                # User role is related to stimuli only
                stimulus_type = message['content']['stimuli'][0]['type']
                stimulus_content = message['content']['stimuli'][0]['content']
                stimulus_source = message['content']['stimuli'][0]['source']

                if stimulus_type in self.rules:
                    reduction.append(self.rules[stimulus_type](focus_agent=agent, source_agent=TinyPerson.get_agent_by_name(stimulus_source), target_agent=agent, kind='stimulus', event=stimulus_type, content=stimulus_content))

            elif message['role'] == 'assistant':
                # Assistant role is related to actions only
                if 'action' in message['content']: 
                    action_type = message['content']['action']['type']
                    action_content = message['content']['action']['content']
                    action_target = message['content']['action']['target']
                    
                    if action_type in self.rules:
                        reduction.append(self.rules[action_type](focus_agent=agent, source_agent=agent, target_agent=TinyPerson.get_agent_by_name(action_target), kind='action', event=action_type, content=action_content))
            
        return reduction

    def reduce_agent_to_dataframe(self, agent: TinyPerson, column_names: list=None) -> pd.DataFrame:
        reduction = self.reduce_agent(agent)
        return pd.DataFrame(reduction, columns=column_names)


################################################################################	
# Convenience mechanisms
################################################################################

# default extractor
default_extractor = InteractionResultsExtractor()