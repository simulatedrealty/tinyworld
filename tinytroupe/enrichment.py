import os
import json
import chevron
import logging
logger = logging.getLogger("tinytroupe")
import pandas as pd

from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld
from tinytroupe.personfactory import TinyPersonFactory

from tinytroupe import openai_utils
import tinytroupe.utils as utils

class Enricher:

    def __init__(self, use_past_results_in_context=False) -> None:
        self.use_past_results_in_context = use_past_results_in_context

        self.context_cache = []
    
    def enrich_content(self, requirements: str, content:str, content_type:str =None, context_info:str ="", context_cache:list=None, verbose:bool=False):

        rendering_configs = {"requirements": requirements,
                             "content": content,
                             "content_type": content_type, 
                             "context_info": context_info,
                             "context_cache": context_cache}

        messages = utils.compose_initial_LLM_messages_with_templates("enricher.system.mustache", "enricher.user.mustache", rendering_configs)
        next_message = openai_utils.client().send_message(messages, temperature=0.4)
        
        debug_msg = f"Enrichment result message: {next_message}"
        logger.debug(debug_msg)
        if verbose:
            print(debug_msg)

        if next_message is not None:
            result = utils.extract_code_block(next_message["content"])
        else:
            result = None

        return result
    
    ###########################################################
    # IO
    ###########################################################

    def to_json(self) -> dict:
        """
        Returns a JSON representation of the object.
        """
        return self.__dict__

    @classmethod
    def from_json(cls, json_dict: dict):
        """
        Loads a JSON representation of the object.
        """
        # instantiates the current class of the object, without specifying the exact class
        new_obj = cls.__new__(cls)
        new_obj.__dict__ = json_dict

        return new_obj
    
