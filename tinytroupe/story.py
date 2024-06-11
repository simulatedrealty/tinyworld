"""
Every simulation tells a story. This module provides helper mechanisms to help with crafting appropriate stories in TinyTroupe.
"""

from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld
import tinytroupe.utils as utils
from tinytroupe import openai_utils

class Story:

    def __init__(self, environment:TinyWorld=None, agent:TinyPerson=None, purpose:str="Be a realistic simulation.") -> None:
        """
        Initialize the story. The story can be about an environment or an agent. It also has a purpose, which
        is used to guide the story generation. Stories are aware that they are related to simulations, so one can
        specify simulation-related purposes.

        Args:
            environment (TinyWorld, optional): The environment in which the story takes place. Defaults to None.
            agent (TinyPerson, optional): The agent in the story. Defaults to None.
            purpose (str, optional): The purpose of the story. Defaults to "Be a realistic simulation.".
        """
        
        # exactly one of these must be provided
        if environment and agent:
            raise Exception("Either 'environment' or 'agent' should be provided, not both")
        if not (environment or agent):
            raise Exception("At least one of the parameters should be provided")
        
        self.environment = environment
        self.agent = agent

        self.purpose = purpose

    def continue_story(self, requirements="Continue the story in an interesting way.", number_of_words:int=100, include_plot_twist:bool=False) -> str:
        """
        Propose a continuation of the story.
        """
        
        rendering_configs = {
                             "purpose": self.purpose,
                             "requirements": requirements,
                             "current_simulation_trace": self._current_story(),
                             "number_of_words": number_of_words,
                             "include_plot_twist": include_plot_twist
                            }

        messages = utils.compose_initial_LLM_messages_with_templates("story.continuation.system.mustache", "story.continuation.user.mustache", rendering_configs)
        next_message = openai_utils.client().send_message(messages, temperature=0.5)

        return next_message["content"]

    def _current_story(self) -> str:
        """
        Get the current story.
        """
        interaction_history = ""
        
        if self.agent is not None:
            interaction_history += self.agent.pretty_current_interactions()
        elif self.environment is not None:
            interaction_history += self.environment.pretty_current_interactions()
        
        return interaction_history
            
        

