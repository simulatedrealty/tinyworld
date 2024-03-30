import os
import json
import chevron
import logging

from tinytroupe import openai_utils
from tinytroupe.agent import TinyPerson
import tinytroupe.utils as utils

class TinyPersonFactory:
    """
    A factory for generating TinyPerson instances using OpenAI's LLM.

    Attributes:
        prompt_template_path (str): The path to the prompt template file.
        current_messages (list): A list of messages exchanged between the system and the user.
        logger (logging.Logger): The logger instance used for logging messages.
    """

    def __init__(self, context_text):
        self.person_prompt_template_path = os.path.join(os.path.dirname(__file__), 'prompts/generate_person.mustache')
        self.context_text = context_text
        self.generated = [] # keep track of the generated persons
        self.logger = logging.getLogger("tinytroupe")

    @staticmethod
    def generate_person_factories(number_of_factories, generic_context_text):
        """
        Generate a list of TinyPersonFactory instances using OpenAI's LLM.

        Args:
            number_of_factories (int): The number of TinyPersonFactory instances to generate.
            generic_context_text (str): The generic context text used to generate the TinyPersonFactory instances.

        Returns:
            list: A list of TinyPersonFactory instances.
        """
        
        logger = logging.getLogger("tinytroupe")
        logger.info(f"Starting the generation of the {number_of_factories} person factories based on that context: {generic_context_text}")
        
        person_factories_prompt = open(os.path.join(os.path.dirname(__file__), 'prompts/generate_person_factory.md')).read()

        messages = []
        messages.append({"role": "system", "content": person_factories_prompt})

        prompt = chevron.render("Please, create {{number_of_factories}} person descriptions based on the following broad context: {{context}}", {
            "number_of_factories": number_of_factories,
            "context": generic_context_text
        })

        messages.append({"role": "user", "content": prompt})

        response = openai_utils.client().send_message(messages)

        if response is not None:
            result = utils.extract_json(response["content"])

            factories = []
            for i in range(number_of_factories):
                logger.debug(f"Generating person factory with description: {result[i]}")
                factories.append(TinyPersonFactory(result[i]))

            return factories

        return None

    def generate_person(self, temperature=1.5):
        """
        Generate a TinyPerson instance using OpenAI's LLM.

        Returns:
            TinyPerson: A TinyPerson instance generated using the LLM.
        """

        self.logger.info(f"Starting the person generation based on that context: {self.context_text}")

        prompt = chevron.render(open(self.person_prompt_template_path).read(), {
            "context": self.context_text,
            "already_generated": [person.minibio() for person in self.generated]
        })

        messages = []
        messages += [{"role": "system", "content": "You are a system that generates specifications of artificial entities."},
                     {"role": "user", "content": prompt}]

        message = openai_utils.client().send_message(messages, temperature=temperature)

        if message is not None:
            result = utils.extract_json(message["content"])

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Generated person parameters:\n{json.dumps(result, indent=4, sort_keys=True)}")

            person = TinyPerson(result["name"])
            for key, value in result["_configuration"].items():
                if isinstance(value, list):
                    person.define_several(key, value)
                else:
                    person.define(key, value)
            
            self.generated.append(person)
            return person
        else:
            return None
