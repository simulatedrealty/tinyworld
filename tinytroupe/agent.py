"""
This module provides the main classes and functions for TinyTroupe's  agents.

Agents are the key abstraction used in TinyTroupe. An agent is a simulated person or entity that can interact with other agents and the environment, by
receiving stimuli and producing actions. Agents have cognitive states, which are updated as they interact with the environment and other agents. 
Agents can also store and retrieve information from memory, and can perform actions in the environment. Different from agents whose objective is to
provide support for AI-based assistants or other such productivity tools, **TinyTroupe agents are aim at representing human-like behavior**, which includes
idiossincracies, emotions, and other human-like traits, that one would not expect from a productivity tool.

The overall underlying design is inspired mainly by cognitive psychology, which is why agents have various internal cognitive states, such as attention, emotions, and goals.
It is also why agent memory, differently from other LLM-based agent platforms, has subtle internal divisions, notably between episodic and semantic memory. 
Some behaviorist concepts are also present, such as the idea of a "stimulus" and "response" in the `listen` and `act` methods, which are key abstractions
to understand how agents interact with the environment and other agents.
"""

import os
import csv
import json
import ast
import textwrap  # to dedent strings
import datetime  # to get current datetime
import chevron  # to parse Mustache templates
import logging
import tinytroupe.utils as utils
from tinytroupe.control import transactional
from rich import print
import copy

from typing import Any, TypeVar, Union

Self = TypeVar("Self", bound="TinyPerson")
AgentOrWorld = Union[Self, "TinyWorld"]

## LLaMa-Index congigs ########################################################
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader


# this will be cached locally by llama-index, in a OS-dependend location
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
###############################################################################


from tinytroupe import openai_utils
from tinytroupe.utils import name_or_empty, break_text_at_length, repeat_on_error

from tinytroupe import config

default_max_content_display_length = config["OpenAI"].getint(
    "TINYPERSON_MAX_CONTENT_DISPLAY_LENGTH", 1024
)


class EpisodicMemory:
    """
    Provides episodic memory capabilities to an agent. Cognitively, episodic memory is the ability to remember specific events,
    or episodes, in the past. This class provides a simple implementation of episodic memory, where the agent can store and retrieve
    messages from memory.
    
    Subclasses of this class can be used to provide different memory implementations.
    """

    def __init__(
        self, fixed_prefix_length: int = 100, lookback_length: int = 100
    ) -> None:
        """
        Initializes the memory.

        Args:
            fixed_prefix_length (int): The fixed prefix length. Defaults to 20.
            lookback_length (int): The lookback length. Defaults to 20.
        """
        self.fixed_prefix_length = fixed_prefix_length
        self.lookback_length = lookback_length

        self.memory = []

    def store(self, value: Any) -> None:
        """
        Stores a value in memory.
        """
        self.memory.append(value)

    def count(self) -> int:
        """
        Returns the number of values in memory.
        """
        return len(self.memory)

    def retrieve_recent(self) -> list:
        """
        Retrieves the n most recent values from memory.
        """
        # compute fixed prefix
        fixed_prefix = self.memory[: self.fixed_prefix_length]

        # how many lookback values remain?
        remaining_lookback = min(
            len(self.memory) - len(fixed_prefix), self.lookback_length
        )

        # compute the remaining lookback values and return the concatenation
        if remaining_lookback <= 0:
            return fixed_prefix
        else:
            return fixed_prefix + self.memory[-remaining_lookback:]

    def retrieve_all(self) -> list:
        """
        Retrieves all values from memory.
        """
        return copy.copy(self.memory)

    def retrieve_relevant(self, relevance_target: str) -> list:
        """
        Retrieves all values from memory that are relevant to a given target.
        """
        # TODO
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_first(self, n: int) -> list:
        """
        Retrieves the first n values from memory.
        """
        return self.memory[:n]

    ###########################################################
    # IO
    ###########################################################

    def to_json(self) -> dict:
        """
        Returns a JSON representation of the memory.
        """
        return self.__dict__

    @staticmethod
    def from_json(json_dict: dict) -> Self:
        """
        Loads a JSON representation of the memory.
        """
        new_memory = EpisodicMemory()
        new_memory.__dict__ = json_dict

        return new_memory

class SemanticMemory:
    """
    Semantic memory is the memory of meanings, understandings, and other concept-based knowledge unrelated to specific experiences.
    It is not ordered temporally, and it is not about remembering specific events or episodes. This class provides a simple implementation
    of semantic memory, where the agent can store and retrieve semantic information.
    """

    def __init__(self, documents_paths: list=None, web_urls: list=None) -> None:
        self.index = None
        
        self.documents_paths = []
        self.documents_web_urls = []

        self.documents = []
        self.filename_to_document = {}

        # load document paths and web urls
        if documents_paths is not None:
            for documents_path in documents_paths:
                self.add_documents_path(documents_path)
        
        if web_urls is not None:
            self.add_web_urls(web_urls)
    
    def retrieve_relevant(self, relevance_target:str, top_k=5) -> list:
        """
        Retrieves all values from memory that are relevant to a given target.
        """
        if self.index is not None:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve("Microsoft's recent major investments")
        else:
            nodes = []

        retrieved = []
        for node in nodes:
            content = "SOURCE: " + node.metadata['file_name']
            content += "\n" + "SIMILARITY SCORE:" + str(node.score)
            content += "\n" + "RELEVANT CONTENT:" + node.text
            retrieved.append(content)
        
        return retrieved
    
    def retrieve_document_content_by_name(self, document_name:str) -> str:
        """
        Retrieves a document by its name.
        """
        if self.filename_to_document is not None:
            doc = self.filename_to_document[document_name]
            if doc is not None:
                content = "SOURCE: " + document_name
                content += "\n" + "CONTENT: " + doc.text[:10000] # TODO a more intelligent way to limit the content
                return content
            else:
                return None
        else:
            return None
    
    def list_documents_names(self) -> list:
        """
        Lists the names of the documents in memory.
        """
        if self.filename_to_document is not None:
            return list(self.filename_to_document.keys())
        else:
            return []
    
    def add_documents_path(self, documents_path:str) -> None:
        """
        Adds a path to a folder with documents used for semantic memory.
        """
        if documents_path not in self.documents_paths:
            self.documents_paths.append(documents_path)
            new_documents = SimpleDirectoryReader(documents_path).load_data()
            self._add_documents(new_documents, lambda doc: doc.metadata["file_name"])
    
    def add_web_urls(self, web_urls:list) -> None:
        """ 
        Adds the data retrieved from the specified URLs to documents used for semantic memory.
        """
        filtered_web_urls = [url for url in web_urls if url not in self.documents_web_urls]
        self.documents_web_urls += filtered_web_urls

        if len(filtered_web_urls) > 0:
            new_documents = SimpleWebPageReader(html_to_text=True).load_data(filtered_web_urls)
            self._add_documents(new_documents, lambda doc: doc.id_)

    def _add_documents(self, new_documents, doc_to_name_func) -> list:
        """
        Adds documents to the semantic memory.
        """
        # index documents by name
        if len(new_documents) > 0:
            # add the new documents to the list of documents
            self.documents += new_documents

            # process documents individually too
            for document in new_documents:
                name = doc_to_name_func(document)
                self.filename_to_document[name] = document

            # index documents for semantic retrieval
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(self.documents)
            else:
                self.index.refresh(self.documents)



    ###########################################################
    # IO
    ###########################################################

    def to_json(self) -> dict:
        """
        Returns a JSON representation of the memory.
        """
        return {"documents_paths": self.documents_paths, "documents_web_urls": self.documents_web_urls}
    
    @staticmethod
    def from_json(json_dict:dict) -> Self:
        """
        Loads a JSON representation of the memory.
        """
        new_memory = SemanticMemory(documents_paths=json_dict["documents_paths"], web_urls=json_dict["documents_web_urls"])

        return new_memory

    

class TinyPerson:
    """A simulated person in the TinyTroupe universe."""

    # The maximum number of actions that an agent is allowed to perform before DONE.
    # This prevents the agent from acting without ever stopping.
    MAX_ACTIONS_BEFORE_DONE = 15

    PP_TEXT_WIDTH = 100

    # A dict of all agents instantiated so far.
    all_agents = {}  # name -> agent

    def __init__(self, name=None, 
                 communication_style="simplified", communication_display=True,
                 simulation_id=None,
                 spec_path=None,
                 episodic_memory=None,
                 semantic_memory=None):
        """
        Creates a TinyPerson.

        Args:
            name (str): The name of the TinyPerson. Either this or spec_path must be specified.
            communication_style (str, optional): The communication style: "simplified" or "full". Defaults to "simplified".
            communication_display (bool, optional): Whether to display the communication or not.
                True is for interactive applications, when we want to see simulation
                outputs as they are produced. Defaults to True.
            simulation_id (str, optional): The simulation ID. Defaults to None.
            spec_path (str, optional): The path to a JSON file containing the TinyPerson's configuration.
                Defaults to None, in which case the TinyPerson is created from scratch. If None, the
                TinyPerson's name must be specified.
            episodic_memory (EpisodicMemory, optional): The memory implementation to use. Defaults to EpisodicMemory().
        """

        self.logger = logging.getLogger("tinytroupe")

        self.communication_style = communication_style
        self.communication_display = communication_display
        self.simulation_id = simulation_id  # will be reset later if the agent is used within a specific simulation scope

        self._prompt_template_path = os.path.join(
            os.path.dirname(__file__), "prompts/basic_agent.mustache"
        )
        self._init_system_message = None  # initialized later

        self.current_messages = []

        
        if episodic_memory is not None:
            self.episodic_memory = episodic_memory
        else:
            # This default value MUST NOT be in the method signature, otherwise it will be shared across all instances.
            # Yeah, I'm programming in Python because I must, not because I want to.
            self.episodic_memory = EpisodicMemory() 
        
        if semantic_memory is not None:
            self.semantic_memory = semantic_memory
        else:
            # This default value MUST NOT be in the method signature, otherwise it will be shared across all instances.
            # Yeah, I'm programming in Python because I must, not because I want to.
            self.semantic_memory = SemanticMemory()

        # the current environment in which the agent is acting
        self.environment = None

        # The list of actions that this agent has performed so far, but which have not been
        # consumed by the environment yet.
        self._actions_buffer = []

        # The list of agents that this agent can currently interact with.
        # This can change over time, as agents move around the world.
        self._accessible_agents = []

        # the buffer of communications that have been displayed so far, used for
        # saving these communications to another output form later (e.g., caching)
        self._displayed_communications_buffer = []

        # load the configuration from a JSON file, if specified
        if spec_path is not None:
            self._load_spec(spec_path)

        # otherwise, create a new configuration
        else:
            assert name is not None, "A TinyPerson must have a name."

            self.name = name
            self._configuration = {
                "name": self.name,
                "age": None,
                "nationality": None,
                "country_of_residence": None,
                "occupation": None,
                "routines": [],
                "occupation_description": None,
                "personality_traits": [],
                "professional_interests": [],
                "personal_interests": [],
                "skills": [],
                "relationships": [],
                "current_datetime": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "current_location": None,
                "current_context": [],
                "current_attention": None,
                "current_goals": [],
                "current_emotions": "Currently you feel calm and friendly.",
                "currently_accessible_agents": [],  # [{"agent": agent_1, "relation": "My friend"}, {"agent": agent_2, "relation": "My colleague"}, ...]
            }

        # register the agent in the global list of agents
        TinyPerson.add_agent(self)

        # start with a clean slate
        self.reset_prompt()

    def generate_agent_specification(self):
        with open(self._prompt_template_path, "r") as f:
            agent_prompt_template = f.read()

        return chevron.render(agent_prompt_template, self._configuration)

    def reset_prompt(self):

        # render the template with the current configuration
        self._init_system_message = self.generate_agent_specification()

        # TODO actually, figure out another way to update agent state without "changing history"

        # reset system message
        self.current_messages = [
            {"role": "system", "content": self._init_system_message}
        ]

        # sets up the actual interaction messages to use for prompting
        self.current_messages += self.episodic_memory.retrieve_recent()

    @transactional
    def define(self, key, value, group=None):
        """
        Define a value to the TinyPerson's configuration.
        If group is None, the value is added to the top level of the configuration.
        Otherwise, the value is added to the specified group.
        """

        # dedent value if it is a string
        if isinstance(value, str):
            value = textwrap.dedent(value)

        if group is None:
            # self.logger.debug(f"[{self.name}] Defining {key}={value} in the person.")
            self._configuration[key] = value
        else:
            if key is not None:
                # self.logger.debug(f"[{self.name}] Adding definition to {group} += [ {key}={value} ] in the person.")
                self._configuration[group].append({key: value})
            else:
                # self.logger.debug(f"[{self.name}] Adding definition to {group} += [ {value} ] in the person.")
                self._configuration[group].append(value)

        # must reset prompt after adding to configuration
        self.reset_prompt()

    def define_several(self, group, records):
        """
        Define several values to the TinyPerson's configuration, all belonging to the same group.
        """
        for record in records:
            self.define(key=None, value=record, group=group)
    
    @transactional
    def define_relationships(self, relationships, replace=True):
        """
        Defines or updates the TinyPerson's relationships.

        Args:
            relationships (list or dict): The relationships to add or replace. Either a list of dicts mapping agent names to relationship descriptions,
              or a single dict mapping one agent name to its relationship description.
            replace (bool, optional): Whether to replace the current relationships or just add to them. Defaults to True.
        """
        
        if (replace == True) and (isinstance(relationships, list)):
            self._configuration['relationships'] = relationships

        elif replace == False:
            current_relationships = self._configuration['relationships']
            if isinstance(relationships, list):
                for r in relationships:
                    current_relationships.append(r)
                
            elif isinstance(relationships, dict) and len(relationships) == 2: #{"Name": ..., "Description": ...}
                current_relationships.append(relationships)

            else:
                raise Exception("Only one key-value pair is allowed in the relationships dict.")

        else:
            raise Exception("Invalid arguments for define_relationships.")

        

    @transactional
    def act(
        self,
        until_done=True,
        n=None,
        return_actions=False,
        max_content_length=default_max_content_display_length,
    ):
        """
        Acts in the environment and updates its internal cognitive state.
        Either acts until the agent is done and needs additional stimuli, or acts a fixed number of times,
        but not both.

        Args:
            until_done (bool): Whether to keep acting until the agent is done and needs additional stimuli.
            n (int): The number of actions to perform. Defaults to None.
            return_actions (bool): Whether to return the actions or not. Defaults to False.
        """

        # either act until done or act a fixed number of times, but not both
        assert not (until_done and n is not None)
        if n is not None:
            assert n < TinyPerson.MAX_ACTIONS_BEFORE_DONE

        contents = []

        # Aux function to perform exactly one action.
        # Occasionally, the model will return JSON missing important keys, so we just ask it to try again
        @repeat_on_error(retries=5, exceptions=[KeyError])
        def aux_act_once():
            # A quick thought before the action. This seems to help with better model responses, perhaps because
            # it interleaves user with assistant messages.
            self.think("I will now act a bit, and then issue DONE.")

          
            role, content = self._produce_message()

            self.episodic_memory.store({'role': role, 'content': content})

            cognitive_state = content["cognitive_state"]


            action = content['action']

            self._actions_buffer.append(action)
            self._update_cognitive_state(goals=cognitive_state['goals'],
                                        attention=cognitive_state['attention'],
                                        emotions=cognitive_state['emotions'])
            
            contents.append(content)          
            if self.communication_display:
                self._display_communication(role=role, content=content, kind='action', simplified=True, max_content_length=max_content_length)
            
            #
            # Some actions induce an immediate stimulus
            #
            if action['type'] == "RECALL" and action['content'] is not None:
                content = action['content']

                semantic_memories = self.semantic_memory.retrieve_relevant(relevance_target=content)

                if len(semantic_memories) > 0:
                    # a string with each element in the list in a new line starting with a bullet point
                    self.think("I have remembered the following information from my semantic memory and will use it to guide me in my subsequent actions: \n" + \
                            "\n".join([f"  - {item}" for item in semantic_memories]))
                else:
                    self.think(f"I can't remember anything about '{content}'.")
            
            elif action['type'] == "CONSULT" and action['content'] is not None:
                content = action['content']

                document_content = self.semantic_memory.retrieve_document_content_by_name(document_name=content)

                if document_content is not None:
                    self.think(f"I have read the following document: \n{document_content}")
                else:
                    self.think(f"I can't find any document with the name '{content}'.")
            
            elif action['type'] == "LIST_DOCUMENTS" and action['content'] is not None:
                documents_names = self.semantic_memory.list_documents_names()

                if len(documents_names) > 0:
                    self.think(f"I have the following documents available to me: {documents_names}")
                else:
                    self.think(f"I don't have any documents available for inspection.")
             
            

        #
        # How to proceed with a sequence of actions.
        #

        ##### Option 1: run N actions ######
        if n is not None:
            for i in range(n):
                aux_act_once()

        ##### Option 2: run until DONE ######
        elif until_done:
            while (len(contents) == 0) or (
                not contents[-1]["action"]["type"] == "DONE"
            ):


                # check if the agent is acting without ever stopping
                if len(contents) > TinyPerson.MAX_ACTIONS_BEFORE_DONE:
                    self.logger.warning(f"[{self.name}] Agent {self.name} is acting without ever stopping. This may be a bug. Let's stop it here anyway.")
                    break
                if len(contents) > 4: # just some minimum number of actions to check for repetition, could be anything >= 3
                    # if the last three actions were the same, then we are probably in a loop
                    if contents[-1]['action'] == contents[-2]['action'] == contents[-3]['action']:
                        self.logger.warning(f"[{self.name}] Agent {self.name} is acting in a loop. This may be a bug. Let's stop it here anyway.")
                        break

                aux_act_once()

        if return_actions:
            return contents

    @transactional
    def listen(
        self,
        speech,
        source: AgentOrWorld = None,
        max_content_length=default_max_content_display_length,
    ):
        """
        Listens to another agent (artificial or human) and updates its internal cognitive state.

        Args:
            speech (str): The speech to listen to.
            source (AgentOrWorld, optional): The source of the speech. Defaults to None.
        """

        return self._observe(
            stimulus={
                "type": "CONVERSATION",
                "content": speech,
                "source": name_or_empty(source),
            },
            max_content_length=max_content_length,
        )

    # TODO obsolete?
    @transactional
    def respond_form(self, questions_csv_file_path, skip_first_row=True):
        """
        Reads a CSV file containing questions and updates the second column with responses generated by the `listen_and_act` method.

        Args:
            questions_csv_file_path (str): The path to the CSV file containing the questions.
            skip_first_row (bool, optional): Whether to skip the first row of the CSV file. Defaults to True.

        Returns:
            None
        """

        self.logger.info(
            f"[{self.name}] Responding to form in {questions_csv_file_path}"
        )

        dialect = None
        with open(questions_csv_file_path, "r", encoding="utf-8") as file:
            dialect = csv.Sniffer().sniff(file.read(1024))
            file.seek(0)
            reader = csv.reader(file, dialect)
            rows = [row for row in reader]

        # now, we can start responding to the questions
        start_index = 1 if skip_first_row else 0
        for i in range(start_index, len(rows)):
            self.logger.info(f"[{self.name}] Responding to question: {rows[i][0]}")
            rows[i][1] = self.listen_and_act(rows[i][0])
            self.logger.info(f"[{self.name}] Response: {rows[i][1]}")

        with open(questions_csv_file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, dialect)
            writer.writerows(rows)

    def socialize(
        self,
        social_description: str,
        source: AgentOrWorld = None,
        max_content_length=default_max_content_display_length,
    ):
        """
        Perceives a social stimulus through a description and updates its internal cognitive state.

        Args:
            social_description (str): The description of the social stimulus.
            source (AgentOrWorld, optional): The source of the social stimulus. Defaults to None.
        """
        return self._observe(
            stimulus={
                "type": "SOCIAL",
                "content": social_description,
                "source": name_or_empty(source),
            },
            max_content_length=max_content_length,
        )

    def see(
        self,
        visual_description,
        source: AgentOrWorld = None,
        max_content_length=default_max_content_display_length,
    ):
        """
        Perceives a visual stimulus through a description and updates its internal cognitive state.

        Args:
            visual_description (str): The description of the visual stimulus.
            source (AgentOrWorld, optional): The source of the visual stimulus. Defaults to None.
        """
        return self._observe(
            stimulus={
                "type": "VISUAL",
                "content": visual_description,
                "source": name_or_empty(source),
            },
            max_content_length=max_content_length,
        )

    def think(self, thought, max_content_length=default_max_content_display_length):
        """
        Forces the agent to think about something and updates its internal cognitive state.

        """
        return self._observe(
            stimulus={
                "type": "THOUGHT",
                "content": thought,
                "source": name_or_empty(self),
            },
            max_content_length=max_content_length,
        )

    def internalize_goal(
        self, goal, max_content_length=default_max_content_display_length
    ):
        """
        Internalizes a goal and updates its internal cognitive state.
        """
        return self._observe(
            stimulus={
                "type": "INTERNAL_GOAL_FORMULATION",
                "content": goal,
                "source": name_or_empty(self),
            },
            max_content_length=max_content_length,
        )

    @transactional
    def _observe(self, stimulus, max_content_length=default_max_content_display_length):
        stimuli = [stimulus]

        content = {"stimuli": stimuli}

        self.logger.debug(f"[{self.name}] Observing stimuli: {content}")

        # whatever comes from the outside will be interpreted as coming from 'user', simply because
        # this is the counterpart of 'assistant'

        self.episodic_memory.store({'role': 'user', 'content': content})

        if self.communication_display:
            self._display_communication(
                role="user",
                content=content,
                kind="stimuli",
                simplified=True,
                max_content_length=max_content_length,
            )

        return self  # allows easier chaining of methods

    @transactional
    def listen_and_act(
        self,
        speech,
        return_actions=False,
        max_content_length=default_max_content_display_length,
    ):
        """
        Convenience method that combines the `listen` and `act` methods.
        """

        self.listen(speech, max_content_length=max_content_length)
        return self.act(
            return_actions=return_actions, max_content_length=max_content_length
        )

    @transactional
    def see_and_act(
        self,
        visual_description,
        return_actions=False,
        max_content_length=default_max_content_display_length,
    ):
        """
        Convenience method that combines the `see` and `act` methods.
        """

        self.see(visual_description, max_content_length=max_content_length)
        return self.act(
            return_actions=return_actions, max_content_length=max_content_length
        )

    @transactional
    def think_and_act(
        self,
        thought,
        return_actions=False,
        max_content_length=default_max_content_display_length,
    ):
        """
        Convenience method that combines the `think` and `act` methods.
        """

        self.think(thought, max_content_length=max_content_length)
        return self.act(return_actions=return_actions, max_content_length=max_content_length)

    def read_documents_from_folder(self, documents_path:str):
        """
        Reads documents from a directory and loads them into the semantic memory.
        """
        self.logger.info(f"Setting documents path to {documents_path} and loading documents.")

        self.semantic_memory.add_documents_path(documents_path)
    
    def read_documents_from_web(self, web_urls:list):
        """
        Reads documents from web URLs and loads them into the semantic memory.
        """
        self.logger.info(f"Reading documents from the following web URLs: {web_urls}")

        self.semantic_memory.add_web_urls(web_urls)
    
    @transactional
    def move_to(self, location, context=[]):
        """
        Moves to a new location and updates its internal cognitive state.
        """
        self._configuration["current_location"] = location

        # context must also be updated when moved, since we assume that context is dictated partly by location.
        self.change_context(context)

    @transactional
    def change_context(self, context: list):
        """
        Changes the context and updates its internal cognitive state.
        """
        self._configuration["current_context"] = {
            "description": item for item in context
        }

        self._update_cognitive_state(context=context)

    @transactional
    def make_agent_accessible(
        self,
        agent: Self,
        relation_description: str = "An agent I can currently interact with.",
    ):
        """
        Makes an agent accessible to this agent.
        """
        if agent not in self._accessible_agents:
            self._accessible_agents.append(agent)
            self._configuration["currently_accessible_agents"].append(
                {"name": agent.name, "relation_description": relation_description}
            )
        else:
            self.logger.warning(
                f"[{self.name}] Agent {agent.name} is already accessible to {self.name}."
            )

    @transactional
    def make_agent_inaccessible(self, agent: Self):
        """
        Makes an agent inaccessible to this agent.
        """
        if agent in self._accessible_agents:
            self._accessible_agents.remove(agent)
        else:
            self.logger.warning(
                f"[{self.name}] Agent {agent.name} is already inaccessible to {self.name}."
            )

    @transactional
    def make_all_agents_inaccessible(self):
        """
        Makes all agents inaccessible to this agent.
        """
        self._accessible_agents = []
        self._configuration["currently_accessible_agents"] = []

    @transactional
    def _produce_message(self):
        # self.logger.debug(f"Current messages: {self.current_messages}")

        # ensure we have the latest prompt (initial system message + selected messages from memory)
        self.reset_prompt()

        messages = [
            {"role": msg["role"], "content": json.dumps(msg["content"])}
            for msg in self.current_messages
        ]

        self.logger.debug(f"[{self.name}] Sending messages to OpenAI API")
        self.logger.debug(f"[{self.name}] Last interaction: {messages[-1]}")

        next_message = openai_utils.client().send_message(messages)

        self.logger.debug(f"[{self.name}] Received message: {next_message}")

        return next_message["role"], utils.extract_json(next_message["content"])

    ###########################################################
    # Internal cognitive state changes
    ###########################################################
    @transactional
    def _update_cognitive_state(
        self, goals=None, context=None, attention=None, emotions=None
    ):
        """
        Update the TinyPerson's cognitive state.
        """

        # update current datetime
        self._configuration["current_datetime"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # update current goals
        if goals is not None:
            self._configuration["current_goals"] = goals

        # update current context
        if context is not None:
            self._configuration["current_context"] = context

        # update current attention
        if attention is not None:
            self._configuration["current_attention"] = attention

        # update current emotions
        if emotions is not None:
            self._configuration["current_emotions"] = emotions

        self.reset_prompt()

    ###########################################################
    # Inspection conveniences
    ###########################################################
    def _display_communication(
        self,
        role,
        content,
        kind,
        simplified=True,
        max_content_length=default_max_content_display_length,
    ):
        """
        Displays the current communication and stores it in a buffer for later use.
        """
        if kind == "stimuli":
            rendering = self._pretty_stimuli(
                role=role,
                content=content,
                simplified=simplified,
                max_content_length=max_content_length,
            )
        elif kind == "action":
            rendering = self._pretty_action(
                role=role,
                content=content,
                simplified=simplified,
                max_content_length=max_content_length,
            )
        else:
            raise ValueError(f"Unknown communication kind: {kind}")

        # if the agent has no parent environment, then it is a free agent and we can display the communication.
        # otherwise, the environment will display the communication instead. This is important to make sure that
        # the communication is displayed in the correct order, since environments control the flow of their underlying
        # agents.
        if self.environment is None:
            self._push_and_display_latest_communication(rendering)
        else:
            self.environment._push_and_display_latest_communication(rendering)

    def _push_and_display_latest_communication(self, rendering):
        """
        Pushes the latest communications to the agent's buffer.
        """
        self._displayed_communications_buffer.append(rendering)
        print(rendering)

    def pop_and_display_latest_communications(self):
        """
        Pops the latest communications and displays them.
        """
        communications = self._displayed_communications_buffer
        self._displayed_communications_buffer = []

        for communication in communications:
            print(communication)

        return communications

    def clear_communications_buffer(self):
        """
        Cleans the communications buffer.
        """
        self._displayed_communications_buffer = []

    @transactional
    def pop_latest_actions(self) -> list:
        """
        Returns the latest actions performed by this agent. Typically used
        by an environment to consume the actions and provide the appropriate
        environmental semantics to them (i.e., effects on other agents).
        """
        actions = self._actions_buffer
        self._actions_buffer = []
        return actions

    @transactional
    def pop_actions_and_get_contents_for(
        self, action_type: str, only_last_action: bool = True
    ) -> list:
        """
        Returns the contents of actions of a given type performed by this agent.
        Typically used to perform inspections and tests.

        Args:
            action_type (str): The type of action to look for.
            only_last_action (bool, optional): Whether to only return the contents of the last action. Defaults to False.
        """
        actions = self.pop_latest_actions()
        # Filter the actions by type
        actions = [action for action in actions if action["type"] == action_type]

        # If interested only in the last action, return the latest one
        if only_last_action:
            return actions[-1].get("content", "")

        # Otherwise, return all contents from the filtered actions
        return "\n".join([action.get("content", "") for action in actions])

    #############################################################################################
    # Formatting conveniences
    #
    # For rich colors,
    #    see: https://rich.readthedocs.io/en/latest/appendix/colors.html#appendix-colors
    #############################################################################################

    def __repr__(self):
        return f"TinyPerson(name='{self.name}')"

    def minibio(self):
        """
        Returns a mini-biography of the TinyPerson.
        """
        return f"{self.name} is a {self._configuration['age']} year old {self._configuration['occupation']}, {self._configuration['nationality']}, currently living in {self._configuration['country_of_residence']}."

    def pp_current_interactions(
        self,
        simplified=True,
        skip_system=True,
        max_content_length=default_max_content_display_length,
    ):
        """
        Pretty prints the current messages.
        """
        print(
            self.pretty_current_interactions(
                simplified=simplified,
                skip_system=skip_system,
                max_content_length=max_content_length,
            )
        )

    def pretty_current_interactions(self, simplified=True, skip_system=True, max_content_length=default_max_content_display_length):
      """
      Returns a pretty, readable, string with the current messages.
      """
      lines = []
      for message in self.episodic_memory.retrieve_all():
        try:
            if not (skip_system and message['role'] == 'system'):
                msg_simplified_type = ""
                msg_simplified_content = ""
                msg_simplified_actor = ""

                if message["role"] == "system":
                    msg_simplified_actor = "SYSTEM"
                    msg_simplified_type = message["role"]
                    msg_simplified_content = message["content"]

                    lines.append(
                        f"[dim] {msg_simplified_type}: {msg_simplified_content}[/]"
                    )

                elif message["role"] == "user":
                    lines.append(
                        self._pretty_stimuli(
                            role=message["role"],
                            content=message["content"],
                            simplified=simplified,
                            max_content_length=max_content_length,
                        )
                    )

                elif message["role"] == "assistant":
                    lines.append(
                        self._pretty_action(
                            role=message["role"],
                            content=message["content"],
                            simplified=simplified,
                            max_content_length=max_content_length,
                        )
                    )
                else:
                    lines.append(f"{message['role']}: {message['content']}")
        except:
            # print(f"ERROR: {message}")
            continue

      return "\n".join(lines)

    def _pretty_stimuli(
        self,
        role,
        content,
        simplified=True,
        max_content_length=default_max_content_display_length,
    ) -> list:
        """
        Pretty prints stimuli.
        """

        lines = []
        msg_simplified_actor = "USER"
        for stimus in content["stimuli"]:
            if simplified:
                if stimus["source"] != "":
                    msg_simplified_actor = stimus["source"]

                else:
                    msg_simplified_actor = "USER"

                msg_simplified_type = stimus["type"]
                msg_simplified_content = break_text_at_length(
                    stimus["content"], max_length=max_content_length
                )

                indent = " " * len(msg_simplified_actor) + "      > "
                msg_simplified_content = textwrap.fill(
                    msg_simplified_content,
                    width=TinyPerson.PP_TEXT_WIDTH,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )

                #
                # Using rich for formatting. Let's make things as readable as possible!
                #
                if msg_simplified_type == "CONVERSATION":
                    rich_style = "bold italic cyan1"
                elif msg_simplified_type == "THOUGHT":
                    rich_style = "dim italic cyan1"
                else:
                    rich_style = "italic"

                lines.append(
                    f"[{rich_style}][underline]{msg_simplified_actor}[/] --> [{rich_style}][underline]{self.name}[/]: [{msg_simplified_type}] \n{msg_simplified_content}[/]"
                )
            else:
                lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _pretty_action(
        self,
        role,
        content,
        simplified=True,
        max_content_length=default_max_content_display_length,
    ) -> str:
        """
        Pretty prints an action.
        """
        if simplified:
            msg_simplified_actor = self.name
            msg_simplified_type = content["action"]["type"]
            msg_simplified_content = break_text_at_length(
                content["action"].get("content", ""), max_length=max_content_length
            )

            indent = " " * len(msg_simplified_actor) + "      > "
            msg_simplified_content = textwrap.fill(
                msg_simplified_content,
                width=TinyPerson.PP_TEXT_WIDTH,
                initial_indent=indent,
                subsequent_indent=indent,
            )

            #
            # Using rich for formatting. Let's make things as readable as possible!
            #
            if msg_simplified_type == "DONE":
                rich_style = "grey82"
            elif msg_simplified_type == "TALK":
                rich_style = "bold green3"
            elif msg_simplified_type == "THINK":
                rich_style = "green"
            else:
                rich_style = "purple"

            return f"[{rich_style}][underline]{msg_simplified_actor}[/] acts: [{msg_simplified_type}] \n{msg_simplified_content}[/]"
        else:
            return f"{role}: {content}"

    ###########################################################
    # IO
    ###########################################################

    def save_spec(self, path, include_memory=False):
        """
        Saves the current configuration to a JSON file.
        """
        spec = {"tinyperson": self._configuration}

        # should we include the memory?
        if include_memory:
            spec["memory"] = self.episodic_memory.to_json()

        with open(path, "w") as f:
            json.dump(self._configuration, f, indent=4, sort_keys=True)

    def _load_spec(self, path):
        """
        Loads a JSON file into the current configuration.
        """
        with open(path, "r") as f:
            spec = json.load(f)
            self._configuration = spec["tinyperson"]

            # some configurations are reflected in the TinyPerson's attributes
            self.name = self._configuration["name"]

            # should we also load the memory?
            if "memory" in spec:
                self.episodic_memory.from_json(spec["memory"])
        
        self.reset_prompt()

    def encode_complete_state(self) -> dict:
        """
        Encodes the complete state of the TinyPerson, including the current messages.
        """
        to_copy = copy.copy(self.__dict__)

        # delete the logger and other attributes that cannot be serialized
        del to_copy["logger"]
        del to_copy["environment"]


        to_copy['episodic_memory'] = self.episodic_memory.to_json()
        to_copy['semantic_memory'] = self.semantic_memory.to_json()

        state = copy.deepcopy(to_copy)

        return state

    def decode_complete_state(self, state: dict) -> Self:
        """
        Loads the complete state of the TinyPerson, including the current messages,
        and produces a new TinyPerson instance.
        """

        self.episodic_memory = EpisodicMemory.from_json(state['episodic_memory'])
        self.semantic_memory = SemanticMemory.from_json(state['semantic_memory'])

        return self
    
    def create_new_agent_from_current_spec(self, new_name:str) -> Self:
        """
        Creates a new agent from the current agent's specification. 

        Args:
            new_name (str): The name of the new agent. Agent names must be unique in the simulation, 
              this is why we need to provide a new name.
        """
        new_agent = TinyPerson(name=new_name, spec_path=None)
        
        new_config = copy.deepcopy(self._configuration)
        new_config['name'] = new_name

        new_agent._configuration = new_config

        return new_agent
        

    @staticmethod
    def add_agent(agent):
        """
        Adds an agent to the global list of agents. Agent names must be unique,
        so this method will raise an exception if the name is already in use.
        """
        if agent.name in TinyPerson.all_agents:
            raise ValueError(f"Agent name {agent.name} is already in use.")
        else:
            TinyPerson.all_agents[agent.name] = agent

    @staticmethod
    def has_agent(agent_name: str):
        """
        Checks if an agent is already registered.
        """
        return agent_name in TinyPerson.all_agents

    @staticmethod
    def set_simulation_for_free_agents(simulation):
        """
        Sets the simulation if it is None. This allows free agents to be captured by specific simulation scopes
        if desired.
        """
        for agent in TinyPerson.all_agents.values():
            if agent.simulation_id is None:
                simulation.add_agent(agent)

    @staticmethod
    def get_agent_by_name(name):
        """
        Gets an agent by name.
        """
        if name in TinyPerson.all_agents:
            return TinyPerson.all_agents[name]
        else:
            return None

    @staticmethod
    def clear_agents():
        """
        Clears the global list of agents.
        """
        TinyPerson.all_agents = {}
