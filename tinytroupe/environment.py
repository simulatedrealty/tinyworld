"""
Environments provide a structured way to define the world in which the
agents interact with each other as well as external entities (e.g., search engines).
"""

import logging
import copy

from tinytroupe.agent import *
from tinytroupe.utils import name_or_empty
import tinytroupe.control as control
from tinytroupe.control import transactional
 
from rich.console import Console

from typing import Any, TypeVar, Union
AgentOrWorld = Union["TinyPerson", "TinyWorld"]

class TinyWorld:
    """
    Base class for environments.
    """

    # A dict of all environments created so far.
    all_environments = {} # name -> environment

    def __init__(self, name: str="A TinyWorld", agents=[], 
                 broadcast_if_no_target=True,
                 communication_display=True,
                 simulation_id=None):
        """
        Initializes an environment.

        Args:
            name (str): The name of the environment.
            agents (list): A list of agents to add to the environment.
            broadcast_if_no_target (bool): If True, broadcast actions if the target of an action is not found.
            communication_display (bool): If True, displays environment information.
            simulation_id (str): The ID of the simulation. Defaults to None.
        """
        self.logger = logging.getLogger("tinytroupe")

        self.name = name
        self.broadcast_if_no_target = broadcast_if_no_target
        self.communication_display = communication_display
        self.simulation_id = simulation_id # will be reset later if the agent is used within a specific simulation scope
        
        
        self.agents = []
        self.name_to_agent = {} # {agent_name: agent, agent_name_2: agent_2, ...}

        # the buffer of communications that have been displayed so far, used for
        # saving these communications to another output form later (e.g., caching)
        self._displayed_communications_buffer = []

        self.console = Console()

        # add the environment to the list of all environments
        TinyWorld.add_environment(self)
        
        self.add_agents(agents)
    
# TODO ??
#    
#   def __deepcopy__(self, memo):
#       """
#       Deep copy the environment.
#       """
#       w = TinyWorld(self.name, agents=self.agents, 
#                broadcast_if_no_target=self.broadcast_if_no_target,
#                communication_display=self.communication_display,
#                simulation_id=self.simulation_id)
#
#       w._displayed_communications_buffer = copy.deepcopy(self._displayed_communications_buffer)
#       
#       return w
    
    #######################################################################
    # Simulation control methods
    #######################################################################
    @transactional
    def _step(self):
        """
        Performs a single step in the environment. This default implementation
        simply calls makes all agents in the environment act and properly
        handle the resulting actions. Subclasses might override this method to implement 
        different policies.
        """
        
        # agents can act
        for agent in self.agents:
            self.logger.debug(f"[{self.name}] Agent {name_or_empty(agent)} is acting.")
            agent.act()
            self._handle_actions(agent, agent.pop_latest_actions())

    @transactional
    def run(self, steps: int):
        """
        Runs the environment for a given number of steps.

        Args:
            steps (int): The number of steps to run the environment for.
        """
        for i in range(steps):
            self.logger.info(f"[{self.name}] Running world simulation step {i+1} of {steps}.")

            if self.communication_display:
                self._display_communication(cur_step=i+1, total_steps=steps, kind='step')

            self._step()

    #######################################################################
    # Agent management methods
    #######################################################################
    def add_agents(self, agents: list):
        """
        Adds a list of agents to the environment.

        Args:
            agents (list): A list of agents to add to the environment.
        """
        for agent in agents:
            self.add_agent(agent)
        
        return self # for chaining

    def add_agent(self, agent: TinyPerson):
        """
        Adds an agent to the environment. The agent must have a unique name within the environment.

        Args:
            agent (TinyPerson): The agent to add to the environment.
        
        Raises:
            ValueError: If the agent name is not unique within the environment.
        """

        # check if the agent is not already in the environment
        if agent not in self.agents:
            self.logger.debug(f"Adding agent {agent.name} to the environment.")
            
            # Agent names must be unique in the environment. 
            # Check if the agent name is already there.
            if agent.name not in self.name_to_agent:
                agent.environment = self
                self.agents.append(agent)
                self.name_to_agent[agent.name] = agent
            else:
                raise ValueError(f"Agent names must be unique, but '{agent.name}' is already in the environment.")
        else:
            self.logger.warn(f"Agent {agent.name} is already in the environment.")
        
        return self # for chaining

    def remove_agent(self, agent: TinyPerson):
        """
        Removes an agent from the environment.

        Args:
            agent (TinyPerson): The agent to remove from the environment.
        """
        self.logger.debug(f"Removing agent {agent.name} from the environment.")
        self.agents.remove(agent)
        del self.name_to_agent[agent.name]

        return self # for chaining
    
    def remove_all_agents(self):
        """
        Removes all agents from the environment.
        """
        self.logger.debug(f"Removing all agents from the environment.")
        self.agents = []
        self.name_to_agent = {}

        return self # for chaining

    def get_agent_by_name(self, name: str) -> TinyPerson:
        """
        Returns the agent with the specified name. If no agent with that name exists in the environment, 
        returns None.

        Args:
            name (str): The name of the agent to return.

        Returns:
            TinyPerson: The agent with the specified name.
        """
        if name in self.name_to_agent:
            return self.name_to_agent[name]
        else:
            return None
        

    #######################################################################
    # Action handlers
    #
    # Specific actions issued by agents are handled by the environment,
    # because they have effects beyond the agent itself.
    #######################################################################
    @transactional
    def _handle_actions(self, source: TinyPerson, actions: list):
        """ 
        Handles the actions issued by the agents.

        Args:
            source (TinyPerson): The agent that issued the actions.
            actions (list): A list of actions issued by the agents. Each action is actually a
              JSON specification.
            
        """
        for action in actions:
            action_type = action["type"] # this is the only required field
            content = action["content"] if "content" in action else None
            target = action["target"] if "target" in action else None

            self.logger.debug(f"[{self.name}] Handling action {action_type} from agent {name_or_empty(source)}. Content: {content}, target: {target}.")

            # only some actions require the enviroment to intervene
            if action_type == "REACH_OUT":
                self._handle_reach_out(source, content, target)
            elif action_type == "TALK":
                self._handle_talk(source, content, target)

    @transactional
    def _handle_reach_out(self, source_agent: TinyPerson, content: str, target: str):
        """
        Handles the REACH_OUT action. This default implementation always allows REACH_OUT to succeed.
        Subclasses might override this method to implement different policies.

        Args:
            source_agent (TinyPerson): The agent that issued the REACH_OUT action.
            content (str): The content of the message.
            target (str): The target of the message.
        """

        # This default implementation always allows REACH_OUT to suceed.
        target_agent = self.get_agent_by_name(target)
        
        source_agent.make_agent_accessible(target_agent)
        target_agent.make_agent_accessible(source_agent)

        source_agent.socialize(f"{name_or_empty(target_agent)} was successfully reached out, and is now available for interaction.", source=self)
        target_agent.socialize(f"{name_or_empty(source_agent)} reached out to you, and is now available for interaction.", source=self)

    @transactional
    def _handle_talk(self, source_agent: TinyPerson, content: str, target: str):
        """
        Handles the TALK action by delivering the specified content to the specified target.

        Args:
            source_agent (TinyPerson): The agent that issued the TALK action.
            content (str): The content of the message.
            target (str, optional): The target of the message.
        """
        target_agent = self.get_agent_by_name(target)

        self.logger.debug(f"[{self.name}] Delivering message from {name_or_empty(source_agent)} to {name_or_empty(target_agent)}.")

        if target_agent is not None:
            target_agent.listen(content, source=source_agent)
        elif self.broadcast_if_no_target:
            self.broadcast(content, source=source_agent)

    #######################################################################
    # Interaction methods
    #######################################################################
    @transactional
    def broadcast(self, speech: str, source: AgentOrWorld=None):
        """
        Delivers a speech to all agents in the environment.

        Args:
            speech (str): The content of the message.
            source (AgentOrWorld, optional): The agent or environment that issued the message. Defaults to None.
        """
        self.logger.debug(f"[{self.name}] Broadcasting message: '{speech}'.")

        for agent in self.agents:
            # do not deliver the message to the source
            if agent != source:
                agent.listen(speech, source=source)

    def make_everyone_accessible(self):
        """
        Makes all agents in the environment accessible to each other.
        """
        for agent_1 in self.agents:
            for agent_2 in self.agents:
                if agent_1 != agent_2:
                    agent_1.make_agent_accessible(agent_2)
            

    ###########################################################
    # Formatting conveniences
    ###########################################################

    # TODO better names for these "display" methods
    def _display_communication(self, cur_step, total_steps, kind):
        """
        Displays the current communication and stores it in a buffer for later use.
        """
        if kind == 'step':
            rendering = self._pretty_step(cur_step=cur_step, total_steps=total_steps) 
        else:
            raise ValueError(f"Unknown communication kind: {kind}")

        self._push_and_display_latest_communication({"content": rendering, "kind": kind})
    
    def _push_and_display_latest_communication(self, rendering):
        """
        Pushes the latest communications to the agent's buffer.
        """
        self._displayed_communications_buffer.append(rendering)
        self._display(rendering)

    def pop_and_display_latest_communications(self):
        """
        Pops the latest communications and displays them.
        """
        communications = self._displayed_communications_buffer
        self._displayed_communications_buffer = []

        for communication in communications:
            self._display(communication)

        return communications    

    def _display(self, communication):
        # unpack the rendering to find more info
        if isinstance(communication, dict):
            content = communication["content"]
            kind = communication["kind"]
        else:
            content = communication
            kind = None
            
        # render as appropriate
        if kind == 'step':
            self.console.rule(content)
        else:
            self.console.print(content)
    
    def clear_communications_buffer(self):
        """
        Cleans the communications buffer.
        """
        self._displayed_communications_buffer = []

    def __repr__(self):
        return f"TinyWorld(name='{self.name}')"

    def _pretty_step(self, cur_step, total_steps):
        rendering = f"{self.name} step {cur_step} of {total_steps}"

        return rendering

    def pp_current_interactions(self, simplified=True, skip_system=True):
        """
        Pretty prints the current messages from agents in this environment.
        """
        print(self.pretty_current_interactions(simplified=simplified, skip_system=skip_system))

    def pretty_current_interactions(self, simplified=True, skip_system=True):
      """
      Returns a pretty, readable, string with the current messages of agents in this environment.
      """
      agent_contents = []

      for agent in self.agents:
          agent_content = f"#### Interactions from the point of view of {agent.name} agent:\n"
          agent_content += f"**BEGIN AGENT {agent.name} HISTORY.**\n "
          agent_content += agent.pretty_current_interactions(simplified=simplified, skip_system=skip_system) + "\n"
          agent_content += f"**FINISHED AGENT {agent.name} HISTORY.**\n "
          agent_contents.append(agent_content)      
          
      return "\n".join(agent_contents)
    
    #######################################################################
    # IO
    #######################################################################

    def encode_complete_state(self) -> dict:
        """
        Encodes the complete state of the environment in a dictionary.

        Returns:
            dict: A dictionary encoding the complete state of the environment.
        """
        to_copy = copy.copy(self.__dict__)

        # remove the logger and other fields
        del to_copy["logger"]
        del to_copy['console']
        del to_copy['agents']
        del to_copy['name_to_agent']

        state = copy.deepcopy(to_copy)

        # agents are encoded separately
        state["agents"] = [agent.encode_complete_state() for agent in self.agents]

        return state
    
    def decode_complete_state(self, state:dict) -> Self:
        """
        Decodes the complete state of the environment from a dictionary.

        Args:
            state (dict): A dictionary encoding the complete state of the environment.

        Returns:
            Self: The environment decoded from the dictionary.
        """
        state = copy.deepcopy(state)

        #################################
        # restore agents in-place
        #################################
        self.remove_all_agents()
        for agent_state in state["agents"]:
            try:
                try:
                    agent = TinyPerson.get_agent_by_name(agent_state["name"])
                except Exception as e:
                    raise ValueError(f"Could not find agent {agent_state['name']} for environment {self.name}.") from e
                
                agent.decode_complete_state(agent_state)
                self.add_agent(agent)
                
            except Exception as e:
                raise ValueError(f"Could not decode agent {agent_state['name']} for environment {self.name}.") from e
        
        # remove the agent states to update the rest of the environment
        del state["agents"]

        # restore other fields
        self.__dict__.update(state)

        return self

    @staticmethod
    def add_environment(environment):
        """
        Adds an environment to the list of all environments. Environment names must be unique,
        so if an environment with the same name already exists, an error is raised.
        """
        if environment.name in TinyWorld.all_environments:
            raise ValueError(f"Environment names must be unique, but '{environment.name}' is already defined.")
        else:
            TinyWorld.all_environments[environment.name] = environment
        

    @staticmethod
    def set_simulation_for_free_environments(simulation):
        """
        Sets the simulation if it is None. This allows free environments to be captured by specific simulation scopes
        if desired.
        """
        for environment in TinyWorld.all_environments.values():
            if environment.simulation_id is None:
                simulation.add_environment(environment)
    
    @staticmethod
    def get_environment_by_name(name: str):
        """
        Returns the environment with the specified name. If no environment with that name exists, 
        returns None.

        Args:
            name (str): The name of the environment to return.

        Returns:
            TinyWorld: The environment with the specified name.
        """
        if name in TinyWorld.all_environments:
            return TinyWorld.all_environments[name]
        else:
            return None
    
    @staticmethod
    def clear_environments():
        """
        Clears the list of all environments.
        """
        TinyWorld.all_environments = {}

class TinySocialNetwork(TinyWorld):

    def __init__(self, name, broadcast_if_no_target=True):
        """
        Create a new TinySocialNetwork environment.

        Args:
            name (str): The name of the environment.
            broadcast_if_no_target (bool): If True, broadcast actions through an agent's available relations
              if the target of an action is not found.
        """
        
        super().__init__(name, broadcast_if_no_target=broadcast_if_no_target)

        self.relations = {}
    
    @transactional
    def add_relation(self, agent_1, agent_2, name="default"):
        """
        Adds a relation between two agents.
        
        Args:
            agent_1 (TinyPerson): The first agent.
            agent_2 (TinyPerson): The second agent.
            name (str): The name of the relation.
        """

        self.logger.debug(f"Adding relation {name} between {agent_1.name} and {agent_2.name}.")

        # agents must already be in the environment, if not they are first added
        if agent_1 not in self.agents:
            self.agents.append(agent_1)
        if agent_2 not in self.agents:
            self.agents.append(agent_2)

        if name in self.relations:
            self.relations[name].append((agent_1, agent_2))
        else:
            self.relations[name] = [(agent_1, agent_2)]

        return self # for chaining
    
    @transactional
    def _update_agents_contexts(self):
        """
        Updates the agents' observations based on the current state of the world.
        """

        # clear all accessibility first
        for agent in self.agents:
            agent.make_all_agents_inaccessible()

        # now update accessibility based on relations
        for relation_name, relation in self.relations.items():
            self.logger.debug(f"Updating agents' observations for relation {relation_name}.")
            for agent_1, agent_2 in relation:
                agent_1.make_agent_accessible(agent_2)
                agent_2.make_agent_accessible(agent_1)

    @transactional
    def _step(self):
        self._update_agents_contexts()

        #call super
        super()._step()
    
    @transactional
    def _handle_reach_out(self, source_agent: TinyPerson, content: str, target: str):
        """
        Handles the REACH_OUT action. This social network implementation only allows
        REACH_OUT to succeed if the target agent is in the same relation as the source agent.

        Args:
            source_agent (TinyPerson): The agent that issued the REACH_OUT action.
            content (str): The content of the message.
            target (str): The target of the message.
        """
            
        # check if the target is in the same relation as the source
        if self.is_in_relation_with(source_agent, self.get_agent_by_name(target)):
            super()._handle_reach_out(source_agent, content, target)
            
        # if we get here, the target is not in the same relation as the source
        source_agent.socialize(f"{target} is not in the same relation as you, so you cannot reach out to them.", source=self)


    # TODO implement _handle_talk using broadcast_if_no_target too

    #######################################################################
    # Utilities and conveniences
    #######################################################################

    def is_in_relation_with(self, agent_1:TinyPerson, agent_2:TinyPerson, relation_name=None) -> bool:
        """
        Checks if two agents are in a relation. If the relation name is given, check that
        the agents are in that relation. If no relation name is given, check that the agents
        are in any relation. Relations are undirected, so the order of the agents does not matter.

        Args:
            agent_1 (TinyPerson): The first agent.
            agent_2 (TinyPerson): The second agent.
            relation_name (str): The name of the relation to check, or None to check any relation.

        Returns:
            bool: True if the two agents are in the given relation, False otherwise.
        """
        if relation_name is None:
            for relation_name, relation in self.relations.items():
                if (agent_1, agent_2) in relation or (agent_2, agent_1) in relation:
                    return True
            return False
        
        else:
            if relation_name in self.relations:
                return (agent_1, agent_2) in self.relations[relation_name] or (agent_2, agent_1) in self.relations[relation_name]
            else:
                return False