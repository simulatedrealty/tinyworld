import pytest
import logging
logger = logging.getLogger("tinytroupe")

import sys
sys.path.append('../../tinytroupe/')
sys.path.append('../../')
sys.path.append('..')


import tinytroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld, TinySocialNetwork
from tinytroupe.personfactory import TinyPersonFactory
from tinytroupe.extraction import InteractionResultsExtractor
from tinytroupe.story import Story

from tinytroupe.examples import lisa, oscar, marcos
from tinytroupe.extraction import default_extractor as extractor
import tinytroupe.control as control
from tinytroupe.control import Simulation

from testing_utils import *

def test_story_continuation(setup, focus_group_world):
    world = focus_group_world

    story_beginning =\
          """
            You were vacationing in the beautiful city of Rio de Janeiro, Brazil. You were walking down the beach when
            the most unexpected thing happened: an Alien spaceship landed right in front of you. The door opened and a
            friendly Alien stepped out. The Alien introduced itself as Zog, and explained that it was on a mission to
            learn more about Earth's cultures. You were intrigued by this encounter and decided to help Zog in its mission.
          """

    world.broadcast(story_beginning)
    
    world.run(2)

    story = Story(world)

    continuation = story.continue_story()

    print("Story continuation: ", continuation)

    assert proposition_holds(f"The following two text blocks could belong to the same story: \n BLOCK 1: '{story_beginning}' and \n BLOCK 2: '{continuation}'"), f"Proposition is false according to the LLM."

