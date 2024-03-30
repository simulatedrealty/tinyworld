import pytest
import os

import sys
sys.path.append('../../tinytroupe/')
sys.path.append('../../')
sys.path.append('..')


from tinytroupe.examples import oscar
from tinytroupe.control import Simulation
import tinytroupe.control as control
import importlib

from testing_utils import *

def test_begin_checkpoint_end(setup):
    control.reset()
    
    assert control._current_simulations["default"] is None, "There should be no simulation running at this point."

    control.begin("control_test.json")
    assert control._current_simulations["default"].status == Simulation.STATUS_STARTED, "The simulation should be started at this point."

    agent = oscar()

    agent.define("age", 19)
    agent.define("nationality", "Brazilian")

    assert control._current_simulations["default"].cached_trace is not None, "There should be a cached trace at this point."
    assert control._current_simulations["default"].execution_trace is not None, "There should be an execution trace at this point."

    control.checkpoint()

    agent.listen_and_act("How are you doing?")

    # check if the file was created
    assert os.path.exists("control_test.json"), "The checkpoint file should have been created."

    control.end()

    assert control._current_simulations["default"].status == Simulation.STATUS_STOPPED, "The simulation should be ended at this point."
