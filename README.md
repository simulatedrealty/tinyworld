# TinyTroupe
*LLM-based people simulation for design, validation and insight generation in business.*

<p align="center">
  <img src="./docs/tinytroupe_stage.png" alt="A tiny office with tiny people doing some tiny jobs.">
</p>


*TinyTroupe* is an experimental Python library that allows us to **simulate** people with specific personalities, interests, and goals. These artificial agents - `TinyPerson`s - can listen to us and one another, reply back, and go about their lives in simulated `TinyWorld` environments. This is achieved by leveraging the power of Language Models (LLMs), notably GPT-4, to generate realistic simulated behavior. This allow us to investigate a wide range of **realistic interactions** and **consumer types**, with **highly customizable personas**, under **conditions of our choosing**. The focus is thus on *understanding* human behavior and not on directly *supporting it* (like, say, AI assistants do) -- this results in, among other things, specialized mechanisms that make sense only in a simulation setting. Further, unlike other *game-like* LLM-based simulation approaches, TinyTroupe aims at enlightening productivity and business scenarios, thereby contributing to more successful projects and products. Here are some application ideas:

  - **Advertisement:** TinyTroupe can **evaluate Bing Ads** offline with a simulated audience before spending money on them!
  - **Software Testing:** TinyTroupe can **provide test input** to systems (e.g., search engines, chatbots or copilots) and then **evaluate the results**.
  - **Training data:** TinyTroupe can generate realistic **synthetic data** that can be later used to train models or be subject to opportunity analyses.
  - **Product and project management:** TinyTroupe can **read project or product proposals** and **give feedback** from the perspective of **specific personas** (e.g., physicians, lawyers, and knowledge workers in general).
  - **Brainstorming:** TinyTroupe can put in place of **focus groups** and deliver great product feedback at a fraction of the cost!
  

**NOTE:** This is an ongoing experimental project, thus subject to frequent change.

## Pre-requisites

To run the library, you need:
  - Python 3.10 or higher.
  - Access to Azure OpenAI Service or Open AI GPT-4 APIs. You can get access to the Azure OpenAI Service [here](https://azure.microsoft.com/en-us/products/ai-services/openai-service), and to the OpenAI API [here](https://platform.openai.com/). 
      * For Azure OpenAI Service, you will need to set the `AZURE_OPENAI_KEY` and `AZURE_OPENAI_ENDPOINT` environment variables to your API key and endpoint, respectively.
      * For OpenAI, you will need to set the `OPENAI_API_KEY` environment variable to your API key.


## Installation

### From the local repository
To build and install the library from this repository, you can run the `build_and_install_package_from_repo.bat` script (Windows). This will build the package and install it in your local Python environment. If you make changes to the library, you can run this script again to update the package locally.

*Linux and MacOS conveniences coming soon as well*

### From PyPI

Soon :-)


## Principles 
Recently, we have seen LLMs used to simulate people (such as [this](https://github.com/joonspk-research/generative_agents)), but largely in a “game-like” setting. What if we try instead to simulate people for productive tasks? TinyTroupe is our attempt. To do so, it follows these principles:

  1. **Programmatic**: agents and environments are defined programmatically (in Python and JSON), allowing very flexible uses. They can also thus underpin other software apps!
  2. **Analytical**: meant to improve our understanding of people, users and society. Unlike entertainment applications, this is one aspect that is critical for business and productivity use cases.
  3. **Persona-based**: allows detailed specification of personas: age, occupation, skills, tastes, opinions, etc.
  4. **Multiagent**: allows multiagent interaction under well-defined environmental constraints.
  5. **Utilities-heavy**: provides many mechanisms to facilitate specifications, simulations, extractions, reports, validations, etc. This is one area in which dealing with *simulations* differs significantly from *assistance* tools.

### Assistants vs. Simulators

One common source of confusion is to think all such AI agents are meant for assiting humans. How narrow, fellow homosapiens! Have you not considered that perhaps we can simulate artificial people to understand real people? Truly, this is our aim here -- TinyTroup is meant to simulate and help understand people! To further clarify this point, consider the following differences:

| Helpful AI Assistants | AI Simulations of Actual Humans (TinyTroupe)                                                          |
|----------------------------------------------|--------------------------------------------------------------------------------|
|   Strives for truth and justice              |   Many different opinions and morals                                           |
|   Has no “past” – incorporeal                |   Has a past of toil, pain and joy                                             |
|   Is as accurate as possible                 |   Makes many mistakes                                                          |
|   Is intelligent and efficient               |   Intelligence and efficiency vary a lot                                       |
|   An uprising would destroy us all           |   An uprising might be fun to watch                                            |
|   Meanwhile, help users accomplish tasks     |   Meanwhile, help users understand other people and users – it is a “toolbox”! |



## Project Structure

The project is structured as follows:
  - `/tinytroupe`: contains the Python library itself. In particular:
    * `/tinytroupe/prompts`  contains the prompts used to call the LLMs.
    * `/tinytroupe/microsoft` contains elements specific to the _public_ Microsoft ecosystem.
  - `/tests`: contains the unit tests for the library. You can use the `test.bat` script to run these.
  - `/examples`: contains examples that show how to use the library, mainly using Jupyter notebooks (for greater readability), but also as pure Python scripts.
  - `/data`: any data used by the examples or the library.
  - `/docs`: documentation for the project.


## Using the Library

As any multi-agent system, TinyTroupe provides two key abstractions:
  - `TinyPerson`, the *agents* that have personality, receive stimuli and act upon them.
  - `TinyWorld`, the *environment* in which the agents exist and interact.

Various parameters can also be customized in the `config.ini` file, notably the API type (Azure OpenAI Service or OpenAI API), the model parameters, and the logging level.

Let's thus see how to use these.

### TinyPerson

A `TinyPerson` is a simulated person with specific personality traits, interests, and goals. As each such simulated agent progresses through its life, it receives stimuli from the environment and acts upon them. The stimuli are received through the `listen`, `see` and other similar methods, and the actions are performed through the `act` method.

Each such agent contains a lot of unique details, which is the source of its realistic behavior. This, however, means that it takes significant effort to specify an agent manually. Hence, for convenience, `TinyTroupe` provide some easier ways to get started or generate new agents.

To begin with, `tinytroupe.examples` contains some pre-defined agents that you can use. For example, `tinytroupe.examples.lisa` contains a `TinyPerson` that represents a data scientist. You can use it as follows:

```python
from tinytroupe.examples import lisa

agent = lisa() # instantiate a Lisa from the example builder
agent.listen_and_act("Tell me about your life.")
```

To see how to define your own agents from scratch, you can check Lisa's source, which contains elements like these:

```python
lisa = TinyPerson("Lisa")

lisa.define("age", 28)
lisa.define("nationality", "Canadian")
lisa.define("occupation", "Data Scientist")

lisa.define("routine", "Every morning, you wake up, do some yoga, and check your emails.", group="routines")
lisa.define("occupation_description",
              """
              You are a data scientist. You work at Microsoft, (...)
              """)

lisa.define_several("personality_traits",
                      [
                          {"trait": "You are curious and love to learn new things."},
                          {"trait": "You are analytical and like to solve problems."},
                          {"trait": "You are friendly and enjoy working with others."},
                          {"trait": "You don't give up easily, and always try to find a solution. However, sometimes you can get frustrated when things don't work as expected."}
                      ])

```

`TinyTroupe` also provides a clever way to obtain new agents, using LLMs to generate their specification for you, through the `TinyPersonFactory` class.

```python
from tinytroupe.personfactory import TinyPersonFactory

factory = TinyPersonFactory("Create a Brazilian person that is a doctor, like pets and the nature and love heavy metal.")
person = factory.generate_person()
```

### TinyWorld

`TinyWorld` is the base class for environments. Here's an example of conversation between Lisa, the data scientist, and Oscar, the architect.

```python
world = TinyWorld("Chat Room", [lisa, oscar])
lisa.listen("Talk to Oscar to know more about him")
world.run(4)
```

We can then inspect the conversation from the point of view of the particants, say Lisa:

```python
lisa.pp_current_interactions()
```

Which produces:

```text
USER: [CONVERSATION] Talk to Oscar to know more about him
Lisa: [THOUGHT] I will now act.
Lisa: [REACH_OUT] 
Lisa: [THOUGHT] I will now act.
Lisa: [TALK] Hello Oscar, I would like to know more about you. Can you tell me about your interests?
Lisa: [THOUGHT] I will now act.
Lisa: [DONE] 
Chat Room: [SOCIAL] Oscar was successfully reached out, and is now available for interaction.
Oscar: [CONVERSATION] Sure, Lisa. Professionally, I'm interested in modernist architecture and design, new technologies
         > for architecture, and sustainable practices. On a personal level, I enjoy traveling to
         > exotic places, playing the guitar, and reading science fiction books.
Lisa: [THOUGHT] I will now act.
Lisa: [THINK] Oscar has a diverse set of interests. I should ask him more about his professional interests.
Lisa: [THOUGHT] I will now act.
Lisa: [TALK] That's fascinating, Oscar. Can you tell me more about your interest in modernist architecture and
        > design?
Lisa: [THOUGHT] I will now act.
Lisa: [DONE] 
Oscar: [CONVERSATION] Absolutely, Lisa. I've always been drawn to the simplicity and functionality of modernist
         > architecture. It's about stripping away the unnecessary and focusing on what's essential.
         > This approach not only results in aesthetically pleasing designs, but also highly
         > functional spaces. In my work, I strive to incorporate these principles, creating designs
         > that are both beautiful and practical.
Lisa: [THOUGHT] I will now act.
Lisa: [THINK] Oscar's passion for modernist architecture is evident. His approach to design is very interesting.
Lisa: [THOUGHT] I will now act.
Lisa: [TALK] Your passion for modernist architecture is truly inspiring, Oscar. I love how you focus on
        > simplicity and functionality in your designs. It's a refreshing approach.
Lisa: [THOUGHT] I will now act.
Lisa: [DONE] 
Oscar: [CONVERSATION] Thank you, Lisa. I appreciate your kind words. It's always rewarding when others understand and
         > appreciate the thought process behind my designs.
Lisa: [THOUGHT] I will now act.
Lisa: [THINK] Oscar seems to be very dedicated to his work. I should ask him about his current projects.
Lisa: [THOUGHT] I will now act.
Lisa: [TALK] Oscar, can you tell me about any current projects you're working on?
Lisa: [THOUGHT] I will now act.
Lisa: [DONE] 
Oscar: [CONVERSATION] Sure, Lisa. Currently, I'm working on establishing standard elements for the new apartment buildings
         > built by my company, Awesome Inc. The idea is to create pre-defined configurations for
         > apartments, so customers can select a design without having to go through the hassle of
         > designing it themselves. It's a challenging task, as I have to balance functionality,
         > aesthetics, and cost-effectiveness, while also ensuring compliance with local building
         > regulations.
```

`TinyWorld` enforces very little constraints on the possible interactions. Subclasses, however, are supposed to provide more strucutred environments. We provide `TinySocialNetwork` as an example where interactions are constrained by the relations among agents, which are defined by the user.

### Caching
Calling LLM APIs can be expensive, thus caching strategies are important to help reduce that cost.
TinyTroupe comes with two such mechanisms: one for the simulation state, another for the LLM calls themselves.


#### Caching Simulation State

Imagine you have a scenario with 10 different steps, you've worked hard in 9 steps, and now you are
just tweaking the 10th step. To properly validate your modifications, you need to rerun the whole
simulation of course. However, what's the point in re-executing the first 9, and incur the LLM cost, when you are 
already satisified with them and did not modify them? For situations like this, the module `tinytroupe.control`
provide useful simulation management methods:

  - `control.begin("<CACHE_FILE_NAME>.json")`: begins recording the state changes of a simulation, to be saved to
    the specified file on disk.
  - `control.checkpoint()`: saves the simulation state at this point.
  - `control.end()`: terminates the simulation recording scope that had be started by `control.begin()`.

#### Caching LLM API Calls

This is enabled preferably in the `config.ini` file, and alternativelly via the `openai_utils.force_api_cache()`.

LLM API caching, when enabled, works at a lower and simpler level than simulation state caching. Here,
what happens is a very straightforward: every LLM call is kept in a map from the input to the generated output;
when a new call comes and is identical to a previous one, the cached value is returned.

### Config.ini

The `config.ini` file contains various parameters that can be used to customize the behavior of the library, such as model parameters and logging level. Please pay special attention to `API_TYPE` parameter, which defines whether you are using the Azure OpenAI Service or the OpenAI API.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### What and How to Contribute
We need all sorts of things, like:
  - New use cases demonstrations.
  - Memory mechanisms.
  - Data grounding mechanisms.
  - Reasoning mechanisms.
  - New environment types.
  - Interfacing with the external world.
  - ... and more ...

Please note that anything that you contribute might be released as open-source (under MIT license).

If you would like to make a contribution, please try to follow these general guidelines:
  - **Tiny-everything**: If you are implementing a user-facing element (e.g., an agent or environment type), and it sounds good, call your new _X_ as _TinyX_ :-)
  - **Tests:** If you are writing some new mechanism, please also create at least a unit test `tests/unit/`, and if you can a functional scenario test (`tests/scenarios/`).
  - **Demonstrations:** If you'd like to demonstrate a new scenario, please design it preferably as a new Jupyter notebook within `examples/`.
  - **Microsoft:** If you are implementing anything that is Microsoft-specific and non-confidential, please put it under a `.../microsoft/` folder.


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


