

# Archived DeepTraffic LLM Reward v0.3

A research sandbox for comparing rule-based, DQN, and language-model policies in a small, headless traffic environment inspired by MIT's DeepTraffic.

## What this project contains

The Python environment v0.3 uses the [DeepTraffic paper v2](https://arxiv.org/abs/1801.02805v2) as its sole scientific specification. Archived website code is not a reference for this implementation. Unspecified details and interpretations of conflicting paper statements are documented explicitly.

Empty observation cells are now zero, as defined in the paper. Following distance
uses vehicle reference positions rather than bumper clearance. Previous v0.1/v0.2
scores and model checkpoints are not interchangeable with this version.

The current environment provides:

- a seven-lane highway with 20 vehicles;
- a `(3, 10)` ego-relative observation grid;
- five discrete actions: accelerate, decelerate, move left, move right, and maintain speed;
- collision-avoidance safety overrides and randomized vehicle respawning;
- deterministic seeding for repeatable experiments; and
- a speed-based reward derived from the ego vehicle's speed.

The exact assumptions, observation and action semantics, reward, episode settings, and known limitations are documented in [`docs/environment_spec.md`](docs/environment_spec.md).

## Project layout

```text
src/deeptraffic_llm_reward/  Environment and example policies
scripts/run_smoke.py         Reproducible policy smoke test
tests/                       Environment and dynamics tests
docs/environment_spec.md     Environment contract and assumptions
network_basic.js             Original MIT DeepTraffic example network
```

## Installation

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Verify the environment

Run the test suite:

```bash
.venv/bin/python -m pytest
```

Run a short smoke test with the included rule-based policy:

```bash
.venv/bin/python scripts/run_smoke.py --policy rule --seeds 3 --frames 10000
```

The smoke test is a technical check of this reconstruction. Its score must not be interpreted as a DQN/LLM benchmark or compared directly with historical MIT DeepTraffic scores.

## Research status

This is an experimental foundation for future DQN and LLM-policy work. The simulator assumptions should be validated before using results as scientific baselines. No trained DQN or LLM policy is included yet.

## License and attribution

The upstream MIT DeepTraffic material is retained under the terms in [`LICENSE`](LICENSE). The Python reconstruction and its documentation are provided in this repository for research and experimentation.

## Problem Statement: Traffic is Terrible

> "Americans will put up with anything provided it doesn’t block traffic." - Dan Rather 

> "Traffic is soul-destroying." - Elon Musk

In the U.S. alone, we spend 6.9 billion hours sitting in traffic each year [1] — roughly 10,000 human lifetimes [2]. Autonomous vehicles will be able to alleviate part (but not all) of the problem. Already, they show promise in reducing phantom traffic jams [3,4].

We’ve designed DeepTraffic to let people (from beginners to experts) explore the design of motion planning algorithms for autonomous vehicles and to inspire the next generation of traffic engineering. We thank the thousands of competitors who have submitted solutions and are actively participating.

## DeepTraffic Layout

<img src="https://selfdrivingcars.mit.edu/wordpress/wp-content/uploads/2018/01/Screenshot-from-2018-01-08-17-45-42.png" 
alt="DeepTraffic" width="600" />

The game page consists of four different areas:

- On the left, you can find a real time simulation of the road with different display options.

- On the upper half of the page, you can find (1) a coding area where you can change the design of the neural network which controls the agents and (2) some buttons for applying your changes, saving/loading, and making a submission.

- Below the coding area, you can find (1) a graph showing a moving average of the center red car’s reward, (2) a visualization of the neural network activations, and (3) buttons for training and testing your network.

- Between the simulated roadway and the graphs, you can find the current image of you vehicle and some options to customize it and create a visualization of your best submission.

The simulation area shows some basic information like the current speed of the car and the number of cars that have been passed since you opened the site. It also allows you to change the way the simulation is displayed.
![Display selection](http://selfdrivingcars.mit.edu/wordpress/wp-content/uploads/2017/01/normal.gif)

## DeepTraffic Simulation & Game

In short, DeepTraffic is a game in which you (a competitor) design your own motion planning algorithm in order to drive a vehicle as fast as possible through dense traffic.

Your algorithm will operate on a 7 lane highway. There are 20 vehicles on the road. Your algorithm controls some vehicles. The game controls the others.

Each autonomous agent runs a copy of your algorithm. Every 30 frames, your algorithm selects 1 of 5 actions:

1. accelerate
1. decelerate
1. change into the left lane
1. change in to the right lane
1. do nothing, i.e. maintain speed in present lane.

Your algorithm will receive, as input, an occupancy grid, representing the free space around the agent. The value of unoccupied cells is set to 80mph. The value of occupied cells is set to the speed of the occupying vehicle. For example here's an occupancy grid `(lanesSide = 1; patchesAhead=10)`:

![learning input](https://selfdrivingcars.mit.edu/wordpress/wp-content/uploads/2017/01/Screen-Shot-2017-01-03-at-16.06.29-151x300.png)

There are a few quirks to DeepTraffic’s dynamics:

### Safety System

Each vehicle has a safety system which prevents it from colliding with other vehicles. This has 2 implications for how you will design your algorithm. First, your algorithm does not need to consider collision avoidance. Second, your path will be overridden when the safety system is activated.

For example, here, the red car cannot accelerate or change into the right lane, because the collision avoidance system has detected vehicles in the way:

![safety system](https://selfdrivingcars.mit.edu/wordpress/wp-content/uploads/2017/01/Screen-Shot-2017-01-03-at-16.28.32-152x300.png)

A vehicle that is 4 cells behind another will immediately slow to match the lead vehicle, regardless of what its algorithm tries to do. (see diagram above)

A vehicle driving beside another will be unable to change into its neighbor’s lane until there is a sufficient gap, regardless of what its algorithm tries to do. (see diagram above)

### Multiple Agents

In version 2.0, the current version, you have the option to deploy a copy of your algorithm on 11 vehicles. You algorithm won’t do multi-agent planning, rather each vehicle makes a greedy choice. The challenge is to design an algorithm which does not get in its own way when controlling several vehicles.

### Where the Highway Ends

DeepTraffic follows just one of the vehicles (the ego vehicle), so you’ll notice some of the vehicles fall off the highway when they drive slower or faster than the ego vehicle. What happens to these vehicles?

When vehicles fall off the road, they are replaced by new vehicles on the opposite end of the highway. When a vehicle is replaced, its speed & lane is chosen randomly.

## Hyperparameters

To do well in DeepTraffic using DQN, you’ll have to choose good hyperparameters. This can be tricky because (1) the full hyperparameter space is rather large and (2) the bigger your network gets, the longer it takes to train which means you’ll explore less of the hyper-parameter space. Therefore, it helps to understand how changing the hyper-parameters will change performance prior to training.

![parameters](https://selfdrivingcars.mit.edu/wordpress/wp-content/uploads/2019/01/deeptraffic_parameters.png)

## Results

### Progress

The plot below shows how the competition progressed over time:

![progress](https://selfdrivingcars.mit.edu/wordpress/wp-content/uploads/2019/01/score_curve_data.png)

### The Structure of Submissions

Below is a t-SNE plot, i.e. submissions originally represented in a vector space spanning patchesAhead, patchesBehind, l2_decay, layer_count, gamma, learning_rate, lanesSide, train_iterations are plotted in a 2 dimensional space which preserves the composition of neighboring points. The color of each dot corresponds to submissions score. An interesting feature of this plot is that several clusters emerge — competitors stumbled upon similar solutions.

![tsne](https://selfdrivingcars.mit.edu/wordpress/wp-content/uploads/2018/06/tsne-scatter.png)

## Help and Documentation

See [Documentation page](https://selfdrivingcars.mit.edu/deeptraffic-documentation/) for more details and hints and how to submit to the competition.

## Team

- [Lex Fridman](https://lexfridman.com/) ([Twitter](https://twitter.com/))
- [Jack Terwilliger](http://www.mit.edu/~jterwill/)
- [Benedikt Jenik](http://bjenik.com/)

## References

- [1] https://static.tti.tamu.edu/tti.tamu.edu/documents/mobility-scorecard-2015.pdf
- [2] (6.9 * 1000000000) / (75 * 365 * 24)
- [3] Horn, Berthold KP. "Suppressing traffic flow instabilities." Intelligent Transportation Systems-(ITSC), 2013 16th International IEEE Conference on. IEEE, 2013. https://people.csail.mit.edu/bkph/articles/Suppressing_Traffic_Flow%20Instabilities_IEEE_ITS_2013.pdf
- [4] Stern, Raphael E., et al. "Dissipation of stop-and-go waves via control of autonomous vehicles: Field experiments." Transportation Research Part C: Emerging Technologies 89 (2018): 205-221. https://arxiv.org/pdf/1705.01693.pdf
- [5] https://link.springer.com/content/pdf/10.1007/BF00992698.pdf
