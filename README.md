# Particle simulation

## Description
This simulation uses attention to simulate particles. Using attention, the lowest entropy configuration of particles is found: min entropy(softmax(x * softmax(x * x))). This model is [Machian](https://en.wikipedia.org/wiki/Mach%27s_principle). The particles are 8 dimensional, and for visualization [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is used to reduce the dimensions down to 2. [Dark energy](https://en.wikipedia.org/wiki/Dark_energy) like behavior is observed. Video of the simulation can be found [here](https://youtu.be/XybMZlyqPdU).

## Citations
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)