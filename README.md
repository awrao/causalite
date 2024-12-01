# causalite
## Overview
**Causalite** is a Python package for 
defining and interacting with causal models, with a particular emphasis on 
**Structural Causal Models** (SCMs). It aims to be an accessible tool that enables users to develop their understanding of
causal models and test their own ideas.

Functionality is provided for:
* Defining an SCM by specifying causal mechanisms for each variable/node in the
corresponding **Directed Acyclic Graph**
* Sampling from the SCM
* Simulating **Randomised Controlled Trials** (RCT) by intervening on a 'treatment' variable
in the SCM
* Applying the **do-operator** to the SCM to generate interventional samples
* Computing deterministic **Counterfactuals** given an SCM and a set of observed data

### Disclaimer
Versions 0.y.z of **causalite** may include breaking changes to the API. Changes to each release are [documented](https://github.com/awrao/causalite/blob/main/CHANGELOG.md).

## Installation
We recommend you install **causalite** using pip. We also recommend that you install it into a virtual environment to avoid
potential conflicts with other packages. Once you have created and activated the virtual environment, you can install **causalite**
as follows:

    $ pip install causalite

## Getting Started
Below we demonstrate how to use **causalite** to create and sample from a Structural Causal Model.
We firstly do some imports:
```python
>>> from causalite import causal_models as cm
>>> from causalite import node_models as nm
>>> import pandas as pd
```

We define an SCM consisting of 3 variables/nodes A, X and Y by specifying
the models representing the causal mechanisms for each node. (We use the words
'variable' and 'node' interchangeably.) 
```python
>>> model = cm.StructuralCausalModel(node_models=[
        nm.NodeAdditiveNoiseModel('A'),
        nm.NodeAdditiveNoiseModel('X', parent_polys={'A': [1., 0., -3.]}),
        nm.NodeAdditiveNoiseModel('Y', parent_polys={'X': [-0.5, 0., 1.2], 'A': [1.4], 'XA': [3.]})
    ])
```

This results in the following SCM:
```python
>>> print(model)
```
    Structural Causal Model
    =======================

    A <-  U_A

    X <-  1.0A - 3.0A^3 + U_X

    Y <-  - 0.5X + 1.2X^3 + 1.4A + 3.0XA + U_Y

Here, U_A, U_X and U_Y represent the exogenous noise for variables A, X and Y respectively. We can draw a 
sample from the SCM and store it in a pandas dataframe as follows:
```python
>>> samples = model.draw_sample(size=50000)
>>> samples.head()
```    


|    |        A |          X |             Y |
|:---|---------:|-----------:|--------------:|
|0 | 1.764052 | -13.080164 |  -2746.102137 |
|1 | 0.400157 |  -0.403826 |      0.142060 |
|2 | 0.978738 |  -2.362115 |    -22.336143 |
|3 | 2.240893 | -32.590699 | -41737.620108 |
|4 | 1.867558 | -16.807889 |  -5782.921438 | 

For more detail and demonstrations of functionality such as simulation of interventions and counterfactual computation, please see
this [notebook](https://github.com/awrao/causalite/blob/main/examples/GettingStarted.ipynb).

## Documentation
Further illustrations of usage will be added to [examples](https://github.com/awrao/causalite/tree/main/examples) in due course.

## Contributing
Please read [this](https://github.com/awrao/causalite/blob/main/CONTRIBUTING.md) if you wish to help or contribute to **causalite** in any way whatsoever.

## License
This project is licensed according to the [Apache 2.0 software license](https://github.com/awrao/causalite/blob/main/LICENSE).
