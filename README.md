


## Setup

Conda

```
conda env create -f environment/environment.yml
```

Docker

```
docker pull dibyaghosh/gcsl:0.1
```


## Example script

```
python experiments/gcsl_example.py
```

If you have, and would like to use, a GPU, you will need to additionally install a GPU-compiled version of PyTorch. To do so, simply run

```
pip uninstall torch && pip install torch==1.1.0
```

## Development Notes

The directory structure currently looks like this:

- gcsl (Contains all code)
    - envs (Contains all environment files and wrappers)
    - algo (Contains all GCSL code)
        - gcsl.py (implements high-level algorithm logic, e.g. data collection, policy update, evaluate, save data)
        - buffer.py (The replay buffer used to *relabel* and *sample* (s,g,a,h) tuples
        - networks.py (Implements neural network policies.)
        - variants.py (Contains relevant hyperparameters for GCSL)

- experiments (Contains all launcher files)
- doodad (We require this old version of doodad)
- dependencies (Contains other libraries like rlkit, rlutil, room_world, multiworld, etc.)
- data (Not synced by github, but this will contain all experiment logs)

Please file an issue if you have trouble running this code.


