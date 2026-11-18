# Non-intrusive modal decomposition of transport-dominated data (W_POD)

This code is the twin brother/sister/siamese twin of our paper, with the above title. The authors are R. Koprinkov, S. Dubreuil, F. Bachoc and N. Bartoli.

- The examples from the paper can be found in the `examples/` directory
- The library can be found in the `src` directory. The 1D (`W_POD`) and 2D (`2D_test`) code are separate
- You can find some unit tests in the `tests` directory (courtesy of Claude Opus 5.5)

The W in W_POD stands for "Wasserstein", because the modes were optimized with respect to the Wasserstein distance (as well as the $L_2$ distance).

To code is split into a library that lets you define your modal decomposition problem and solve it and into a few files that use the library to calculate the modal decomposition for the cases from the paper. You can find the library under `src/W_POD` (1D) or `src/src_wildfire` (2D). The cases from the paper can be found in the `examples` folder.

The library lets you:
- Store your data
- Define an ansatz
- Define how you want to measure the reconstruction error
- Solve the optimization problem

## Installation
To run the 1D examples:
1. Run `pip install -e .` in the home directory of this repo (or install it in your python env)
2. Run the case file: `python cw_exp1_formulation_B_budget.py`

To run the 2D examples:
1. Run `pip install -e .` in the home directory of this repo (or install it in your python env)
2. Copy the `src_wildfire` files next to the `formulation_B_8_4.py` file. The directory setup should look like this:
```text
.
├── formulation_B_8_4.py
└── src_wildfire/
    ├── visu.py
    ├── losses.py
    └── ansatz.py
```
3. Run the case file: `python formulation_B_8_4.py`


