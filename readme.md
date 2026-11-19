# Non-intrusive modal decomposition of transport-dominated data (`W_POD`)

This code is the twin brother/sister/siamese twin of our paper, with the above title. The authors are R. Koprinkov, S. Dubreuil, F. Bachoc and N. Bartoli.

- The examples from the paper can be found in the `examples/` directory
- The library can be found in the `src` directory. The 1D (`W_POD`) and 2D (`2D_test`) code are separate
- You can find some unit tests in the `tests` directory (courtesy of Claude Opus 5.5)

The W in `W_POD` stands for "Wasserstein", because the modes were optimized with respect to the Wasserstein distance (as well as the $L_2$ distance).

To code is split into a library that lets you define your modal decomposition problem and solve it and into a few files that use the library to calculate the modal decomposition for the cases from the paper. You can find the library under `src/W_POD` (1D) or `src/src_wildfire` (2D). The cases from the paper can be found in the `examples` folder.

The library lets you:
- Store your data
- Define an ansatz
- Define how you want to measure the reconstruction error
- Solve the optimization problem

## Installation

To faithfully reproduce the environment in which the code was run, I recommend to make separate GPU/CPU environments. You can install `W_POD` with the requirements for CPU by running
```
pip install -r requirements_cpu.txt -e .
```
at the project root. For GPU, you can run
```
pip install -r requirements_gpu.txt -e .
```
also at the project root. If you have an existing environment and are fine with using newer versions of the requirements (or do not care for reproducing the environment exactly), you can run
```
pip install -e .
```
which should install `W_POD` into the environment that `pip` belongs to.

To run the 1D examples:
1. Install the CPU/GPU environments using the above instructions.
2. Run the case file: `python cw_exp1_formulation_B_budget.py`

To run the 2D examples:
1. Install the CPU/GPU environments using the above instructions.
2. Run the case file: `python formulation_B_8_4.py [seed]`

