# Ising Model: Metropolis vs. Gibbs

A Python implementation of the two-dimensional Ising model using **Metropolis** and **Gibbs sampling**. The project simulates interacting spins on a square lattice and compares both Monte Carlo methods through thermodynamic observables.

## Overview

Each spin takes one of two values, $S_{i,j} \in {-1,+1}$. The system is described by the Hamiltonian

$$
H = -J\sum_{\langle i,j\rangle} S_iS_j - h\sum_i S_i.
$$

The simulation uses a $60 \times 60$ lattice with periodic boundary conditions and temperatures from $T=1.0$ to $T=4.0$. For $J=k_B=1$, the exact critical temperature is

$$
T_c = \frac{2J}{k_B\ln(1+\sqrt{2})} \approx 2.269.
$$

## Features

* Two-dimensional square lattice with periodic boundary conditions
* Metropolis and Gibbs Monte Carlo algorithms
* Multiple independent simulation runs
* Equilibration before measurements
* Energy, magnetization, and heat-capacity estimates
* Comparison of both sampling methods
* Spin-configuration visualization and animation
* Numba JIT compilation for computationally intensive Monte Carlo and measurement loops

## Monte Carlo Methods

### Metropolis sampling

For each update, a random spin is selected and a flip is proposed. The local energy difference is

$$
\Delta E = 2S_{i,j}\left(J\sum_{\text{neighbors}}S+h\right).
$$

Energy-lowering flips are accepted. Otherwise, the flip is accepted with probability

$$
P = e^{-\Delta E/(k_BT)}.
$$

One sweep performs $N^2$ random spin updates.

### Gibbs sampling

Gibbs sampling resamples the selected spin directly from its local conditional distribution:

$$
P(S_{i,j}=+1) = \frac{1}{1+e^{-2H_{\mathrm{loc}}/(k_BT)}}.
$$

## Simulation Process

For each temperature and sampling method, the program:

1. Creates a random spin configuration.
2. Equilibrates the lattice.
3. Performs measurement sweeps.
4. Records energy and absolute magnetization.
5. Repeats the simulation for 3 independent runs.
6. Averages the measured quantities.

The simulation evaluates 30 temperatures between $T=1.0$ and $T=4.0$ for each Monte Carlo method.

## Observables

### Energy

The total energy uses right and down neighbors to avoid double-counting interactions. Results are reported per spin.

### Magnetization

The absolute magnetization per spin is

$$
|M| = \frac{1}{N^2}\left|\sum_i S_i\right|.
$$

### Heat Capacity

Heat capacity is estimated from energy fluctuations:

$$
C = \frac{\langle E^2\rangle - \langle E\rangle^2}{k_BT^2}.
$$

## Results

The program generates:

* Spin configurations at $T=1.0$, $T=2.0$, $T=T_c$, $T=3.0$, and $T=4.0$
* An animation of spin evolution near the critical temperature
* Temperature-dependent comparisons of energy, absolute magnetization, and heat capacity for Metropolis and Gibbs sampling

The thermodynamic observables obtained with Metropolis and Gibbs sampling are compared over the full temperature range to examine the consistency of both Monte Carlo approaches.

The temperature dependence of heat capacity is used to study the phase transition of the Ising model. The critical temperature $T_c \approx 2.269$ is used as a reference point for analyzing the behavior of the thermodynamic observables.

At low temperatures, the system generally forms ordered spin regions with higher magnetization. As the temperature increases, thermal fluctuations become stronger and the spin configuration becomes more disordered. The behavior of heat capacity and magnetization near the critical temperature provides information about the phase transition.

## Installation

```bash
pip install numpy matplotlib numba
```

## Usage

```bash
3.0.lab.py
```

The script performs the simulations and generates numerical results, plots, and an animation.

## Main Parameters

```python
N = 60
J = 1.0
h = 0.0
kB = 1.0
N_RUNS = 3
TEMPS = np.linspace(1.0, 4.0, 30)
```

Here, `N` is the lattice size, `J` is the interaction strength, `h` is the external magnetic field, `kB` is the Boltzmann constant, and `N_RUNS` is the number of independent simulations per temperature.

## Technologies

* Python
* NumPy
* Matplotlib
* Numba
* Monte Carlo methods
* Statistical mechanics




