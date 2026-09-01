# Priority-Aware Mean-Field Edge Computing Simulator

## Overview

This project is a simulation framework for evaluating priority-aware task allocation and load balancing in heterogeneous edge computing environments.

The simulator compares two resource allocation strategies:

* **Least-Loaded Baseline Policy**
* **Priority-Aware Mean-Field Policy**

The objective is to study how different allocation strategies perform under varying workload conditions, congestion levels, and utility configurations.

The framework evaluates multiple system-level performance metrics, including utility, response time, throughput, success ratio, resource utilization, load variance, fairness, queue length, and priority-aware task success.

The project also includes equilibrium diagnostics, reproducibility checks, repeated experiments, paired statistical comparisons, and utility sensitivity analysis.

---

# Research Motivation

Edge computing systems must allocate computational tasks across distributed and heterogeneous edge nodes.

A simple allocation strategy may assign a task to the currently least-loaded node. Although computationally simple, such an approach does not explicitly model the aggregate behavior of a large population of competing tasks and users.

This project investigates a priority-aware Mean-Field approach in which allocation decisions are influenced by the aggregate state of the system.

The simulator is designed to answer questions such as:

* How does a Mean-Field allocation policy compare with a traditional least-loaded strategy?
* Does the policy improve system utility?
* How does performance change under moderate and high congestion?
* What trade-offs occur between resource utilization, latency, fairness, queueing, and load balancing?
* Does the equilibrium solver converge to a stable solution?
* Are the experimental results reproducible?
* Are observed performance differences statistically significant?

---

# Implemented Features

## Edge Computing Simulation

The simulation environment includes:

* IoT task generation
* Multiple task priority classes
* Heterogeneous edge nodes
* Resource-aware node feasibility filtering
* CPU utilization tracking
* Memory utilization tracking
* Bandwidth utilization tracking
* Queue modeling
* Energy-related state modeling
* Task allocation and execution simulation
* Congestion scenarios
* Performance metric collection

---

# Allocation Policies

## 1. Least-Loaded Baseline Policy

The baseline policy selects the feasible edge node with the lowest current load.

The selection primarily considers:

* Node load ratio
* Queue length

This policy represents a conventional load-balancing approach.

### Selection Principle

The baseline selects:

```text
Feasible Node with Minimum Load
```

Queue length is used as an additional tie-breaking or diagnostic factor.

---

## 2. Priority-Aware Mean-Field Policy

The proposed policy uses a Mean-Field equilibrium to guide task allocation.

Instead of modeling detailed pairwise interactions between every individual participant, the Mean-Field approach represents the aggregate behavior of the system as a population distribution.

Allocation decisions consider system state information such as:

* Resource utilization
* Queue pressure
* Energy pressure
* Task priority
* Aggregate Mean-Field state
* Equilibrium control values

The objective is to model large-scale collective system behavior while maintaining computational scalability.

---

# Mean-Field Equilibrium

The Mean-Field component solves for a stable equilibrium between:

* Population state distribution
* Control policy

The equilibrium process iteratively updates the system until convergence criteria are satisfied.

The simulator records:

* Convergence status
* Number of iterations
* Distribution residual
* Policy residual
* Raw residual values
* Priority-specific solver iterations
* Priority-specific residuals
* Priority-specific control statistics
* Equilibrium construction time

A successful experiment reports:

```text
converged = True
```

Small residual values indicate that the iterative equilibrium solution has stabilized.

---

# Priority-Specific Controls

The Mean-Field model supports multiple priority populations.

Each priority class can produce different equilibrium control behavior depending on the utility profile and aggregate system state.

The diagnostics include:

* Minimum control value
* Maximum control value
* Mean control value
* Standard deviation
* Control values at representative state points
* Low saturation ratio
* High saturation ratio

This allows the experiment to verify that the policy responds differently across priority classes.

---

# Utility Profiles

The simulator evaluates multiple utility configurations.

## Balanced

The balanced profile assigns relatively balanced importance to the system objectives.

It provides a general-purpose configuration for evaluating overall system behavior.

---

## Priority Latency

This profile gives greater importance to:

* Task priority
* Response time

It is intended for workloads where delay-sensitive or important tasks require stronger consideration.

---

## Priority Latency Queue

This profile considers:

* Task priority
* Response time
* Queue conditions

It introduces queue-related penalties into the allocation objective.

---

## Priority Latency Queue Energy

This profile considers:

* Task priority
* Response time
* Queue conditions
* Energy-related pressure

This configuration represents a more complex multi-objective optimization problem.

---

# Experimental Scenarios

Each utility profile is evaluated under multiple workload conditions.

## Default

Represents normal operating conditions.

## Moderate Congestion

Represents increased workload and resource contention.

## High Congestion

Represents heavy workload and significant competition for limited edge resources.

Testing multiple congestion levels allows the simulator to evaluate policy behavior under different operating conditions.

---

# Performance Metrics

The framework evaluates the following metrics.

## Utility Mean

Measures the average utility achieved by the allocation policy.

Higher values generally indicate better performance according to the selected utility configuration.

---

## Response Time Mean

Measures the average task response time.

Lower response time is generally preferred.

---

## Throughput

Measures the amount of successfully processed workload over the experiment.

Higher throughput is generally preferred.

---

## Success Ratio

Measures the proportion of tasks successfully processed.

Higher values are preferred.

---

## Rejected Tasks

Measures the number of tasks that could not be allocated or processed.

Lower values are preferred.

---

## Resource Utilization

Measures how effectively the available edge resources are used.

Higher utilization may indicate more efficient use of available resources, although excessive utilization can also contribute to congestion.

---

## Load Variance

Measures how unevenly workload is distributed across edge nodes.

Lower load variance generally indicates better load balancing.

---

## Jain's Fairness Index

Measures how fairly workload or resources are distributed.

Values closer to:

```text
1.0
```

indicate greater fairness.

---

## Average Queue Length

Measures the average amount of queued workload.

Lower queue length generally indicates reduced waiting and congestion.

---

## Priority Success Ratio

Measures the success behavior of tasks with respect to their priority classes.

This metric helps evaluate whether the allocation mechanism behaves consistently with priority-aware objectives.

---

# Experimental Design

The simulator compares:

```text
Least-Loaded Baseline Policy
            VS
Priority-Aware Mean-Field Policy
```

The experiments are repeated multiple times for each combination of:

* Utility profile
* Congestion scenario
* Allocation policy

The current sensitivity experiments use:

```text
n = 10 repeated runs
```

for each policy comparison.

The results are summarized using:

* Mean
* Standard deviation
* Median
* Minimum
* Maximum
* 95% confidence interval

---

# Statistical Comparison

The project performs paired comparisons between the baseline and Mean-Field policies.

The comparison output includes:

```text
baseline_mean
comparison_mean
mean_difference
relative_change_percent
std_difference
ci95_low
ci95_high
p_value
```

The significance threshold used in the experiments is:

```text
p < 0.05
```

A statistically significant result indicates that the observed difference is unlikely to be explained only by experimental variation under the selected statistical test.

A positive numerical difference without statistical significance is reported as a possible observed improvement, but it is not treated as conclusive evidence.

---

# Current Experimental Findings

The current experiments show that the Mean-Field policy does not universally outperform the least-loaded baseline across every metric and configuration.

Instead, the results reveal measurable trade-offs.

## Resource Utilization

The Mean-Field policy produced statistically significant improvements in resource utilization under congestion.

Examples include:

```text
Balanced / High Congestion

Baseline:   0.8263
Mean-Field: 0.8367
Change:     +1.27%
P-value:    0.0113
```

```text
Balanced / Moderate Congestion

Baseline:   0.7412
Mean-Field: 0.7507
Change:     +1.28%
P-value:    0.0017
```

These results indicate that the Mean-Field policy can use available resources more effectively under congested conditions.

---

## Load Variance Trade-Off

Under moderate congestion, some experiments showed statistically significant increases in load variance.

Example:

```text
Baseline:   0.0256
Mean-Field: 0.0282
Change:     +10.12%
P-value:    0.0285
```

Since lower load variance generally represents more balanced workload distribution, this result demonstrates a trade-off:

```text
Higher Resource Utilization
        VS
More Uneven Load Distribution
```

---

## Utility Trade-Off

For the Priority-Latency-Queue-Energy profile under default conditions:

```text
Baseline:   0.3540
Mean-Field: 0.3388
Change:     -4.28%
P-value:    0.0038
```

The Mean-Field policy performed significantly worse in overall utility for this configuration.

This demonstrates that adding multiple objectives can create competing optimization pressures.

The proposed policy should therefore not be interpreted as universally superior.

---

## Response Time Trade-Off

For the Priority-Latency-Queue-Energy profile under default conditions:

```text
Baseline:   2.9314
Mean-Field: 3.0103
Change:     +2.69%
P-value:    0.0047
```

The Mean-Field policy produced a statistically significant increase in response time for this configuration.

---

## Queue Length Trade-Off

For the Priority-Latency-Queue-Energy profile under default conditions:

```text
Baseline:   0.5837
Mean-Field: 0.5996
Change:     +2.73%
P-value:    0.0031
```

The Mean-Field policy produced a statistically significant increase in average queue length.

---

## Fairness Trade-Off

For the Priority-Latency-Queue-Energy profile under moderate congestion:

```text
Baseline Fairness:   0.9495
Mean-Field Fairness: 0.9458
Change:              -0.39%
P-value:             0.0485
```

The Mean-Field policy showed a small but statistically significant reduction in fairness.

---

# Interpretation of Results

The experiments demonstrate that there is no single policy that dominates every performance metric.

The Mean-Field policy can improve:

* Resource utilization
* Aggregate system adaptation under congestion

However, depending on the utility profile, it may also introduce trade-offs involving:

* Utility
* Response time
* Queue length
* Load variance
* Fairness

Therefore, the primary contribution of the framework is not to claim that Mean-Field allocation is always better.

Instead, the framework provides a reproducible environment for studying multi-objective trade-offs in priority-aware edge computing systems.

---

# Reproducibility

The project includes deterministic reproducibility testing.

Two experiment runs are captured:

```text
run1.txt
run2.txt
```

Execution-time values are normalized because runtime measurements can vary due to:

* Operating system scheduling
* Background processes
* CPU activity
* Other environmental factors

The normalized outputs can be compared using PowerShell.

```powershell
Compare-Object `
    (Get-Content run1_normalized.txt) `
    (Get-Content run2_normalized.txt)
```

If the command produces no output, the normalized files contain no differences.

This demonstrates that the deterministic experiment results are reproducible under the same configuration.

---

# Project Structure

The project follows a modular Python structure.

```text
edge_game_simulator/
│
├── src/
│   └── edge_game/
│       ├── algorithms/
│       │   ├── mean_field.py
│       │   ├── policy.py
│       │   └── state.py
│       │
│       ├── entities/
│       │   ├── edge_node.py
│       │   └── task.py
│       │
│       ├── metrics/
│       │   └── collector.py
│       │
│       ├── models/
│       │   └── mean_field_model.py
│       │
│       ├── config.py
│       ├── environment.py
│       └── main.py
│
├── tests/
│   ├── test_experiments.py
│   ├── test_foundation.py
│   └── test_visualization.py
│
├── outputs/
│   └── utility_sensitivity/
│       ├── utility_sensitivity_summary.csv
│       ├── utility_sensitivity_paired_comparison.csv
│       └── equilibrium_diagnostics.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

The exact structure may evolve as new experiments and research modules are added.

---

# Installation

Clone or download the repository.

Create a Python virtual environment:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

---

# Running Experiments

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Run the project using the configured experiment entry point.

Example:

```powershell
python -m edge_game.main
```

The exact experiment commands may vary depending on the selected simulation configuration.

---

# Viewing Utility Sensitivity Results

The experiment summary can be viewed using:

```powershell
Import-Csv outputs\utility_sensitivity\utility_sensitivity_summary.csv |
    Format-Table -AutoSize
```

---

# Viewing Policy Comparisons

The paired policy comparison can be displayed using:

```powershell
Import-Csv outputs\utility_sensitivity\utility_sensitivity_paired_comparison.csv |
    Select-Object profile, scenario, metric,
        @{Name="Baseline";Expression={[math]::Round([double]$_.baseline_mean,4)}},
        @{Name="MeanField";Expression={[math]::Round([double]$_.comparison_mean,4)}},
        @{Name="Difference";Expression={[math]::Round([double]$_.mean_difference,4)}},
        @{Name="ChangePercent";Expression={[math]::Round([double]$_.relative_change_percent,2)}},
        @{Name="PValue";Expression={[math]::Round([double]$_.p_value,4)}} |
    Format-Table -AutoSize -Wrap
```

---

# Viewing Utility Results Only

To compare overall utility:

```powershell
Import-Csv outputs\utility_sensitivity\utility_sensitivity_paired_comparison.csv |
    Where-Object {$_.metric -eq "utility_mean"} |
    Select-Object profile, scenario,
        @{Name="Baseline";Expression={[math]::Round([double]$_.baseline_mean,4)}},
        @{Name="MeanField";Expression={[math]::Round([double]$_.comparison_mean,4)}},
        @{Name="Difference";Expression={[math]::Round([double]$_.mean_difference,4)}},
        @{Name="ChangePercent";Expression={[math]::Round([double]$_.relative_change_percent,2)}},
        @{Name="PValue";Expression={[math]::Round([double]$_.p_value,4)}} |
    Format-List
```

---

# Viewing Statistically Significant Results

To display only statistically significant results:

```powershell
Import-Csv outputs\utility_sensitivity\utility_sensitivity_paired_comparison.csv |
    Where-Object {[double]$_.p_value -lt 0.05} |
    Select-Object profile, scenario, metric,
        @{Name="Baseline";Expression={[math]::Round([double]$_.baseline_mean,4)}},
        @{Name="MeanField";Expression={[math]::Round([double]$_.comparison_mean,4)}},
        @{Name="Difference";Expression={[math]::Round([double]$_.mean_difference,4)}},
        @{Name="ChangePercent";Expression={[math]::Round([double]$_.relative_change_percent,2)}},
        @{Name="PValue";Expression={[math]::Round([double]$_.p_value,4)}} |
    Format-List
```

---

# Viewing Equilibrium Diagnostics

The equilibrium diagnostics are stored in:

```text
outputs/utility_sensitivity/equilibrium_diagnostics.csv
```

Example PowerShell command:

```powershell
Import-Csv outputs\utility_sensitivity\equilibrium_diagnostics.csv |
    Select-Object -First 1 |
    Format-List *
```

Important fields include:

```text
converged
iterations
distribution_residual
policy_residual
priority-specific control statistics
equilibrium_build_seconds
```

---

# Testing

The project contains automated tests.

The test suite includes:

```text
tests/test_foundation.py
tests/test_experiments.py
tests/test_visualization.py
```

Run the tests using:

```powershell
pytest
```

---

# Key Contributions

The current implementation provides:

1. A heterogeneous edge computing simulation environment.
2. A conventional least-loaded baseline allocation policy.
3. A priority-aware Mean-Field allocation policy.
4. Equilibrium-based control and population distribution analysis.
5. Multiple utility profiles.
6. Multiple congestion scenarios.
7. Comprehensive system performance metrics.
8. Repeated experimental evaluation.
9. Paired policy comparison.
10. Statistical significance analysis.
11. Equilibrium convergence diagnostics.
12. Deterministic reproducibility checks.
13. A framework for studying multi-objective trade-offs in edge computing.

---

# Limitations

The current results should be interpreted as simulation results under the implemented configuration.

The framework does not claim universal superiority of the Mean-Field policy.

Performance depends on factors such as:

* Utility weights
* Congestion level
* Resource heterogeneity
* Task characteristics
* Priority distribution
* Queue behavior
* Energy-related parameters

Future experiments may investigate additional configurations and larger workloads.

---

# Future Work

Possible extensions include:

* Larger-scale edge node populations
* Additional priority classes
* Dynamic workload arrival patterns
* Adaptive utility weights
* Additional congestion models
* More extensive sensitivity analysis
* Larger experimental sample sizes
* Additional baseline algorithms
* Reinforcement learning comparisons
* Hierarchical multi-server coordination
* Stackelberg-game-based resource coordination
* Inter-server workload migration
* Weighted Nash Bargaining
* Real-world workload traces
* Energy consumption models with physical measurements

---

# Conclusion

This project provides a reproducible simulation framework for comparing conventional least-loaded task allocation with a priority-aware Mean-Field approach in heterogeneous edge computing environments.

The experiments demonstrate that the Mean-Field policy can improve resource utilization under congestion while also introducing measurable trade-offs in other metrics depending on the selected utility profile.

Rather than assuming that one policy is universally optimal, the framework enables systematic analysis of the relationships between:

```text
Utility
Latency
Queueing
Resource Utilization
Load Balancing
Fairness
Task Priority
Congestion
```

The current results therefore support the use of the simulator as a research and experimental platform for studying multi-objective resource allocation in edge computing systems.

---

## Version

Current experimental version:

```text
0.3.0
```

The project is under active development, and the architecture, experiments, and evaluation framework may evolve as additional research components are implemented.
