# Model and Method

## Baseline propagation
The baseline dynamic uses a simple linear rule:

x_{t+1} = A x_t

This is the undamped propagation benchmark. It describes how a shock moves if there is no absorption or substitution capacity.

## Damped propagation
The thesis extends the baseline with a damping matrix D. In the current implementation, damping is applied to propagated loss rather than to total output:

x_{t+1} = baseline_{t+1} - A(I - D) loss_t

where loss_t = baseline_t - x_t.

This is important because it makes the comparison between baseline and damped outcomes economically meaningful. The damped system reduces the transmitted shock, not the production base itself.

## Damping interpretation
D is built from two ingredients:
- structural substitutability, captured by upstream vulnerability V_j
- institutional absorption capacity, proxied by normalized logistics performance phi_j

The combined damping coefficient is:

d_j = (1 - V_j) * phi_j

## Relative loss metric
Relative loss is reported as:

RL = (baseline - shocked) / baseline

This metric is used to compare shock intensity across nodes and scenarios.

## Why this method is defendable
The method is transparent, traceable, and scenario-specific. Every result can be tied back to a network structure, a shock choice, and a damping assumption.
