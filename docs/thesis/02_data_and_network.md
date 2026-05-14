# Data and Network Construction

## Data source
The analysis uses the 2018 world input-output table and associated final demand data. Each node in the network is a country-sector pair such as `USA_C19` or `DEU_M`.

## Network construction
The raw inter-industry flow matrix is converted into a technical coefficient matrix A. Each column represents the input structure of one sector, so A captures how much production in one node depends on all supplying nodes.

## Output vector
The baseline output vector X is computed from intermediate demand plus final demand rather than copied from an external output field. This makes the model internally consistent with the constructed network.

## Why the network matters
The production system is not a flat collection of sectors. It is a directed weighted network with concentrated suppliers, hub-like countries, and strong asymmetries in connectivity. Those structural features determine how quickly shocks spread.

## Empirical network pattern
The data show a fat-tailed supplier structure and a strong asymmetry between supply-side and demand-side connectivity. That supports a network-based shock propagation framework instead of a representative-agent style model.
