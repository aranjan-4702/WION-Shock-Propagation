"""
src/dynamics/compute_phi.py
============================
Computes phi_j: Institutional Absorption Capacity for each sector j.

Economic meaning
----------------
phi_j measures how capable sector j's country is of executing
a supplier switch when an alternative supplier exists in the network.

Even if topological substitutes exist (captured by 1 - V_j),
a sector can only actually switch suppliers if its institutional
environment supports rapid onboarding of new trade relationships.
LPI captures exactly this — customs efficiency, infrastructure
quality, tracking ability, timeliness.

Formula
-------
phi_j = LPI_{c(j)}  (normalized to [0,1])

where:
    c(j)        = country of sector j (e.g. 'USA' from 'USA_C19')
    LPI_{c(j)}  = World Bank Logistics Performance Index score
                  for country c(j) in 2018
    normalized  = (LPI - LPI_min) / (LPI_max - LPI_min)

Why country-level (not sector-level)?
--------------------------------------
LPI is inherently a country-level measure — it captures the
quality of a country's trade infrastructure, customs procedures,
and logistics networks. These apply uniformly to all sectors
within a country when executing cross-border supplier switches.

A more granular sector-level institutional measure does not
exist in the literature at this scale (81 countries × 55 sectors).
This is a known limitation acknowledged in the thesis.

Why LPI specifically?
----------------------
LPI is the World Bank's operationalization of trade facilitation
barriers — the same barriers that determine how quickly a firm
can onboard a new international supplier after a shock.
It is the standard proxy for institutional trade capacity
in the supply chain resilience literature.

Reference: Arvis et al. (2016), World Bank LPI methodology.
Anderson & van Wincoop (2003) gravity model establishes that
trade costs (of which logistics is a key component) determine
bilateral trade flow feasibility.

Paper note
----------
"phi_j is the normalized LPI score of sector j's country,
capturing the institutional capacity to execute cross-border
supplier substitution. Sectors in high-LPI countries can
onboard alternative suppliers faster, absorbing a larger
fraction of incoming supply shocks."

Relationship to V_j
--------------------
V_j  = topological factor    = how substitutable are j's suppliers structurally?
phi_j = institutional factor = can j's country operationalize that substitution?

d_j = (1 - V_j) * phi_j
    = structural resilience × institutional capacity
    = total absorption capacity of sector j
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Country name → ISO3 mapping
# Source: World Bank LPI 2018 country names matched to ICIO ISO3 codes
# ─────────────────────────────────────────────────────────────────────────────

LPI_NAME_TO_ISO3 = {
    'Germany'                  : 'DEU',
    'Sweden'                   : 'SWE',
    'Belgium'                  : 'BEL',
    'Austria'                  : 'AUT',
    'Japan'                    : 'JPN',
    'Netherlands'              : 'NLD',
    'Singapore'                : 'SGP',
    'Denmark'                  : 'DNK',
    'United Kingdom'           : 'GBR',
    'Finland'                  : 'FIN',
    'United Arab Emirates'     : 'ARE',
    'Hong Kong SAR, China'     : 'HKG',
    'Switzerland'              : 'CHE',
    'United States'            : 'USA',
    'New Zealand'              : 'NZL',
    'France'                   : 'FRA',
    'Spain'                    : 'ESP',
    'Australia'                : 'AUS',
    'Italy'                    : 'ITA',
    'Canada'                   : 'CAN',
    'Norway'                   : 'NOR',
    'Czech Republic'           : 'CZE',
    'Portugal'                 : 'PRT',
    'Luxembourg'               : 'LUX',
    'Korea, Rep.'              : 'KOR',
    'China'                    : 'CHN',
    'Taiwan, China'            : 'TWN',
    'Poland'                   : 'POL',
    'Ireland'                  : 'IRL',
    'Hungary'                  : 'HUN',
    'Thailand'                 : 'THA',
    'South Africa'             : 'ZAF',
    'Chile'                    : 'CHL',
    'Slovenia'                 : 'SVN',
    'Estonia'                  : 'EST',
    'Israel'                   : 'ISR',
    'Vietnam'                  : 'VNM',
    'Iceland'                  : 'ISL',
    'Malaysia'                 : 'MYS',
    'Greece'                   : 'GRC',
    'India'                    : 'IND',
    'Cyprus'                   : 'CYP',
    'Indonesia'                : 'IDN',
    'Turkey'                   : 'TUR',
    'Romania'                  : 'ROU',
    'Croatia'                  : 'HRV',
    "Côte d'Ivoire"            : 'CIV',
    'Mexico'                   : 'MEX',
    'Bulgaria'                 : 'BGR',
    'Slovak Republic'          : 'SVK',
    'Lithuania'                : 'LTU',
    'Saudi Arabia'             : 'SAU',
    'Brazil'                   : 'BRA',
    'Colombia'                 : 'COL',
    'Philippines'              : 'PHL',
    'Argentina'                : 'ARG',
    'Ecuador'                  : 'ECU',
    'Iran, Islamic Rep.'       : 'IRN',
    'Ukraine'                  : 'UKR',
    'Egypt, Arab Rep.'         : 'EGY',
    'Malta'                    : 'MLT',
    'Latvia'                   : 'LVA',
    'Kazakhstan'               : 'KAZ',
    'Costa Rica'               : 'CRI',
    'Russian Federation'       : 'RUS',
    'Peru'                     : 'PER',
    'Jordan'                   : 'JOR',
    'Morocco'                  : 'MAR',
    'Nigeria'                  : 'NGA',
    'Cambodia'                 : 'KHM',
    'Bangladesh'               : 'BGD',
    'Tunisia'                  : 'TUN',
    'Pakistan'                 : 'PAK',
    'Belarus'                  : 'BLR',
    'Myanmar'                  : 'MMR',
    'Senegal'                  : 'SEN',
    'Angola'                   : 'AGO',
    'Congo, Dem. Rep.'         : 'COD',
    'Cameroon'                 : 'CMR',
    'Lao PDR'                  : 'LAO',
    'São Tomé and Principe'    : 'STP',
    'Brunei Darussalam'        : 'BRN',
}

# ICIO countries not in LPI — will be imputed with global mean
# ROW = Rest of World aggregate — no LPI equivalent
LPI_MISSING = ['ROW']


# ─────────────────────────────────────────────────────────────────────────────
# Block 1: Load and parse LPI data
# ─────────────────────────────────────────────────────────────────────────────

def load_lpi_scores(
    lpi_path : str,
    year     : int = 2018,
) -> pd.Series:
    """
    Load raw LPI Excel file and return scores indexed by ISO3 country code.

    File structure (World Bank LPI):
        Sheet name : year as string (e.g. '2018')
        Header row : row index 2 (header=2)
        Columns    : 'Country', 'score', and others

    Parameters
    ----------
    lpi_path : str   path to LPI Excel file
    year     : int   LPI year to load

    Returns
    -------
    lpi_iso3 : pd.Series
        LPI scores indexed by ISO3 code.
        Only covers countries successfully mapped from LPI names.
    """
    # Read Excel — header on row 2
    df_raw = pd.read_excel(lpi_path, sheet_name=str(year), header=2)

    # Keep only country name and overall LPI score
    df_raw = df_raw[['Country', 'score']].copy()
    df_raw = df_raw.dropna(subset=['Country', 'score'])

    # Map country names to ISO3 codes
    df_raw['iso3'] = df_raw['Country'].map(LPI_NAME_TO_ISO3)

    # Report unmapped countries — these are LPI countries not in ICIO
    unmapped = df_raw[df_raw['iso3'].isna()]['Country'].tolist()
    if unmapped:
        print(f"  LPI countries not mapped to ICIO ({len(unmapped)}): {unmapped}")

    # Keep only successfully mapped countries
    df_mapped = df_raw.dropna(subset=['iso3']).copy()
    df_mapped = df_mapped.set_index('iso3')['score']
    df_mapped.name = 'lpi_score'

    print(f"  LPI year        : {year}")
    print(f"  Total LPI rows  : {len(df_raw)}")
    print(f"  Mapped to ISO3  : {len(df_mapped)}")
    print(f"  Score range     : [{df_mapped.min():.3f}, {df_mapped.max():.3f}]")

    return df_mapped


# ─────────────────────────────────────────────────────────────────────────────
# Block 2: Align LPI to ICIO country list and impute missing
# ─────────────────────────────────────────────────────────────────────────────

def align_lpi_to_icio(
    lpi_scores    : pd.Series,
    icio_countries: list,
) -> pd.Series:
    """
    Align LPI scores to the ICIO country list.

    Two types of missing values:
    1. ICIO countries with no LPI data (e.g. ROW, IRN)
       → impute with global mean of available LPI scores
    2. LPI countries not in ICIO
       → already excluded in load_lpi_scores()

    Why impute with global mean?
        ROW (rest of world) is an ICIO aggregate — no single
        LPI score is appropriate. Global mean is the most
        neutral assumption, equivalent to saying ROW has
        average institutional capacity.
        This is a known limitation documented in the thesis.

    Parameters
    ----------
    lpi_scores     : pd.Series  LPI scores indexed by ISO3
    icio_countries : list       list of ISO3 codes in ICIO dataset

    Returns
    -------
    phi_aligned : pd.Series
        LPI scores for all ICIO countries.
        Missing values imputed with global mean.
    """
    # Reindex to ICIO country list
    phi_aligned = lpi_scores.reindex(icio_countries)

    # Identify missing
    missing     = phi_aligned[phi_aligned.isna()].index.tolist()
    global_mean = lpi_scores.mean()

    if missing:
        print(f"  ICIO countries missing LPI ({len(missing)}): {missing}")
        print(f"  Imputing with global mean: {global_mean:.4f}")
        phi_aligned = phi_aligned.fillna(global_mean)

    print(f"  ICIO countries covered : {len(icio_countries)}")
    print(f"  After imputation NaN   : {phi_aligned.isna().sum()}")

    return phi_aligned


# ─────────────────────────────────────────────────────────────────────────────
# Block 3: Normalize to [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def normalize_lpi(phi_raw: pd.Series) -> pd.Series:
    """
    Normalize LPI scores to [0, 1].

    Formula: phi = (LPI - LPI_min) / (LPI_max - LPI_min)

    Why normalize?
        Raw LPI scores range ~[1.95, 4.20].
        Normalization maps them to [0,1] so phi_j
        is directly interpretable as a probability-like
        weight in d_j = (1 - V_j) * phi_j.
        phi_j = 1 → perfect institutional capacity
        phi_j = 0 → zero institutional capacity

    Parameters
    ----------
    phi_raw : pd.Series  raw LPI scores (country-level)

    Returns
    -------
    phi : pd.Series  normalized to [0,1]
    """
    LPI_MAX_POSSIBLE = 5.0
    phi = phi_raw / LPI_MAX_POSSIBLE
    phi.name = 'phi_normalized'

    assert phi.isna().sum() == 0, "NaN in phi after normalization"
    assert phi.min() >= 0,        "phi < 0"
    assert phi.max() <= 1 + 1e-9, "phi > 1"

    print(f"  Normalized range : [{phi.min():.4f}, {phi.max():.4f}]")
    print(f"  Mean             : {phi.mean():.4f}")
    print(f"  Median           : {phi.median():.4f}")

    return phi


# ─────────────────────────────────────────────────────────────────────────────
# Block 4: Map country-level phi to node-level
# ─────────────────────────────────────────────────────────────────────────────

def map_phi_to_nodes(
    phi_country : pd.Series,
    nodes       : pd.Index,
) -> pd.Series:
    """
    Map country-level phi to each node (country-sector pair).

    Every sector in a country gets the same phi value.
    e.g. USA_C19, USA_M, USA_G all get phi = phi['USA']

    Why the same phi for all sectors in a country?
        LPI is a country-level measure — it captures the
        quality of a country's trade infrastructure which
        applies to all cross-border transactions regardless
        of sector. A more granular sector-level institutional
        measure does not exist at this scale.

    Parameters
    ----------
    phi_country : pd.Series  normalized phi per ISO3 country
    nodes       : pd.Index   all node labels (e.g. 'USA_C19')

    Returns
    -------
    phi_node : pd.Series (n,)
        phi value per node. Same value for all nodes in same country.
    """
    # Extract country from each node label
    # 'USA_C19' → 'USA', 'ROW_B06' → 'ROW'
    node_countries = pd.Series(
        [n.split('_')[0] for n in nodes],
        index=nodes,
        name='country'
    )

    # Map phi to each node via country
    phi_node = node_countries.map(phi_country)
    phi_node.name = 'phi_node'

    # Check for unmapped nodes
    missing_nodes = phi_node[phi_node.isna()].index.tolist()
    if missing_nodes:
        print(f"  WARNING: {len(missing_nodes)} nodes have no phi value")
        print(f"  First 5: {missing_nodes[:5]}")
        # Impute with mean as fallback
        phi_node = phi_node.fillna(phi_node.mean())

    print(f"  Nodes mapped     : {(~phi_node.isna()).sum()}")
    print(f"\n  Top 5 highest phi (best institutional capacity):")
    # Show country-level top 5
    top5 = phi_country.nlargest(5)
    for country, val in top5.items():
        print(f"    {country}: {val:.4f}")
    print(f"\n  Bottom 5 lowest phi (weakest institutional capacity):")
    bot5 = phi_country.nsmallest(5)
    for country, val in bot5.items():
        print(f"    {country}: {val:.4f}")

    return phi_node


# ─────────────────────────────────────────────────────────────────────────────
# Block 5: Master function
# ─────────────────────────────────────────────────────────────────────────────

def compute_phi(
    lpi_path : str,
    nodes    : pd.Index,
    year     : int = 2018,
) -> tuple[pd.Series, pd.Series]:
    """
    Master function: compute phi_j for all nodes.

    Pipeline:
        LPI Excel
         → load raw scores (country name → ISO3)
         → align to ICIO country list (impute missing with mean)
         → normalize to [0,1]
         → map country-level phi to each node

    Parameters
    ----------
    lpi_path : str        path to LPI Excel file
    nodes    : pd.Index   all node labels from A matrix
    year     : int        LPI year (default 2018)

    Returns
    -------
    phi_node    : pd.Series (n,)   phi per node, normalized [0,1]
    phi_country : pd.Series        phi per ISO3 country (for EDA/plotting)
    """
    print("=" * 60)
    print("Computing phi_j: Institutional Absorption Capacity")
    print("Source: World Bank Logistics Performance Index (LPI)")
    print("=" * 60)

    # Step 1: Load raw LPI scores
    print(f"\n[1/4] Loading LPI {year} from {lpi_path}...")
    lpi_scores = load_lpi_scores(lpi_path, year)

    # Step 2: Get unique ICIO countries from node labels
    print(f"\n[2/4] Extracting ICIO country list from nodes...")
    icio_countries = sorted(set(n.split('_')[0] for n in nodes))
    print(f"  ICIO countries : {len(icio_countries)}")

    # Step 3: Align to ICIO, impute missing
    print(f"\n[3/4] Aligning LPI to ICIO country list...")
    phi_raw     = align_lpi_to_icio(lpi_scores, icio_countries)

    # Step 4: Normalize to [0,1]
    print(f"\n[4/4] Normalizing LPI to [0,1]...")
    phi_country = normalize_lpi(phi_raw)

    # Step 5: Map to nodes
    print(f"\n[5/5] Mapping country-level phi to {len(nodes)} nodes...")
    phi_node    = map_phi_to_nodes(phi_country, nodes)

    print("\n" + "=" * 60)
    print("phi_j computation complete.")
    print("=" * 60)

    return phi_node, phi_country