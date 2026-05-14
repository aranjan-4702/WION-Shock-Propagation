import numpy as np
import pandas as pd
from pathlib import Path


def compute_herfindahl(input_dist: pd.DataFrame) -> pd.Series:
    """
    Compute Herfindahl concentration index for each sector j.

    H_j = sum_i (input_dist[i,j])^2

    Where input_dist[i,j] = share of j's intermediate inputs from i.
    Each column of input_dist already sums to 1.

    Parameters
    ----------
    input_dist : pd.DataFrame (n x n)
        Normalized input distribution matrix.
        Column j = probability distribution over suppliers for sector j.

    Returns
    -------
    H : pd.Series (n,)
        Herfindahl index per sector. Range [1/n, 1].
        High H_j = concentrated = fragile.
        Low H_j  = diversified  = resilient.
    """
    H = (input_dist ** 2).sum(axis=0)
    H.name = "herfindahl"
    return H



# Mapping from World Bank LPI country names to ICIO ISO3 codes
LPI_NAME_TO_ISO3 = {
    'Germany': 'DEU',
    'Sweden': 'SWE',
    'Belgium': 'BEL',
    'Austria': 'AUT',
    'Japan': 'JPN',
    'Netherlands': 'NLD',
    'Singapore': 'SGP',
    'Denmark': 'DNK',
    'United Kingdom': 'GBR',
    'Finland': 'FIN',
    'United Arab Emirates': 'ARE',
    'Hong Kong SAR, China': 'HKG',
    'Switzerland': 'CHE',
    'United States': 'USA',
    'New Zealand': 'NZL',
    'France': 'FRA',
    'Spain': 'ESP',
    'Australia': 'AUS',
    'Italy': 'ITA',
    'Canada': 'CAN',
    'Norway': 'NOR',
    'Czech Republic': 'CZE',
    'Portugal': 'PRT',
    'Luxembourg': 'LUX',
    'Korea, Rep.': 'KOR',
    'China': 'CHN',
    'Taiwan, China': 'TWN',
    'Poland': 'POL',
    'Ireland': 'IRL',
    'Hungary': 'HUN',
    'Thailand': 'THA',
    'South Africa': 'ZAF',
    'Chile': 'CHL',
    'Slovenia': 'SVN',
    'Estonia': 'EST',
    'Israel': 'ISR',
    'Vietnam': 'VNM',
    'Iceland': 'ISL',
    'Malaysia': 'MYS',
    'Greece': 'GRC',
    'India': 'IND',
    'Cyprus': 'CYP',
    'Indonesia': 'IDN',
    'Turkey': 'TUR',
    'Romania': 'ROU',
    'Croatia': 'HRV',
    "Côte d'Ivoire": 'CIV',
    'Mexico': 'MEX',
    'Bulgaria': 'BGR',
    'Slovak Republic': 'SVK',
    'Lithuania': 'LTU',
    'Saudi Arabia': 'SAU',
    'Brazil': 'BRA',
    'Colombia': 'COL',
    'Philippines': 'PHL',
    'Argentina': 'ARG',
    'Ecuador': 'ECU',
    'Iran, Islamic Rep.': 'IRN',
    'Ukraine': 'UKR',
    'Egypt, Arab Rep.': 'EGY',
    'Malta': 'MLT',
    'Latvia': 'LVA',
    'Kazakhstan': 'KAZ',
    'Costa Rica': 'CRI',
    'Russian Federation': 'RUS',
    'Peru': 'PER',
    'Jordan': 'JOR',
    'Morocco': 'MAR',
    'Nigeria': 'NGA',
    'Cambodia': 'KHM',
    'Bangladesh': 'BGD',
    'Tunisia': 'TUN',
    'Pakistan': 'PAK',
    'Belarus': 'BLR',
    'Myanmar': 'MMR',
    'Senegal': 'SEN',
    'Angola': 'AGO',
    'Congo, Dem. Rep.': 'COD',
    'Cameroon': 'CMR',
    'Lao PDR': 'LAO',
    'São Tomé and Principe': 'STP',
    'Brunei Darussalam': 'BRN',
}

# ICIO countries not in LPI — assign global mean
LPI_MISSING = ['ROW']


def load_lpi(lpi_path: str,
             country_codes: list,
             year: int = 2018) -> pd.Series:
    # Read sheet with row 1 as header
    df = pd.read_excel(lpi_path,
                       sheet_name=str(year),
                       header=2)

    # Rename the country column
    #df = df.rename(columns={'Unnamed: 0': 'Country'})

    # Keep country name and overall score only
    df = df[['Country', 'score']].copy()
    df = df.dropna(subset=['Country', 'score'])

    # Map country names to ISO3
    df['iso3'] = df['Country'].map(LPI_NAME_TO_ISO3)

    # Warn about unmapped LPI countries
    unmapped = df[df['iso3'].isna()]['Country'].tolist()
    if unmapped:
        print(f"NOTE: {len(unmapped)} LPI countries "
              f"not mapped to ICIO — ignored.")

    # Keep only mapped countries
    df = df.dropna(subset=['iso3'])
    df = df.set_index('iso3')['score']

    # Align to ICIO country list
    phi_raw = df.reindex(country_codes)

    # Impute missing with global mean
    missing = phi_raw[phi_raw.isna()].index.tolist()
    if missing:
        global_mean = phi_raw.mean()
        print(f"WARNING: LPI missing for: {missing}")
        print(f"Imputing with global mean: "
              f"{global_mean:.4f}")
        phi_raw = phi_raw.fillna(global_mean)

    # Normalize to [0,1]
    phi = (phi_raw - phi_raw.min()) / \
          (phi_raw.max() - phi_raw.min())
    phi.name = "phi_lpi"

    return phi


def compute_damping(H: pd.Series,
                    phi: pd.Series,
                    node_labels: list) -> pd.Series:
    """
    Compute sector-level damping coefficients.

    d_j = (1 - H_j) * phi_c(j)

    Where:
        H_j     = Herfindahl concentration of sector j's inputs
        phi_c   = normalized LPI score of sector j's country
        c(j)    = country extracted from node label e.g. 'USA_C20' -> 'USA'

    Parameters
    ----------
    H : pd.Series (n,)
        Herfindahl index per sector. Indexed by node label.
    phi : pd.Series (81,)
        Normalized LPI score per country. Indexed by ISO3 code.
    node_labels : list
        List of node labels e.g. ['USA_C20', 'CHN_C26', ...]

    Returns
    -------
    d : pd.Series (n,)
        Damping coefficient per sector.
        Range [0, 1].
        High d_j = high absorption capacity.
        Low d_j  = low absorption capacity = fragile.
    """
    # Extract country code from node label
    # e.g. 'USA_C20' -> 'USA', 'ROW_B06' -> 'ROW'
    countries = pd.Series(
        [label.split('_')[0] for label in node_labels],
        index=node_labels,
        name='country'
    )

    # Map phi to each sector via country code
    phi_mapped = countries.map(phi)

    # Warn if any sectors have unmapped phi
    missing = phi_mapped[phi_mapped.isna()].index.tolist()
    if missing:
        print(f"WARNING: phi unmapped for "
              f"{len(missing)} sectors.")
        print(f"First 5: {missing[:5]}")
        phi_mapped = phi_mapped.fillna(phi_mapped.mean())

    # Compute damping
    d = (1 - H) * phi_mapped
    d.name = "damping"

    # Validate
    assert d.min() >= 0, "Negative damping detected"
    assert d.max() <= 1, "Damping exceeds 1"
    assert d.isna().sum() == 0, "NaN in damping"

    print(f"Damping coefficients computed for "
          f"{len(d)} sectors.")
    print(f"  Mean   : {d.mean():.4f}")
    print(f"  Median : {d.median():.4f}")
    print(f"  Min    : {d.min():.4f}")
    print(f"  Max    : {d.max():.4f}")

    return d


def build_damping_matrix(d: pd.Series) -> np.ndarray:
    """
    Build diagonal damping matrix D from sector-level
    damping coefficients.

    D = diag(d_1, d_2, ..., d_n)

    Parameters
    ----------
    d : pd.Series (n,)
        Damping coefficients per sector.

    Returns
    -------
    D : np.ndarray (n x n)
        Diagonal matrix with d_j on diagonal.
        Off-diagonal entries are zero.
    """
    D = np.diag(d.values)

    # Validate
    assert D.shape == (len(d), len(d)), \
        "D shape mismatch"
    assert np.allclose(np.diag(D), d.values), \
        "Diagonal entries don't match d"

    print(f"Damping matrix D built: "
          f"{D.shape[0]} x {D.shape[1]}")

    return D