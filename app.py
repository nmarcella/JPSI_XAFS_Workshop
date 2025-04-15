# app.py
import streamlit as st
import plotly.graph_objects as go # Import Plotly
import plotly.colors as pcolors # Import plotly colors
from plotly.subplots import make_subplots # Import make_subplots (needed for Scattering Properties plot)
import pandas as pd # Explicitly import pandas
from larch.io import read_ascii
# Use larch.xafs for most functions as requested
from larch.xafs import pre_edge, autobk, xftf, feffpath, path2chi, ff2chi
from larch.xafs import feffit, feffit_transform, feffit_dataset, feffit_report
# Import Parameter tools separately (likely still needed from fitting or base larch)
try:
    from larch.fitting import Parameter, Parameters
except ImportError:
    from larch import Parameter, Parameters # Fallback import location

from larch import Group # Import Group for type checking if needed, or creating temp groups
import os # Import os to handle file paths and listing directories
import glob # Use glob to find feff files
import numpy as np # Needed for number inputs potentially and gradient, interp
import tempfile # Needed for caching uploaded files
import copy # Import the copy module
import itertools # For color cycling
from collections import defaultdict # For grouping shells

# --- Page Configuration ---
st.set_page_config(page_title="XAFS Demo", layout="wide")

# --- Title ---
st.title("XAFS Data Processing & Modeling Demo (Larch)")
st.write("This app demonstrates XAFS data processing, modeling, and fitting using Larch.")

# --- Initialize Session State ---
if 'path_modifications' not in st.session_state:
    st.session_state.path_modifications = {} # {path_key: [mod1, mod2,...]}
# if 'fit_result' not in st.session_state: # Obsolete - use fit_result_obj
#     st.session_state.fit_result = None # To store fit output Parameters group
if 'fit_result_obj' not in st.session_state:
    st.session_state.fit_result_obj = None # To store full fit result object if needed
if 'current_data_basename' not in st.session_state:
    st.session_state.current_data_basename = None # Track current data
# if 'fit_history' not in st.session_state: # Obsolete - use fit_result_obj
#      st.session_state.fit_history = []

# --- Explanation Expanders ---
with st.expander("Step 1: What does 'Pre-edge Subtraction' do?"):
    st.markdown("""
    The `pre_edge` function performs several standard steps in XAFS (X-ray Absorption Fine Structure) analysis:

    1.  **Find E0:** Determines the absorption edge energy (E0), often from the maximum of the derivative of the absorption data (`mu`), if not provided by the user.
    2.  **Fit Pre-edge:** Fits a line (or sometimes a low-order polynomial) to the region *before* the absorption edge (`pre1` to `pre2` relative to E0). This represents background absorption not related to the element of interest.
    3.  **Fit Post-edge:** Fits a polynomial (typically 1st to 3rd order, `nnorm`) to the region *after* the absorption edge (`norm1` to `norm2` relative to E0). This models the atomic absorption background.
    4.  **Calculate Edge Step:** Extrapolates the pre-edge line and post-edge polynomial to E0 to estimate the height of the absorption jump (`edge_step`).
    5.  **Subtract & Normalize:**
        * Subtracts the extrapolated pre-edge line from the entire `mu` spectrum.
        * Divides the result by the calculated `edge_step` (derived from the post-edge fit extrapolated to E0) to normalize the data, typically making the jump height equal to 1.
    6.  **Flatten (Optional):** Can calculate a "flattened" spectrum (`dat.flat`) by removing residual low-frequency oscillations from the normalized data using another polynomial fit.

    The function stores results like the calculated `e0`, `edge_step`, the `pre_edge` line, the `post_edge` normalization curve, and the final `norm`alized data array back into the data group (`dat` in this app).

    [More Information](https://xraypy.github.io/xraylarch/xafs_preedge.html)
    """)

with st.expander("Step 2: What does 'Autobk Background Subtraction' do?"):
    st.markdown("""
    The `autobk` function empirically determines the smooth, isolated-atom absorption background (`μ₀(k)`) to extract the EXAFS oscillations (`χ(k)`).

    Key Concepts:
    * **Goal:** Isolate the `χ(k)` signal, which contains structural information, from the overall absorption `μ(E)`.
    * **Method:** It uses the AUTOBK algorithm, fitting a flexible spline function to the data in k-space (photoelectron wavenumber, related to energy above E₀).
    * **Low-R Filtering:** The core idea is that the true EXAFS signal (`χ`) primarily comes from scattering atoms at distances > ~1 Å. Any signal appearing at very short distances (low-R) in the Fourier Transform of `χ(k)` is likely due to imperfections in the initial background subtraction or atomic effects not related to bonding. Autobk adjusts the spline to minimize these low-R components below a cutoff distance (`rbkg`).
    * **Parameters:**
        * `rbkg`: The cutoff distance (Å) in R-space. Components below this value are minimized during the spline fit. A typical value is ~1 Å.
        * `kweight`: How the `χ(k)` data is weighted during the process. Higher weights emphasize the signal at higher k-values. Common values are 1, 2, or 3.
    * **Output:** Creates `k` (wavenumber array), `chi` (EXAFS oscillations array, stored as `chi_exp` in this app), and `bkg` (the calculated μ₀ background array vs energy) within the data group (`dat`).
    """)

with st.expander("Step 3: What does 'Fourier Transform (k -> R)' do?"):
    st.markdown("""
    The `xftf` function performs a Fourier Transform (FT) on the EXAFS data `χ(k)` to convert it into `χ(R)`, which represents the signal as a function of distance (R) from the absorbing atom. This helps visualize the contributions from different shells of neighboring atoms.

    Key Concepts:
    * **Goal:** Transform the oscillatory `χ(k)` signal into R-space to identify bond distances. Peaks in `|χ(R)|` roughly correspond to atomic shells (uncorrected for phase shifts).
    * **Process:**
        1.  Applies a k-weighting (`k^kweight`) to `χ(k)` to emphasize certain parts of the signal.
        2.  Applies a window function (`window`) over a specified k-range (`kmin` to `kmax`) with tapering (`dk`, `dk2`). This smoothly brings the weighted `χ(k)` signal to zero at the edges, reducing artifacts (ringing) in the resulting `χ(R)`.
        3.  Performs a Fast Fourier Transform (FFT) on the weighted, windowed `χ(k)`.
    * **Parameters:**
        * `kmin`, `kmax`: The range in k (Å⁻¹) over which the FT is performed.
        * `kweight`: The exponent for k-weighting (often 1, 2, or 3).
        * `window`: The type of window function used (e.g., 'hanning', 'kaiser'). Hanning is common.
        * `dk`, `dk2`: Tapering parameters for the window function edges.
    * **Output:** Creates `r` (distance array in Å), `chir_mag` (magnitude of `χ(R)`), `chir_re` (real part), `chir_im` (imaginary part), and `kwin` (the window function used) in the data group (`dat`). The magnitude `chir_mag` is typically plotted.
    """)

with st.expander("Step 4: What is EXAFS Modeling with Feff Paths?"):
    st.markdown("""
    EXAFS modeling aims to reproduce the experimental `χ(k)` signal by summing theoretical signals calculated for individual photoelectron scattering paths using Feff.

    Key Concepts:
    * **EXAFS Equation:** The `χ(k)` signal is mathematically described as a sum over all possible scattering paths the photoelectron can take. Each path's contribution depends on factors like the path length (R), the number of atoms involved (degeneracy N), atomic scattering properties (amplitude F(k) and phase shift δ(k)), and disorder terms (like mean-square displacement σ²).
    * **Feff Calculation:** Feff calculates the theoretical scattering amplitude F(k) and phase shift δ(k) for specific paths within a given atomic structure. These results are stored in `feffNNNN.dat` files.
    * **FeffPath Object:** Larch reads a `feffNNNN.dat` file into a `FeffPath` object using `feffpath()`. This object contains the theoretical Feff results (`amp`, `pha`, `lam`, etc.) and adjustable parameters (`N`, `E₀`, `ΔR`, `σ²`, etc.).
    * **Calculating Path Chi:** `path2chi()` calculates the `χ(k)` contribution for a *single* FeffPath object using the EXAFS equation and the current values of its adjustable parameters.
    * **Summing Paths:** `ff2chi()` takes a *list* of FeffPath objects, calculates `χ(k)` for each (using their current parameters), and sums them to produce a total model `χ(k)`.
    * **Goal:** By selecting relevant Feff paths and potentially adjusting their parameters (N, E₀, ΔR, σ²), one can try to match the summed model `χ(k)` to the experimental `χ(k)`.

    **EXAFS Equation (Larch/Feff Implementation):**

    The `χ(k)` for a single path is calculated using the following equation (or a close variant):
    """)
    st.latex(r'''
    \chi(k) = \text{Im} \left[
        \frac{N S_0^2 f_{\text{eff}}(k)}{k (R_{\text{eff}} + \Delta R)^2}
        e^{-2p'' R_{\text{eff}}}
        e^{-2p^2 \sigma^2}
        e^{\frac{2}{3} p^4 c_4}
        e^{i \left( 2k R_{\text{eff}} + \delta(k) + 2p(\Delta R - \frac{2\sigma^2}{R_{\text{eff}}}) - \frac{4}{3} p^3 c_3 \right)}
    \right]
    ''')
    # Note: Larch often uses the Real part in practice for chi(k), but this is the formal complex equation.
    st.markdown("""
    Where:
    * `χ(k)`: EXAFS oscillation for the path vs. wavenumber `k`.
    * `Im[...]`: Imaginary part of the complex expression.
    * `k`: Photoelectron wavenumber (Å⁻¹) used for calculations. It's related to the Feff wavenumber $k_{\mathrm{feff}}$ and the edge shift $E_0$ by the equation below.
    """)
    st.latex(r'''k = \sqrt{k_{\mathrm{feff}}^2 - \frac{2m_e E_0}{\hbar^2}}''')
    st.markdown("""
    * `N`: Path degeneracy (coordination number, adjustable parameter `degen`).
    * `S₀²`: Amplitude reduction factor (accounts for many-body effects, adjustable parameter `s02`).
    * `f_eff(k)`: Effective scattering amplitude calculated by Feff (`_feffdat.amp`).
    * `R_eff`: Nominal path length calculated by Feff (`path.reff`).
    * `ΔR`: Change in path length from nominal (adjustable parameter `deltar`).
    * `p`: Complex photoelectron wavenumber (`p = p' + i p''`), includes core-hole lifetime and inelastic losses (`_feffdat.rep`, `_feffdat.lam`) and potential imaginary energy shift (`ei`).
    * `p''`: Imaginary part of `p`, related to the mean free path `λ(k)`.
    * `σ²`: Mean-square displacement in path length (Debye-Waller factor, adjustable parameter `sigma2`).
    * `c₃`: Third cumulant of path length distribution (accounts for asymmetry, adjustable parameter `third`).
    * `c₄`: Fourth cumulant of path length distribution (adjustable parameter `fourth`).
    * `δ(k)`: Total phase shift calculated by Feff (`_feffdat.pha`).
    * `Eᵢ`: Imaginary energy shift (adjustable parameter `ei`), incorporated into `p`.
    """)

with st.expander("Step 5: What is EXAFS Fitting with Feffit?"):
    st.markdown("""
    Fitting involves adjusting the parameters of the theoretical Feff paths (like bond distances, coordination numbers, disorder factors) to make the summed model `χ(k)` (or its Fourier Transform `χ(R)`) match the experimental data as closely as possible.

    Key Concepts:
    * **Goal:** Quantify structural parameters by finding the parameter values that minimize the difference between the model and the data.
    * **Process (`feffit`):**
        1.  **Define Parameters:** Create Larch `Parameters` for the variables you want to fit (e.g., `S₀²`, `E₀`, `ΔR` or `α`, `σ²`). Parameters can be fixed (`vary=False`) or allowed to vary (`vary=True`) with an initial guess. You can also define constraints between parameters.
        2.  **Assign Parameters to Paths:** Link the parameters defined in step 1 to the corresponding attributes of the FeffPath objects (e.g., `path.s02 = 's02_global'`, `path.deltar = 'alpha * reff'`).
        3.  **Define Transform & Fit Range:** Specify how the data and model should be compared (e.g., in R-space) and over what range (e.g., `rmin`, `rmax`). This uses a `FeffitTransform` object.
        4.  **Create Dataset:** Combine the experimental data (`k`, `chi_exp`), the list of FeffPath objects (with parameters assigned), and the transform settings into a `FeffitDataset`.
        5.  **Run Fit:** Call `feffit()` with the parameter group and the dataset(s). Larch uses optimization algorithms (like Levenberg-Marquardt) to find the best-fit parameter values.
    * **Output:** The fit returns the parameter group updated with best-fit values, uncertainties, correlations, and fit statistics (`chi_square`, R-factor). A `model` group containing the calculated best-fit `chi(k)` and `chi(R)` is also added to the dataset.
    * **Information Limit:** EXAFS data has a limited number of independent data points (~`2ΔkΔR/π`). The number of varying parameters in a fit should generally not exceed this limit to ensure meaningful results. Fitting in R-space helps focus the fit on specific structural features.
    """)

# --- Constants ---
DATA_DIR = 'data'
FEFF_DIR_BASE = 'feff' # Base directory for Feff files
UPLOAD_OPTION = "Upload Your Own File"
MAX_MODIFICATIONS = 5 # Max modifications to show on plot

# --- Utility Functions ---
def get_element_color(symbol):
    """Returns a color based on element symbol."""
    # Simple color mapping, extend as needed
    color_map = {
        'Fe': 'orange', 'O': 'red', 'Cu': 'brown', 'Zn': 'grey',
        'C': 'black', 'N': 'blue', 'S': 'yellow', 'P': 'purple',
        # Add more common elements
    }
    return color_map.get(symbol, 'grey') # Default to grey

# --- Cached Data Loading Function ---
@st.cache_data(show_spinner="Loading data...") # Cache the data loading
def load_data(source_identifier, source_type='path', labels='energy mu i0'):
    """Loads data from a file path or bytes content using Larch."""
    print(f"Cache miss: Loading data from {source_type}") # Debug print
    if source_type == 'path':
        if os.path.exists(source_identifier):
            return read_ascii(source_identifier, labels=labels)
        else:
            raise FileNotFoundError(f"Example file not found: {source_identifier}")
    elif source_type == 'upload':
        # For uploaded files, write bytes to a temporary file for read_ascii
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
            tmp_file.write(source_identifier) # source_identifier is bytes content here
            tmp_filename = tmp_file.name
        try:
            data = read_ascii(tmp_filename, labels=labels)
        finally:
            os.remove(tmp_filename) # Clean up the temporary file
        return data
    else:
        raise ValueError("Invalid source_type for load_data")

# --- Cached Feff Path Loading Function ---
@st.cache_data(show_spinner="Loading Feff paths...")
def load_feff_paths(feff_directory):
    """Finds and reads all feffNNNN.dat files in a directory."""
    print(f"Cache miss: Loading Feff paths from {feff_directory}") # Debug print
    paths = {}
    if os.path.isdir(feff_directory):
        feff_files = sorted(glob.glob(os.path.join(feff_directory, 'feff????.dat')))
        if not feff_files:
            st.warning(f"No 'feff????.dat' files found in {feff_directory}")
            return paths # Return empty dict
        for ffile in feff_files:
            try:
                path_obj = feffpath(ffile)
                paths[os.path.basename(ffile)] = path_obj
            except Exception as e:
                st.warning(f"Could not read Feff file {os.path.basename(ffile)}: {e}")
    else:
        st.warning(f"Feff directory not found: {feff_directory}")
    return paths

# --- Cached feff.inp Atom/Shell Parsing Function ---
@st.cache_data(show_spinner="Parsing feff.inp...")
def parse_feff_inp_atoms(feff_inp_path):
    """Parses the ATOMS section of feff.inp to get absorber and shells."""
    print(f"Cache miss: Parsing {feff_inp_path}") # Debug print
    atoms = []
    shells = defaultdict(list)
    absorber_info = None # Store full info for absorber
    in_atoms_section = False

    if not os.path.exists(feff_inp_path):
        st.warning(f"feff.inp not found at {feff_inp_path}")
        return None, None # Return None if file not found

    try:
        with open(feff_inp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('ATOMS'):
                    in_atoms_section = True
                    continue
                if line.startswith('END') and in_atoms_section:
                    break
                if in_atoms_section and line and not line.startswith('*'):
                    try:
                        # Format: x, y, z, ipot, tag, distance, *site_info
                        parts = line.split()
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        ipot = int(parts[3])
                        tag = parts[4] # Element symbol
                        distance = float(parts[5])
                        atom_data = {'x': x, 'y': y, 'z': z, 'ipot': ipot, 'tag': tag, 'distance': distance}
                        atoms.append(atom_data)
                        if ipot == 0:
                            absorber_info = atom_data # Store absorber info
                    except (IndexError, ValueError) as parse_err:
                        st.warning(f"Could not parse atom line in {feff_inp_path}: '{line}'. Error: {parse_err}")
                        continue # Skip malformed lines

        if absorber_info is None:
            st.warning(f"Absorbing atom (ipot=0) not found in ATOMS section of {feff_inp_path}")
            return None, None

        # Group by distance (rounded) and assign shells
        dist_groups = defaultdict(list)
        for atom in atoms:
            if atom['ipot'] != 0: # Exclude absorber itself
                dist = atom['distance']
                dist_groups[round(dist, 3)].append({'x': atom['x'], 'y': atom['y'], 'z': atom['z'], 'tag': atom['tag']})

        sorted_distances = sorted(dist_groups.keys())
        for i, dist in enumerate(sorted_distances):
            shell_num = i + 1
            shells[shell_num] = {'distance': dist, 'atoms': dist_groups[dist]}

        return absorber_info, shells

    except Exception as e:
        st.error(f"Error reading or parsing {feff_inp_path}: {e}")
        return None, None


# --- Find Example Files ---
example_files = []
try:
    if os.path.isdir(DATA_DIR):
        # List files in the data directory
        all_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
        example_files = sorted(all_files) # Sort alphabetically
    else:
        st.sidebar.warning(f"'{DATA_DIR}' directory not found. Only upload option available.")
except Exception as e:
    st.sidebar.error(f"Error scanning '{DATA_DIR}' directory: {e}")

# --- Sidebar Controls ---
st.sidebar.header("1. Data Source")
# Combine example files (prefixed) and the upload option
data_source_options = [f"Example: {f}" for f in example_files] + [UPLOAD_OPTION]

# Handle empty data directory case
if not example_files:
    selected_source = UPLOAD_OPTION # Default to upload if no examples found
    st.sidebar.info("No example files found. Please upload a file.")
else:
    # Set default index if necessary or handle potential errors
    default_data_index = 0 # Or choose a specific default
    if 'selected_source_index' not in st.session_state:
        st.session_state.selected_source_index = default_data_index # Default to first example
    try:
        selected_source = st.sidebar.selectbox(
            "Choose data source:",
            options=data_source_options,
            index=st.session_state.selected_source_index, # Use index to maintain selection
            key='data_source_selector' # Use a key to help manage state
        )
        # Update session state with the new index if selection changes
        st.session_state.selected_source_index = data_source_options.index(selected_source)
    except ValueError: # Handle case where index might be invalid after options change
        st.session_state.selected_source_index = default_data_index
        selected_source = st.sidebar.selectbox(
            "Choose data source:",
            options=data_source_options,
            index=st.session_state.selected_source_index,
            key='data_source_selector_fallback' # Use a different key if needed
        )


uploaded_file = None
source_id = None
source_type = None
display_filename = "N/A"
data_basename = None # To find corresponding feff folder

if selected_source == UPLOAD_OPTION:
    uploaded_file = st.sidebar.file_uploader("Choose a data file (.xmu, .dat, etc.)", type=None)
    if uploaded_file is not None:
        source_id = uploaded_file.getvalue() # Use file content bytes as identifier for caching
        source_type = 'upload'
        display_filename = uploaded_file.name
        data_basename = os.path.splitext(display_filename)[0] # Get name without extension
        st.sidebar.success(f"Using uploaded file: `{display_filename}`")
        # Clear session state when file changes IF it's different from current
        if st.session_state.get('current_data_basename') != data_basename or st.session_state.get('current_source_type') != 'upload':
            st.session_state.path_modifications = {}
            st.session_state.fit_result_obj = None # Use the correct state variable
            # st.session_state.fit_history = [] # Removed
            st.session_state.current_data_basename = data_basename
            st.session_state.current_source_type = 'upload'
    else:
        st.sidebar.warning("Please upload a data file.")
        source_id = None # Ensure source_id is None if no file uploaded
        data_basename = None
        # Also clear state if no file is uploaded after selecting upload
        if st.session_state.get('current_source_type') == 'upload':
            st.session_state.path_modifications = {}
            st.session_state.fit_result_obj = None
            st.session_state.current_data_basename = None
            st.session_state.current_source_type = None
else:
    # Extract filename from the selected "Example: filename" string
    example_filename = selected_source.replace("Example: ", "", 1)
    source_id = os.path.join(DATA_DIR, example_filename) # Use file path as identifier
    source_type = 'path'
    display_filename = example_filename
    data_basename = os.path.splitext(display_filename)[0] # Get name without extension
    st.sidebar.info(f"Using example file: `{display_filename}`")
    # Clear session state when file changes IF it's different from current
    if st.session_state.get('current_data_basename') != data_basename or st.session_state.get('current_source_type') != 'path':
        st.session_state.path_modifications = {}
        st.session_state.fit_result_obj = None # Use the correct state variable
        # st.session_state.fit_history = [] # Removed
        st.session_state.current_data_basename = data_basename
        st.session_state.current_source_type = 'path'


# --- Load Feff Paths and Parse feff.inp based on selected data ---
feff_paths_dict = {}
absorber_atom_info = None
shell_data = None
feff_dir = None
if data_basename:
    feff_dir = os.path.join(FEFF_DIR_BASE, data_basename)
    if os.path.isdir(feff_dir):
        feff_paths_dict = load_feff_paths(feff_dir)
        feff_inp_file = os.path.join(feff_dir, 'feff.inp')
        absorber_atom_info, shell_data = parse_feff_inp_atoms(feff_inp_file)
    else:
        st.sidebar.warning(f"Feff directory not found for '{data_basename}': {feff_dir}")
        feff_paths_dict = {} # Ensure it's empty
        absorber_atom_info = None
        shell_data = None
else:
    # Explicitly clear if no data_basename
    feff_paths_dict = {}
    absorber_atom_info = None
    shell_data = None


# --- Parameter Control Panel ---
st.sidebar.header("2. Pre-edge Parameters")
use_custom_e0 = st.sidebar.checkbox("Set Custom E0?", value=False, key="use_e0")
e0_val = None
if use_custom_e0:
    e0_val = st.sidebar.number_input("E0 (eV)", value=st.session_state.get('e0_input', 7112.0), format="%.2f", help="Absorption edge energy. If unchecked, Larch estimates it.", key="e0_input")
else:
    st.sidebar.markdown("_(E0 will be estimated by Larch)_")
use_custom_step = st.sidebar.checkbox("Set Custom Edge Step?", value=False, key="use_step")
step_val = None
if use_custom_step:
    step_val = st.sidebar.number_input("Edge Step", value=st.session_state.get('step_input', 1.0), format="%.3f", help="Absorption edge step height. If unchecked, Larch estimates it.", key="step_input")
else:
    st.sidebar.markdown("_(Edge step will be estimated by Larch)_")
use_custom_pre = st.sidebar.checkbox("Set Custom Pre-edge Range?", value=False, key="use_pre")
pre1_val = None
pre2_val = None
if use_custom_pre:
    pre1_val = st.sidebar.number_input("Pre-edge Start (relative to E0)", value=st.session_state.get('pre1_input', -150.0), format="%.1f", help="Start energy for pre-edge fit, relative to E0.", key="pre1_input")
    pre2_val = st.sidebar.number_input("Pre-edge End (relative to E0)", value=st.session_state.get('pre2_input', -30.0), format="%.1f", help="End energy for pre-edge fit, relative to E0.", key="pre2_input")
else:
    st.sidebar.markdown("_(Pre-edge range will use Larch defaults)_")
use_custom_norm = st.sidebar.checkbox("Set Custom Normalization Range?", value=False, key="use_norm")
norm1_val = None
norm2_val = None
if use_custom_norm:
    norm1_val = st.sidebar.number_input("Normalization Start (relative to E0)", value=st.session_state.get('norm1_input', 150.0), format="%.1f", help="Start energy for normalization fit, relative to E0.", key="norm1_input")
    norm2_val = st.sidebar.number_input("Normalization End (relative to E0)", value=st.session_state.get('norm2_input', 800.0), format="%.1f", help="End energy for normalization fit, relative to E0.", key="norm2_input")
else:
    st.sidebar.markdown("_(Normalization range will use Larch defaults)_")
nnorm_options = {"Auto (Default)": None, "0 (Constant)": 0, "1 (Linear)": 1, "2 (Quadratic)": 2, "3 (Cubic)": 3}
selected_nnorm_key = st.sidebar.selectbox("Normalization Polynomial Degree (nnorm)", options=list(nnorm_options.keys()), index=0, help="Degree of polynomial for post-edge normalization.", key="nnorm_select")
nnorm_val = nnorm_options[selected_nnorm_key]

# --- Autobk Parameters ---
st.sidebar.header("3. Autobk Parameters")
rbkg_val = st.sidebar.number_input("Rbkg (Å)", value=st.session_state.get('rbkg_input', 1.0), min_value=0.1, max_value=3.0, step=0.1, format="%.1f", help="Low-R cutoff for background removal.", key="rbkg_input")
kweight_options_autobk = {1: 1, 2: 2, 3: 3} # Common k-weights
selected_kweight_autobk_key = st.sidebar.selectbox("Autobk k-weight", options=list(kweight_options_autobk.keys()), index=1, help="Weighting (k^?) applied to chi(k) for spline fit.", key="kweight_autobk_select")
kweight_autobk_val = kweight_options_autobk[selected_kweight_autobk_key]

# --- Fourier Transform Parameters ---
st.sidebar.header("4. Fourier Transform Parameters")
ft_kmin_val = st.sidebar.number_input("k Min (Å⁻¹)", value=st.session_state.get('ft_kmin', 3.0), min_value=0.0, format="%.1f", help="Minimum k for FT window.", key="ft_kmin")
ft_kmax_val = st.sidebar.number_input("k Max (Å⁻¹)", value=st.session_state.get('ft_kmax', 15.0), min_value=ft_kmin_val+0.1, format="%.1f", help="Maximum k for FT window.", key="ft_kmax") # Ensure kmax > kmin

kweight_options_ft = {0: 0, 1: 1, 2: 2, 3: 3} # Common k-weights for FT
selected_kweight_ft_key = st.sidebar.selectbox("FT k-weight", options=list(kweight_options_ft.keys()), index=2, # Default k^2
                                                help="Weighting (k^?) applied to chi(k) before FT.", key="kweight_ft_select")
kweight_ft_val = kweight_options_ft[selected_kweight_ft_key]

window_options = ['hanning', 'parzen', 'welch', 'kaiser', 'gaussian', 'sine']
window_val = st.sidebar.selectbox("Window Type", options=window_options, index=0, help="Window function for FT.", key="ft_window")

ft_dk_val = st.sidebar.number_input("Window Taper dk (Å⁻¹)", value=st.session_state.get('ft_dk', 1.0), min_value=0.0, format="%.1f", help="Taper width for FT window.", key="ft_dk")

# --- Feff Path Selection ---
st.sidebar.header("5. Feff Path Selection (for Model/Fit)")
selected_feff_paths = []
if feff_paths_dict:
    available_paths = list(feff_paths_dict.keys())
    # Ensure default selection doesn't exceed available paths
    default_selection = available_paths[:min(1, len(available_paths))]
    selected_feff_paths = st.sidebar.multiselect(
        "Select Feff paths to sum:",
        options=available_paths,
        default=default_selection,
        key="feff_path_select"
    )
else:
    st.sidebar.info(f"No Feff paths found or loaded. Cannot perform modeling.")
    available_paths = [] # Ensure available_paths is empty list

# --- Fitting Setup ---
st.sidebar.header("6. Fitting Setup")
# Ensure options for multiselect are available paths, even if selected_feff_paths (for model) is different
paths_for_fit = st.sidebar.multiselect(
    "Select paths to INCLUDE in fit:",
    options=available_paths, # Base options on all loaded paths
    default=selected_feff_paths, # Default selection based on model paths
    key="fit_paths"
)
fit_rmin = st.sidebar.number_input("Fit R Min (Å)", value=st.session_state.get('fit_rmin', 1.0), min_value=0.0, format="%.2f", key="fit_rmin")
fit_rmax = st.sidebar.number_input("Fit R Max (Å)", value=st.session_state.get('fit_rmax', 3.0), min_value=fit_rmin+0.1, format="%.2f", key="fit_rmax")

st.sidebar.markdown("**Define Fit Parameters:**")
# Use dictionary to store parameter settings per path
fit_param_settings = {}
with st.sidebar.expander("Set Parameters for Paths in Fit", expanded=True): # Expand by default
    if not paths_for_fit:
        st.write("_Select paths for fit above._")
    else:
        # Make sure we iterate over the currently selected paths_for_fit
        # to generate the controls and populate fit_param_settings
        for path_key in paths_for_fit:
            st.markdown(f"**{path_key}**")
            path_params = {}
            # Get default path object if available, otherwise use safe defaults
            default_path_obj = feff_paths_dict.get(path_key) # Use .get for safety
            default_s02 = float(getattr(default_path_obj, 's02', 1.0)) if default_path_obj else 1.0
            default_degen = float(getattr(default_path_obj, 'degen', 1.0)) if default_path_obj else 1.0
            default_sigma2 = float(getattr(default_path_obj, 'sigma2', 0.002)) if default_path_obj else 0.002
            default_deltar = 0.0 # deltar usually defaults to 0
            default_e0 = 0.0 # e0 shift usually defaults to 0

            # S02
            vary_s02 = st.checkbox(f"Vary S₀²?", value=True, key=f"vary_s02_{path_key}")
            guess_s02 = st.number_input(f"Guess/Value S₀²", value=default_s02, format="%.3f", key=f"guess_s02_{path_key}")
            path_params['s02'] = {'vary': vary_s02, 'guess': guess_s02}

            # Degen
            vary_degen = st.checkbox(f"Vary N (degen)?", value=False, key=f"vary_degen_{path_key}") # Usually fixed or tied
            guess_degen = st.number_input(f"Guess/Value N", value=default_degen, format="%.2f", key=f"guess_degen_{path_key}")
            path_params['degen'] = {'vary': vary_degen, 'guess': guess_degen}

            # Delta R
            vary_deltar = st.checkbox(f"Vary ΔR?", value=True, key=f"vary_deltar_{path_key}")
            guess_deltar = st.number_input(f"Guess/Value ΔR (Å)", value=default_deltar, format="%.4f", step=0.001, key=f"guess_deltar_{path_key}")
            path_params['deltar'] = {'vary': vary_deltar, 'guess': guess_deltar}

            # Sigma2
            vary_sigma2 = st.checkbox(f"Vary σ²?", value=True, key=f"vary_sigma2_{path_key}")
            guess_sigma2 = st.number_input(f"Guess/Value σ² (Å²)", value=default_sigma2, min_value=0.0, format="%.5f", step=0.0001, key=f"guess_sigma2_{path_key}")
            path_params['sigma2'] = {'vary': vary_sigma2, 'guess': guess_sigma2}

            # E0 shift
            vary_e0 = st.checkbox(f"Vary E₀ Shift?", value=True, key=f"vary_e0_{path_key}")
            guess_e0 = st.number_input(f"Guess/Value E₀ Shift (eV)", value=default_e0, format="%.2f", step=0.1, key=f"guess_e0_{path_key}")
            path_params['e0'] = {'vary': vary_e0, 'guess': guess_e0}

            # Store settings for this path IMMEDIATELY after creating controls
            fit_param_settings[path_key] = path_params

# --- Fit Execution Trigger ---
run_fit = st.sidebar.button("Run Fit")
reset_fit = st.sidebar.button("Reset Fit State", key="reset_fit") # Add Reset Button

if reset_fit:
    st.session_state.fit_result_obj = None
    # st.session_state.fit_history = [] # Removed
    st.success("Fit results cleared.")
    # Need rerun to refresh plots that depend on fit_result_obj
    st.rerun()

# --- Single Path Modification ---
st.sidebar.header("7. Single Path Modification")
path_to_modify = None
# Allow modification only for paths included in the AVAILABLE feff paths
if available_paths: # Base modification selection on all available paths
    # Check if currently selected fit paths are valid options
    valid_paths_for_mod = [p for p in paths_for_fit if p in available_paths]
    if valid_paths_for_mod:
         mod_options = valid_paths_for_mod
         mod_index = 0
    else: # Fallback if no fit paths selected or valid, use first available path
         mod_options = available_paths
         mod_index = 0

    path_to_modify = st.sidebar.selectbox(
         "Select Path to Modify:",
         options=mod_options,
         index=mod_index,
         key="path_mod_select"
     )
else:
    st.sidebar.info("Load Feff paths to enable modification.")

if path_to_modify and path_to_modify in feff_paths_dict:
    default_path = feff_paths_dict[path_to_modify]
    # Use defaults from the path object if available, otherwise use generic defaults
    default_mod_s02 = float(getattr(default_path,'s02', 1.0))
    default_mod_degen = float(getattr(default_path,'degen', 1.0))
    default_mod_deltar = 0.0
    default_mod_sigma2 = float(getattr(default_path, 'sigma2', 0.002))
    default_mod_e0 = 0.0

    s02_mod_val = st.sidebar.number_input("S₀²", min_value=0.0, value=default_mod_s02, step=0.05, format="%.3f", key=f"s02_mod_{path_to_modify}")
    degen_mod_val = st.sidebar.number_input("N (degen)", min_value=0.0, value=default_mod_degen, step=0.5, format="%.2f", key=f"degen_mod_{path_to_modify}")
    deltar_mod_val = st.sidebar.number_input("ΔR (Å)", value=default_mod_deltar, step=0.01, format="%.3f", key=f"deltar_mod_{path_to_modify}")
    sigma2_mod_val = st.sidebar.number_input("σ² (Å²)", min_value=0.0, value=default_mod_sigma2, step=0.0005, format="%.4f", key=f"sigma2_mod_{path_to_modify}")
    e0_mod_val = st.sidebar.number_input("E₀ Shift (eV)", value=default_mod_e0, step=0.1, format="%.2f", key=f"e0_mod_{path_to_modify}")

    col1_mod, col2_mod = st.sidebar.columns(2)
    with col1_mod:
        add_mod = st.button("Add Modification to Plot", key="add_mod_button")
    with col2_mod:
        reset_mod = st.button("Reset Mod Plot", key="reset_mod_button")

    if add_mod:
        mod_params = {
            's02': s02_mod_val, 'degen': degen_mod_val, 'deltar': deltar_mod_val,
            'sigma2': sigma2_mod_val, 'e0': e0_mod_val
        }
        if path_to_modify not in st.session_state.path_modifications:
            st.session_state.path_modifications[path_to_modify] = []

        # Limit number of modifications shown
        if len(st.session_state.path_modifications[path_to_modify]) >= MAX_MODIFICATIONS:
            st.session_state.path_modifications[path_to_modify].pop(0) # Remove oldest

        st.session_state.path_modifications[path_to_modify].append({'params': mod_params})
        # Rerun to update plot with modification
        st.rerun()

    if reset_mod:
        if path_to_modify in st.session_state.path_modifications:
            st.session_state.path_modifications[path_to_modify] = []
            # Rerun to update plot after reset
            st.rerun()


# --- Plotting Options ---
st.sidebar.header("9. Plotting Options") # Renumbered
with st.sidebar.expander("Plots 1 & 2 Options (Energy Space)"):
    plot_start_rel = st.number_input("Plot Start (relative to E0)", value=st.session_state.get('plot_start', -50.0), format="%.1f", help="Start energy for plot display, relative to E0.", key="plot_start")
    plot_end_rel = st.number_input("Plot End (relative to E0)", value=st.session_state.get('plot_end', 200.0), format="%.1f", help="End energy for plot display, relative to E0.", key="plot_end")

with st.sidebar.expander("Plot 3 Options (k Space)"):
    k_min_plot_val = st.number_input("Plot Start (k)", value=st.session_state.get('k_min_plot', 0.0), min_value=0.0, format="%.1f", help="Minimum k value for chi(k) plot display.", key="k_min_plot")
    k_max_plot_val = st.number_input("Plot End (k)", value=st.session_state.get('k_max_plot', 15.0), min_value=k_min_plot_val+0.1, format="%.1f", help="Maximum k value for chi(k) plot display.", key="k_max_plot")

with st.sidebar.expander("Plot 4 & 5 Options (R Space)"): # Combined R-space options
    r_min_plot_val = st.number_input("Plot Start (R)", value=st.session_state.get('r_min_plot', 0.0), min_value=0.0, format="%.1f", help="Minimum R value for |chi(R)| plot display.", key="r_min_plot")
    r_max_plot_val = st.number_input("Plot End (R)", value=st.session_state.get('r_max_plot', 6.0), min_value=r_min_plot_val+0.1, format="%.1f", help="Maximum R value for |chi(R)| plot display.", key="r_max_plot")
    show_chir_re = st.checkbox("Show Real Part (Re[χ(R)])", value=False, key="show_re")
    show_chir_im = st.checkbox("Show Imaginary Part (Im[χ(R)])", value=False, key="show_im")

shell_visibility = {} # Initialize here
with st.sidebar.expander("Step 4 Optional Plots"):
    show_geom = st.checkbox("Show Path Geometries & Shells", value=False, key="show_geom")
    if show_geom and shell_data:
        st.write("Toggle Shell Visibility:")
        max_shells_to_show = 5 # Limit number of toggles
        shell_keys = sorted(shell_data.keys())[:max_shells_to_show]
        for i, shell_num in enumerate(shell_keys):
            shell_info_item = shell_data[shell_num] # Use different name
            default_visible = (i < 1) # Show 1st shell by default
            # Ensure unique key for checkbox if shell_num can repeat (unlikely here)
            is_visible = st.checkbox(f"Shell {shell_num} (R≈{shell_info_item['distance']:.2f} Å, {len(shell_info_item['atoms'])} atoms)", value=default_visible, key=f"show_shell_{shell_num}_{data_basename}") # Add basename to key
            shell_visibility[shell_num] = is_visible
    elif show_geom:
        st.write("_Load data and corresponding feff folder to see shell toggles._")

    show_scat = st.checkbox("Show Scattering Properties", value=False, key="show_scat")
    # Show individual chi plot by default
    show_indiv_chi = st.checkbox("Show Individual Path χ(k) / χ(R)", value=True, key="show_indiv_chi")


# --- Function to Process Data and Generate Plot ---
# Note: This function now takes the pre-loaded 'dat' object AND feff paths dict AND shell data
def process_and_plot(dat, feff_paths, selected_path_keys, shell_info, absorber_info, filename_for_title, params, plot_options, fit_result_obj=None): # Changed to fit_result_obj
    """Performs pre-edge, autobk, FT, Feff modeling, and plotting. Optionally plots fit results."""
    if dat is None:
        st.warning("No data loaded to process.")
        return

    try:
        # --- Step 1: Pre-processing ---
        st.subheader("Step 1: Pre-edge Subtraction & Normalization")
        pre_edge(dat, group=dat, e0=params['e0'], step=params['step'], pre1=params['pre1'], pre2=params['pre2'], norm1=params['norm1'], norm2=params['norm2'], nnorm=params['nnorm'])
        e0_used = dat.e0 if hasattr(dat, 'e0') else None
        step_used = dat.edge_step if hasattr(dat, 'edge_step') else None
        e0_display = f"{e0_used:.2f}" if isinstance(e0_used, (int, float)) else 'N/A (Estimated)'
        step_display = f"{step_used:.3f}" if isinstance(step_used, (int, float)) else 'N/A (Estimated)'
        st.write(f"Pre-edge subtraction applied. Using E0 = {e0_display} eV, Edge Step = {step_display}")

        # --- Step 2: Autobk Background Subtraction ---
        st.subheader("Step 2: Autobk Background Subtraction")
        autobk(dat, group=dat, rbkg=params['rbkg'], kweight=params['autobk_kweight'])
        # Store experimental chi before it might be overwritten
        if hasattr(dat, 'chi'):
            dat.chi_exp = copy.deepcopy(dat.chi) # Make a copy
        else:
            st.warning("Autobk did not produce 'chi' array.")
            dat.chi_exp = None # Ensure chi_exp is None if chi wasn't created

        st.write(f"Autobk applied with Rbkg = {params['rbkg']:.1f} Å, k-weight = {params['autobk_kweight']}.")

        # --- Step 3: Fourier Transform (k -> R) of Experimental Data ---
        st.subheader("Step 3: Fourier Transform (k -> R) of Experimental Data")
        # Perform FT on experimental chi (dat.chi_exp)
        if hasattr(dat, 'k') and dat.k is not None and len(dat.k)>0 and dat.chi_exp is not None:
            try:
                xftf(dat, group=dat, chi='chi_exp', # Specify input chi array name
                     kmin=params['ft_kmin'], kmax=params['ft_kmax'], dk=params['ft_dk'],
                     window=params['ft_window'], kweight=params['ft_kweight'])
                st.write(f"Forward FT applied to experimental data: k=[{params['ft_kmin']:.1f}, {params['ft_kmax']:.1f}], k-weight={params['ft_kweight']}, window='{params['ft_window']}', dk={params['ft_dk']:.1f}")
            except Exception as ft_exp_err:
                 st.error(f"Error during experimental data FT: {ft_exp_err}")
                 # Clear potentially invalid FT results
                 for attr in ['r', 'chir_mag', 'chir_re', 'chir_im']:
                     if hasattr(dat, attr): delattr(dat, attr)
        else:
            st.warning("Cannot perform Fourier Transform: Missing valid 'k' or 'chi_exp' from Autobk.")


        # --- Step 4: Calculate Model EXAFS from Feff Paths ---
        st.subheader("Step 4: Calculate Model EXAFS from Feff Paths")
        model_chi = None
        individual_path_chis = {} # Store individual path chi for optional plot
        individual_path_fts = {} # Store individual path FT results
        path_objects_for_model = [] # Store copies used for model plot
        k_present_for_model = hasattr(dat, 'k') and dat.k is not None and len(dat.k) > 0

        if selected_path_keys and feff_paths and k_present_for_model:
            for key in selected_path_keys:
                if key in feff_paths:
                    path_obj_copy = copy.deepcopy(feff_paths[key]) # Work on a copy
                    # --- Apply per-path parameters here if implemented ---
                    # For now, use defaults from file for initial model plot
                    path_objects_for_model.append(path_obj_copy) # Add to list for ff2chi

                    # Calculate individual chi
                    try:
                        path2chi(path_obj_copy, _larch=None, k=dat.k) # Explicitly pass k
                        if hasattr(path_obj_copy, 'chi'):
                            individual_path_chis[key] = path_obj_copy.chi
                            # Calculate individual FT if needed for plotting later
                            if plot_options['show_indiv_chi']:
                                temp_ft_group = Group(k=dat.k, chi=path_obj_copy.chi)
                                xftf(temp_ft_group, group=temp_ft_group, chi='chi', # Use chi from temp group
                                     kmin=params['ft_kmin'], kmax=params['ft_kmax'], dk=params['ft_dk'],
                                     window=params['ft_window'], kweight=params['ft_kweight'])
                                individual_path_fts[key] = temp_ft_group # Store the whole group
                        else:
                             # path2chi might fail silently sometimes without `chi`
                             st.warning(f"path2chi for path {key} did not produce 'chi' attribute.")

                    except Exception as e_indiv:
                        st.warning(f"Could not calculate chi/FT for path {key}: {e_indiv}")

            if path_objects_for_model:
                # Calculate initial model using default path params
                try:
                    ff2chi(path_objects_for_model, group=dat, k=dat.k) # Pass k explicitly
                    if hasattr(dat, 'chi'):
                        dat.chi_mod = dat.chi # Store model chi separately
                        model_chi = dat.chi_mod # Flag that model was calculated
                        st.write(f"Calculated initial model χ(k) from {len(selected_path_keys)} selected path(s).")
                    else:
                        st.warning("ff2chi did not produce 'chi' output for initial model.")
                except Exception as e_ff2chi:
                     st.error(f"Error during ff2chi (model calculation): {e_ff2chi}")
                     model_chi = None # Ensure model_chi is None if ff2chi fails
            else:
                st.warning("No valid paths selected or processed for modeling.")

        elif not selected_path_keys:
            st.info("No Feff paths selected for modeling.")
        elif not feff_paths:
            st.warning("Feff paths dictionary is empty.")
        elif not k_present_for_model:
            st.warning("Cannot calculate model χ(k): Missing 'k' from Autobk.")

        # --- Get Best Fit Data if available ---
        best_fit_chi = None
        best_fit_ft_group = None # Store the Group containing FT results
        if fit_result_obj is not None and hasattr(fit_result_obj, 'datasets') and len(fit_result_obj.datasets) > 0:
            fit_dset = fit_result_obj.datasets[0]
            if hasattr(fit_dset, 'model'):
                fit_model_group = fit_dset.model
                # Check for chi(k)
                if hasattr(fit_model_group, 'k') and hasattr(fit_model_group, 'chi'):
                    # Ensure k grids match roughly before assigning chi
                    if len(fit_model_group.k) == len(dat.k) and np.allclose(fit_model_group.k, dat.k):
                         best_fit_chi = fit_model_group.chi
                    else:
                         st.warning("k-grid mismatch between data and fit model chi. Cannot plot best fit chi(k).")

                # Check for chi(R) - store the whole group
                if hasattr(fit_model_group, 'r') and hasattr(fit_model_group, 'chir_mag'):
                    best_fit_ft_group = fit_model_group # Store the group

        # --- Plotting Results ---
        st.subheader("Interactive Data Visualization")

        # --- Plot 1: Original vs. Pre/Post Edge Fits & Background ---
        st.markdown("**Plot 1: Original Data and Background Fits**")
        fig1 = go.Figure()
        # Plot only if data arrays exist and are not None
        if hasattr(dat, 'energy') and dat.energy is not None and hasattr(dat, 'mu') and dat.mu is not None:
            fig1.add_trace(go.Scatter(x=dat.energy, y=dat.mu, mode='lines', name='μ(E) Original', line=dict(color='royalblue')))
        if hasattr(dat, 'energy') and dat.energy is not None and hasattr(dat, 'pre_edge') and dat.pre_edge is not None:
            fig1.add_trace(go.Scatter(x=dat.energy, y=dat.pre_edge, mode='lines', name='Pre-edge Line', line=dict(color='darkorange', dash='dot')))
        if hasattr(dat, 'energy') and dat.energy is not None and hasattr(dat, 'post_edge') and dat.post_edge is not None:
            fig1.add_trace(go.Scatter(x=dat.energy, y=dat.post_edge, mode='lines', name='Post-edge Curve', line=dict(color='seagreen', dash='dot')))
        if hasattr(dat, 'energy') and dat.energy is not None and hasattr(dat, 'bkg') and dat.bkg is not None:
            fig1.add_trace(go.Scatter(x=dat.energy, y=dat.bkg, mode='lines', name='μ₀(E) Background', line=dict(color='grey', dash='longdash')))

        boundary_points_x = []
        boundary_points_y = []
        point_labels = []
        if isinstance(e0_used, (int, float)) and hasattr(dat, 'energy') and dat.energy is not None and len(dat.energy) > 0 and hasattr(dat, 'mu') and dat.mu is not None:
            safe_interp = lambda x, xp, yp: np.interp(x, xp, yp, left=np.nan, right=np.nan)
            if params['use_custom_pre'] and params['pre1'] is not None and params['pre2'] is not None:
                pre_start_abs = e0_used + params['pre1']
                pre_end_abs = e0_used + params['pre2']
                pre_start_y = safe_interp(pre_start_abs, dat.energy, dat.mu)
                pre_end_y = safe_interp(pre_end_abs, dat.energy, dat.mu)
                if not np.isnan(pre_start_y): boundary_points_x.append(pre_start_abs); boundary_points_y.append(pre_start_y); point_labels.append(f"Pre Start ({pre_start_abs:.1f} eV)")
                if not np.isnan(pre_end_y): boundary_points_x.append(pre_end_abs); boundary_points_y.append(pre_end_y); point_labels.append(f"Pre End ({pre_end_abs:.1f} eV)")
            if params['use_custom_norm'] and params['norm1'] is not None and params['norm2'] is not None:
                norm_start_abs = e0_used + params['norm1']
                norm_end_abs = e0_used + params['norm2']
                norm_start_y = safe_interp(norm_start_abs, dat.energy, dat.mu)
                norm_end_y = safe_interp(norm_end_abs, dat.energy, dat.mu)
                if not np.isnan(norm_start_y): boundary_points_x.append(norm_start_abs); boundary_points_y.append(norm_start_y); point_labels.append(f"Norm Start ({norm_start_abs:.1f} eV)")
                if not np.isnan(norm_end_y): boundary_points_x.append(norm_end_abs); boundary_points_y.append(norm_end_y); point_labels.append(f"Norm End ({norm_end_abs:.1f} eV)")

        if boundary_points_x:
            fig1.add_trace(go.Scatter(x=boundary_points_x, y=boundary_points_y, mode='markers', name='Fit Boundaries', marker=dict(color='red', symbol='diamond', size=10), text=point_labels, hoverinfo='text+x+y'))

        plot_range_set_e = False
        if isinstance(e0_used, (int, float)):
            fig1.add_vline(x=e0_used, line_width=1, line_dash="dot", line_color="grey", annotation_text=f"E0={e0_used:.2f}", annotation_position="top left")
            plot_start_abs_e = e0_used + plot_options['plot_start_rel']
            plot_end_abs_e = e0_used + plot_options['plot_end_rel']
            fig1.update_xaxes(range=[plot_start_abs_e, plot_end_abs_e])
            plot_range_set_e = True
        fig1.update_layout(title=f'Pre-edge Subtraction & Background: {filename_for_title}', xaxis_title='Energy (eV)', yaxis_title='Absorption (arbitrary units)', legend_title='Legend', hovermode='x unified', height=500, showlegend=True)
        st.plotly_chart(fig1, use_container_width=True)

        # --- Plot 2: Flattened Normalized Data ---
        if hasattr(dat, 'flat') and dat.flat is not None and hasattr(dat, 'energy') and dat.energy is not None:
            st.markdown("**Plot 2: Flattened Normalized Data**")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dat.energy, y=dat.flat, mode='lines', name='μ(E) Flattened', line=dict(color='firebrick')))
            if isinstance(e0_used, (int, float)):
                fig2.add_vline(x=e0_used, line_width=1, line_dash="dot", line_color="grey", annotation_text=f"E0={e0_used:.2f}", annotation_position="top left")
                if plot_range_set_e:
                    fig2.update_xaxes(range=[plot_start_abs_e, plot_end_abs_e]) # Apply Energy range
            fig2.update_layout(title=f'Flattened Normalized Data: {filename_for_title}', xaxis_title='Energy (eV)', yaxis_title='Normalized Absorption (Flattened)', legend_title='Legend', hovermode='x unified', height=500, showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Flattened data (`dat.flat`) not found or generated.")

        # --- Plot 3: chi(k) Comparison ---
        st.markdown(f"**Plot 3: Experimental vs Model χ(k) * k^{params['ft_kweight']}**")
        fig3 = go.Figure()
        plot_k_weight = params['ft_kweight'] # Use FT k-weight for plotting consistency
        k_present = hasattr(dat, 'k') and dat.k is not None and len(dat.k) > 0

        # Plot Experimental chi(k)
        if k_present and dat.chi_exp is not None:
            chi_exp_plot = dat.chi_exp * (dat.k ** plot_k_weight)
            fig3.add_trace(go.Scatter(x=dat.k, y=chi_exp_plot, mode='lines', name=f'Exp χ(k)*k^{plot_k_weight}', line=dict(color='purple')))
        else:
            st.warning("Experimental chi(k) (`dat.k`, `dat.chi_exp`) not found or invalid.")

        # Plot Initial Model chi(k) if calculated, hidden by default
        if k_present and model_chi is not None:
            chi_mod_plot = model_chi * (dat.k ** plot_k_weight)
            fig3.add_trace(go.Scatter(x=dat.k, y=chi_mod_plot, mode='lines', name=f'Model χ(k)*k^{plot_k_weight}', line=dict(color='orange', dash='dash'), visible='legendonly'))
        elif selected_path_keys and k_present:
            st.warning("Initial Model chi(k) (`dat.chi_mod`) not calculated.")

        # Plot Best Fit chi(k) if available, hidden by default
        if k_present and best_fit_chi is not None:
            chi_fit_plot = best_fit_chi * (dat.k ** plot_k_weight)
            # Show fit chi(k) by default if it exists
            fig3.add_trace(go.Scatter(x=dat.k, y=chi_fit_plot, mode='lines', name=f'Best Fit χ(k)*k^{plot_k_weight}', line=dict(color='red', dash='solid'), visible=True))

        # Apply k-range from plot options
        if k_present:
            fig3.update_xaxes(range=[plot_options['k_min'], plot_options['k_max']])
        fig3.update_layout(title=f"Experimental vs Model k-Weighted χ(k): {filename_for_title}", xaxis_title="k (Å⁻¹)", yaxis_title=f"k^{plot_k_weight} * χ(k)", legend_title='Legend', hovermode='x unified', height=500, showlegend=True)
        st.plotly_chart(fig3, use_container_width=True)


        # --- Plot 4: chi(R) Comparison ---
        fig4 = go.Figure()
        st.markdown("**Plot 4: Fourier Transform |χ(R)|**")
        r_present = hasattr(dat, 'r') and dat.r is not None and len(dat.r) > 0

        # Plot Experimental FT Magnitude
        if r_present and hasattr(dat, 'chir_mag'):
            fig4.add_trace(go.Scatter(x=dat.r, y=dat.chir_mag, mode='lines', name='|χ(R)| Exp', line=dict(color='darkgreen')))
            # Conditionally plot Experimental Real part (starts hidden)
            if plot_options['show_re'] and hasattr(dat, 'chir_re'):
                fig4.add_trace(go.Scatter(x=dat.r, y=dat.chir_re, mode='lines', name='Re[χ(R)] Exp', line=dict(color='red', dash='dashdot'), visible='legendonly'))
            # Conditionally plot Experimental Imaginary part (starts hidden)
            if plot_options['show_im'] and hasattr(dat, 'chir_im'):
                fig4.add_trace(go.Scatter(x=dat.r, y=dat.chir_im, mode='lines', name='Im[χ(R)] Exp', line=dict(color='blue', dash='dashdot'), visible='legendonly'))
        else:
            st.warning("Experimental FT data (`dat.r`, `dat.chir_mag`) not found or generated.")

        # --- Add FT of Initial Model ---
        if k_present_for_model and model_chi is not None:
            temp_group_mod = Group(k=dat.k, chi=model_chi)
            try:
                xftf(temp_group_mod, group=temp_group_mod, chi='chi', kmin=params['ft_kmin'], kmax=params['ft_kmax'], dk=params['ft_dk'], window=params['ft_window'], kweight=params['ft_kweight'])
                # Plot Model Magnitude (hidden by default)
                if hasattr(temp_group_mod, 'r') and hasattr(temp_group_mod, 'chir_mag'):
                    fig4.add_trace(go.Scatter(x=temp_group_mod.r, y=temp_group_mod.chir_mag, mode='lines', name='|χ(R)| Model', line=dict(color='orange', dash='dash'), visible='legendonly'))
                # Plot Model Real (hidden by default, only if main toggle is on)
                if plot_options['show_re'] and hasattr(temp_group_mod, 'chir_re'):
                    fig4.add_trace(go.Scatter(x=temp_group_mod.r, y=temp_group_mod.chir_re, mode='lines', name='Re[χ(R)] Model', line=dict(color='pink', dash='dash'), visible='legendonly'))
                # Plot Model Imaginary (hidden by default, only if main toggle is on)
                if plot_options['show_im'] and hasattr(temp_group_mod, 'chir_im'):
                    fig4.add_trace(go.Scatter(x=temp_group_mod.r, y=temp_group_mod.chir_im, mode='lines', name='Im[χ(R)] Model', line=dict(color='lightblue', dash='dash'), visible='legendonly'))
            except Exception as ft_err:
                st.warning(f"Could not calculate Fourier Transform of initial model chi: {ft_err}")
        elif selected_path_keys and k_present_for_model:
             st.warning("Initial Model FT not calculated.")

        # --- Add FT of Best Fit ---
        if best_fit_ft_group is not None: # Use the pre-calculated group
            # Plot Best Fit Magnitude (show by default if fit exists)
            if hasattr(best_fit_ft_group, 'r') and hasattr(best_fit_ft_group, 'chir_mag'):
                fig4.add_trace(go.Scatter(x=best_fit_ft_group.r, y=best_fit_ft_group.chir_mag, mode='lines', name='|χ(R)| Best Fit', line=dict(color='red', dash='solid'), visible=True))
            # Plot Best Fit Real (hidden by default, only if main toggle is on)
            if plot_options['show_re'] and hasattr(best_fit_ft_group, 'chir_re'):
                fig4.add_trace(go.Scatter(x=best_fit_ft_group.r, y=best_fit_ft_group.chir_re, mode='lines', name='Re[χ(R)] Best Fit', line=dict(color='#FF7F7F', dash='solid'), visible='legendonly')) # Lighter red
            # Plot Best Fit Imaginary (hidden by default, only if main toggle is on)
            if plot_options['show_im'] and hasattr(best_fit_ft_group, 'chir_im'):
                fig4.add_trace(go.Scatter(x=best_fit_ft_group.r, y=best_fit_ft_group.chir_im, mode='lines', name='Im[χ(R)] Best Fit', line=dict(color='#ADD8E6', dash='solid'), visible='legendonly')) # Lighter blue

        # Finalize Plot 4 Layout only if some data was added
        if len(fig4.data) > 0:
             if r_present: fig4.update_xaxes(range=[plot_options['r_min'], plot_options['r_max']]) # Apply R range
             fig4.update_layout(title=f"Fourier Transform: {filename_for_title}", xaxis_title="R (Å)", yaxis_title="χ(R) Magnitude (arb. units)", legend_title='Legend', hovermode='x unified', height=500, showlegend=True)
             st.plotly_chart(fig4, use_container_width=True)
        # No need for else, handled by warnings above


        # --- Optional Plots for Feff Paths ---
        st.markdown("---")
        st.subheader("Optional Feff Path Details")

        if plot_options['show_geom'] and selected_path_keys and feff_paths:
            st.markdown("**Path Geometries & Local Structure**")
            # --- Enhanced Geometry Plot ---
            fig_geom = go.Figure()
            unique_atoms_data = {}
            path_line_traces = []
            path_color_cycle = itertools.cycle(pcolors.qualitative.Plotly)
            max_dist_for_opacity = 8.0
            min_opacity = 0.15

            # 1. Process Shells
            if shell_info:
                max_visible_shell_dist = 0
                visible_shell_count = 0
                for shell_num, show_shell in plot_options.get('shell_visibility', {}).items():
                    if show_shell and shell_num in shell_info:
                        max_visible_shell_dist = max(max_visible_shell_dist, shell_info[shell_num]['distance'])
                        visible_shell_count += 1
                if max_visible_shell_dist <= 0 or visible_shell_count == 0:
                    max_visible_shell_dist = max_dist_for_opacity

                for shell_num, show_shell in plot_options.get('shell_visibility', {}).items():
                    if show_shell and shell_num in shell_info:
                        shell = shell_info[shell_num]
                        distance = shell['distance']
                        opacity = max(min_opacity, 0.8 - 0.65 * (distance / max_visible_shell_dist))
                        for atom in shell['atoms']:
                            x, y, z, tag = atom['x'], atom['y'], atom['z'], atom['tag']
                            coord_str = f"{x:.4f}_{y:.4f}_{z:.4f}"
                            if coord_str not in unique_atoms_data:
                                unique_atoms_data[coord_str] = {'x': x, 'y': y, 'z': z, 'symbol': tag, 'type': f'shell_{shell_num}', 'opacity': opacity, 'size': 6, 'color': get_element_color(tag)}

            # 2. Process Paths
            for key in selected_path_keys:
                if key in feff_paths and hasattr(feff_paths[key], 'geom'):
                    path_obj = feff_paths[key]
                    geom = path_obj.geom
                    if geom:
                        path_x, path_y, path_z = [], [], []
                        path_color = next(path_color_cycle)
                        valid_geom_path = True
                        for i, atom_info in enumerate(geom):
                            try:
                                symbol, Z, ipot, mass, x_str, y_str, z_str = atom_info
                                x, y, z = float(x_str), float(y_str), float(z_str)
                            except (ValueError, TypeError, IndexError) as e:
                                st.warning(f"Unexpected geometry format for atom {i} in path {key}: {atom_info}. Error: {e}. Skipping this path geometry.")
                                path_x, path_y, path_z = [], [], []
                                valid_geom_path = False
                                break

                            if not valid_geom_path: continue

                            coord_str = f"{x:.4f}_{y:.4f}_{z:.4f}"
                            if coord_str not in unique_atoms_data:
                                unique_atoms_data[coord_str] = {'x': x, 'y': y, 'z': z, 'symbol': symbol, 'type': 'path', 'opacity': 1.0, 'size': 8, 'color': get_element_color(symbol)}
                            else:
                                unique_atoms_data[coord_str]['type'] = 'path'
                                unique_atoms_data[coord_str]['opacity'] = 1.0
                                unique_atoms_data[coord_str]['size'] = 8

                            path_x.append(x)
                            path_y.append(y)
                            path_z.append(z)

                        if path_x and valid_geom_path:
                            path_line_traces.append(go.Scatter3d(
                                x=path_x, y=path_y, z=path_z,
                                mode='lines', line=dict(width=5, color=path_color), name=f"{key} (Reff={path_obj.reff:.3f}Å)"
                            ))

            # 3. Plot Atoms
            if unique_atoms_data:
                grouped_plot_atoms = defaultdict(lambda: {'x': [], 'y': [], 'z': []})
                for data_atom in unique_atoms_data.values():
                    group_key = (data_atom['symbol'], data_atom['size'], data_atom['opacity'])
                    grouped_plot_atoms[group_key]['x'].append(data_atom['x'])
                    grouped_plot_atoms[group_key]['y'].append(data_atom['y'])
                    grouped_plot_atoms[group_key]['z'].append(data_atom['z'])

                for group_key, coords in grouped_plot_atoms.items():
                    symbol, size, opacity = group_key
                    fig_geom.add_trace(go.Scatter3d(
                        x=coords['x'], y=coords['y'], z=coords['z'],
                        mode='markers', marker=dict(size=size, color=get_element_color(symbol), opacity=opacity),
                        name=f'{symbol} Atoms (Size:{size}, Opacity:{opacity:.2f})', showlegend=False
                    ))

            # 4. Add Path Lines
            for trace in path_line_traces:
                fig_geom.add_trace(trace)

            # 5. Layout
            if unique_atoms_data or path_line_traces:
                fig_geom.update_layout(
                    title="Selected Path Geometries & Local Structure", margin=dict(l=0, r=0, b=0, t=40),
                    scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)', aspectmode='data'),
                    height=600, showlegend=True, legend_title_text='Paths'
                )
                st.plotly_chart(fig_geom, use_container_width=True)
            else:
                st.info("No valid geometry data to plot.")
            # --- End Enhanced Geometry Plot ---

        if plot_options['show_scat'] and selected_path_keys and feff_paths:
            st.markdown("**Scattering Properties vs k**")
            fig_scat = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                     subplot_titles=("Feff Amplitude |Feff(k)|", "Phase Shift δ(k)", "Mean Free Path λ(k)"))
            colors = pcolors.qualitative.Plotly
            paths_plotted_scat = 0
            for i, key in enumerate(selected_path_keys):
                if key in feff_paths and hasattr(feff_paths[key], '_feffdat'):
                    feffdat = feff_paths[key]._feffdat
                    color = colors[i % len(colors)]
                    has_amp = hasattr(feffdat, 'k') and hasattr(feffdat, 'amp') and feffdat.k is not None and feffdat.amp is not None
                    has_pha = hasattr(feffdat, 'k') and hasattr(feffdat, 'pha') and feffdat.k is not None and feffdat.pha is not None
                    has_lam = hasattr(feffdat, 'k') and hasattr(feffdat, 'lam') and feffdat.k is not None and feffdat.lam is not None
                    show_legend_scat = True # Show legend for amplitude trace

                    if has_amp:
                        fig_scat.add_trace(go.Scatter(x=feffdat.k, y=feffdat.amp, mode='lines', name=key, legendgroup=key, line=dict(color=color), showlegend=show_legend_scat), row=1, col=1)
                        paths_plotted_scat +=1
                    if has_pha:
                        fig_scat.add_trace(go.Scatter(x=feffdat.k, y=feffdat.pha, mode='lines', name=key, legendgroup=key, line=dict(color=color), showlegend=False), row=2, col=1)
                    if has_lam:
                        fig_scat.add_trace(go.Scatter(x=feffdat.k, y=feffdat.lam, mode='lines', name=key, legendgroup=key, line=dict(color=color), showlegend=False), row=3, col=1)
                else:
                    st.warning(f"No Feff data (`_feffdat`) found for path {key}.")

            if paths_plotted_scat > 0:
                fig_scat.update_layout(title="Feff Path Scattering Properties", height=600, hovermode='x unified', legend_title_text='Path')
                fig_scat.update_yaxes(title_text="|Feff(k)|", row=1, col=1)
                fig_scat.update_yaxes(title_text="Phase δ(k)", row=2, col=1)
                fig_scat.update_yaxes(title_text="MFP λ(k)", row=3, col=1)
                fig_scat.update_xaxes(title_text="k (Å⁻¹)", range=[plot_options['k_min'], plot_options['k_max']], row=3, col=1) # Apply k range
                st.plotly_chart(fig_scat, use_container_width=True)
            else:
                st.info("No valid scattering properties found to plot.")


        if plot_options['show_indiv_chi'] and selected_path_keys and feff_paths and k_present_for_model:
             # --- Individual Path k and R plots ---
             st.markdown(f"**Individual Path χ(k) * k^{params['ft_kweight']} and |χ(R)|**")
             col1, col2 = st.columns(2)
             paths_plotted_k = 0
             paths_plotted_r = 0

             with col1:
                 fig_indiv_k = go.Figure()
                 k_weight_plot = params['ft_kweight']
                 colors_k = pcolors.qualitative.Plotly
                 for i, key in enumerate(selected_path_keys):
                      if key in individual_path_chis:
                           chi_plot_indiv = individual_path_chis[key] * (dat.k ** k_weight_plot)
                           fig_indiv_k.add_trace(go.Scatter(x=dat.k, y=chi_plot_indiv, mode='lines', name=f"{key}", line=dict(color=colors_k[i % len(colors_k)])))
                           paths_plotted_k += 1

                 if paths_plotted_k > 0:
                     fig_indiv_k.update_xaxes(range=[plot_options['k_min'], plot_options['k_max']])
                     fig_indiv_k.update_layout(title=f"Individual Path k-Weighted χ(k)", xaxis_title="k (Å⁻¹)", yaxis_title=f"k^{k_weight_plot} * χ(k)", legend_title='Path', hovermode='x unified', height=500, showlegend=True)
                     st.plotly_chart(fig_indiv_k, use_container_width=True)
                 else:
                     st.info("No individual path χ(k) data calculated or available to plot.")

             with col2:
                 fig_indiv_r = go.Figure()
                 colors_r = pcolors.qualitative.Plotly
                 for i, key in enumerate(selected_path_keys):
                     if key in individual_path_fts:
                         ft_group = individual_path_fts[key]
                         if hasattr(ft_group, 'r') and hasattr(ft_group, 'chir_mag'):
                             fig_indiv_r.add_trace(go.Scatter(x=ft_group.r, y=ft_group.chir_mag, mode='lines', name=f"{key}", line=dict(color=colors_r[i % len(colors_r)])))
                             paths_plotted_r += 1

                 if paths_plotted_r > 0:
                     fig_indiv_r.update_xaxes(range=[plot_options['r_min'], plot_options['r_max']])
                     fig_indiv_r.update_layout(title=f"Individual Path |χ(R)|", xaxis_title="R (Å)", yaxis_title="|χ(R)| (arb. units)", legend_title='Path', hovermode='x unified', height=500, showlegend=True)
                     st.plotly_chart(fig_indiv_r, use_container_width=True)
                 else:
                     st.info("No individual path |χ(R)| data calculated or available to plot.")
             # --- End Individual Path k and R plots ---

        # --- Single Path Modification Plot ---
        if params['path_to_modify'] and params['path_to_modify'] in feff_paths and k_present_for_model:
            st.markdown("---")
            st.subheader(f"Single Path Modification Analysis: {params['path_to_modify']}")
            fig_mod = go.Figure()
            mod_colors = pcolors.qualitative.Set2

            # Plot original selected path (from individual_path_chis)
            if params['path_to_modify'] in individual_path_chis:
                orig_chi = individual_path_chis[params['path_to_modify']]
                k_weight_plot = params['ft_kweight']
                chi_plot_orig = orig_chi * (dat.k ** k_weight_plot)
                fig_mod.add_trace(go.Scatter(x=dat.k, y=chi_plot_orig, mode='lines', name='Original Path', line=dict(color='black', width=3)))

                # Plot modifications stored in session state
                mod_list = st.session_state.path_modifications.get(params['path_to_modify'], [])
                for i, mod in enumerate(mod_list):
                    mod_params = mod['params']
                    # Recalculate chi for this modification on the fly
                    temp_path = copy.deepcopy(feff_paths[params['path_to_modify']])
                    temp_path.s02 = mod_params['s02']
                    temp_path.degen = mod_params['degen']
                    temp_path.deltar = mod_params['deltar']
                    temp_path.sigma2 = mod_params['sigma2']
                    temp_path.e0 = mod_params['e0']
                    try:
                        path2chi(temp_path, k=dat.k) # Pass k explicitly
                        if hasattr(temp_path, 'chi'):
                            chi_plot_mod = temp_path.chi * (dat.k ** k_weight_plot)
                            param_str = f"S02={mod_params['s02']:.2f}, N={mod_params['degen']:.1f}, ΔR={mod_params['deltar']:.3f}, σ²={mod_params['sigma2']:.4f}, E₀={mod_params['e0']:.1f}"
                            fig_mod.add_trace(go.Scatter(x=dat.k, y=chi_plot_mod, mode='lines', name=f"Mod {i+1}",
                                                         line=dict(color=mod_colors[i % len(mod_colors)]),
                                                         customdata=[param_str]*len(dat.k), # Store params for hover
                                                         hovertemplate='<b>Mod '+str(i+1)+'</b><br>k=%{x:.2f}<br>k^w*chi=%{y:.3f}<br>Params: %{customdata}<extra></extra>'))
                    except Exception as e_mod_chi:
                        st.warning(f"Could not calculate chi for modification {i+1} of path {params['path_to_modify']}: {e_mod_chi}")

                fig_mod.update_xaxes(range=[plot_options['k_min'], plot_options['k_max']])
                fig_mod.update_layout(title=f"Effect of Parameters on Path {params['path_to_modify']}",
                                      xaxis_title="k (Å⁻¹)", yaxis_title=f"k^{k_weight_plot} * χ(k)",
                                      legend_title='Modification', hovermode='x unified', height=500, showlegend=True)
                st.plotly_chart(fig_mod, use_container_width=True)

            else:
                st.warning(f"Original chi data not available for selected path: {params['path_to_modify']}")


        # --- Optional: Display Processed Data Table ---
        st.markdown("---") # Add a separator
        if st.checkbox("Show Processed Data Table"):
            st.subheader("Processed Data")
            display_data = {}
            # Use getattr for safer access
            if getattr(dat, 'energy', None) is not None: display_data['Energy'] = dat.energy
            if getattr(dat, 'mu', None) is not None: display_data['mu'] = dat.mu
            if getattr(dat, 'pre_edge', None) is not None: display_data['Pre-Edge'] = dat.pre_edge
            if getattr(dat, 'post_edge', None) is not None: display_data['Post-Edge'] = dat.post_edge
            if getattr(dat, 'bkg', None) is not None: display_data['Background'] = dat.bkg
            if getattr(dat, 'norm', None) is not None: display_data['Normalized'] = dat.norm
            if getattr(dat, 'flat', None) is not None: display_data['Flattened'] = dat.flat
            if getattr(dat, 'k', None) is not None: display_data['k'] = dat.k
            if getattr(dat, 'chi_exp', None) is not None: display_data['chi_exp'] = dat.chi_exp # Exp chi
            if getattr(dat, 'chi_mod', None) is not None: display_data['chi_mod'] = dat.chi_mod # Model chi
            if getattr(dat, 'r', None) is not None: display_data['r'] = dat.r
            if getattr(dat, 'chir_mag', None) is not None: display_data['chir_mag'] = dat.chir_mag
            if getattr(dat, 'chir_re', None) is not None: display_data['chir_re'] = dat.chir_re
            if getattr(dat, 'chir_im', None) is not None: display_data['chir_im'] = dat.chir_im

            scalar_params = {}
            if hasattr(dat, 'e0'): scalar_params['E0'] = dat.e0
            if hasattr(dat, 'edge_step'): scalar_params['Edge_Step'] = dat.edge_step

            if display_data:
                try:
                    # Filter only array-like data and ensure they are not None
                    array_data = {k: v for k, v in display_data.items() if hasattr(v, '__len__') and v is not None}
                    if array_data:
                        # Check for empty arrays before calculating min_len
                        valid_arrays = {k: v for k, v in array_data.items() if len(v) > 0}
                        if valid_arrays:
                            min_len = min(len(v) for v in valid_arrays.values())
                            # Only include valid arrays in the DataFrame
                            df_display = pd.DataFrame({k: v[:min_len] for k, v in valid_arrays.items()})
                            st.dataframe(df_display)
                        else:
                            st.warning("Found data arrays, but they are empty.")

                        if scalar_params:
                            st.write("Calculated Parameters:")
                            st.json({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in scalar_params.items()})
                    elif scalar_params:
                        st.write("Calculated Parameters:")
                        st.json({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in scalar_params.items()})
                    else:
                        st.warning("No array data found to display in table.")
                except Exception as df_err:
                    st.warning(f"Could not display data table: {df_err}")
            elif scalar_params:
                st.write("Calculated Parameters:")
                st.json({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in scalar_params.items()})
            else:
                st.warning("No data attributes found to display in table.")

    except Exception as e:
        # Catch errors during processing/plotting specifically
        st.error(f"An error occurred during data processing or plotting for {filename_for_title}:")
        st.exception(e) # Shows the full traceback


# --- Main Execution Logic ---
if source_id and source_type:
    try:
        # Load data using the cached function
        dat_loaded = load_data(source_id, source_type)

        # --- Perform Fit Execution FIRST if button is pressed ---
        # This ensures fit results are available for the subsequent plot rendering pass
        if run_fit:
            if not paths_for_fit:
                st.error("Cannot run fit: No paths selected for fitting.")
            elif not feff_paths_dict:
                 st.error("Cannot run fit: Feff paths dictionary not loaded.")
            elif dat_loaded is None:
                st.error("Cannot run fit: Data failed to load.")
            else:
                # Ensure data has been processed enough to have k and chi_exp
                temp_data_group = copy.deepcopy(dat_loaded) # Use original loaded data
                fit_preprocessed_ok = False
                try:
                    # --- Rerun Preprocessing for Fit ---
                    # Use parameters directly from sidebar widgets for fit consistency
                    # Need to get these values again or ensure they are up-to-date
                    # (This section might need refinement if sidebar params change between page load and fit button press)
                    # Using the values derived earlier (e.g., e0_val, step_val) should be okay for now.
                    st.write("--- Running Preprocessing for Fit ---") # Debug
                    pre_edge(temp_data_group, group=temp_data_group,
                             e0=e0_val, step=step_val, pre1=pre1_val, pre2=pre2_val,
                             norm1=norm1_val, norm2=norm2_val, nnorm=nnorm_val)
                    autobk(temp_data_group, group=temp_data_group,
                           rbkg=rbkg_val, kweight=kweight_autobk_val)
                    if hasattr(temp_data_group, 'k') and hasattr(temp_data_group, 'chi') and len(temp_data_group.k)>0:
                        temp_data_group.chi_exp = temp_data_group.chi # Store exp chi here for dataset
                        fit_preprocessed_ok = True
                        st.write("Fit Preprocessing Succeeded.") # Debug
                    else:
                        st.error("Fit pre-processing failed: Could not generate valid k or chi array.")

                except Exception as proc_err:
                    st.error(f"Error during pre-processing for fit: {proc_err}")
                    temp_data_group = None # Prevent fit if pre-processing failed

                if fit_preprocessed_ok and temp_data_group:
                    st.session_state.fit_result_obj = None # Clear previous result object

                    params_group = Parameters()
                    fit_path_objs = []
                    valid_paths_for_fit_found = False

                    st.write("--- Building Fit Parameters and Paths ---") # Debug
                    st.write("Paths selected for fit:", paths_for_fit) # Debug
                    st.write("Parameter Settings Dict:", fit_param_settings) # Debug

                    # Define parameters and paths for fit
                    # Iterate over the paths selected in the multiselect FOR THIS RUN
                    for path_key in paths_for_fit:
                        if path_key in feff_paths_dict:
                            p = copy.deepcopy(feff_paths_dict[path_key])
                            # Use the fit_param_settings dict populated when controls were drawn
                            path_settings = fit_param_settings.get(path_key)
                            path_prefix = path_key.replace(".dat", "") # Unique prefix for params

                            if path_settings:
                                st.write(f"Processing path {path_key} with settings: {path_settings}") # Debug
                                # Add parameters to group and assign names to path object
                                for param_name in ['s02', 'degen', 'deltar', 'sigma2', 'e0']:
                                    settings = path_settings.get(param_name)
                                    if settings:
                                        param_larch_name = f"{param_name}_{path_prefix}"
                                        params_group.add(param_larch_name, value=settings['guess'], vary=settings['vary'])
                                        if param_name == 's02': params_group[param_larch_name].min = 0
                                        if param_name == 'degen': params_group[param_larch_name].min = 0
                                        if param_name == 'sigma2': params_group[param_larch_name].min = 0
                                        setattr(p, param_name, param_larch_name)
                                fit_path_objs.append(p)
                                valid_paths_for_fit_found = True
                            else:
                                st.warning(f"Settings NOT found for path {path_key} in fit_param_settings dictionary. Skipping this path in fit.") # Changed Debug
                        else:
                            st.warning(f"Path key '{path_key}' selected for fit not found in loaded Feff paths. Skipping.")

                    if valid_paths_for_fit_found:
                        st.write("--- Creating Transform and Dataset ---") # Debug
                        # Create transform for R-space fit
                        transform = feffit_transform(kmin=ft_kmin_val, kmax=ft_kmax_val, # Use FT params from sidebar
                                                     kw=kweight_ft_val, dk=ft_dk_val,  # CORRECTED variable name here
                                                     window=window_val, rmin=fit_rmin, rmax=fit_rmax,
                                                     fitspace='r')
                        # Create dataset
                        dset = feffit_dataset(data=temp_data_group, pathlist=fit_path_objs, transform=transform)

                        # Run the fit
                        st.write("--- Running Feffit ---") # Debug
                        st.write("Parameters before fit:", params_group) # Debug
                        try:
                            with st.spinner("Running fit..."):
                                result = feffit(params_group, dset)
                            st.session_state.fit_result_obj = result # Store the full fit result object
                            st.success("Fit completed!")
                            st.write("Fit Result Object:", result) # Debug
                            st.rerun() # Rerun immediately to show results
                        except Exception as fit_err:
                            st.error(f"Error during feffit execution: {fit_err}")
                            st.exception(fit_err) # Show traceback
                            st.session_state.fit_result_obj = None # Ensure it's None if fit fails

                    else:
                        st.error("Fit aborted: No valid paths with parameter settings found.")
                elif not fit_preprocessed_ok:
                    # Error message handled above
                    pass
        # --- End of Fit Execution Block ---

        # --- Processing and Plotting (Runs on every script run) ---
        if dat_loaded is not None:
            dat_to_process = copy.deepcopy(dat_loaded) # Use a fresh copy for plotting

            # Collect parameters from sidebar controls FOR PLOTTING
            current_params = {
                'e0': e0_val, 'step': step_val, 'pre1': pre1_val, 'pre2': pre2_val,
                'norm1': norm1_val, 'norm2': norm2_val, 'nnorm': nnorm_val,
                'use_custom_pre': use_custom_pre, 'use_custom_norm': use_custom_norm,
                'rbkg': rbkg_val, 'autobk_kweight': kweight_autobk_val,
                'ft_kmin': ft_kmin_val, 'ft_kmax': ft_kmax_val, 'ft_dk': ft_dk_val,
                'ft_window': window_val, 'ft_kweight': kweight_ft_val,
                'path_to_modify': path_to_modify,
                'paths_for_fit': paths_for_fit # Technically not needed by plot func, but good to have context
            }
            # Collect plotting options (including shell visibility)
            current_plot_options = {
                'plot_start_rel': plot_start_rel, 'plot_end_rel': plot_end_rel,
                'k_min': k_min_plot_val, 'k_max': k_max_plot_val,
                'r_min': r_min_plot_val, 'r_max': r_max_plot_val,
                'show_re': show_chir_re, 'show_im': show_chir_im,
                'show_geom': show_geom, 'show_scat': show_scat, 'show_indiv_chi': show_indiv_chi,
                'shell_visibility': shell_visibility # Pass shell toggles
            }

            # Call the processing and plotting function
            process_and_plot(dat_to_process, feff_paths_dict, selected_feff_paths, # Use selected_feff_paths for MODEL plot
                             shell_data, absorber_atom_info,
                             display_filename, current_params, current_plot_options,
                             fit_result_obj=st.session_state.fit_result_obj) # Pass fit result object
        else:
             st.warning("Data not loaded, cannot process or plot.")


    except FileNotFoundError as e:
        st.error(f"Error loading data file '{display_filename}': {e}")
        # Clear potentially problematic state if file not found
        st.session_state.current_data_basename = None
        st.session_state.current_source_type = None
        st.session_state.fit_result_obj = None
    except Exception as e:
        st.error(f"An unexpected error occurred in the main execution block:")
        st.exception(e) # Shows the full traceback

else:
    # Only show this if user upload is selected but no file is uploaded yet,
    # OR if an example was selected but the path became invalid
    if selected_source == UPLOAD_OPTION and uploaded_file is None:
        st.info("Upload a data file using the sidebar to begin processing.")
    elif selected_source != UPLOAD_OPTION and source_type == 'path' and (source_id is None or not os.path.exists(source_id)):
        st.error(f"Selected example file path seems invalid or file not found: {display_filename}")
    else:
        # Default message if no source is identified for any other reason
        st.info("Select or upload a data source in the sidebar to begin.")


# --- Display Fit Report and Comparison Plot ---
# This section runs AFTER the main plotting section in the Streamlit execution flow
st.markdown("---")
st.subheader("Step 5: Fitting Results")
# Check the fit_result_obj for the actual fit object
if st.session_state.get('fit_result_obj') is not None: # Use .get for safety
    st.markdown("**Fit Statistics and Best-Fit Parameters:**")
    try:
        # Pass the full fit result object to the report function
        report_str = feffit_report(st.session_state.fit_result_obj)
        st.text(report_str)

        # Add a dedicated plot comparing data and fit in R space
        st.markdown("**Fit Comparison in R-Space**")
        fig5 = go.Figure()
        fit_obj = st.session_state.fit_result_obj # Get the full fit result object

        if hasattr(fit_obj, 'datasets') and len(fit_obj.datasets)>0:
            dset = fit_obj.datasets[0] # Assuming one dataset for now
            data_group_present = hasattr(dset, 'data')
            model_group_present = hasattr(dset, 'model')

            if data_group_present and model_group_present:
                # Check attributes needed for plotting exist
                data_r_present = hasattr(dset.data, 'r') and hasattr(dset.data, 'chir_mag') and dset.data.r is not None
                model_r_present = hasattr(dset.model, 'r') and hasattr(dset.model, 'chir_mag') and dset.model.r is not None

                # Plot Data Magnitude
                if data_r_present:
                    fig5.add_trace(go.Scatter(x=dset.data.r, y=dset.data.chir_mag, mode='lines', name='|χ(R)| Exp', line=dict(color='darkgreen')))
                else: st.warning("Experimental |χ(R)| not found in fit dataset for plotting.")

                # Plot Fit Magnitude
                if model_r_present:
                    fig5.add_trace(go.Scatter(x=dset.model.r, y=dset.model.chir_mag, mode='lines', name='|χ(R)| Best Fit', line=dict(color='red')))
                else: st.warning("Best Fit |χ(R)| not found in fit dataset for plotting.")

                # Get plot options (might not be available if script reran after fit)
                # It's safer to use the fit range directly if needed
                show_re_opt = show_chir_re # Use variable defined earlier in script run
                show_im_opt = show_chir_im

                # Optionally add Real parts if toggled in plot options
                if show_re_opt:
                    if data_r_present and hasattr(dset.data, 'chir_re'):
                        fig5.add_trace(go.Scatter(x=dset.data.r, y=dset.data.chir_re, mode='lines', name='Re[χ(R)] Exp', line=dict(color='darkgreen', dash='dot'), visible='legendonly'))
                    if model_r_present and hasattr(dset.model, 'chir_re'):
                        fig5.add_trace(go.Scatter(x=dset.model.r, y=dset.model.chir_re, mode='lines', name='Re[χ(R)] Best Fit', line=dict(color='red', dash='dot'), visible='legendonly'))
                # Optionally add Imaginary parts if toggled
                if show_im_opt:
                    if data_r_present and hasattr(dset.data, 'chir_im'):
                        fig5.add_trace(go.Scatter(x=dset.data.r, y=dset.data.chir_im, mode='lines', name='Im[χ(R)] Exp', line=dict(color='darkgreen', dash='dashdot'), visible='legendonly'))
                    if model_r_present and hasattr(dset.model, 'chir_im'):
                        fig5.add_trace(go.Scatter(x=dset.model.r, y=dset.model.chir_im, mode='lines', name='Im[χ(R)] Best Fit', line=dict(color='red', dash='dashdot'), visible='legendonly'))

                # Only update axis range and plot if data was plotted
                if data_r_present or model_r_present:
                    # Use fit_rmin/fit_rmax defined earlier in script run
                    fig5.update_xaxes(range=[fit_rmin, fit_rmax])
                    fig5.update_layout(title=f"Fit Comparison in R-Space: {display_filename}", xaxis_title="R (Å)", yaxis_title="χ(R) Magnitude (arb. units)", legend_title='Legend', hovermode='x unified', height=500, showlegend=True)
                    st.plotly_chart(fig5, use_container_width=True)
                else:
                    st.warning("No R-space data found in fit dataset to plot comparison.")

            else:
                st.warning("Could not find data or model groups within the fit dataset.")
        else:
            st.warning("Could not find dataset information in the fit result object.")

    except Exception as report_err:
        st.error(f"Could not generate fit report or plot: {report_err}")
        # Try to display raw parameters if report fails
        if st.session_state.fit_result_obj and hasattr(st.session_state.fit_result_obj, 'params'):
            try:
                st.write("Fit Parameters object (raw):")
                st.write(st.session_state.fit_result_obj.params)
            except Exception as e_raw:
                 st.error(f"Could not display raw fit parameters: {e_raw}")

else:
    st.info("Run a fit using the 'Fitting Setup' controls in the sidebar to see results here.")