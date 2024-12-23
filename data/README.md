## How to use Graph Exporter
1. Convert raw data to EDF (if not already in EDF format)
- Use the WFDB package to read MATLAB files.
- Use the `EdfWriter` function from the `pyedflib` package to convert them into EDF format.
```
import wfdb
from pyedflib import EdfWriter
import numpy as np

# Example: Converting a MATLAB file to EDF
mat_file = "path_to_mat_file.mat"
edf_file = "output_file.edf"

# Load data using WFDB (update keys based on your MATLAB file structure)
data = wfdb.io.loadmat(mat_file)
signal_data = data['signal']  # Replace with the actual key for signal data
channel_labels = ['ch1', 'ch2', 'ch3']  # Replace with your actual channel names

# EDF writer setup
n_channels = len(channel_labels)
signal_headers = [
    {
        "label": label,
        "dimension": "uV",
        "sample_rate": 100,  # Update with actual sampling rate
        "physical_min": -200,
        "physical_max": 200,
        "digital_min": -32768,
        "digital_max": 32767,
        "transducer": "",
        "prefilter": ""
    } for label in channel_labels
]

with EdfWriter(edf_file, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
    writer.setSignalHeaders(signal_headers)
    writer.writeSamples(np.array(signal_data).T)

print(f"Converted {mat_file} to {edf_file}")
```
2. Create a configuration file for data alignment
- Open the EDF data file to verify channel names and types.
- Create a custom configuration file for adjusting channel positions.

3. 3. Build and run the container
- For KISS dataset
```bash
docker build -t graph-exporter-convert . && \
docker run -it --name graph-exporter-convert \
-e TYPE=KISS \
-v /edfdata_dir:/input \
-v /img_dir:/output \
graph-exporter-convert
```

- For SHHS2 dataset
```bash
docker build -t graph-exporter-convert . && \
docker run -it --name graph-exporter-convert \
-e TYPE=shhs2 \
-v /edfdata_dir:/input \
-v /img_dir:/output \
graph-exporter-convert
```
