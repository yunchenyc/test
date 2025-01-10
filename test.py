import h5py
from pynwb import NWBHDF5IO
import os

# dirpath = '/AMAX/cuihe_lab/share_rw/Neucyber-NC-2024-A-01/Abel/Data_recording/20240924_Interception_001/formatted_data'
# filepath = os.path.join(dirpath, 'neural_data.nwb')
# io = NWBHDF5IO(filepath, mode='r')

# nwbfile = io.read()

dirpath = '/AMAX/cuihe_lab/share_rw/CuiLab-Database/interception/Abel/data_recording/20240501_Interception_001/formatted_data'
filepath = os.path.join(dirpath, 'standard_data.nwb')
io = NWBHDF5IO(filepath, mode='r')

nwbfile = io.read()

# event_marker = nwbfile.processing['behavior']['MonkeyLogicEvents'].data[:]
# event_time = nwbfile.processing['behavior']['MonkeyLogicEvents'].timstamps[:]
# trials = nwbfile.trials.to_dataframe()

print(1)