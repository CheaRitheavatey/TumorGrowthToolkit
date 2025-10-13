#%%
from TumorGrowthToolkit.FK import Solver as FKSolver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage
import nibabel as nib

# Apply a Gaussian filter for smooth transitions
wm_data = nib.load('/mnt/8tb_slot8/jonas/datasets/tau_pet_Archive/sub-6377/ses-2018-08-17/sub-6377_ses-2018-08-17_WMProb.nii.gz').get_fdata()*1.0
gm_data = nib.load('/mnt/8tb_slot8/jonas/datasets/tau_pet_Archive/sub-6377/ses-2018-08-17/sub-6377_ses-2018-08-17_GMProb.nii.gz').get_fdata()*1.0
tau_data = nib.load('/mnt/8tb_slot8/jonas/datasets/tau_pet_Archive/sub-6377/ses-2018-08-17/sub-6377_ses-2018-08-17_pet-AV1451_INFCER_SUVR.nii.gz').get_fdata()*1.0

#all nan to 0 in tau data
tau_data = np.nan_to_num(tau_data, nan=0.0)
wm_data = np.nan_to_num(wm_data, nan=0.0)
gm_data = np.nan_to_num(gm_data, nan=0.0)

brainmask = (wm_data + gm_data) > 0.01

tau_data = tau_data * brainmask - 1 
tau_data[tau_data < 0] = 0
tau_data[tau_data > 1] = 1

#set all the border pixels to 0
dilationMask = scipy.ndimage.binary_dilation(1.0-brainmask, iterations=2)

tau_data[dilationMask==1] = 0
testGM = gm_data * (1.0 - dilationMask)

gm_data[brainmask==0] = 0
wm_data[brainmask==0] = 0


plt.imshow(tau_data[:,:,50], cmap='hot')
plt.title("Tau Data Slice")
plt.colorbar()
#
plt.imshow((gm_data -testGM)[:,:,50], cmap='gray')

#%%
plt.imshow((wm_data + gm_data)[:,:,50], cmap='gray')
plt.colorbar()
#%%
# Set up parameters
parameters = {
    'Dw': 0.7,          # Diffusion coefficient for white matter
    'rho': 0.30,         # Proliferation rate
    'RatioDw_Dg': 1.0,  # Ratio of diffusion coefficients in white and grey matter
    'gm': gm_data,      # Grey matter data
    'wm': wm_data,      # White matter data
    'NxT1_pct': 0.3,    # tumor position [%]
    'NyT1_pct': 0.7,
    'NzT1_pct': 0.5,
    'init_scale': tau_data, # this is a gaussian at the starting location or a 3D array with the initial condition
    'resolution_factor': 1, #resultion scaling for calculations
    'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
    'verbose': True, #printing timesteps 
    'time_series_solution_Nt': 10, # number of timesteps in the output
    'stopping_time': 10 # in days
    
}

# Create custom color maps
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)

cmap2 = plt.get_cmap('jet')

# Calculate the slice index
NzT = int(parameters['NzT1_pct'] * gm_data.shape[2])

# Plotting function
def plot_tumor_states(wm_data, initial_state, final_state, slice_index):
    plt.figure(figsize=(12, 6))

    # Plot initial state
    plt.subplot(1, 2, 1)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(initial_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Initial Tumor State")

    # Plot final state
    plt.subplot(1, 2, 2)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(final_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Final Tumor State")
    plt.show()
    

def plot_time_series(wm_data, time_series_data, slice_index, coronal_slice_index):
    plt.figure(figsize=(24, 12))

    # Generate 8 indices evenly spaced across the time series length
    time_points = np.linspace(0, time_series_data.shape[0] - 1, 8, dtype=int)

    for i, t in enumerate(time_points):
        plt.subplot(2, 4, i + 1)  # 2 rows, 4 columns, current subplot index
        plt.imshow(np.rot90(wm_data[:, :, slice_index], k=1), cmap=cmap1, vmin=0, vmax=1, alpha=1)
        plt.imshow(np.rot90(time_series_data[t, :, :, slice_index], k=1), cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
        plt.title(f"Axial Time Slice {t + 1}")

    plt.tight_layout()
    plt.show()

    # Add coronal plot along y-axis
     
    plt.figure(figsize=(24, 12))

    for i, t in enumerate(time_points):
        plt.subplot(2, 4, i + 1)  # 2 rows, 4 columns, current subplot index
        plt.imshow(np.rot90(wm_data[:, coronal_slice_index, :], k=1), cmap=cmap1, vmin=0, vmax=1, alpha=1)
        plt.imshow(np.rot90(time_series_data[t, :, coronal_slice_index, :], k=1), cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
        plt.title(f"Coronal Time Slice {t + 1}")

    plt.tight_layout()
    plt.show()
# %%
# Run the FK_solver and plot the results
start_time = time.time()
fk_solver = FKSolver(parameters)
result = fk_solver.solve()
end_time = time.time()  # Store the end time
execution_time = int(end_time - start_time)  # Calculate the difference

#%%
NzT = 32

print(f"Execution Time: {execution_time} seconds")
if result['success']:
    print("Simulation successful!")
    #plot_tumor_states(wm_data, result['initial_state'], result['final_state'], NzT)
    plot_time_series(wm_data,result['time_series'], NzT, 50)
else:
    print("Error occurred:", result['error'])

#save nii file for the first 3 images of the time series
nib.save(nib.Nifti1Image(result['time_series'][0], np.eye(4)), 'initial_tumor.nii.gz')
nib.save(nib.Nifti1Image(result['time_series'][1], np.eye(4)), '1_mid_tumor.nii.gz')
nib.save(nib.Nifti1Image(result['time_series'][2], np.eye(4)), '2_mid_tumor.nii.gz') 
# %%
result
# %%
