import numpy as np
import mrcfile
import cv2
import matplotlib.pyplot as plt
import h5py

def save_to_hdf5(image_path, mask_path, output_hdf5_path):
    # Load image and mask using PIL or any other image loading library
    image = mrcfile.read(image_path)
    mask = mrcfile.read(mask_path)

    # Convert image and mask to NumPy arrays
    image_array = np.array(image)
    mask_array = np.array(mask)

    # Create or open an HDF5 file
    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        # Create datasets for image and mask
        hdf5_file.create_dataset('raw', data=image_array, dtype='float32')
        hdf5_file.create_dataset('label', data=mask_array, dtype='float32')
    hdf5_file.close()

if __name__ == "__main__":
    # Change path
    image_path = './out/Gaussian/out_rec3d_G=2.5_type=1.mrc'
    mask_path = './SNR_0.5_5t2c_5lqw_202401160153/tomos/tomo_den_0.mrc'
    output_hdf5_path = './data.h5'

    save_to_hdf5(image_path, mask_path, output_hdf5_path)

