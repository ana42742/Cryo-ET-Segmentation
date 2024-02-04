import h5py
import matplotlib.pyplot as plt

def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as hdf5_file:
        # List all datasets in the HDF5 file
        print("Datasets in HDF5 file:")
        for dataset_name in hdf5_file.keys():
            print(dataset_name)

            # Read data from a specific dataset (replace 'dataset_name' with the actual dataset name)
            dataset = hdf5_file[dataset_name]
            data = dataset[:]  # Read all data from the dataset into a NumPy array
            # Alternatively, you can use dataset[0:10] to read a specific range of data

            # Print information about the dataset
            print("\nInformation about the dataset:")
            print("Shape:", data.shape)
            print("Data type:", data.dtype)

            #Pick any slice of the 3D data
            plt.imshow(data[203], cmap='gray')
            # Close the HDF5 file (automatically done using 'with' statement)


hdf5_file_path = 'path/to/file'
read_hdf5(hdf5_file_path)