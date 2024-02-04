import os, sys
import aitom.filter.gaussian as G
import numpy as np
import aitom.io.mrcfile_proxy as mrcfile_proxy
import aitom.io.file as io_file
import time
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import aitom.image.vol.util as GV
import mrcfile

save_dir = './out'

def write_with_spacing(data, voxel_spacing, path, overwrite=True):
    # only for 3D array
    assert data.ndim == 3
    data = data.astype('float32')
    # transpose data according to tomominer.image.vol.eman2_util.em2numpy
    data = data.transpose([2, 1, 0])
    with mrcfile.new(path, overwrite=overwrite) as m:
        m.set_data(data)
        m.voxel_size = voxel_spacing

def g_denoising(G_type, a, voxel_spacing, name, gaussian_sigma, save_flag=False):
    b_time = time.time()

    if G_type == 1:
        a = G.smooth(a, gaussian_sigma)
    elif G_type == 2:
        a = G.dog_smooth(a, gaussian_sigma)
    elif G_type == 3:
        a = G.dog_smooth__large_map(a, gaussian_sigma)

    end_time = time.time()
    print('Gaussian de-noise takes', end_time - b_time, 's', ' sigma=', gaussian_sigma)

    if save_flag:
        img = (a[:, :, int(a.shape[2] / 2)]).copy()
        img_path = save_dir + '/Gaussian/' + str(name) + '_G=' + \
                   str(gaussian_sigma) + '_type=' + str(G_type) + '.png'
        plt.imsave(img_path, img, cmap='gray')

        mrc_path = save_dir + '/Gaussian/' + str(name) + '_G=' + \
                   str(gaussian_sigma) + '_type=' + str(G_type) + '.mrc'
        io_file.put_mrc_data(a, mrc_path)
        write_with_spacing(a, voxel_spacing, mrc_path)

        return img


def bandpass_denoising(a, voxel_spacing, name, save_flag=False):
    b_time = time.time()
    grid = GV.grid_displacement_to_center(a.shape, GV.fft_mid_co(a.shape))
    rad = GV.grid_distance_to_center(grid)
    rad = np.round(rad).astype(int)

    # create a mask that only center frequencies components will be left
    curve = np.zeros(rad.shape)
    # TODO: change the curve value as desired
    curve[int(rad.shape[0] / 8) * 3: int(rad.shape[0] / 8) * 5, int(rad.shape[1] / 8) * 3: int(rad.shape[1] / 8) * 5,
    int(rad.shape[2] / 8) * 3: int(rad.shape[2] / 8) * 5] = 1

    #perform FFT and filter the data with the mask and then transform the filtered data back
    vf = ifftn(ifftshift((fftshift(fftn(a)) * curve)))
    vf = np.real(vf)
    
    end_time = time.time()
    print('Bandpass de-noise takes', end_time - b_time, 's')

    if save_flag:
        img = (vf[:, :, int(vf.shape[2] / 2)]).copy()
        # TODO: Change the image and tomogram saving path
        img_path = save_dir + '/Bandpass/' + str(name) + '_BP.png'
        plt.imsave(img_path, img, cmap='gray')

        mrc_path = save_dir + '/Bandpass/' + str(name) + '_BP.mrc'
        io_file.put_mrc_data(vf, mrc_path)
        write_with_spacing(vf, voxel_spacing, mrc_path)

        return img

if __name__ == "__main__":
    path = './SNR_0.5_5t2c_5lqw_202401160153/tem/out_rec3d.mrc'
    name = 'out_rec3d'
    G_type = 1

    # read the volume data as numpy array
    original = mrcfile_proxy.read_data(path)

    mrc_header = io_file.read_mrc_header(path)
    voxel_spacing = mrc_header['MRC']['xlen'] / mrc_header['MRC']['nx']
    print(voxel_spacing)

    plt.imsave('./out/'+name+'.png', original[:, :, int(original.shape[2] / 2)], cmap='gray')

    g_fimg = g_denoising(G_type, a=original, voxel_spacing=voxel_spacing, name=name, gaussian_sigma=2.5, save_flag=True)

    bp_fimg = bandpass_denoising(a=original, voxel_spacing=voxel_spacing, name=name, save_flag=True)