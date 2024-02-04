# Cryo-ET Segmentation

## Steps to perform 3D segmentation using U-Net:
-  Clone repository:
    ```
    git clone https://github.com/ana42742/Cryo-ET-Segmentation.git
    cd Cryo-ET-Segmentation
    ```
    Create an environment and install dependencies from `requirements.txt`. Activate the environment.

-  Download the dataset:
    ```
    gdown 16ee1_hY_PLgkSCgXTi58eLmh99KUOvuR
    gdown 1V1Ov9DyXwDB3DTtmSOnZVfhJhH-pfcgN
    gdown 1rYsulp0IghqmzsuWO8c_aHtyf9fc9i7z

    unzip ./SNR_0.5_new.zip
    rm -R ./SNR_0.5_new.zip
    ```

-  Setup 3D-Unet
    ```
    cd ..
    git clone https://github.com/wolny/pytorch-3dunet.git
    cd pytorch-3dunet
    python setup.py install
    ```

    Install all required dependencies (as Import errors arise)

-  Convert original mrcfile to hdf5 format (make sure to update file paths in `process.py`)
    ```
    cd ../Cryo-ET-Segmentation
    python3 process.py
    ```

-  Download the pre-trained weights for the model from [here](https://oc.embl.de/index.php/s/61s67Mg5VQy7dh9/download?path=%2FArabidopsis-Ovules%2Funet_bce_dice_ds2x&files=best_checkpoint.pytorch).
    Update `model_path`, `output_dir` and `file_paths` parameters in the `test_config.yml` file in `pytorch-3dunet`. Run the following to perform predictions (update number of CUDA devices as per preference):
    ```
    cd ../pytorch-3dunet
    CUDA_VISIBLE_DEVICES=2, predict3dunet --config test_config.yml
    ```

- Denoise dataset by running the following (make sure to update file paths in the scripts):
    ```
    cd ../Cryo-ET-Segmentation
    python3 Denoise_3D.py
    python3 process.py
    ```
- `cd` into `pytorch-3dunet`, edit the `file_paths` in `test_config.yml` to link to the denoised file and run predictions again as per the previous step.

NOTE: You can visualise your results by running `python3 visualise_hdf.py`

    