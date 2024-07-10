# Accurate Prediction of NMR Chemical Shifts: Integrating DFT Calculations with Three-Dimensional Graph Neural Networks 
This is the official implementation of the CSTShift model from: [Accurate Prediction of NMR Chemical Shifts: Integrating DFT Calculations with Three-Dimensional Graph Neural Networks](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00422).

Workflow of our CSTShift model:
<figure>
  <img
  src="fig/fig2.svg">
</figure>

## Environment setup
We recommand to use conda to create a new environment first. Here's the command to create a new environment and install the required packages on Linux: 

```bash
conda create -n cstshift python=3.9
conda activate cstshift
conda install -c conda-forge rdkit
conda install numpy pandas PyYAML tensorboard
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch_geometric==2.1.0.post1 torch-sparse==0.6.15 torch-cluster==1.6.1 torch-scatter==2.0.9
```

If you need to install the package of a specific version or on other platforms, please refer to the official website of the package for more details.

## Data perparation
CHESHIRE dataset is provided in data/CHESHIRE. Other datasets of NMRShiftDB2, TIC-10 and NHP will be provided later. You can also implement your own dataset class similar to `NMRDataset` if you want to train or evaluate the model on your own dataset.

```bash
custom_dataset
├── processed
│   ├── processed.pt
│   ├── split.pt
├── raw
│   ├── NMR_output
│   │   ├── a.log
│   │   ├── ...
│   ├── nmr_shift.csv
```

To use the model to predict chemical shifts for your own dataset, you need to provide the DFT calculation of 3D geometry and shielding tensors using Gaussian or other quantum chemistry software. Recommanded process to obtain the calculated 3D geometry and shielding tensors is provided in the jupyter notebook `data_preparation_example.ipynb`.


## Run prediction with processed dataset

With our provided trained models, you can directly run the prediction on the processed dataset by setting all args in a yaml file and run with the following command. The trained models are accessible at https://yzhang.hpc.nyu.edu/IMA . You may want to create a new folder `models` under the root directory and download the trained models to the folder, or you can look into the config file and change the corresponding args.

```bash
python src/test.py --test_config configs/embC_test_cheshire.yaml
```

A new folder will be generated, where the args, log file and predicted results will be saved. You can find the example yaml file under `configs/`. The prediction could be accessed in the csv file if save_csv is set to True, or in the `.pt` file as the torch tensor. `nmrshiftdb2_example_results_C/` and `nmrshiftdb2_example_results_H/` give examples of the test results of ensembled concat_emb models on NMRShiftDB2-DFT test set. Set `labeled_data=False` if you want to provide prediction on your own generated dataset without given experimental shift value.

## Train a new model and run prediction
CSTShift model architecture details：
<figure>
  <img
  src="fig/SIfig_model.svg">
</figure>

You can use `train.py` to train a new model: 
  
  ```bash
  python src/train.py --config_name configs/concat_emb.yaml
  ```

`test.py` is used to run prediction on the test set:

  ```bash
  python src/test.py --test_config YOUR_TEST_CONFIG
  ```
