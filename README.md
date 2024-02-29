# Improving NMR chemical shift prediction by integrating DFT calculation with 3D graph neural network 

<figure>
  <img
  src="fig/fig2.svg">
  <figcaption>Workflow of our CSTShift model</figcaption>
  <style>
    figure {
      text-align: center;
    }
    figcaption {
      text-align: center;
    }
  </style>
</figure>

## Environment setup
### If you are using Linux and Conda

### If you are on other systems or are using other package managers
please install the following packages by checking their websites.

## Data perparation
Datasets of NMRShiftDB2, CHESHIRE, TIC-10 and NHP will be provided later. Check the case study dataset provided here for a general idea of the data format. You can also implement your own dataset class similar to `NMRDataset` if you want to train or evaluate the model on your own dataset.

```bash
custom_dataset
├── processed
│   ├── processed.pt
│   ├── split.pt
├── raw
│   ├── NMR_output
│   │   ├── TIC10a.log
│   │   ├── ...
│   ├── experimental_data.csv
```

To use the model to predict chemical shifts for your own dataset, you need to provide the DFT calculation of 3D geometry and shielding tensors using Gaussian or other quantum chemistry software. Recommanded process to obtain the calculated 3D geometry and shielding tensors is provided in the jupyter notebook `data_preparation_example.ipynb`.


## Run prediction with processed dataset

With our provided trained models, you can directly run the prediction on the processed dataset by setting all args in a yaml file and run with the following command.

```bash
python src/predict.py --ens_config configs/ens_test.yaml
```

A new folder will be generated, where the args, log file and predicted results will be saved. You can find the example yaml file under `configs/`. The prediction could be accessed in the csv file if save_csv is set to True, or in the `.pt` file as the torch tensor.

## Train a new model and run prediction

<figure>
  <img
  src="fig/SI_fig_model.drawio.svg">
  <figcaption>CSTShift model architecture details</figcaption>
  <style>
    figure {
      text-align: center;
    }
    figcaption {
      text-align: center;
    }
  </style>
</figure>

You can use `train.py` to train a new model: 
  
  ```bash
  python src/train.py --config_name configs/train.yaml
  ```

`test.py` is used to run prediction on the test set (ensemble model is not supported here):

  ```bash
  python src/test.py --folder_name TRAINED_MODEL_FOLDER --explicit_ds_config configs/test_case_study.yaml --ignore_val
  ```