# SRToANet

Codes for "Super-Resolution ToA estimation using Neural Networks", EUSIPCO 2020

## 🎓 Enhanced Features for Undergraduate Thesis

This repository has been extended with the following improvements:

### 1. **Attention Mechanism** (`--use_attention`)
- **Description**: Incorporates Attention Gates into the U-Net skip connections, allowing the model to automatically focus on critical signal features (e.g., first-path components) while suppressing noise and multipath interference.
- **Benefit**: Improves ToA estimation accuracy, especially in low SNR scenarios.

### 2. **End-to-End Joint Optimization** (`--use_e2e`)
- **Description**: After the standard two-stage training (SR + Regressor), performs an additional joint fine-tuning phase where all networks (G, RA, RB) are optimized together. This allows the super-resolution network to learn features that are specifically beneficial for localization, rather than just waveform reconstruction.
- **Benefit**: Bridges the gap between SR quality and ToA precision, often yielding lower RMSE.

### 3. **Reproducibility** (`--seed`)
- **Description**: Global random seed control for all random libraries (Python, NumPy, PyTorch CPU/GPU) to ensure experimental reproducibility.
- **Benefit**: Essential for academic research and thesis validation.

## Install

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> `requirements_old.txt` is the original code repository runnable environment, but the old torch version is incompatible with modern gpu's, so it's only kept here as a backup.

## Data Preperation

All the information of channels (time delays, complex attenuation) are stored in `data/Pathset_train.mat`, `data/Pathset_test.mat`, and `data/Pathset_test_802.mat`

Simply run `data/CIR_Generation.m` will generate the dataset for training and testing, stored in `data/traindata` and `data/testdata` respectively

## Interpolation module

Require installation of Torchinterp1d. See [https://github.com/aliutkus/torchinterp1d](https://github.com/aliutkus/torchinterp1d)

## Train the model

Run `train.py` to train the model. It will first train the super-resolution network and then the two regressors. Optionally, you can enable attention mechanism and/or end-to-end joint optimization.

### Basic Usage
```bash
# Baseline training (original method)
python train.py --name baseline_exp --snr high --device gpu

# Training with Attention Mechanism
python train.py --name attention_exp --snr high --use_attention True --device gpu

# Training with End-to-End Optimization
python train.py --name e2e_exp --snr high --use_e2e True --e_e2e 100 --lr_e2e 1e-4 --device gpu

# Training with Both Enhancements
python train.py --name full_exp --snr high --use_attention True --use_e2e True --device gpu
```

### Resume Training
The training script automatically detects existing weights:
- If original weights (`sr.w`, `ra.w`, `rb.w`) exist and `--use_e2e=False`, training is skipped.
- If original weights exist and `--use_e2e=True`, the script loads them and proceeds to E2E fine-tuning.
- If E2E weights (`sr_e2e.w`, etc.) already exist, training is skipped entirely.

### Command-line Arguments

    usage: train.py [-h] [--name NAME] [--snr {high,low}] [--bandwidth BANDWIDTH]
                [--e_sr E_SR] [--lr_sr LR_SR] [--batch_sr BATCH_SR]
                [--lam LAM] [--up UP] [--e_reg E_REG] [--lr_reg LR_REG]
                [--batch_reg BATCH_REG] [--use_ori [USE_ORI]]
                [--window_len WINDOW_LEN] [--window_index WINDOW_INDEX]
                [--device {cpu,gpu}] [--print_interval PRINT_INTERVAL]

    optional arguments:
      -h, --help            show this help message and exit
      --name NAME           name for the experiment folder
      --snr {high,low}      select high SNR scenario or low SNR scenario
      --bandwidth BANDWIDTH
                        define the system bandwidth
      --e_sr E_SR           number of epochs to train the super resolution network
      --lr_sr LR_SR         learning rate for the super resolution network
      --batch_sr BATCH_SR   batchsize for the super resolution network
      --lam LAM             weight for the time domain loss
      --up UP               upsamping rate for the super resolution net
      --e_reg E_REG         number of epochs to train the super resolution network
      --lr_reg LR_REG       learning rate for the super resolution network
      --batch_reg BATCH_REG
                        batchsize for the super resolution network
      --use_ori [USE_ORI]   whether to include the original observation for the
                        regressors
      --window_len WINDOW_LEN
                        window size for the second regressor
      --window_index WINDOW_INDEX
                        number of samples for the window for the second
                        regressor
      --device {cpu,gpu}    choose the device to run the code
      --print_interval PRINT_INTERVAL
                        number of iterations between each loss print
      --use_attention [USE_ATTENTION]
                        whether to use attention mechanism for the SR network
      --use_e2e [USE_E2E]
                        whether to use end-to-end joint optimization
      --e_e2e E_E2E     number of epochs for end-to-end training (default: 100)
      --lr_e2e LR_E2E   learning rate for end-to-end training (default: 1e-4)
      --seed SEED       random seed for reproducibility (default: 42)

## Test 

Run `test.py` to test the model. You can test the customized channel model and the 802.15.4a channel model.

### Basic Usage
```bash
# Test baseline model
python test.py --name baseline_exp --snr 30 --device gpu

# Test model with attention
python test.py --name attention_exp --snr 30 --use_attention True --device gpu

# Test end-to-end optimized model
python test.py --name e2e_exp --snr 30 --use_e2e True --device gpu

# Test model with both enhancements
python test.py --name full_exp --snr 30 --use_attention True --use_e2e True --device gpu
```

### Command-line Arguments

    usage: test.py [-h] [--name NAME] [--snr SNR] [--bandwidth BANDWIDTH]
               [--up UP] [--use_ori [USE_ORI]] [--use_802 [USE_802]]
               [--window_len WINDOW_LEN] [--window_index WINDOW_INDEX]
               [--device {cpu,gpu}] [--num_test NUM_TEST]

    optional arguments:
      -h, --help            show this help message and exit
      --name NAME           name for the experiment folder
      --snr SNR             select high SNR scenario or low SNR scenario
      --bandwidth BANDWIDTH
                        define the system bandwidth
      --up UP               upsamping rate for the super resolution net
      --use_ori [USE_ORI]   whether to include the original observation for the
                        regressors
      --use_802 [USE_802]   whether to test the 802.15.4a channel
      --window_len WINDOW_LEN
                        window size for the second regressor
      --window_index WINDOW_INDEX
                        number of samples for the window for the second
                        regressor
      --device {cpu,gpu}    choose the device to run the code
      --num_test NUM_TEST   number of test cirs
      --use_attention [USE_ATTENTION]
                        whether to use attention mechanism for the SR network
      --use_e2e [USE_E2E]
                        whether to load end-to-end optimized weights
      --seed SEED       random seed for reproducibility (default: 42)

## Experimental Comparison for Thesis

For a comprehensive undergraduate thesis, it is recommended to conduct ablation studies comparing different configurations:

| Configuration | Command Example | Expected Benefit |
|---------------|-----------------|------------------|
| Baseline | `--use_attention False --use_e2e False` | Original method performance |
| + Attention | `--use_attention True --use_e2e False` | Better noise suppression |
| + E2E | `--use_attention False --use_e2e True` | Task-specific SR optimization |
| + Both | `--use_attention True --use_e2e True` | Maximum performance |

### Performance Metrics to Report
- **RMSE (Root Mean Square Error)**: Coarse and fine ToA estimation errors
- **CDF (Cumulative Distribution Function)**: Percentage of samples below certain error thresholds
- **Inference Time**: Model computational efficiency
- **Visualization**: Compare reconstructed CIR waveforms and ToA predictions

## Example figures
### CIR comparison
<img src="example1.png" alt="drawing" width="600"/>

### ToA estimation
<img src="example2.png" alt="drawing" width="600"/>

## Reference 
> Yao-shan Hsiao*, Mingyu Yang*, Hun-Seok Kim, "Super-Resolution Time-of-Arrival Estimation using Neural Networks", EUSIPCO 2020

## Citation
If you use the enhanced features (attention mechanism or end-to-end optimization) in your research, please acknowledge this extended implementation in your thesis or publication.

## Contact & Support
For questions regarding the new features or undergraduate thesis guidance, please open an issue on this repository.
