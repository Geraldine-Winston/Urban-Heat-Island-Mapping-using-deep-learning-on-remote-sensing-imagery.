# Urban Heat Island Mapping using Deep Learning on Remote Sensing Imagery

"""
This repository presents a comprehensive deep learning framework for mapping Urban Heat Islands (UHI) using remote sensing imagery.

Urban Heat Islands are critical indicators of climate stress in urban environments, and accurately identifying these zones supports informed urban planning and sustainable development efforts. The project leverages Convolutional Neural Networks (CNNs) to process satellite images and detect UHI-affected regions, delivering efficient, automated analysis beyond traditional methods. With the capacity to handle complex patterns and large datasets, the model enhances the detection of thermal anomalies across diverse urban landscapes.

## Project Structure
- **data/**: Contains satellite images and corresponding label masks indicating UHI regions.
- **models/**: Directory to save trained CNN models.
- **outputs/**: Folder storing generated UHI prediction maps and training history visualizations.
- **uhi_mapping_cnn.py**: Main Python script for loading data, training the CNN, evaluating, and generating outputs.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

## Usage
1. Organize your satellite images in `data/images/` and corresponding binary label masks in `data/labels/`.

2. Execute the script:
```bash
python uhi_mapping_cnn.py
```

## Workflow
The script performs:
- Data loading and preprocessing (image resizing, normalization).
- CNN model construction and training.
- Evaluation on a validation set.
- Generation of sample UHI prediction maps.
- Visualization of training and validation performance.

## Model Architecture
- 3 Convolutional layers with increasing filters (32, 64, 128).
- MaxPooling layers to downsample feature maps.
- Dense layers with Dropout for regularization.
- Binary cross-entropy loss function.
- Adam optimizer for efficient training.

## Output
- Trained CNN model saved in `models/`.
- Training history plots (accuracy over epochs) saved in `outputs/`.
- Sample UHI prediction visualizations.

## Author
Ayebawanaemi Geraldine Winston
"""

