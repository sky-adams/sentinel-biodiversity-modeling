# Sentinel Biodiversity Modeling 

## Table of contents
1. [Overview](#overview)
2. [Data](#data)
3. [Method](#mthd)
4. [Results](#results)
5. [Discussion and Limitations](#discussion)
6. [Repository Structure](#structure)
7. [Libraries Used](#libraries)
8. [How to Run](#run)

## Overview<a name="overview"></a>
This project predicts the Biodiversity Intactness Index (BII) of Sentinel-2 satellite image patches over Santa Barbara 
County. The goal is to develop a model that can estimate biodiversity-related signals from remote sensing data. Such a 
model could provide valuable information more quickly and at a lower cost than current methods used to determine the 
BII of an area.

The project includes code to download the necessary satellite and BII data in a GeoTIFF file from Google Earth Engine. The 
pipeline loads 64x64 patches from the GeoTIFF, filters out patches with too many missing values, trains a convolutional 
neural network regressor, and evaluates performance on a held-out test split. 

This repository includes the data export script, model code, training pipeline, and experiment summaries for predicting 
BII from Sentinel-2 imagery.

## Data<a name="data"></a>
The data used to train the model are downloaded from Google Earth Engine and use the Biodiversity Intactness Index (BII) as the 
target variable. BII measures how local terrestrial biodiversity responds to human pressures and compares the abundance 
and composition of native species present to a "pristine" intact baseline. BII scores range from 0 to 1. A BII of 1 means 
that the ecosystem is fully intact and undisturbed, while a BII of 0 means that it is completely degraded and no native 
species remain. The BII data used here are sourced from the [Google Earth Engine community catalog](https://gee-community-catalog.org/projects/bii/).

The Santa Barbara County bounding box used to define the region of interest comes from the 
[2019 TIGER/Line Shapefile](https://catalog.data.gov/dataset/tiger-line-shapefile-2019-county-santa-barbara-county-ca-topological-faces-polygons-with-all-ge) 
published by the U.S. Census Bureau. Due to the presence of the Channel Islands in Santa Barbara County, much of the region 
of interest contains ocean, for which there is no BII score.

The GeoTIFF file contains five bands:
1. B2 = blue
2. B3 = green
3. B4 = red
4. B8 = near-infrared
5. bii_label = Biodiversity Intactness Index (the target)

The pipeline uses bands 1–4 as inputs and band 5 as the regression target.

## Method<a name="mthd"></a>
The model is a convolutional neural network regressor built in PyTorch. In addition to the raw Sentinel-2 bands, the 
pipeline also uses three vegetation indices computed from those bands: NDVI, GNDVI, and MSAVI2. More information about 
these indices can be found [here](https://www.auravant.com/en/articles/precision-agriculture/vegetation-indices-and-their-interpretation-ndvi-gndvi-msavi2-ndre-and-ndwi/).

Training uses a patch-based dataset split into train, validation, and test sets. The model is evaluated using RMSE, 
MAE, and R².

## Results<a name="results"></a>
**Best-performing configuration:**

Features: 4 raw bands + NDVI + GNDVI + MSAVI2.

Learning rate: 1e-3.

Dropout: 0.3.

**Best test metrics:**

RMSE: 0.0724

MAE: 0.0557

R²: 0.5375

**Baseline metrics:**

Only the 4 raw bands were included in the baseline, and the metrics were computed prior to tuning the learning rate 
and dropout.

RMSE: 0.0890

MAE: 0.0623

R²: 0.3029

**Features tested but not included in the best model:**

The features in the table below were tested in combination with the 4 raw bands + NDVI + GNDVI + MSAVI2. 
The metrics were computed prior to tuning the learning rate and dropout, so the numbers are not directly comparable to the final best model.

| Feature | RMSE | MAE | R² | Notes|
| -------- | -------- | -------- | -------- | -------- |
| EVI | 0.08346 | 0.0610 | 0.3866 | Decrease in performance likely due to correlation with NDVI. |
| GLCM entropy on NDVI computed as one value per patch | 0.0892 | 0.0614 | 0.2994 | |
| Mean of each of the 4 raw bands | 0.2091 | 0.1799 | -2.8484 | All predictions were 1.0 due to the means being significantly larger values than the other inputs. |
| Normalized raw-band means | 0.1040 | 0.0743 | 0.0478 | Normalization was done using the training set mean and standard deviation, and means were computed per patch. |
| Normalized raw-band means and standard deviations | 0.0820 | 0.0590 | 0.4082 | |

## Discussion and Limitations<a name="discussion"></a>
The best model explains almost 54% of the variance in BII, which is a decent result for a small remote-sensing 
regression problem. It shows that the Sentinel-2 bands and vegetation indices contain useful signal, but there 
is still room for improvement.

The main limitations are the small dataset size and the fact that the data were split in contiguous order rather 
than with a fully spatial block split. The current splitting technique reduces but does not completely eliminate 
spatial dependence between train and test sets.

Future work could test more random seeds, try alternative models, use more explicitly spatial splits, explore 
more engineered features, and evaluate different patch sizes to study the tradeoff between spatial context and 
sample count.

## Repository Structure<a name="structure"></a>

```text
.
├── README.md
├── data/
│   ├── sentinel_bii_download.js
│   └── santa_barbara_sentinel_bii.tif
├── outputs/
│   └── pred_vs_actual.png
├── biodiversity_predictor/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── requirements.txt
└── config.yaml
```

The outputs/ folder contains generated artifacts from training, including the predicted-vs-actual plot. These files 
are reproducible and are not tracked in version control.

## Libraries Used<a name="libraries"></a>
* Python
* PyTorch
* Rasterio
* NumPy
* Pandas
* scikit-learn
* Plotly

## How to Run<a name="run"></a>
1. **Download the Data from Google Earth Engine**
	1. Create a Google Earth Engine Project.
	2. Create a new script in the project and run the code from data/sentinel_bii_download.js.
	3. In the Google Earth Engine "Tasks" tab, run the export task. The export took about 11 minutes when I ran it.
	4. Download the resulting GeoTIFF file and place it in the data/ folder of this project.
2. **Install Dependencies**
    1. Install the required Python modules using `pip install -r requirements.txt`.
3. **Train and Evaluate the Model**
	1. Run the Python code from the project root using `python -m biodiversity_predictor.train`. Results will 
	be printed to the terminal and also saved in the `outputs/` folder along with a plot of the actual 
	vs. predicted BII scores of the test set.