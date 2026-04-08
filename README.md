#Biodiversity Predictor Using 

##Data
The data used to train the model are downloaded from EarthEngine and use the Biodiversity Intactness Index. Include info about the BII and the proper citation here. https://gee-community-catalog.org/projects/bii/
Coordinates for the bounding box for Santa Barbara County are from the 2019 TIGER/Line Shapefile published by the US Census Bureau here https://catalog.data.gov/dataset/tiger-line-shapefile-2019-county-santa-barbara-county-ca-topological-faces-polygons-with-all-ge

##Repository Structure
├── README.md
├── data/
│   └── sentinel_bii_download.js
│   └── santa_barbara_sentinel_bii.tif
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── requirements.txt
└── config.yaml

##How to Run the Project
1. Download the Data from EarthEngine:
	1. Create an EarthEngine Project.
	2. Create a new script in the project and run the code from data/sentinel_bii_download.js.
	3. In the "Tasks" tab, run the task. Note that doing this step took 11 minutes when I ran it.
	4. Download and save the GeoTIFF file into the data folder of this project.
2. Run the Code
	1. Install the required Python modules using pip install -r requirements.txt.
	2. 
