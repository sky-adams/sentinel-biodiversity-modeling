// 1. Santa Barbara County bounding box 
var roi = ee.Geometry.Rectangle([-120.734382, 33.411024, -118.962728, 35.114678]);

// 2. Load BII (2017-2020 mean)
var bii = ee.ImageCollection("projects/ebx-data/assets/earthblox/IO/BIOINTACT")
  .filterDate('2017-01-01', '2020-12-31')
  .mean()
  .select('BioIntactness');

// 3. Sentinel-2 2019 median composite (low cloud)
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate('2019-01-01', '2019-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median()
  .select(['B2','B3','B4','B8']); // Blue, Green, Red, NIR

// 4. Stack into 5-band image
var stacked = s2.float().addBands(bii.rename('bii_label').float()).clip(roi);

// 5. Visualize
Map.centerObject(roi, 9);
Map.addLayer(s2, {min:0, max:3000, bands:['B4','B3','B2']}, 'Sentinel RGB');
Map.addLayer(bii, {min:0, max:1, palette:['red','yellow','green']}, 'BII');

// 6. Export directly to Drive
Export.image.toDrive({
  image: stacked,
  description: 'santa_barbara_bii',
  folder: 'earthengine',
  fileNamePrefix: 'santa_barbara_sentinel_bii',
  region: roi,
  scale: 100,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});
