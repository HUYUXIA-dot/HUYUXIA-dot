var roi = table
Map.centerObject(roi, 10);
Map.addLayer(roi, {color:"black"}, "roi");
var l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
          .filterDate('2019-01-01', '2019-12-31')
          .filterBounds(roi)
          .map(roiClip);

function roiClip(image){
  return image.clip(roi)
}
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}
var withNDVI = l8.map(addNDVI);

var viz = {min:-1, max:1, palette:'blue, white, green'};
Map.addLayer(withNDVI.select('NDVI'), viz, 'NDVI');

var ndviTransform = function(img){ 
  var ndvi = img.normalizedDifference(['B4', 'B3']) // calculate normalized dif between band 4 and band 3 (B4-B3/B4_B3)
                .multiply(1000) // scale results by 1000
                .select([0], ['NDVI']) // name the band
                .set('system:time_start', img.get('system:time_start'));
  return ndvi;
};
var geometry = ee.FeatureCollection('users');
Map.centerObject(geometry,6);
var dataset = ee.ImageCollection('MODIS/006/MOD17A3HGF').filter(ee.Filter.date('2014-01-01', '2014-12-31'));
print(dataset)
var npp = dataset.select('Npp');
var nppVis = {
  min: 0.0,
  max: 19000.0,
  palette: ['bbe029', '0a9501', '074b03'],
};
Map.addLayer(npp.mean().clip(geometry), nppVis, 'NPP');
