import ee
import numpy as np


def export_img(img,desc,asset_id,aoi,scale,crs):
    """Export image to imageCollection"""
    
    task = ee.batch.Export.image.toAsset(
        image=ee.Image(img).clip(aoi),
        description=desc,
        assetId=asset_id, 
        region=aoi, 
        scale=scale, 
        crs=crs, 
        maxPixels=1e13)

    task.start()
    print(f"Export Started for {asset_id}")
    
def ee_treatments(hucs:ee.FeatureCollection,
                  prescription:ee.Number,
                  unit_size:ee.Number,
                  radius:ee.Number,
                  pixel_value:ee.Number,
                  constraint_layer:ee.Image,
                  seed:ee.Number=101010):
  """
  generates a treated area raster within boundaries of one or multiple HUCs
  
  args:
    hucs: ee.FeatureCollection = boundary(ies) to generate treated pixels in
    prescription: Integer/ee.Number = total area to treat in m^2
    unit_size: Integer/ee.Number = size of the treatment units in m^2
    radius: radius of a square (approximate) in meters
    pixel_value: Integer/ee.Number = pixel value of output image
    constraint_layer: ee.Image = raster mask to constrain treatment centroids to
    seed: ee.Number = random seed to pass to sampling functions
  returns:
    ee.Image
  """

  # get how many treatments to place given total prescribed area and our unit size
  treatment_count = ee.Number(prescription).divide(unit_size).round()
  
  # mask constraining pt sample placement
  constraintMask = ee.Image(constraint_layer).clip(hucs)
  
  # generate treatment centroid location pts
  overshoot = 2
  pts = constraintMask.selfMask().sample(**{
    'region':hucs,
    'scale':ee.Number(radius).divide(2).round(),
    'projection':'EPSG:5070',
    'numPixels':ee.Number(treatment_count).multiply(overshoot).round(),
    'seed':seed,
    'dropNulls':True,
    'tileScale':1,
    'geometries':True}) 
  
  # remove pts too close together
  pt_spacing = 1.5
  pts_f = distanceFilter(pts,ee.Number(radius).multiply(pt_spacing)).limit(treatment_count)
  ptsSize= pts_f.size()
  
  # treated areas  
  areas = (ee.Image().paint(pts_f).Not().unmask(0)
  .distance(ee.Kernel.euclidean(radius,'meters'))
  .gte(0)
  .clip(hucs)
  .unmask(0)
  )
  
  huc_group_size = hucs.size()
  return (
    # we return the image, setting QA info as properties
    ee.Image(areas).multiply(pixel_value).toByte().selfMask().rename('Tx')
    .set(
    'prescriptionAreaSqm',prescription,
    'prescriptionAreaAc',ee.Number(prescription).multiply(0.000247105),
    'treatmentsGenerated',ptsSize,
    'treatmentsNeeded',treatment_count,
    'ptOvershoot',overshoot,
    'hucGroupSize',huc_group_size,
    'seed',seed,
    'yearInterval',pixel_value,),
    
    # we also return the QA info as a stand-alone dictionary so that we can retrieve it if the image's properties 
    # this does not solve the fact that if multiple of these returned dictionaries were to be set as properties
    # on a mosaicked set of these returned images, the property names would not be unique
    # see the remap_keys() func which solves that
    ee.Dictionary({'prescriptionAreaSqm':prescription,
                  'prescriptionAreaAc':ee.Number(prescription).multiply(0.000247105),
                  'treatmentsGenerated':ptsSize,
                  'treatmentsNeeded':treatment_count,
                  'ptOvershoot':overshoot,
                  'hucGroupSize':huc_group_size,
                  'seed':seed,
                  'yearInterval':pixel_value,
                  })

    

  )
  
def distanceFilter(pts,distance):
    withinDistance = distance 

    ## From the User Guide: https:#developers.google.com/earth-engine/joins_spatial
    ## add extra filter to eliminate self-matches
    distFilter = ee.Filter.And(ee.Filter.withinDistance(**{
      'distance': withinDistance,
      'leftField': '.geo',
      'rightField': '.geo', 
      'maxError': 1
    }), ee.Filter.notEquals(**{
      'leftField': 'system:index',
      'rightField': 'system:index',

    }))
    
    distSaveAll = ee.Join.saveAll(**{
                  'matchesKey': 'points',
                  'measureKey': 'distance'
    })
    # Apply the join.
    spatialJoined = distSaveAll.apply(pts, pts, distFilter)

    # Check the number of matches.
    # We're only interested if nmatches > 0.
    spatialJoined = spatialJoined.map(lambda f: f.set('nmatches', ee.List(f.get('points')).size()) )
    spatialJoined = spatialJoined.filterMetadata('nmatches', 'greater_than', 0)

    # The real matches are only half the total, because if p1.withinDistance(p2) then p2.withinDistance(p1)
    # Use some iterative logic to clean up
    def unpack(l): 
        return ee.List(l).map(lambda f: ee.Feature(f).id())

    def iterator_f(f,list):
        key = ee.Feature(f).id()
        list = ee.Algorithms.If(ee.List(list).contains(key), list, ee.List(list).cat(unpack(ee.List(f.get('points')))))
        return list
    
    ids = spatialJoined.iterate(iterator_f,ee.List([]))
    ##print("Removal candidates' IDs", ids)

    # Clean up 
    cleaned_pts = pts.filter(ee.Filter.inList('system:index', ids).Not())
    return cleaned_pts