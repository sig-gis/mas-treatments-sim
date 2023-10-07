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
    
def export_image_to_drive(
    image,
    description="myExportImageTask",
    folder=None,
    fileNamePrefix=None,
    dimensions=None,
    region=None,
    scale=None,
    crs=None,
    crsTransform=None,
    maxPixels=None,
    shardSize=None,
    fileDimensions=None,
    skipEmptyTiles=None,
    fileFormat=None,
    formatOptions=None,
    **kwargs,
):
    """Creates a batch task to export an Image as a raster to Google Drive.

    Args:
        image: The image to be exported.
        description: Human-readable name of the task.
        folder: The name of a unique folder in your Drive account to
            export into. Defaults to the root of the drive.
        fileNamePrefix: The Google Drive filename for the export.
            Defaults to the name of the task.
        dimensions: The dimensions of the exported image. Takes either a
            single positive integer as the maximum dimension or "WIDTHxHEIGHT"
            where WIDTH and HEIGHT are each positive integers.
        region: The lon,lat coordinates for a LinearRing or Polygon
            specifying the region to export. Can be specified as a nested
            lists of numbers or a serialized string. Defaults to the image's
            region.
        scale: The resolution in meters per pixel. Defaults to the
            native resolution of the image assset unless a crsTransform
            is specified.
        crs: The coordinate reference system of the exported image's
            projection. Defaults to the image's default projection.
        crsTransform: A comma-separated string of 6 numbers describing
            the affine transform of the coordinate reference system of the
            exported image's projection, in the order: xScale, xShearing,
            xTranslation, yShearing, yScale and yTranslation. Defaults to
            the image's native CRS transform.
        maxPixels: The maximum allowed number of pixels in the exported
            image. The task will fail if the exported region covers more
            pixels in the specified projection. Defaults to 100,000,000.
        shardSize: Size in pixels of the tiles in which this image will be
            computed. Defaults to 256.
        fileDimensions: The dimensions in pixels of each image file, if the
            image is too large to fit in a single file. May specify a
            single number to indicate a square shape, or a tuple of two
            dimensions to indicate (width,height). Note that the image will
            still be clipped to the overall image dimensions. Must be a
            multiple of shardSize.
        skipEmptyTiles: If true, skip writing empty (i.e. fully-masked)
            image tiles. Defaults to false.
        fileFormat: The string file format to which the image is exported.
            Currently only 'GeoTIFF' and 'TFRecord' are supported, defaults to
            'GeoTIFF'.
        formatOptions: A dictionary of string keys to format specific options.
        **kwargs: Holds other keyword arguments that may have been deprecated
            such as 'crs_transform', 'driveFolder', and 'driveFileNamePrefix'.
    """

    if not isinstance(image, ee.Image):
        raise ValueError("Input image must be an instance of ee.Image")

    task = ee.batch.Export.image.toDrive(
        image,
        description,
        folder,
        fileNamePrefix,
        dimensions,
        region,
        scale,
        crs,
        crsTransform,
        maxPixels,
        shardSize,
        fileDimensions,
        skipEmptyTiles,
        fileFormat,
        formatOptions,
        **kwargs,
    )
    task.start()
    print(f"Export Started (Drive): {fileNamePrefix}")

def ee_treatments(hucs:ee.FeatureCollection,
                  prescription:ee.Number,
                  unit_size:ee.Number,
                  radius:ee.Number,
                  pixel_value:ee.Number,
                  constraint_layer:ee.Image,
                  seed:ee.Number=101010,
                  ptOvershoot:ee.Number=2):
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
    ptOvershoot: ee.Number = multiplicative factor of how many random points to place over the requirement.
  returns:
    ee.Image
  """

  # get how many treatments to place given total prescribed area and our unit size
  treatment_count = ee.Number(prescription).divide(unit_size).round()
  
  # mask constraining pt sample placement
  constraintMask = ee.Image(constraint_layer).clip(hucs)
  
  # generate treatment centroid location pts
  pts = constraintMask.selfMask().sample(**{
    'region':hucs,
    'scale':ee.Number(radius).divide(2).round(),
    'projection':'EPSG:5070',
    'numPixels':ee.Number(treatment_count).multiply(ptOvershoot).round(),
    'seed':seed,
    'dropNulls':True,
    'tileScale':1,
    'geometries':True}) 
  
  # remove pts too close together
  pt_spacing = 1.5 # this could technically be another user-adjusted dial but trying not overcomplicate things.
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
    'ptOvershoot',ptOvershoot,
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
                  'ptOvershoot':ptOvershoot,
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