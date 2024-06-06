#!/usr/bin/env python3
import os
import sys
import itertools
import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from Utils import gdalCommonUtils as gutils
from Utils import IO as io
from Utils import tileHelpers as tileHelper

import tensorflow as tf
from tensorflow.keras import models
from osgeo import gdal, ogr, osr
import cv2 

import time
import click
import requests

from loguru import logger
import rasterio
from rasterio.enums import Resampling
import os
#import earthpy.spatial as es
#import richdem as rd
import numpy as np
from osgeo import gdal
from rio_stac import stac
import json

def open_and_rescale_to_baseResolution(img_path, shape, bbox=None, crop_to_bbox=None):
    img = gutils.readGDAL2numpy(img_path)
    img = tileHelper.rescaleInput(img, shape)
    if crop_to_bbox is not None:
        yMin, yMax, xMin, xMax = bbox
        img = img[yMin: yMax, xMin: xMax]
    return img

def normalize(image):
    new_img = np.array(image, dtype = np.float32)
    new_img /= 127.5
    new_img -= 1.
    return new_img
    

def normalizeArray(image):
    # post image
    image[ : , :, :3] = normalize(image[ : , :, :3])
    #
    # pre image
    image[ : , :, 3:6] = normalize(image[ : , :, 3:6])
    #
    # HS image
    image[ : , :, 6:7] = normalize(image[ : , :, 6:7])
    #
    # Slope Image
    image[ : , :, -1] = (image[ : , :, -1] / 45.0) - 1.0
    #
    return image



def save_stac(out_hillshade,output_type):
    with rasterio.open(os.path.join(out_hillshade), 'r') as dst:
        res=stac.create_stac_item(
                source=dst,
                asset_roles=["data", "visual", output_type],
                with_proj=True,
                with_raster=False,
            )
        local_dict=res.to_dict()
        local_dict["assets"]["asset"]["href"]=local_dict["assets"]["asset"]["href"].replace(os.getcwd(),"").replace("myMockStacItem/","")
        logger.info("Write stac item ...")
        open("myMockStacItem/"+output_type+".json", "w").write(json.dumps(local_dict))

@click.command(
    short_help="Generate slope and hillshade",
    help="Generate slope and hillshade"
)
@click.option(
    "--model",
    "model_file",
    help="The model to use",
    required=True
)
@click.option(
    "--dem",
    "dem_path",
    help="DEM image to use",
    required=True
)
@click.option(
    "--hillshade",
    "hs_path",
    help="DEM image to use",
    required=True
)
@click.option(
    "--slope",
    "slope_path",
    help="DEM image to use",
    required=True
)
@click.option(
    "--pre-image",
    "pre_image_path",
    help="DEM image to use",
    required=True
)
@click.option(
    "--post-image",
    "post_image_path",
    help="DEM image to use",
    required=True
)
@click.option(
    "--mask",
    "mask_path",
    help="DEM image to use",
    required=True
)
@click.option(
    "--boundary",
    "boundary_path",
    help="DEM image to use",
    required=True
)

def main(model_file,dem_path,hs_path,slope_path,pre_image_path,post_image_path,mask_path,boundary_path):

    # Create a directory
    os.makedirs("myMockStacItem", exist_ok=True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    PROJECT_PATH = '/tmp/'
    OUTPUT_PATH  = "myMockStacItem" #os.path.join(*["/tmp", 'Mapping_results'])

    io.createDirectory(OUTPUT_PATH, emptyExistingFiles = True, verbose = True)     

    BASE_RESOLUTION = 6

    SAVED_MODEL = os.path.join(*[PROJECT_PATH, 'M_ALL_006.hdf5'])

    # Download the model file to use
    params = {'apikey': 'xxxxxxxxxxxxxxxxxxx', 'hash':'xxxxxxxxxxxxxxxxxxxxxxxxx', 'stream':True}
    response = requests.get(model_file, params)
    totalbits = 0
    if response.status_code == 200:
        with open(SAVED_MODEL, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    totalbits += 1024
                    f.write(chunk)


    RAW_DATA_PATH = "/vsicurl/"
    #RAW_DATA_PATH = os.path.join(*[PROJECT_PATH, 'RawData', 'AOI_E_S2-S2'])

    DEM_PATH   = os.path.join(*[RAW_DATA_PATH, dem_path])
    HS_PATH    = os.path.join(*[RAW_DATA_PATH, hs_path])
    SLOPE_PATH = os.path.join(*[RAW_DATA_PATH, slope_path])

    POST_IMAGE_PATH = os.path.join(*[RAW_DATA_PATH, pre_image_path])
    PRE_IMAGE_PATH  = os.path.join(*[RAW_DATA_PATH, post_image_path])
    NO_DATA_MASK    = os.path.join(*[RAW_DATA_PATH, mask_path])

    TEST_ROI_PATH   = os.path.join(*[RAW_DATA_PATH, boundary_path])

    bbox, binaryMask, newGeoT, proj = gutils.getBoundingBox(rasterPath = TEST_ROI_PATH, returnBinaryMask = True)
    nscn, npix = binaryMask.shape

    ## generate a mask from no data file
    logger.info(f"Load  NO_DATA_MASK ...")
    try:
        maskImage = 1 - open_and_rescale_to_baseResolution(img_path=NO_DATA_MASK, shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
    except:
        logger.info(f"Create NO_DATA_MASK ith Zeros ...")
        maskImage = np.zeros(shape=(nscn, npix))
    
    ## append in mask file where test area Image == 0
    maskImage[binaryMask == 0] = 0

    maskFileName = os.path.join(*[OUTPUT_PATH, 'MaskImage.tif'])
    gutils.writeNumpyArr2Geotiff(maskFileName, 1-maskImage, newGeoT, proj, GDAL_dtype=gdal.GDT_Byte, noDataValue=0)

    # Block #
    imageSize = 224
    overlapFactor = 2
    fetchSize = int(imageSize / 2)
    skipPx = int(imageSize / overlapFactor)

    ## Generates all possible bounding boxes for tiling
    ## The center location of every box is the anchor point defined in list 'locXY'
    Y = [y for y in range(fetchSize + 1, nscn - fetchSize - 1, skipPx)]
    X = [x for x in range(fetchSize + 1, npix - fetchSize - 1, skipPx)]
    locXY = list(itertools.product(X, Y))

    ## extract all the valid boxes
    ## i.e. which are in the study area and landslide mask
    ## Mask Image == 1 for valid regions
    ## threshold of 0.75 --> 75% region is valid
    validLocXY = [currLoc for currLoc in locXY if tileHelper.isValidTile(maskImage, imageSize, currLoc, threshold=0.50)]

    # BLOCK #
    logger.info(f"Load  POST_IMAGE_PATH ...")
    postImage = open_and_rescale_to_baseResolution(img_path=POST_IMAGE_PATH, shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
    logger.info(f"Load  PRE_IMAGE_PATH ...")
    preImage  = open_and_rescale_to_baseResolution(img_path=PRE_IMAGE_PATH,  shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)

    logger.info(f"Load  HS_PATH ...")
    hs    = open_and_rescale_to_baseResolution(img_path=HS_PATH,    shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)
    logger.info(f"Load  SLOPE_PATH ...")
    slope = open_and_rescale_to_baseResolution(img_path=SLOPE_PATH, shape=(nscn, npix), bbox=bbox, crop_to_bbox=bbox)


    showMask = True

    # display images
    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12,18)) 

    ax[0,0].imshow(postImage)
    ax[0,0].set_title('Sentinel-2 RGB (POST-EVENT)')

    ax[0,1].imshow(preImage)
    ax[0,1].set_title('Sentinel-2 RGB (PRE-EVENT)')

    ax[1,0].imshow(hs, cmap='gray', vmin=0, vmax=255)
    ax[1,0].set_title('HILLSHADE')

    ax[1,1].imshow(slope, cmap='seismic', vmin=0, vmax=90)
    ax[1,1].set_title('SLOPE')

    if showMask:
        maskForDisplay = 1 - maskImage.copy()
        maskForDisplay = np.float32(maskForDisplay)
        maskForDisplay[maskForDisplay == 0] = np.nan
        
        ax[0,0].imshow(maskForDisplay, alpha=0.35, cmap='cool', vmin=0, vmax=1)
        ax[0,1].imshow(maskForDisplay, alpha=0.35, cmap='cool', vmin=0, vmax=1)
        ax[1,0].imshow(maskForDisplay, alpha=0.35, cmap='cool', vmin=0, vmax=1)
        ax[1,1].imshow(maskForDisplay, alpha=0.35, cmap='cool', vmin=0, vmax=1)
        
    plt.savefig(os.path.join(*[OUTPUT_PATH, 'output.png']))

    stack = []

    stack.append(postImage)
    stack.append(preImage)
    stack.append(hs[:, :, np.newaxis])
    stack.append(slope[:, :, np.newaxis])

    testImage = np.concatenate(stack, axis=2)
    del stack

    testImage = np.nan_to_num(testImage)
    testImage = normalizeArray(testImage)

    inputChannel = testImage.shape[2]

    predictMask = np.zeros((testImage.shape[0], testImage.shape[1]), np.uint8)

    fetchSize_half = int(fetchSize/2)

    model = models.load_model(SAVED_MODEL, compile=False)

    model.compile(optimizer='adam', loss='binary_crossentropy')

    for x, y in tqdm(validLocXY):
        #
        img = testImage[y - fetchSize: y + fetchSize, x - fetchSize: x + fetchSize, :]
        #
        img = img.reshape(1, imageSize, imageSize, inputChannel)
        predict_image_list = [img[:, :, :, 3:6],  # pre-image
                            img[:, :, :, :3],  # post-image
                            img[:, :, :, -2:]  # topo-image
                                ]
        predicted_label = model.predict(predict_image_list)[0][0]
        #
        predictMask[y - fetchSize_half: y + fetchSize_half, x - fetchSize_half: x +
            fetchSize_half] = predicted_label[fetchSize_half: -fetchSize_half, fetchSize_half: -fetchSize_half, 0] * 100

    predictMask = predictMask * maskImage
    logger.info("Produce Predict_LS_Conf.tif ...")
    gutils.writeNumpyArr2Geotiff(OUTPUT_PATH + '/Predict_LS_Conf.tif', predictMask, newGeoT, proj, GDAL_dtype= gdal.GDT_Byte, noDataValue = 0)    

    threshold = 50
    predictMask_highConf = predictMask.copy()
    predictMask_highConf[predictMask_highConf < threshold] = 0
    predictMask_highConf[predictMask_highConf >= threshold] = 1

    logger.info("Produce Predict_LS_HighConf_50.tif ...")
    rasterPath = OUTPUT_PATH + '/Predict_LS_HighConf_{}.tif'.format(str(threshold))
    gutils.writeNumpyArr2Geotiff(rasterPath, predictMask, newGeoT, proj, GDAL_dtype=gdal.GDT_Byte, noDataValue=0)

    showMask = True

    ## Zoom to Extent
    xmin, xmax = 0, 1250
    ymin, ymax = -2000, -750


    LSForDisplay = predictMask_highConf.copy()
    LSForDisplay = np.float32(LSForDisplay)
    LSForDisplay[LSForDisplay == 0] = np.nan


    # display images
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,15)) 


    ax[0].imshow(preImage[ymin:ymax,xmin:xmax])
    ax[0].set_title('Sentinel-2 RGB (PRE-EVENT)')
    ax[0].axis('off')

    ax[1].imshow(postImage[ymin:ymax,xmin:xmax])
    ax[1].set_title('Sentinel-2 RGB (POST-EVENT)')
    ax[1].axis('off')

    ax[2].imshow(postImage[ymin:ymax,xmin:xmax], alpha= 0.35)
    ax[2].imshow(LSForDisplay[ymin:ymax,xmin:xmax], alpha=1, cmap='bwr', vmin=0, vmax=1)
    ax[2].set_title('MAPPED LANDSLIDE IN RED')
    ax[2].axis('off')


    if showMask:
        maskForDisplay = 1 - maskImage.copy()
        maskForDisplay = np.float32(maskForDisplay)
        maskForDisplay[maskForDisplay == 0] = np.nan
        
        ax[0].imshow(maskForDisplay[ymin:ymax,xmin:xmax], alpha=0.35, cmap='cool', vmin=0, vmax=1)
        ax[1].imshow(maskForDisplay[ymin:ymax,xmin:xmax], alpha=0.35, cmap='cool', vmin=0, vmax=1)
        ax[2].imshow(maskForDisplay[ymin:ymax,xmin:xmax], alpha=0.35, cmap='cool', vmin=0, vmax=1)

    fig.tight_layout()
    plt.show()
    plt.savefig(OUTPUT_PATH + '/predictions.png')

    save_stac(rasterPath,'LS_Predict_Conf')
    save_stac(rasterPath,"LS_Predict_50")
    #save_stac(rasterPath,fname,src,"slope")

    return 
    # Open the DEM and read the first (and only) band
    with rasterio.open("/vsicurl/"+dem_path) as src:
        out_hillshade = os.getcwd()
        elevation = src.read(1)
        fname = dem_path.split("/")[-1].split(".")[0]
        logger.info(f"Compute hillshade")
        hillshade = es.hillshade(elevation, azimuth=315, altitude=45)
        # # Create a slope raster
        ds = gdal.Open("/vsicurl/"+dem_path)
        nparray = rd.rdarray(np.array(ds.GetRasterBand(1).ReadAsArray()), no_data=-9999)
        logger.info("Create a slope raster ...")
        slope = rd.TerrainAttribute(nparray, attrib='slope_riserun')

        # # Create a aspect raster
        logger.info("Create a aspect raster ...")
        aspect = rd.TerrainAttribute(nparray, attrib='aspect')
        
        # Write the hillshade and slope rasters to new files
        with rasterio.open(os.path.join(out_hillshade,"myMockStacItem", fname+"_hillshade.tif"), 'w', **src.profile) as dst:
            dst.write(hillshade,1)
            #dst.close()

        save_gdal(out_hillshade,slope,src,fname,"slope")
        save_gdal(out_hillshade,aspect,src,fname,"aspect")
        
        logger.info(f"Produce stac items ...")
        save_stac(out_hillshade,fname,src,"hillshade")
        save_stac(out_hillshade,fname,src,"slope")
        save_stac(out_hillshade,fname,src,"aspect")
        logger.info(f"Over")


if __name__ == "__main__":
    main()
