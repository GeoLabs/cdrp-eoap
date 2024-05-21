#!/usr/bin/env python3
import os
import time
import click

from loguru import logger
import rasterio
from rasterio.enums import Resampling
import os
import earthpy.spatial as es
import richdem as rd
import numpy as np
from osgeo import gdal
from rio_stac import stac
import json


def save_stac(out_hillshade,fname,src,output_type):
    with rasterio.open(os.path.join(out_hillshade,"myMockStacItem", fname+"_"+output_type+".tif"), 'r', **src.profile) as dst:
        res=stac.create_stac_item(
                source=dst,
                asset_roles=["data", "visual", output_type],
                with_proj=True,
                with_raster=False,
            )
        local_dict=res.to_dict()
        local_dict["assets"]["asset"]["href"]=local_dict["assets"]["asset"]["href"].replace(os.getcwd(),"").replace("/myMockStacItem/","")
        logger.info("Write stac item ...")
        open("myMockStacItem/"+output_type+".json", "w").write(json.dumps(local_dict))

def save_gdal(out_hillshade,obj,src,fname,output_type):
    obj.crs=src.crs
    obj.geotransform=src.get_transform()
    rd.SaveGDAL(os.path.join(out_hillshade,"myMockStacItem", fname+"_"+output_type+".tif"),obj)

@click.command(
    short_help="Generate slope and hillshade",
    help="Generate slope and hillshade"
)
@click.option(
    "--input-item",
    "dem_path",
    help="DEM image to use",
    required=True
)

def main(dem_path):

    # Create a directory
    os.makedirs("myMockStacItem", exist_ok=True)

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
