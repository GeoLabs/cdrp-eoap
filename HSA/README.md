# OGC EOAP for Hillshade, Slope and Aspect

For a given DEM, this EOAP computes Slope and Aspect using the [RichDEM](https://richdem.readthedocs.io/en/latest/) library [[1]](#references). Hillshade is relying on [EarthPy](https://earthpy.readthedocs.io/en/latest/). 

## Usage

First, build the required docker image and tag it with `geolabs/cdrp-hsa-package:1.0`.

````
docker build . -t geolabs/cdrp-hsa-package:1.0
````

Run the EOAP from the command line.

````
cwltool app.yml#geolabs_cdrp_hsa --item <YOUR_DEM_URL>
````

To deploy the EOAP on the ZOO-Project-DRU, use the following requests.

````
curl -X POST \
     -H "accept: application/json" \
     -H "Content-type: application/cwl+yaml" \
     -d @app.yml \
     <YOUR_ZOO-Project-DRU_URL>/processes
````

To execute the EOAP you can use the following request.

````
curl -X 'POST' \
  '<YOUR_ZOO-Project-DRU_URL>/processes/geolabs_cdrp_slope/execution' \
  -H 'accept: /*' \
  -H 'Prefer: respond-async;return=representation' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": {
    "item": "<YOUR_DEM_URL>"
  }
}'
````

From the response, you can find the job identifier, which you can use to follow the process execution progress. Once the execution ends, you can access the result endpoint to access the produced stac catalog, which is a collection of hillshade, slope, and aspect features.


## References

[1] Barnes, Richard. 2016. RichDEM: Terrain Analysis Software. http://github.com/r-barnes/richdem