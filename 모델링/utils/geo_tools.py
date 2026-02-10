
import ee
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def initialize_gee(project_id):
    """
    Initializes Google Earth Engine.
    """
    try:
        ee.Initialize(project=project_id)
        return True
    except Exception as e:
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            return True
        except Exception as e2:
            print(f"GEE Initialization failed: {e2}")
            return False

def get_roi_geometry(lat, lon, buffer_meters):
    """
    Returns a GEE Geometry buffer around a point.
    """
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(buffer_meters)

def reproject_raster(src_path, dst_path, target_crs='EPSG:32652', resolution=10):
    """
    Reprojects a raster to a target CRS and resolution.
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=resolution
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
    return dst_path
