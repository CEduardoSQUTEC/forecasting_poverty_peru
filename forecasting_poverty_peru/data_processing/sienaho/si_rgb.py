from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd
import requests
from tqdm import tqdm

from forecasting_poverty_peru.config import PROCESSED_PATH, RAW_PATH


SIENAHO_PATH = PROCESSED_PATH / 'sienaho_rgb'
BUFFER_SIZE = 1120
MAX_WORKERS = 8
MAX_RETRIES = 3
RETRY_DELAY = 5
EE_PROJECT = 'forecasting-poverty'
IMAGE_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
BANDS = ['B4', 'B3', 'B2']


def initialize_earth_engine() -> None:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)


def create_directory_structure() -> None:
    for cls in ['adequate', 'inadequate']:
        (SIENAHO_PATH / cls).mkdir(parents=True, exist_ok=True)


def download_image(url: str, output_path: Path) -> bool:
    for attempt in range(MAX_RETRIES):
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        output_path.write_bytes(response.content)

        if output_path.stat().st_size == 0:
            return False
        return True
    return False


def process_conglomerate(conglomerate: pd.Series) -> None:
    point = ee.Geometry.Point(conglomerate.longitude, conglomerate.latitude)
    region = point.buffer(BUFFER_SIZE).bounds()

    start_date = ee.Date.fromYMD(conglomerate.year, conglomerate.month, 1)
    end_date = start_date.advance(1, 'month')

    collection = (
        ee.ImageCollection(IMAGE_COLLECTION)
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .sort('CLOUDY_PIXEL_PERCENTAGE')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    )

    if collection.size().getInfo() == 0:
        return

    image = collection.first().select(BANDS).clip(region)

    download_url = image.getDownloadURL({
        'region': region,
        'scale': 10,
        'crs': 'EPSG:4326',
        'format': 'GEO_TIFF'
    })

    cls = 'adequate' if conglomerate.adequate else 'inadequate'
    output_path = SIENAHO_PATH / cls / f'{conglomerate.Index:04}.tif'
    download_image(download_url, output_path)


def main() -> None:
    initialize_earth_engine()
    create_directory_structure()

    df = pd.read_pickle(SIENAHO_PATH / 'class_names.pkl')

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = [
            executor.submit(process_conglomerate, row)
            for row in df.itertuples()
        ]
        with tqdm(
                total=len(future_to_id),
                desc='Downloading images',
                unit='image'
        ) as pbar:
            for future in as_completed(future_to_id):
                try:
                    future.result()
                finally:
                    pbar.update(1)


if __name__ == '__main__':
    main()
