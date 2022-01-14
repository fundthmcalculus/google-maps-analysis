import json
import logging
import os
import sys
from functools import cache
from typing import Union, List

import kmlutilities
from arcgisconnection import ArcGisProcessor


@cache
def load_config() -> dict:
    with open('config.json') as fid:
        return json.load(fid)


@cache
def get_config(key: str = None) -> Union[dict, list, str]:
    config = load_config()
    if key is not None:
        return config[key]
    return config


def owner_report() -> None:
    # Parse the kml file
    kml_file: str = get_config('kml_file')

    logging.info(f"Parsing KML file={kml_file} for owner report")
    doc = kmlutilities.parse_file(kml_file)
    processor = ArcGisProcessor(get_config('arcgis_servers'), get_config('layer_fields'), doc)
    processor.load()


def trail_length() -> None:
    summary_columns: List[str] = get_config('summary_columns')
    kml_file: str = get_config('kml_file')

    logging.info(f"Parsing KML file={kml_file} for trail length, summary columns={summary_columns}")
    doc = kmlutilities.parse_file(kml_file)

    for polygon_format in get_config('polygon_output_formats'):
        processor = kmlutilities.ReportProcessor(doc, polygon_format)
        processor.generate_report(summary_columns, get_config('report_file'))


def main():
    dir_path: str = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(dir_path, 'maps-analysis.log')
    logging.basicConfig(filename=log_file, format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='w', level=logging.DEBUG)
    logging.debug('Started')
    try:
        # trail_length()
        owner_report()
        logging.debug('Finished')

        sys.exit(0)
    except Exception as err:
        logging.exception('Fatal error in main')
        raise err


if __name__ == '__main__':
    main()
