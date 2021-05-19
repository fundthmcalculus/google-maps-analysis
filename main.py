import argparse
import logging
import os
import sys
from typing import Dict, Callable

import requests as requests

import kmlutilities


def create_function_map() -> Dict[str, Callable]:
    return {
            'traillength': trail_length
            }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help="Perform an action", choices=create_function_map().keys())
    parser.add_argument('--reportfile', help="Output csv file")
    parser.add_argument('--summarycolumns', help="Summary columns for report, separated by commas `,`")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--kmlfile', help="Get the KML file")
    group.add_argument('--mapurl', help="Map URL")
    return parser.parse_args()


def trail_length() -> None:
    args = parse_arguments()
    summary_columns = args.summarycolumns.split(',')

    if args.mapurl:
        r = requests.get(args.mapurl, allow_redirects=True)
        open('kml_file.kml', 'wb').write(r.content)
        args.kmlfile = 'kml_file.kml'

    logging.info(f"Parsing KML file={args.kmlfile}, summary columns={summary_columns}")
    doc = kmlutilities.parse_file(args.kmlfile)
    processor = kmlutilities.ReportProcessor()
    processor.load(doc)
    processor.generate_report(summary_columns, args.reportfile)


def main():
    dir_path: str = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(dir_path, 'maps-analysis.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='w',
                        level=logging.DEBUG)
    logging.debug('Started')
    try:
        args = parse_arguments()
        func = create_function_map()[args.command]
        func()
        logging.debug('Finished')

        sys.exit(0)
    except Exception as err:
        logging.exception('Fatal error in main:', exc_info=True)
        raise err


if __name__ == '__main__':
    main()
