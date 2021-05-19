import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pykml.parser
from shapely.geometry import LineString, Polygon


def process_kml_data(doc) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    # Since KML is always a data table, every element has the same extended data
    trail_columns = ['Trail Name', 'Length [mile]']
    extended_columns = get_extended_data(doc.Document.Placemark[0].ExtendedData).keys()
    trail_columns.extend(extended_columns)
    trail_data = pd.DataFrame(columns=trail_columns)
    report_data = pd.DataFrame(columns=['Region Name', 'Area [mi^2]'])

    all_trails = [placemark for placemark in doc.Document.Placemark if hasattr(placemark, 'LineString')]
    all_trails.extend([placemark for placemark in doc.Document.Placemark
                       if hasattr(placemark, 'Polygon') and not is_report(placemark)])
    report_polygons: List = [placemark for placemark in doc.Document.Placemark
                             if hasattr(placemark, 'Polygon') and is_report(placemark)]

    trail_data = calculate_trail_data(trail_data, all_trails)
    report_data, trails_report = calculate_report_data(report_data, report_polygons, all_trails, trail_columns)

    return trail_data, report_data, trails_report


def create_pivot_table_summaries(trail_data: pd.DataFrame, summary_columns: List[str], report_file: str) -> None:
    # Summarize by type, status, and official
    for column in summary_columns:
        summary_pivot_table(trail_data, column, report_file)


def write_report_files(trail_data: pd.DataFrame, report_data: pd.DataFrame, trails_report: Dict[str, pd.DataFrame],
                       report_file: str) -> None:
    # Report polygons
    report_data.to_csv(report_file.replace('.csv', f'_report_polygon.csv'), index=False)
    trail_data.to_csv(report_file, index=False)
    for (report_name, report) in trails_report.items():
        report.to_csv(report_file.replace('.csv', f'_{report_name}.csv'), index=False)


def get_shapely_shape(placemark):
    if hasattr(placemark, 'LineString'):
        return LineString(ECEF_to_ENU(geodetic_to_ECEF(parse_coordinates(placemark.LineString.coordinates))))
    elif hasattr(placemark, 'Polygon'):
        return Polygon(
            ECEF_to_ENU(geodetic_to_ECEF(parse_coordinates(placemark.Polygon.outerBoundaryIs.LinearRing.coordinates))))
    else:
        raise NotImplementedError


def calculate_report_data(report_data: pd.DataFrame, report_polygons: List, trail_shapes: List,
                          trail_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    report_dict: Dict[str, pd.DataFrame] = dict()
    for report in report_polygons:
        report_name = report.name.text.strip()
        logging.info(f'Polygon: {report_name}')
        report_poly = get_shapely_shape(report)
        row = [report_name, report_poly.area / 2589988.110336]
        report_data = report_data.append(pd.DataFrame([row], columns=report_data.columns))

        # Trim to shape and report
        trails_report = pd.DataFrame(columns=trail_columns)
        trails_report = calculate_trail_data(trails_report, trail_shapes, report_poly)
        report_dict[report_name] = trails_report

    return report_data, report_dict


def calculate_trail_data(data, trail_lines, trim_poly=None):
    for placemark in trail_lines:
        placemark_name = placemark.name.text.strip()
        logging.info(f'Trail: {placemark_name} \\ {trim_poly}')
        ext_data = get_extended_data(placemark.ExtendedData)
        shapely_trail = get_shapely_shape(placemark)
        shapely_trail = shapely_trail.intersection(trim_poly) if trim_poly else shapely_trail
        distance = shapely_trail.length / 1609  # m -> mile
        if distance == 0.0:
            continue

        row = [placemark_name, distance]
        row.extend(ext_data.values())
        data = data.append(pd.DataFrame([row], columns=data.columns))
    return data


def summary_pivot_table(data: pd.DataFrame, pivot_column: str, report_file: str, data_column: str = 'Length [mile]') -> None:
    official = data.pivot_table(index=pivot_column, values=data_column, aggfunc='sum')
    official = official.rename({'': '[BLANK]'})

    official.to_csv(report_file.replace('.csv', f'_pivot_{pivot_column}.csv'))


def save_dataframe(file_path: str, data: pd.DataFrame) -> None:
    data.to_csv(file_path, index=False)


def parse_coordinates(coordinates) -> np.ndarray:
    # Lat, Lon, h
    coord_strings = split_string(coordinates.text)
    coords = np.zeros((len(coord_strings), 3))
    for ij in range(len(coord_strings)):
        coords[ij, :] = geo_coords(coord_strings[ij])
    return coords


def geodetic_to_ECEF(coords: np.ndarray) -> np.ndarray:
    a = 6378137.0  # m
    b = 6356752.3  # m
    e2 = 1 - b ** 2 / a ** 2

    N_phi = lambda phi: a / np.sqrt(1 - e2 * np.sin(np.radians(phi)) ** 2)
    X = lambda phi, lamb: N_phi(phi) * cosd(phi) * cosd(lamb)
    Y = lambda phi, lamb: N_phi(phi) * cosd(phi) * sind(lamb)
    Z = lambda phi, alt: (b ** 2 / a ** 2 * N_phi(phi) + alt) * sind(phi)

    ecef = np.zeros(coords.shape)
    if len(ecef.shape) > 1:
        for ij in range(coords.shape[0]):
            p = coords[ij, 0]
            l = coords[ij, 1]
            h = coords[ij, 2]
            ecef[ij, :] = np.array([X(p, l), Y(p, l), Z(p, h)])
    else:
        p = coords[0]
        l = coords[1]
        h = coords[2]
        ecef = np.array([X(p, l), Y(p, l), Z(p, h)])

    return ecef


def ECEF_to_ENU(coords: np.ndarray) -> np.ndarray:
    # Reference point is Reser Bicycle Outfitters
    ref_lat = 39.09029667468314  # deg
    ref_lon = -84.49260971579635  # deg
    ref_alt = 156.058  # m
    ref_XYZ = geodetic_to_ECEF(np.array([ref_lat, ref_lon, ref_alt]))
    # Construct the matrix
    mat_transform = np.array([[-sind(ref_lon), cosd(ref_lon), 0],
                              [-sind(ref_lat) * cosd(ref_lon), -sind(ref_lat) * sind(ref_lon), cosd(ref_lat)],
                              [cosd(ref_lat) * cosd(ref_lon), cosd(ref_lat) * sind(ref_lon), sind(ref_lat)]])
    enu_coords = np.zeros(coords.shape)
    for ij in range(coords.shape[0]):
        enu_coords[ij, :] = np.dot(mat_transform, coords[ij, :] - ref_XYZ)

    return enu_coords


def is_report(placemark) -> bool:
    return get_extended_data(placemark.ExtendedData).get('official', '').upper() == 'REPORT'


def parse_file(file_name: str):
    with open(file_name) as f:
        return pykml.parser.parse(f).getroot()


def geo_coords(s: str) -> np.ndarray:
    # Lat, Lon, h
    return np.array(tuple(map(float, s.split(','))))[[1, 0, 2]]


def split_string(x: str) -> List[str]:
    return [line.strip() for line in x.split('\n') if line.strip()]


def get_extended_data(x) -> Dict[str, str]:
    return dict([(d.attrib.values()[0], csv_format_string((d.value.text or "").strip())) for d in x.Data])


def csv_format_string(x: str) -> str:
    return x.replace('\n', ' ').replace('\r', ' ').replace('\f', ' ').replace(',', '')


def sind(x: float) -> float:
    return np.sin(np.radians(x))


def cosd(x: float) -> float:
    return np.cos(np.radians(x))
