from pathlib import Path
from typing import Dict, List

import pandas as pd

from forecasting_poverty_peru.config import PROCESSED_PATH, RAW_PATH

COLUMNS_MAPPING: Dict[str, str] = {
    'ano': 'year',
    'mes': 'month',
    'conglome': 'conglomerate',
    'vivienda': 'house',
    'hogar': 'household',
    'latitud': 'latitude',
    'longitud': 'longitude',
    'nbi1': 'adequate'
}

DTYPES_MAPPING: Dict[str, str] = {
    'year': 'int16',
    'month': 'int8',
    'conglomerate': 'category',
    'house': 'category',
    'household': 'category',
    'latitude': 'float64',
    'longitude': 'float64',
    'adequate': 'category'
}

SELECTED_COLUMNS: List[str] = list(COLUMNS_MAPPING.keys())


def clean_columns_names(df: pd.DataFrame) -> pd.DataFrame:
    replacements = {'$': '_', 'Ñ': 'N'}
    return df.rename(columns=lambda x: (
        x.translate(str.maketrans(replacements))
         .lower()
         .strip()
    ))


def transform_adequate_column(s: pd.Series) -> pd.Series:
    adequacy_mapping = {
        'Vivienda adecuada': True,
        'Vivienda inadecuada': False
    }
    return s.map(adequacy_mapping).astype('boolean')


def process_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns_names(raw_df)
    df = df[SELECTED_COLUMNS].rename(columns=COLUMNS_MAPPING)
    df = df.dropna().copy()
    df = df.astype(DTYPES_MAPPING)
    df['adequate'] = transform_adequate_column(df['adequate'])
    return (
        df.groupby(
            ['year', 'month', 'conglomerate', 'longitude', 'latitude'],
            observed=True
        )
        .agg(adequate=('adequate', 'all'))
        .reset_index()
    )


def main():
    input_filepath = RAW_PATH / 'enaho' / '2023' / 'data.sav'
    output_filepath = PROCESSED_PATH / 'sienaho_rgb' / 'class_names.pkl'

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_spss(input_filepath)
    processed_df = process_data(raw_df)
    processed_df.to_pickle(output_filepath)


if __name__ == '__main__':
    main()
