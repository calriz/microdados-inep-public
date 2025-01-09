# https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar

import dask.dataframe as dd
import numpy as np

print("Reading DF...")

df = dd.read_csv(
    "../microdados_ed_basica_2023.csv",
    encoding="windows-1252",
    delimiter=";",
    low_memory=False,
    dtype={
        "CO_ORGAO_REGIONAL": "string",
        "NU_DDD": "string",
        "NU_ENDERECO": "string",
    },
)

columns = [
    "NO_REGIAO",
    "CO_REGIAO",
    "SG_UF",
    "CO_UF",
    "NO_MUNICIPIO",
    "CO_MUNICIPIO",
    "CO_ENTIDADE",
    "QT_MAT_BAS_FEM",
    "QT_MAT_BAS_MASC",
    "QT_MAT_BAS_ND",
    "QT_MAT_BAS_BRANCA",
    "QT_MAT_BAS_PRETA",
    "QT_MAT_BAS_PARDA",
    "QT_MAT_BAS_AMARELA",
    "QT_MAT_BAS_INDIGENA",
    "QT_MAT_BAS_0_3",
    "QT_MAT_BAS_4_5",
    "QT_MAT_BAS_6_10",
    "QT_MAT_BAS_11_14",
    "QT_MAT_BAS_15_17",
    "QT_MAT_BAS_18_MAIS",
]

df = df[columns]

print("Writing DF...")

df.to_csv(
    "./microdados_ed_basica_2023_filtered.csv",
    index=False,
    single_file=True,
)
