import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import math


def _expand_rows_for_chunk(df_chunk, row_counts_chunk, chunk_index):
    """
    Process a subset of rows in one process.
    Each row i in df_chunk corresponds to row_counts_chunk[i] students.
    Return a concatenated DataFrame of per-student data for that chunk.
    """

    print(
        f"[CHUNK {chunk_index:3d}] Initializing processing for {len(df_chunk)} aggregated rows..."
    )

    df_list = []

    # Extract arrays for quick access
    fem_array = df_chunk["QT_MAT_BAS_FEM"].values
    masc_array = df_chunk["QT_MAT_BAS_MASC"].values
    nd_array = df_chunk["QT_MAT_BAS_ND"].values

    branca_array = df_chunk["QT_MAT_BAS_BRANCA"].values
    preta_array = df_chunk["QT_MAT_BAS_PRETA"].values
    parda_array = df_chunk["QT_MAT_BAS_PARDA"].values
    amarela_array = df_chunk["QT_MAT_BAS_AMARELA"].values
    indigena_array = df_chunk["QT_MAT_BAS_INDIGENA"].values

    a_0_3_array = df_chunk["QT_MAT_BAS_0_3"].values
    a_4_5_array = df_chunk["QT_MAT_BAS_4_5"].values
    a_6_10_array = df_chunk["QT_MAT_BAS_6_10"].values
    a_11_14_array = df_chunk["QT_MAT_BAS_11_14"].values
    a_15_17_array = df_chunk["QT_MAT_BAS_15_17"].values

    tot_array = df_chunk["TOTAL_ESTUDANTES"].values

    # Identify location columns (if they exist)
    location_cols = [
        col
        for col in [
            "NO_REGIAO",
            "CO_REGIAO",
            "SG_UF",
            "CO_UF",
            "NO_MUNICIPIO",
            "CO_MUNICIPIO",
            "CO_ENTIDADE",
        ]
        if col in df_chunk.columns
    ]
    loc_data = df_chunk[location_cols].values if location_cols else None

    gender_labels = ["F", "M", "ND"]
    race_labels = ["BRANCA", "PRETA", "PARDA", "AMARELA", "INDIGENA"]
    age_buckets = [
        (0, 3),
        (4, 5),
        (6, 10),
        (11, 14),
        (15, 17),
    ]

    def sample_ints(low, high, n):
        return np.random.randint(low, high + 1, size=n)

    for i in range(len(df_chunk)):
        k_i = row_counts_chunk[i]
        if k_i == 0:
            continue

        row_total = tot_array[i]

        # Gender distribution
        p_gender = (
            np.array(
                [
                    fem_array[i],
                    masc_array[i],
                    nd_array[i],
                ]
            )
            / row_total
        )
        gender_counts = np.random.multinomial(k_i, p_gender)

        # Race distribution
        p_race = (
            np.array(
                [
                    branca_array[i],
                    preta_array[i],
                    parda_array[i],
                    amarela_array[i],
                    indigena_array[i],
                ]
            )
            / row_total
        )
        race_counts = np.random.multinomial(k_i, p_race)

        # Age distribution
        p_age = (
            np.array(
                [
                    a_0_3_array[i],
                    a_4_5_array[i],
                    a_6_10_array[i],
                    a_11_14_array[i],
                    a_15_17_array[i],
                ]
            )
            / row_total
        )
        age_counts = np.random.multinomial(k_i, p_age)

        # Grades
        grades = np.random.normal(loc=50, scale=15, size=k_i)
        grades = np.around(np.clip(grades, 0, 100), 2)

        # Repeated lists for gender, race, age
        gender_list = []
        for g_label, gcount in zip(gender_labels, gender_counts):
            gender_list.extend([g_label] * gcount)

        race_list = []
        for r_label, rcount in zip(race_labels, race_counts):
            race_list.extend([r_label] * rcount)

        age_list = []
        for (lowA, highA), acount in zip(age_buckets, age_counts):
            if acount > 0:
                age_list.extend(sample_ints(lowA, highA, acount))

        np.random.shuffle(gender_list)
        np.random.shuffle(race_list)
        np.random.shuffle(age_list)

        df_temp = pd.DataFrame(
            {
                "NU_NOTA_REDACAO": grades,
                "TP_SEXO": gender_list,
                "TP_RACA": race_list,
                "NU_IDADE": age_list,
            }
        )

        # Add location columns if present
        if loc_data is not None:
            row_loc = loc_data[i]
            for col_idx, col_name in enumerate(location_cols):
                df_temp[col_name] = row_loc[col_idx]

        # Add chunk-based row index or similar
        # We can store the source aggregated row index if needed
        # If df_chunk still had the original index:
        #    df_temp["SOURCE_ROW"] = df_chunk.index[i]
        # or we can store i + an offset if needed
        df_temp["SOURCE_ROW"] = df_chunk.index[i]

        # 1) CLIP AGE between 6 and 18:
        df_temp["NU_IDADE"] = np.clip(df_temp["NU_IDADE"], 6, 18)

        # 2) ADD STUDENT_YEAR:
        df_temp["NO_ETAPA_ENSINO"] = np.select(
            [
                df_temp["NU_IDADE"].between(6, 10),
                df_temp["NU_IDADE"].between(11, 14),
                df_temp["NU_IDADE"].between(15, 18),
            ],
            ["FUNDAMENTAL I", "FUNDAMENTAL II", "ENSINO MÉDIO"],
            default="DESCONHECIDO",
        )

        df_list.append(df_temp)

    chunk_students = row_counts_chunk.sum()
    print(f"[CHUNK {chunk_index:3d}] Assigned {chunk_students} students...")

    # Concatenate all partial results for this chunk
    if len(df_list) > 0:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame(
            columns=["NU_NOTA_REDACAO", "TP_SEXO", "TP_RACA", "NU_IDADE"]
            + location_cols
            + ["SOURCE_ROW"]
        )


def generate_per_student_dataset_parallel(df_agg, N=100_000, n_jobs=16):
    """
    Parallel version of per-student dataset generator.
    Uses joblib to distribute the row-level expansion across multiple processes.
    """

    df_agg = df_agg.copy()
    # Fill NaNs in relevant columns
    cols_gender = ["QT_MAT_BAS_FEM", "QT_MAT_BAS_MASC", "QT_MAT_BAS_ND"]
    cols_race = [
        "QT_MAT_BAS_BRANCA",
        "QT_MAT_BAS_PRETA",
        "QT_MAT_BAS_PARDA",
        "QT_MAT_BAS_AMARELA",
        "QT_MAT_BAS_INDIGENA",
    ]
    cols_age = [
        "QT_MAT_BAS_0_3",
        "QT_MAT_BAS_4_5",
        "QT_MAT_BAS_6_10",
        "QT_MAT_BAS_11_14",
        "QT_MAT_BAS_15_17",
        "QT_MAT_BAS_18_MAIS",
    ]

    for c in cols_gender + cols_race + cols_age:
        df_agg[c] = df_agg[c].fillna(0)

    # Ensure TOTAL_ESTUDANTES
    if "TOTAL_ESTUDANTES" not in df_agg.columns:
        df_agg["TOTAL_ESTUDANTES"] = (
            df_agg["QT_MAT_BAS_FEM"]
            + df_agg["QT_MAT_BAS_MASC"]
            + df_agg["QT_MAT_BAS_ND"]
        )
    df_agg = df_agg[df_agg["TOTAL_ESTUDANTES"] > 0].copy()

    # Probability for each row
    total_sum = df_agg["TOTAL_ESTUDANTES"].sum()
    df_agg["p_row"] = df_agg["TOTAL_ESTUDANTES"] / total_sum

    # Reset index so row IDs are 0..(len(df_agg)-1)
    df_agg = df_agg.reset_index(drop=True)

    # 1) Single global choice for row assignment
    row_ids = np.random.choice(df_agg.index, size=N, p=df_agg["p_row"].values)
    # 2) Count how many students in each row
    row_counts = np.bincount(row_ids, minlength=len(df_agg))

    # 3) Split df_agg + row_counts into chunks
    num_rows = len(df_agg)
    chunk_size = math.ceil(num_rows / n_jobs)

    chunks = []
    for job_i in range(n_jobs):
        start = job_i * chunk_size
        end = min((job_i + 1) * chunk_size, num_rows)

        if start >= end:
            break

        df_chunk = df_agg.iloc[start:end].copy()
        row_counts_chunk = row_counts[start:end]
        chunks.append((df_chunk, row_counts_chunk, job_i))

    # 4) Process each chunk in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_expand_rows_for_chunk)(
            df_chunk=chunk[0], row_counts_chunk=chunk[1], chunk_index=chunk[2]
        )
        for chunk in chunks
    )

    # 5) Concatenate the partial DataFrames
    df_students = pd.concat(results, ignore_index=True)
    return df_students


import time

print("Loading microdata...")
microdados_censo_2023 = pd.read_csv(
    "https://raw.githubusercontent.com/calriz/microdados-inep-public/refs/heads/main/microdados_ed_basica_2023_filtered.csv"
)
print("Done.")

print("Generating students...")
start = time.time()

N = 100_000
enem_2023_df = generate_per_student_dataset_parallel(
    microdados_censo_2023, N, n_jobs=16
)
print(f"Generated in {time.time() - start:.2f}s.")

enem_2023_df.to_csv("./enem23.csv", index=False)
print(enem_2023_df.head(5))
