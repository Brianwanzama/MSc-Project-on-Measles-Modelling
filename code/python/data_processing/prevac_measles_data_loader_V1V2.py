# ============================================================
# prevac_measles_data_loader_V1V2.py
# SFNN feature engineering — West Bengal measles
# V1+V2 version: adds MCV2 features + V1V2 susceptibles
#
# CHANGES FROM prevac_measles_data_loader.py:
#   1. v2_data_loc parameter added
#   2. V2.csv loaded -> v2_lag_26, v2_lag_52 features
#   3. susc_data_loc -> tsir_susceptibles_V1V2.csv
#   4. output -> prefit_cases_V1V2/
#   5. drops v2_* cols from susc_df (V1V2 file has extra cols)
#
# All other features identical to V1 version for fair comparison
# ============================================================

import numpy as np
import pandas as pd
import os

np.random.seed(2)

base             = "/home/brain/Msc_project/"
output_directory = base + "output/data/prefit_cases_V1V2/"
os.makedirs(output_directory, exist_ok=True)


def create_measles_data(
        k,
        t_lag,
        cases_data_loc,
        pop_data_loc,
        coords_data_loc,
        birth_data_loc,
        susc_data_loc,
        v1_data_loc,
        v2_data_loc,           # NEW — MCV2 coverage
        write_to_file=False,
        write_loc=base + "output/data/prefit_cases_V1V2/",
        top_12_cities=False,
        verbose=False,
        current_births=False,
        cutoff_year=2017
        ):

    # ── LOAD DATA ─────────────────────────────────────────────
    cases      = pd.read_csv(cases_data_loc).rename(
                     columns={"Unnamed: 0": "time"})
    population = pd.read_csv(pop_data_loc).rename(
                     columns={"Unnamed: 0": "time"})
    coords     = pd.read_csv(coords_data_loc, index_col=0).T

    births = (pd.read_csv(birth_data_loc, index_col=0)
                .reset_index()
                .rename(columns={"index": "year"}))

    # Susceptibles from V1+V2 TSIR reconstruction
    susc_df = pd.read_csv(susc_data_loc)
    if "Unnamed: 0" in susc_df.columns:
        susc_df = susc_df.rename(columns={"Unnamed: 0": "time"})

    # MCV1
    v1      = pd.read_csv(v1_data_loc)
    v1_long = pd.melt(v1, id_vars='Year',
                      var_name='city', value_name='v1')
    v1_long.rename(columns={'Year': 'year'}, inplace=True)

    # MCV2 — NEW
    v2      = pd.read_csv(v2_data_loc)
    v2_long = pd.melt(v2, id_vars='Year',
                      var_name='city', value_name='v2')
    v2_long.rename(columns={'Year': 'year'}, inplace=True)

    # Strip whitespace
    coords.index       = coords.index.str.strip()
    cases.columns      = cases.columns.str.strip()
    population.columns = population.columns.str.strip()

    if top_12_cities:
        top_n_city_names = population.iloc[0, 1:].nlargest(12).index
        cases      = cases[top_n_city_names.insert(0, 'time')]
        population = population[top_n_city_names.insert(0, 'time')]
        coords     = coords.loc[top_n_city_names]
        susc_df    = susc_df[susc_df['city'].isin(top_n_city_names)]
        births     = births[[col for col in
                             top_n_city_names.insert(0, 'year').tolist()
                             if col in births.columns]]
        v1_long    = v1_long[v1_long['city'].isin(top_n_city_names)]
        v2_long    = v2_long[v2_long['city'].isin(top_n_city_names)]

    # ── CASES ─────────────────────────────────────────────────
    cases_long = pd.melt(cases, id_vars='time',
                         var_name='city', value_name='cases')
    cases_long['city']        = cases_long['city'].str.strip()
    cases_long['time']        = np.round(cases_long['time'], 5)
    cases_long['cases_trans'] = np.log(cases_long['cases'] + 1)

    cases_groups = cases_long.groupby(['city'])
    cases_mean   = cases_groups.transform("mean")
    cases_std    = cases_groups.transform("std")
    cases_long['cases_trans'] = ((cases_long['cases_trans'] -
                                   cases_mean['cases_trans']) /
                                  cases_std['cases_trans'])

    cases_transform_output = cases_long[['time', 'city']].copy()
    cases_transform_output['cases_mean'] = cases_mean['cases_trans']
    cases_transform_output['cases_std']  = cases_std['cases_trans']

    temp_dfs = []
    for i in range(k, t_lag + 1):
        lag_df      = cases_long.groupby('city')['cases_trans'].shift(i)
        lag_df.name = "cases_lag_" + str(i)
        temp_dfs.append(lag_df)
    cases_long = cases_long.join(pd.concat(temp_dfs, axis=1))

    # ── POPULATION ────────────────────────────────────────────
    pop_long = pd.melt(population, id_vars='time',
                       var_name='city', value_name='pop')
    pop_long['city']    = pop_long['city'].str.strip()
    pop_long['time']    = np.round(pop_long['time'], 5)
    pop_long['pop_std'] = ((pop_long['pop'] - pop_long['pop'].mean()) /
                            pop_long['pop'].std())

    k_temp = ((k + 25) // 26) * 26
    pop_long["pop_lag_" + str(k_temp)] = (pop_long
        .groupby('city')["pop_std"]
        .shift(k_temp))
    pop_long.drop(columns='pop_std', inplace=True)

    cases_long = cases_long.merge(pop_long, how='left',
                                  on=['time', 'city'])

    # ── SUSCEPTIBLES (V1V2 reconstruction) ────────────────────
    # Drop all non-essential columns — handle V1V2 extra columns
    cols_to_drop = [c for c in
                    ['births', 'births_original', 'births_adjusted',
                     'pop', 'cases',
                     'v1_annual', 'v1_biweekly', 'v1_lagged',
                     'v2_annual', 'v2_biweekly', 'v2_lagged']
                    if c in susc_df.columns]
    susc_df.drop(columns=cols_to_drop, inplace=True)

    if 'city' in susc_df.columns:
        susc_df['city'] = susc_df['city'].str.strip()

    susc_df['time'] = np.round(susc_df['time'], 5)

    susc_groups = susc_df.groupby(['city'])
    susc_mean   = susc_groups.transform("mean")
    susc_std    = susc_groups.transform("std")
    susc_df['susc_trans'] = ((susc_df['susc'] - susc_mean['susc']) /
                               susc_std['susc'])
    susc_df['susc_trans'] = np.where(pd.isna(susc_df['susc_trans']),
                                     susc_mean['susc'],
                                     susc_df['susc_trans'])

    temp_dfs = []
    for i in range(k, t_lag + 1):
        lag_df      = susc_df.groupby('city')['susc_trans'].shift(i)
        lag_df.name = "susc_lag_" + str(i)
        temp_dfs.append(lag_df)
    susc_df    = susc_df.join(pd.concat(temp_dfs, axis=1))
    cases_long = cases_long.merge(susc_df, how='left',
                                  on=['time', 'city'])

    # ── BIRTHS ────────────────────────────────────────────────
    births      = births.fillna(births.mean())
    births      = np.round(births)
    births_long = pd.melt(births, id_vars='year',
                          var_name='city', value_name='births')
    births_long['city']   = births_long['city'].str.strip()
    births_long['births'] = births_long['births'] / 26

    births_long['births_std'] = ((births_long['births'] -
                                   births_long['births'].mean()) /
                                  births_long['births'].std())

    cases_long['year'] = np.floor(cases_long['time']).astype(int)
    cases_long = cases_long.merge(births_long, how='left',
                                  on=['year', 'city'])

    # ── MCV1 FEATURES ─────────────────────────────────────────
    v1_long['city']   = v1_long['city'].str.strip()
    v1_long['v1_std'] = ((v1_long['v1'] - v1_long['v1'].mean()) /
                          v1_long['v1'].std())

    cases_long = cases_long.merge(
        v1_long[['city', 'year', 'v1_std']],
        how='left', on=['city', 'year']
    )
    cases_long = cases_long.sort_values(['city', 'time'])

    cases_long['v1_std'] = (cases_long
        .groupby('city')['v1_std']
        .transform(lambda s: s.interpolate(
            method='linear', limit_direction='both')))

    v1_col_1 = 'v1_lag_' + str(k_temp)
    v1_col_2 = 'v1_lag_' + str(k_temp + 26)

    cases_long[v1_col_1] = (cases_long
        .groupby('city')['v1_std']
        .shift(k_temp))
    cases_long[v1_col_2] = (cases_long
        .groupby('city')['v1_std']
        .shift(k_temp + 26))

    for col in [v1_col_1, v1_col_2]:
        cases_long[col] = (cases_long
            .groupby('city')[col]
            .transform(lambda s: s.bfill()))

    cases_long.drop(columns='v1_std', inplace=True)

    if verbose:
        print(f"  V1 features: {v1_col_1}, {v1_col_2}")

    # ── MCV2 FEATURES — NEW ───────────────────────────────────
    # Same lag structure as V1: two lags at k_temp and k_temp+26
    # V2 = 0 for 2008-2010, national estimates 2011-2016,
    #      district-level 2017-2019
    v2_long['city']   = v2_long['city'].str.strip()
    v2_long['v2']     = pd.to_numeric(v2_long['v2'],
                                       errors='coerce').fillna(0)
    v2_long['v2_std'] = ((v2_long['v2'] - v2_long['v2'].mean()) /
                          (v2_long['v2'].std() + 1e-8))

    cases_long = cases_long.merge(
        v2_long[['city', 'year', 'v2_std']],
        how='left', on=['city', 'year']
    )
    cases_long = cases_long.sort_values(['city', 'time'])

    cases_long['v2_std'] = (cases_long
        .groupby('city')['v2_std']
        .transform(lambda s: s.interpolate(
            method='linear', limit_direction='both')))
    cases_long['v2_std'] = cases_long['v2_std'].fillna(0)

    v2_col_1 = 'v2_lag_' + str(k_temp)
    v2_col_2 = 'v2_lag_' + str(k_temp + 26)

    cases_long[v2_col_1] = (cases_long
        .groupby('city')['v2_std']
        .shift(k_temp))
    cases_long[v2_col_2] = (cases_long
        .groupby('city')['v2_std']
        .shift(k_temp + 26))

    for col in [v2_col_1, v2_col_2]:
        cases_long[col] = (cases_long
            .groupby('city')[col]
            .transform(lambda s: s.bfill()))

    cases_long.drop(columns='v2_std', inplace=True)

    if verbose:
        print(f"  V2 features: {v2_col_1}, {v2_col_2}")

    # ── BIRTHS LAG ────────────────────────────────────────────
    cases_long.drop(columns='year', inplace=True)

    cases_long['births_std'] = np.where(
        pd.isna(cases_long['births_std']), 0,
        cases_long['births_std'])

    cases_long["births_lag_" + str(k_temp)] = (cases_long
        .groupby(["city"])["births_std"]
        .shift(k_temp))

    if current_births:
        cases_long["births_lag_0"] = cases_long["births_std"]

    cases_long.drop(columns='births_std', inplace=True)

    # ── TOP 7 BIG CITY CASES ──────────────────────────────────
    top_7_cities = population.iloc[0, 1:].nlargest(7).index

    for i in top_7_cities:
        temp_cases_all = cases_long[cases_long['city'] == i]
        temp_cases     = temp_cases_all.filter(
            regex="cases_lag_.*", axis=1)
        safe_name = i.lower().replace(" ", "_")
        temp_cases = temp_cases.rename(
            columns=lambda x: x.replace(
                "cases_lag_", "cases_" + safe_name + "_lag_"))
        temp_cases['time'] = temp_cases_all['time'].values
        cases_long = cases_long.merge(temp_cases,
                                      how='left', on=['time'])
        if verbose:
            print(f"  {i} cases joined")

    # ── DISTANCES ─────────────────────────────────────────────
    def spherical_dist(pos1, pos2, r=3958.75):
        pos1      = pos1 * np.pi / 180
        pos2      = pos2 * np.pi / 180
        cos_lat1  = np.cos(pos1[..., 0])
        cos_lat2  = np.cos(pos2[..., 0])
        cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
        cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
        return r * np.arccos(
            cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

    coord_dist = spherical_dist(
        coords.values[:, None], coords.values)

    for i in top_7_cities:
        safe_name   = i.lower().replace(" ", "_")
        city_i_ind  = np.where(i == coords.index)[0][0]
        city_i_dist = coord_dist[city_i_ind]
        col_name    = 'dist_' + safe_name
        pd_temp     = pd.DataFrame(
            data={'city': coords.index, col_name: city_i_dist})
        pd_temp[col_name] = ((pd_temp[col_name] -
                               pd_temp[col_name].mean()) /
                              pd_temp[col_name].std())
        cases_long = cases_long.merge(pd_temp,
                                      how='left', on=['city'])

    # ── NEAREST BIG CITY LAGS ─────────────────────────────────
    city_locs = [np.where(coords.index == c)[0][0]
                 for c in top_7_cities]

    nearest_big_city_locs      = []
    nearest_big_city_distances = []
    for i in range(coord_dist.shape[0]):
        distances = coord_dist[i][city_locs]
        min_idx   = np.argmin(distances)
        nearest_big_city_distances.append(distances[min_idx])
        nearest_big_city_locs.append(city_locs[min_idx])

    nearest_big_city = coords.index[nearest_big_city_locs]

    cases_long         = cases_long.reset_index(drop=True)
    city_array         = cases_long['city'].values
    coords_index_array = np.array(coords.index)

    nearest_big_city_ordered = [
        np.where(city_array[i] == coords_index_array)[0][0]
        for i in range(len(city_array))
    ]

    cases_long['nearest_big_city'] = (
        nearest_big_city[nearest_big_city_ordered])
    nbc_dist_arr = np.array(nearest_big_city_distances)
    cases_long['nearest_big_city_distances'] = (
        (nbc_dist_arr[nearest_big_city_ordered] -
         nbc_dist_arr.mean()) / nbc_dist_arr.std())

    lag_vec            = ["cases_lag_" + str(i)
                          for i in range(k, t_lag + 1)]
    to_join_columns    = np.concatenate((['time', 'city'], lag_vec))
    cases_long_to_join = cases_long[to_join_columns].copy()
    cases_long_to_join.columns = [
        x.replace('cases_lag_', 'cases_nbc_lag_')
        for x in cases_long_to_join.columns
    ]
    cases_long_to_join.rename(
        columns={'city': 'nearest_big_city'}, inplace=True)

    cases_long = cases_long.merge(
        cases_long_to_join, how='left',
        on=['time', 'nearest_big_city'])
    cases_long = cases_long.drop('nearest_big_city', axis=1)

    # ── NEAREST 10 CITIES ─────────────────────────────────────
    coord_dist_pd           = pd.DataFrame(
        coord_dist, columns=coords.index)
    coord_dist_pd['city_a'] = coords.index
    coord_dist_long         = coord_dist_pd.melt(
        id_vars=['city_a'], var_name='city_b', value_name='dist')

    nearest_10_idx   = np.argsort(coord_dist, axis=1)[:, 1:11]
    coords_names     = np.array(coords.index)
    nearest_10_names = coords_names[nearest_10_idx]

    nearest_10_cities_pd = pd.DataFrame(
        nearest_10_names,
        columns=['nearest_' + str(j) + '_city'
                 for j in range(1, 11)]
    )
    nearest_10_cities_pd['city'] = coords.index

    cases_long         = cases_long.merge(
        nearest_10_cities_pd, how='left', on=['city'])
    lag_vec            = ["cases_lag_" + str(i)
                          for i in range(k, t_lag + 1)]
    to_join_columns    = np.concatenate((['time', 'city'], lag_vec))
    cases_long_to_join = cases_long[to_join_columns]

    for j in range(1, 11):
        cases_long_to_join_temp = cases_long_to_join.copy()
        cases_long_to_join_temp.columns = [
            x.replace('cases_lag_', 'cases_nc_' + str(j) + '_lag_')
            for x in cases_long_to_join_temp.columns
        ]
        cases_long_to_join_temp.rename(
            columns={'city': 'nearest_' + str(j) + '_city'},
            inplace=True)

        cases_long = cases_long.merge(
            cases_long_to_join_temp, how='left',
            on=['time', 'nearest_' + str(j) + '_city'])

        cases_long = cases_long.merge(
            coord_dist_long, how='left',
            left_on=['city', 'nearest_' + str(j) + '_city'],
            right_on=['city_a', 'city_b'])
        cases_long.drop(['city_a', 'city_b'], axis=1, inplace=True)
        cases_long['dist'] = ((cases_long['dist'] -
                                cases_long['dist'].mean()) /
                               cases_long['dist'].std())
        cases_long.rename(
            columns={'dist': 'nearest_' + str(j) + '_city_dist'},
            inplace=True)
        cases_long.drop(
            'nearest_' + str(j) + '_city', axis=1, inplace=True)

    # ── TRIMMING ──────────────────────────────────────────────
    cases_long = cases_long[cases_long.time < 2020]

    drop_times = np.round(cases['time'][0:t_lag].values, 5)
    cases_long = cases_long[
        ~cases_long['time'].isin(drop_times)]

    cases_long['split'] = np.where(
        cases_long['time'] >= cutoff_year, 'test', 'train')

    if verbose:
        n_train = (cases_long['split'] == 'train').sum()
        n_test  = (cases_long['split'] == 'test').sum()
        n_v1    = sum(1 for c in cases_long.columns if 'v1_lag' in c)
        n_v2    = sum(1 for c in cases_long.columns if 'v2_lag' in c)
        print(f"  Train: {n_train} | Test: {n_test} | "
              f"Total: {len(cases_long)}")
        print(f"  V1 features: {n_v1} | V2 features: {n_v2}")

    # ── SAVE ──────────────────────────────────────────────────
    if write_to_file:
        out_name = f"k{k}_tlag{t_lag}.gzip"
        trn_name = f"k{k}_tlag{t_lag}_cases_transform_output.gzip"

        cases_long.to_parquet(
            write_loc + out_name, compression="gzip")
        cases_transform_output.to_parquet(
            write_loc + trn_name, compression="gzip")

        if verbose:
            print(f"  Saved: {out_name} "
                  f"({len(cases_long)} rows, "
                  f"{len(cases_long.columns)} cols)")
    else:
        return cases_long, cases_transform_output


def run_experiments():

    base = "/home/brain/Msc_project/"

    cases_data_loc  = base + "data/cases_biweekly.csv"
    pop_data_loc    = base + "data/inferred_popn.csv"
    coords_data_loc = base + "data/coordinates.csv"
    # V1V2 susceptibles
    susc_data_loc   = base + "output/data/tsir/tsir_susceptibles_V1V2.csv"
    birth_data_loc  = base + "data/Births.csv"
    v1_data_loc     = base + "data/V1.csv"
    v2_data_loc     = base + "data/V2.csv"   # NEW
    write_loc       = base + "output/data/prefit_cases_V1V2/"

    os.makedirs(write_loc, exist_ok=True)

    k_values     = [1, 4, 12, 20, 34]
    t_lag_values = [26, 52]

    total = sum(1 for k in k_values
                for t in t_lag_values if k < t)
    count = 0

    print(f"Grid: k={k_values}, t_lag={t_lag_values}")
    print(f"susc: tsir_susceptibles_V1V2.csv")
    print(f"V2:   {v2_data_loc}")
    print(f"Output: {write_loc}")
    print(f"Total combinations: {total}\n")

    for k in k_values:
        for t_lag in t_lag_values:
            if k < t_lag:
                count += 1
                print(f"\n[{count}/{total}] k={k}, t_lag={t_lag}")
                create_measles_data(
                    k               = k,
                    t_lag           = t_lag,
                    cases_data_loc  = cases_data_loc,
                    pop_data_loc    = pop_data_loc,
                    coords_data_loc = coords_data_loc,
                    susc_data_loc   = susc_data_loc,
                    birth_data_loc  = birth_data_loc,
                    v1_data_loc     = v1_data_loc,
                    v2_data_loc     = v2_data_loc,
                    write_to_file   = True,
                    write_loc       = write_loc,
                    cutoff_year     = 2017,
                    verbose         = True
                )

    print(f"\nAll {total} combinations complete.")
    print(f"Output: {write_loc}")
    print("Files produced:")
    for k in k_values:
        for t_lag in t_lag_values:
            if k < t_lag:
                print(f"  k{k}_tlag{t_lag}.gzip")
                print(f"  k{k}_tlag{t_lag}_cases_transform_output.gzip")


if __name__ == '__main__':
    run_experiments()
