import math
import os
import glob
import time

import networkx as nx
import pandas as pd
import polars as pl

def time_execution(method):
    """
    Decorator that measures the execution time of the decorated method.
    """
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = method(self, *args, **kwargs)
        end = time.time()
        elapsed = end - start
        print(f"{method.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper


class DataLoader:
    def __init__(
        self,
        hs_code: int = None,
        data_path_pattern=os.path.join("dataset", "BACI*.csv"),
        country_codes_path=os.path.join("dataset", "country_codes_V202401b.csv"),
        product_codes_path=os.path.join("dataset", "product_codes_HS22_V202401b.csv"),
        gravity_path=os.path.join("gravity", "Gravity_V202211.csv")
    ):
        self.hs_code = hs_code

        self.data_path_pattern = data_path_pattern
        self.country_codes_path = country_codes_path
        self.product_codes_path = product_codes_path
        self.gravity_path = gravity_path

        self._initialize_data()

    @time_execution
    def _initialize_data(self):
        # Loading base data
        self.df = self._load_baci_data()
        self.country_codes = self._load_country_codes()
        self.product_codes = self._load_product_codes()
        self.gravity = self._preprocess_gravity_data()

        # Merge names and gravity
        self.df = self._merge_country_and_product_names_with_gravity()

        # Final cleaning
        self.df = self._clean_data()


    def _load_baci_data(self) -> pl.DataFrame:
        all_files = glob.glob(self.data_path_pattern)
        dataframes = [pl.read_csv(file) for file in all_files]
        df = pl.concat(dataframes)
        df = df.with_columns([
            pl.col("i").cast(pl.Int64),
            pl.col("j").cast(pl.Int64),
            pl.col("k").cast(pl.Int64)
        ])

        if self.hs_code is not None:
            df = df.filter(pl.col("k") == self.hs_code)

        return df


    def _load_country_codes(self) -> pl.DataFrame:
        cc = pl.read_csv(self.country_codes_path)
        cc = cc.with_columns(pl.col("country_code").cast(pl.Int64))
        return cc


    def _load_product_codes(self) -> pl.DataFrame:
        pc = pl.read_csv(self.product_codes_path)
        pc = pc.with_columns(pl.col("code").cast(pl.Int64))
        return pc


    def _preprocess_gravity_data(self) -> pl.DataFrame:
        gravity = pl.read_csv(self.gravity_path, ignore_errors=True)
        gravity = gravity.select([
            "iso3_o", "iso3_d", "year", "gdpcap_d", "gdpcap_o", "tradeflow_baci",
            "diplo_disagreement", "comrelig", "distw_harmonic", "pop_o", "pop_d",
            "wto_o", "wto_d", "eu_o", "eu_d", "entry_cost_o", "entry_cost_d",
            "comlang_off", "gatt_o", "gatt_d", "fta_wto", "tradeflow_imf_o",
            "tradeflow_imf_d"
        ])

        gravity = gravity.filter(pl.col("year") > 2000)
        gravity = gravity.unique(subset=["iso3_o", "iso3_d", "year"], keep="first")

        # Merge Country Names Into Gravity
        gravity = gravity.join(
            self.country_codes.select(["country_name", "country_iso3"]),
            left_on="iso3_o",
            right_on="country_iso3",
            how="left"
        ).rename({"country_name": "country_o"})

        gravity = gravity.join(
            self.country_codes.select(["country_name", "country_iso3"]),
            left_on="iso3_d",
            right_on="country_iso3",
            how="left"
        ).rename({"country_name": "country_d"})

        return gravity


    def _merge_country_and_product_names_with_gravity(
        self,
        export_col="i",
        import_col="j",
        product_col="k",
        time_col="t"
    ) -> pl.DataFrame:

        # Merge export country names
        df = self.df.join(
            self.country_codes.select(["country_code", "country_name"]),
            left_on=export_col,
            right_on="country_code",
            how="left"
        ).rename({"country_name": "export_country"})

        # Merge Import Country Names
        df = df.join(
            self.country_codes.select(["country_code", "country_name"]),
            left_on=import_col,
            right_on="country_code",
            how="left"
        ).rename({"country_name": "import_country"})

        # Merge Product Descriptions
        df = df.join(
            self.product_codes.select(["code", "description"]),
            left_on=product_col,
            right_on="code",
            how="left"
        )

        # Merge Gravity Data
        df = df.join(
            self.gravity,
            left_on=[time_col, "export_country", "import_country"],
            right_on=["year", "country_o", "country_d"],
            how="left"
        )

        return df


    def _clean_data(self) -> pl.DataFrame:
        df = self.df.with_columns([
            pl.col("v").fill_null(0.0),
            pl.col("q").fill_null(0.0),
            pl.col("gdpcap_d").fill_null(0.0),
            pl.col("gdpcap_o").fill_null(0.0)
        ])

        df = df.with_columns([
            pl.when(pl.col("v").str.strip_chars() == "NA")
              .then(0.0)
              .otherwise(pl.col("v"))
              .alias("v_clean"),

            pl.when(pl.col("q").str.strip_chars() == "NA")
              .then(0.0)
              .otherwise(pl.col("q"))
              .alias("q_clean")
        ])

        df = df.with_columns([
            pl.col("v_clean").str.strip_chars().cast(pl.Float64).alias("v"),
            pl.col("q_clean").str.strip_chars().cast(pl.Float64).alias("q"),
            pl.col("t").cast(pl.Int64)
        ])

        df = df.drop(["v_clean", "q_clean"])
        return df


    def get_data(self, hs_code: int = None, start_year: int = None, end_year: int = None) -> pl.DataFrame:
        df_filtered = self.df.__copy__()

        # Filter by HS code
        if hs_code is not None:
            df_filtered = df_filtered.filter(pl.col("k") == hs_code)

        # Filter by Year Range
        if start_year is not None:
            df_filtered = df_filtered.filter(pl.col("t") >= start_year)
        if end_year is not None:
            df_filtered = df_filtered.filter(pl.col("t") <= end_year)

        df_filtered = df_filtered.filter(pl.col("v") > 0)
        df_filtered = df_filtered.filter(pl.col("q") > 0)

        return df_filtered

    def get_baseline(self, load_precompute=False):
        if load_precompute:
            baseline_df = pd.read_csv(os.path.join('dataset', 'summed_base_data.csv'))
            baseline_df = baseline_df[baseline_df["v"] > 0]
            return baseline_df

        return self.df.group_by(['t', 'export_country', 'import_country']).agg([pl.sum('q').alias('q_sum')])

    def _aggregate_and_build_graph(self, df):
        # Aggregate trade values by exporter-importer pairs
        df_agg = (
            df
            .groupby(["export_country", "import_country"], as_index=False)["v"]
            .sum()
        )
        G = nx.Graph()
        for _, row in df_agg.iterrows():
            exporter = row["export_country"]
            importer = row["import_country"]
            value = row["v"]

            if exporter != importer:
                if value <= 0:
                    continue
                if G.has_edge(exporter, importer):
                    G[exporter][importer]['weight'] += value
                    G[exporter][importer]['inverse_weight'] = 1 / G[exporter][importer]['weight']
                    G[exporter][importer]['log_weight'] = math.log(G[exporter][importer]['weight']+1)
                else:
                    G.add_edge(exporter, importer, weight=value, inverse_weight=1 / value)

        return G

    def get_yearly_baseline_graphs(self, df, years):
        yearly_graphs = {}
        for y in years:
            df_year = df[df["t"] == y]
            G = self._aggregate_and_build_graph(df_year)
            yearly_graphs[y] = G

        return yearly_graphs

    def get_yearly_graphs(self, years):
        df = self.df.to_pandas()
        yearly_graphs = {}
        for y in years:
            df_year = df[df["t"] == y]

            if self.hs_code is not None:
                df_year = df_year[df_year["k"] == self.hs_code]

            G = self._aggregate_and_build_graph(df_year)
            yearly_graphs[y] = G

        return yearly_graphs


if __name__ == "__main__":
    loader = DataLoader(hs_code=282520)
    polar_df = loader.get_data()
    df = polar_df.to_pandas()
    loader.get_yearly_graphs([2015, 2016, 2017, 2018, 2019, 2020])