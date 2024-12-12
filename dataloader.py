import os
import glob
import time
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
        data_path_pattern=os.path.join("dataset", "BACI*.csv"),
        country_codes_path=os.path.join("dataset", "country_codes_V202401b.csv"),
        product_codes_path=os.path.join("dataset", "product_codes_HS22_V202401b.csv"),
        gravity_path=os.path.join("gravity", "Gravity_V202211.csv")
    ):
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
            "wto_o", "wto_d", "eu_o", "eu_d", "entry_cost_o", "entry_cost_d"
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
            pl.col("q_clean").str.strip_chars().cast(pl.Float64).alias("q")
        ])

        df = df.drop(["v_clean", "q_clean"])
        return df


    def get_data(self, hs_code: int = None, start_year: int = None, end_year: int = None) -> pl.DataFrame:
        df_filtered = self.df

        # Filter by HS code
        if hs_code is not None:
            df_filtered = df_filtered.filter(pl.col("k") == hs_code)

        # Filter by Year Range
        if start_year is not None:
            df_filtered = df_filtered.filter(pl.col("t") >= start_year)
        if end_year is not None:
            df_filtered = df_filtered.filter(pl.col("t") <= end_year)

        return df_filtered