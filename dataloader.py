import math
import os
import glob
import time

import networkx as nx
import pandas as pd
import polars as pl
import pycountry
from fuzzywuzzy import process
import geopandas as gpd


def time_execution(method):
    """
    Decorator that measures and prints the execution time of the decorated method.

    Parameters
    ----------
    method : callable
        The method to be decorated.

    Returns
    -------
    callable
        The wrapped method that prints execution time upon completion.
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
    """
    A class to handle loading, merging, and cleaning trade data along with
    country codes, product codes, and gravity variables. Prepares a combined
    dataset ready for further analysis, including building yearly graphs.
    """
    def __init__(
        self,
        hs_code: int = None,
        data_path_pattern=os.path.join("data/dataset", "BACI*.csv"),
        country_codes_path=os.path.join("data/dataset", "country_codes_V202401b.csv"),
        product_codes_path=os.path.join("data/dataset", "product_codes_HS22_V202401b.csv"),
        gravity_path=os.path.join("data/gravity", "Gravity_V202211.csv")
    ):
        """
         Initialize the DataLoader instance.

         Parameters
         ----------
         hs_code : int, optional
             The specific HS code (product) to filter on. If None, all products are included.
         data_path_pattern : str, optional
             Glob pattern to locate BACI dataset CSV files.
         country_codes_path : str, optional
             Path to the CSV file containing country code mappings.
         product_codes_path : str, optional
             Path to the CSV file containing product code mappings.
         gravity_path : str, optional
             Path to the Gravity dataset CSV file.
         """
        self.hs_code = hs_code

        self.data_path_pattern = data_path_pattern
        self.country_codes_path = country_codes_path
        self.product_codes_path = product_codes_path
        self.gravity_path = gravity_path

        self.country_name_mapping = self.load_country_mapping()

        self._initialize_data()

    @time_execution
    def _initialize_data(self):
        """
        Internal method to load and prepare all datasets:
        - Loads BACI trade data (optionally filtered by hs_code)
        - Loads country and product codes
        - Loads and preprocesses gravity data
        - Merges all data sources
        - Cleans the final merged DataFrame
        """
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
        """
         Load and concatenate all BACI dataset CSV files matching the given pattern.
         Optionally filters by the specified hs_code.

         Returns
         -------
         pl.DataFrame
             A Polars DataFrame containing BACI trade data with columns cast to Int64 where appropriate.
         """
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
        """
        Load the country codes CSV and ensure 'country_code' is cast to Int64.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame of country codes.
        """
        cc = pl.read_csv(self.country_codes_path)
        cc = cc.with_columns(pl.col("country_code").cast(pl.Int64))
        return cc


    def _load_product_codes(self) -> pl.DataFrame:
        """
        Load the product codes CSV and cast 'code' to Int64.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame of product codes.
        """
        pc = pl.read_csv(self.product_codes_path)
        pc = pc.with_columns(pl.col("code").cast(pl.Int64))
        return pc


    def _preprocess_gravity_data(self) -> pl.DataFrame:
        """
        Load and preprocess the Gravity dataset:
        - Selects relevant columns
        - Filters for years > 2000
        - Ensures uniqueness of (iso3_o, iso3_d, year)
        - Joins country names and performs forward-fill to handle missing values over time

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame containing gravity data with stable attributes forward-filled.
        """
        gravity = pl.read_csv(self.gravity_path, ignore_errors=True)
        gravity = gravity.select([
            "iso3_o", "iso3_d",
            "year",
            "gdpcap_d", "gdpcap_o",
            "tradeflow_baci",
            "diplo_disagreement",
            "comrelig",
            "distw_harmonic",
            "pop_o", "pop_d",
            "wto_o", "wto_d",
            "eu_o", "eu_d",
            "entry_cost_o", "entry_cost_d",
            "comlang_off",
            "gatt_o", "gatt_d",
            "fta_wto",
            "tradeflow_imf_o", "tradeflow_imf_d",
            "comlang_ethno",
            "comcol", "col45",
            "comleg_pretrans", "comleg_posttrans",
            "col_dep_ever",
            "empire",
            "sibling_ever",
            "scaled_sci_2021"
        ])

        gravity = gravity.filter(pl.col("year") > 2000)
        gravity = gravity.unique(subset=["iso3_o", "iso3_d", "year"], keep="first")

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

        gravity_pd = gravity.to_pandas()
        gravity_pd = gravity_pd.sort_values(["iso3_o", "iso3_d", "year"])

        gravity_pd = gravity_pd.groupby(["iso3_o", "iso3_d"]).apply(lambda g: g.ffill())
        gravity_pd = gravity_pd.reset_index(drop=True)

        gravity = pl.from_pandas(gravity_pd)
        return gravity


    def _merge_country_and_product_names_with_gravity(
        self,
        export_col="i",
        import_col="j",
        product_col="k",
        time_col="t"
    ) -> pl.DataFrame:
        """
         Merge export/import country names, product descriptions, and gravity data
         into the main DataFrame.

         Parameters
         ----------
         export_col : str, optional
             Column name representing the exporter country code.
         import_col : str, optional
             Column name representing the importer country code.
         product_col : str, optional
             Column name representing the product code.
         time_col : str, optional
             Column name representing the time dimension (year).

         Returns
         -------
         pl.DataFrame
             The merged DataFrame including country names, product descriptions,
             and gravity attributes.
         """
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
        """
        Clean the merged DataFrame by:
        - Filling null values in key columns
        - Converting columns 'v' and 'q' from strings 'NA' to 0 if needed
        - Casting 't' to Int64

        Returns
        -------
        pl.DataFrame
            The cleaned DataFrame ready for further filtering or aggregation.
        """
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
        """
         Retrieve filtered data based on optional hs_code and year range filters.
         Ensures 'v' and 'q' are positive.

         Parameters
         ----------
         hs_code : int, optional
             Filter by a specific HS code if provided.
         start_year : int, optional
             Include records from this year onwards.
         end_year : int, optional
             Include records up to this year.

         Returns
         -------
         pl.DataFrame
             The filtered DataFrame.
         """
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

        # Adapt country names for used libraries
        df_filtered = df_filtered.to_pandas()
        df_filtered["export_country"] = df_filtered["export_country"].apply(lambda x: self.replace_countries(x))
        df_filtered["import_country"] = df_filtered["import_country"].apply(lambda x: self.replace_countries(x))

        return df_filtered

    def get_baseline(self, load_precompute=False) -> pl.DataFrame:
        """
        Get a baseline aggregated dataset. If 'load_precompute' is True, load from a precomputed file.
        Otherwise, aggregate over 't', 'export_country', 'import_country' summing 'q'.

        Parameters
        ----------
        load_precompute : bool, optional
            If True, attempt to load precomputed baseline data from 'summed_base_data.csv'.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            The baseline aggregated dataset.
        """
        if load_precompute:
            baseline_df = pd.read_csv(os.path.join('data/dataset', 'summed_base_data.csv'))
            baseline_df = baseline_df[baseline_df["v"] > 0]
            return baseline_df

        return self.df.group_by(['t', 'export_country', 'import_country']).agg([pl.sum('q').alias('q_sum')])

    def _aggregate_and_build_graph(self, df) -> nx.Graph:
        """
        Aggregate trade values by exporter-importer pairs and build a weighted undirected graph.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame (Pandas) with at least 'export_country', 'import_country', and 'v' columns.

        Returns
        -------
        nx.Graph
            A NetworkX graph with edges weighted by the sum of 'v', plus inverse and log-transformed weights.
        """
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

    def get_yearly_baseline_graphs(self, df, years) -> dict[int, nx.Graph]:
        """
        Build yearly baseline graphs from a given DataFrame and a list of years.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 't', 'export_country', 'import_country', and 'v'.
        years : list of int
            List of years to build graphs for.

        Returns
        -------
        dict
            A dictionary keyed by year, with values as NetworkX graphs.
        """
        yearly_graphs = {}
        for y in years:
            df_year = df[df["t"] == y]
            G = self._aggregate_and_build_graph(df_year)
            yearly_graphs[y] = G

        for year, G in yearly_graphs.items():
            mapping = {node: self.replace_countries(node) for node in G.nodes()}
            yearly_graphs[year] = nx.relabel_nodes(G, mapping)

        return yearly_graphs

    def get_yearly_graphs(self, years):
        """
        Build yearly graphs directly from the DataLoader's main dataset (self.df).
        Optionally filtered by the instance's hs_code if provided.

        Parameters
        ----------
        years : list of int
            List of years to build graphs for.

        Returns
        -------
        dict
            A dictionary keyed by year, with values as NetworkX graphs of trade flows.
        """
        df = self.df.to_pandas()
        yearly_graphs = {}
        for y in years:
            df_year = df[df["t"] == y]

            if self.hs_code is not None:
                df_year = df_year[df_year["k"] == self.hs_code]

            G = self._aggregate_and_build_graph(df_year)
            yearly_graphs[y] = G

        for year, G in yearly_graphs.items():
            mapping = {node: self.replace_countries(node) for node in G.nodes()}
            yearly_graphs[year] = nx.relabel_nodes(G, mapping)

        return yearly_graphs

    def load_country_mapping(self):
        """
        Load a mapping of country names to ISO Alpha-2 codes.

        Returns
        -------
        dict
            A dictionary mapping country names to ISO Alpha-2 codes.
        """

        return {
            "USA": "United States of America",
            "Türkiye": "Turkey",
            'China, Hong Kong SAR': 'China',
            'Other Asia, nes': 'China',
            'Dem. Rep. of the Congo': 'Democratic Republic of the Congo',
            'Serbia': "Republic of Serbia",
            'Malta': "Italy",
            'Russian Federation': "Russia",
            'Eswatini': "eSwatini",
            'Rep. of Korea': "South Korea",
            'Czech Republic': 'Czechia',
            'United States': "United States of America",
            'Bolivia (Plurinational State of)': "Bolivia",
            'Rep. of Moldova': "Moldova",
            'United Rep. of Tanzania': "United Republic of Tanzania",
            "Viet Nam": "Vietnam",
            "Dem. People's Rep. of Korea": "South Korea",
            "FS Micronesia": "Micronesia",
            "Sudan (...2011)": "Sudan",
            "rep. of korea": "South Korea"
        }

    def generate_fuzzy_mapping(self, unmatched_countries, valid_countries):
        mapping = {}
        for country in unmatched_countries:
            match, score = process.extractOne(country, valid_countries)
            if score > 70:
                mapping[country] = match
            else:
                mapping[country] = None
        return mapping

    def fuzzy_match_country(self, country_name, country_list):
        match, score = process.extractOne(country_name, country_list)
        return match if score > 70 else None

    def replace_countries(self, name: str) -> str:
        """
        Replace country names with standardized names based on a predefined mapping.

        Parameters:
            name (str): Original country name.

        Returns:
            str: Standardized country name.
        """
        return self.country_name_mapping.get(name, name)

    def normalize_country_names(self, data, column_name, ):
        return data[column_name].replace(self.country_name_mapping)

    def get_country_code(self, country_name):
        """
        Extract the ISO Alpha-2 country code for a given country name.

        Parameters:
            country_name (str): The full name of the country.

        Returns:
            str: ISO Alpha-2 country code, or None if not found.
        """
        try:
            # Use country name mapping to standardize names first
            standardized_name = self.country_name_mapping.get(country_name, country_name)

            # Try to fetch the country code directly
            country_info = pycountry.countries.get(name=standardized_name)
            if country_info:
                return country_info.alpha_2

            # If not found, attempt fuzzy matching with pycountry
            matches = pycountry.countries.search_fuzzy(standardized_name)
            if matches:
                return matches[0].alpha_2

            # Final fallback: fuzzy matching with predefined valid countries
            valid_countries = [country.name for country in pycountry.countries]
            fuzzy_match = self.fuzzy_match_country(standardized_name, valid_countries)
            if fuzzy_match:
                country_info = pycountry.countries.get(name=fuzzy_match)
                return country_info.alpha_2 if country_info else None

        except LookupError:
            print(f"Country code not found for: {country_name}")
            return None

        return None

    def preprocess_culture_and_country_names(self, df):
        shapefile_path = os.path.join("data/dataset", "110m_cultural", "ne_110m_admin_0_countries.shp")
        world = gpd.read_file(shapefile_path)
        valid_countries = world['ADMIN'].tolist()

        export_data = df.groupby('export_country')['v'].sum().reset_index()
        import_data = df.groupby('import_country')['v'].sum().reset_index()

        export_data['export_country'] = self.normalize_country_names(export_data, 'export_country')
        import_data['import_country'] = self.normalize_country_names(import_data, 'import_country')

        unmatched_exports = set(export_data['export_country']) - set(valid_countries)
        unmatched_imports = set(import_data['import_country']) - set(valid_countries)

        if unmatched_exports:
            export_mapping = self.generate_fuzzy_mapping(unmatched_exports, valid_countries)
            export_data['export_country'] = export_data['export_country'].replace(export_mapping)
            print(f"Unmapped Names: {', '.join([name for name in export_mapping.keys() if not export_mapping[name]])}")

        if unmatched_imports:
            import_mapping = self.generate_fuzzy_mapping(unmatched_imports, valid_countries)
            import_data['import_country'] = import_data['import_country'].replace(import_mapping)
            print(f"Unmapped Names: {', '.join([name for name in import_mapping.keys() if not import_mapping[name]])}")

        export_data = export_data.groupby('export_country')['v'].sum().reset_index()
        import_data = import_data.groupby('import_country')['v'].sum().reset_index()

        world_export = world.merge(export_data, left_on="ADMIN", right_on="export_country", how="left")
        world_import = world.merge(import_data, left_on="ADMIN", right_on="import_country", how="left")

        return world_export, world_import

