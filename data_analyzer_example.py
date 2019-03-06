import data_analyzer as da

# Example usage
res = da.get_results_from_session()
df = da.results_to_dataframe(res)
print(df)