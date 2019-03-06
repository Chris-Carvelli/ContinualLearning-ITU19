import data_analyzer as da
import seaborn as sns
import matplotlib.pyplot as plt

# Example usage
res = da.get_results_from_session()
df = da.results_to_dataframe(res)
print(df)
sns.lineplot(x='generation', y='max_score', hue='run', data=df,
                 err_style="bars", ci='run')
plt.legend()

plt.show()

print(df)