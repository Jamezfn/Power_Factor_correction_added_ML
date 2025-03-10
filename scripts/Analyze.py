import pandas as pd
import matplotlib.pyplot as plt
import config

data_path = config.CONFIG['data_path']
df = pd.read_csv(data_path)
plt.hist(df['Power'], bins=50)
plt.title('Power Distribution')
plt.xlabel('Power (W)')
plt.ylabel('Frequency')
plt.show()