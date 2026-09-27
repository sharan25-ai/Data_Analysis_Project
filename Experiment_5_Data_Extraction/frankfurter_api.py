import pandas as pd
import requests
import matplotlib.pyplot as plt

url = "https://api.frankfurter.app/latest"

response = requests.get(url)

print("Status Code:", response.status_code)

data = response.json()

base_currency = data["base"]
rates = data["rates"]

df = pd.DataFrame(
    list(rates.items()),
    columns=["Currency", "Exchange_Rate"]
)

print("\nBase Currency:", base_currency)

print("\nExchange Rates:")
print(df)

print("\nFive Highest Exchange Rate Values:")
print(df.nlargest(5, "Exchange_Rate"))

selected_currencies = ["USD", "EUR", "INR", "GBP", "JPY"]

print("\nSelected Currency Exchange Rates:")

selected = df[df["Currency"].isin(selected_currencies)]

print(selected)

df.to_csv("exchange_rates.csv", index=False)

print("\nDataset saved as exchange_rates.csv")

plt.figure(figsize=(12, 6))
plt.bar(df["Currency"], df["Exchange_Rate"])
plt.xlabel("Currency")
plt.ylabel("Exchange Rate")
plt.title("Exchange Rates Relative to EUR")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

strongest = df.loc[df["Exchange_Rate"].idxmax()]
weakest = df.loc[df["Exchange_Rate"].idxmin()]

print("\nStrongest Currency Relative to Base Currency:")
print(strongest)

print("\nWeakest Currency Relative to Base Currency:")
print(weakest)