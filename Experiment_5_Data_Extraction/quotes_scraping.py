import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

quotes = []

for page in range(1, 11):
    url = f"https://quotes.toscrape.com/page/{page}/"

    response = requests.get(url)

    print("Page", page, "Status Code:", response.status_code)

    soup = BeautifulSoup(response.text, "html.parser")

    for quote in soup.select(".quote"):
        text = quote.select_one(".text").text.strip()
        author = quote.select_one(".author").text.strip()

        tags = [
            tag.text.strip()
            for tag in quote.select(".tags .tag")
        ]

        quotes.append({
            "Quote": text,
            "Author": author,
            "Tags": ", ".join(tags)
        })

df = pd.DataFrame(quotes)

print("\nTotal Quotes:", len(df))

print("\nQuotes per Author:")
author_counts = df["Author"].value_counts()
print(author_counts)

print("\nTop Five Tags:")
all_tags = []

for tags in df["Tags"]:
    all_tags.extend(tags.split(", "))

tag_counts = pd.Series(all_tags).value_counts()
print(tag_counts.head(5))

print("\nLongest Quotation:")
longest_quote = df.loc[df["Quote"].str.len().idxmax()]
print(longest_quote)

df.to_csv("quotes_dataset.csv", index=False)

print("\nDataset saved as quotes_dataset.csv")

plt.figure(figsize=(10, 6))
author_counts.plot(kind="bar")
plt.xlabel("Author")
plt.ylabel("Number of Quotes")
plt.title("Number of Quotes per Author")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()