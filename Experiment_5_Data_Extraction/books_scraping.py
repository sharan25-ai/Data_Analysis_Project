import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

books = []

for page in range(1, 6):
    url = f"https://books.toscrape.com/catalogue/page-{page}.html"

    response = requests.get(url)

    print("Page", page, "Status Code:", response.status_code)

    soup = BeautifulSoup(response.text, "html.parser")

    for book in soup.select("article.product_pod"):
        title = book.h3.a["title"]
        price = book.select_one(".price_color").text.strip()
        availability = book.select_one(".availability").text.strip()
        rating = book.select_one(".star-rating")["class"][1]

        rating_values = {
            "One": 1,
            "Two": 2,
            "Three": 3,
            "Four": 4,
            "Five": 5
        }

        rating = rating_values[rating]

        books.append({
            "Title": title,
            "Price": float(price.replace("Â£", "").replace("£", "")),
            "Availability": availability,
            "Rating": rating
        })

df = pd.DataFrame(books)

print("\nTotal Books Extracted:", len(df))

print("\nBooks DataFrame:")
print(df.head())

print("\nFive Least Expensive Books:")
print(df.nsmallest(5, "Price"))

print("\nFive Most Expensive Books:")
print(df.nlargest(5, "Price"))

average_price = df["Price"].mean()

print("\nAverage Book Price:", round(average_price, 2))

df.to_csv("books_dataset.csv", index=False)

print("\nDataset saved as books_dataset.csv")

plt.figure(figsize=(8, 5))
plt.hist(df["Price"], bins=10)
plt.xlabel("Price")
plt.ylabel("Number of Books")
plt.title("Distribution of Book Prices")
plt.show()