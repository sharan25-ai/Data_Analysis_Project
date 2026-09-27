import pandas as pd
import requests

url = "https://openlibrary.org/search.json?q=python"

response = requests.get(url)

print("Status Code:", response.status_code)

data = response.json()

books = []

for book in data["docs"]:
    title = book.get("title", "")
    
    authors = book.get("author_name", [])
    author = authors[0] if authors else "Unknown"
    
    publication_year = book.get("first_publish_year", None)
    
    publishers = book.get("publisher", [])
    publisher = publishers[0] if publishers else "Unknown"
    
    isbns = book.get("isbn", [])
    isbn = isbns[0] if isbns else "Not Available"

    books.append({
        "Title": title,
        "Author": author,
        "Publication_Year": publication_year,
        "Publisher": publisher,
        "ISBN": isbn
    })

df = pd.DataFrame(books)

print("\nOpen Library Books:")
print(df.head())

print("\nTotal Books Returned:", len(df))

print("\nBooks Published After 2020:")
print(df[df["Publication_Year"] > 2020])

print("\nNumber of Books per Author:")
print(df["Author"].value_counts())

df.to_csv("openlibrary_books.csv", index=False)

print("\nDataset saved as openlibrary_books.csv")

print("\nTen Most Recently Published Books:")
recent_books = df.sort_values(
    "Publication_Year",
    ascending=False
)

print(recent_books.head(10))

print("\nSummary Statistics for Publication Years:")
print(df["Publication_Year"].describe())