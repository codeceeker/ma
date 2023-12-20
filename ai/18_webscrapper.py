import requests # if not installed : pip install requests
from bs4 import BeautifulSoup #if not installed : pip install bs4

def fetch_site(link):

    response = requests.request("GET", url)

    if response.status_code == 200:
        return response
    else:
        raise Exception("Failed to fetch_site site")


url = "https://books.toscrape.com/catalogue/page-1.html"

response = fetch_site(url)

soup = BeautifulSoup(response.text , 'html.parser')

total_pages = soup.select_one('.current').text.strip()
total_pages = int(total_pages.split()[-1])

current_page = 47
books= []
while current_page <= total_pages:
    print(f"fetching pg no {current_page}")
    url = f"https://books.toscrape.com/catalogue/page-{current_page}.html"
    response = fetch_site(url)
    soup = BeautifulSoup(response.text , 'html.parser')

    records = soup.select('.product_pod')
    for record in records:
        book_title = record.select_one('.product_pod > h3 > a')["title"]
        book_price = record.select_one('.product_pod > .product_price > .price_color').text

        books.append({
            "title" : book_title,
            "price" : book_price
        })

        print(f"Title : {book_title} , Price : {book_price}")
    current_page += 1


