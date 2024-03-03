import requests
from bs4 import BeautifulSoup
r = requests.get("https://www.xcontest.org/world/en/")
type(r)
soup = BeautifulSoup(r.text)
print(soup.find('h1').get_text())
print(soup.h2)

#error 429 ist blocked cause of to many requests