from bs4 import BeautifulSoup
import requests
# ----------------------------------------------------------------------------------------------------------------------
def crawler(url):
    pages_crawled = []
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    links = soup.find_all('a')

    for link in links:
        if 'href' in link.attrs:
            if link['href'].startswith('/wiki') and ':' not in link['href']:
                if link['href'] not in pages_crawled:
                    new_link = f"https://en.wikipedia.org{link['href']}"
                    pages_crawled.append(link['href'])
                    # try:
                    #     with open('data.csv', 'a') as file:
                    #         file.write(f'{soup.title.text}; {soup.h1.text}; {link["href"]}\n')
                    #     crawler(new_link)
                    # except:
                    #     continue
    return pages_crawled
# ----------------------------------------------------------------------------------------------------------------------
def getURL(page):

    start_link = page.find("href")
    if start_link == -1:
        return None, 0
    start_quote = page.find('"', start_link)
    end_quote = page.find('"', start_quote + 1)
    url = page[start_quote + 1: end_quote]
    return url, end_quote


# ----------------------------------------------------------------------------------------------------------------------
url = 'https://en.wikipedia.org'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #pages_crawled = crawler(url)

    response = requests.get(url)
    page = str(BeautifulSoup(response.content,features="html.parser"))

    while True:
        url, n = getURL(page)
        page = page[n:]
        if url:
            print(url)
        else:
            break
