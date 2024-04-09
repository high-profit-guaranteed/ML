import time
import requests
import feedparser
from goose3 import Goose
from bs4 import BeautifulSoup
from selenium import webdriver

# Get links from google news

def getDateAndLinks(keyword, term):
    url = 'https://news.google.com/rss/search?q=' + keyword + \
        '+when:' + str(term) + 'd&hl=en-US&gl=US&ceid=US:en'
    text = getData(url)
    datas = feedparser.parse(text).entries
    links = []
    driver = webdriver.Chrome()
    for data in datas:
        date = data.published
        driver.get(data.link)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        linkTag = soup.find('a')
        if linkTag == None:
            continue
        links.append([date, linkTag.text])
    driver.quit()
    return links

def getDateAndLinks(keyword, t1, t2):
    url = 'https://news.google.com/rss/search?q=' + keyword + \
        '+after:' + t1 + '+before:' + t2 + '&ceid=US:en&hl=en-US&gl=US'
    text = getData(url)
    datas = feedparser.parse(text).entries
    links = []
    driver = webdriver.Chrome()
    for data in datas:
        date = data.published
        driver.get(data.link)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        linkTag = soup.find('a')
        if linkTag == None:
            continue
        links.append([date, linkTag.text])
    driver.quit()
    return links

# Get data from link


def getData(link):
    response = requests.get(link)
    return response.text

# Get article from link


def getArticle(link, g):
    try:
        article = g.extract(url=link)
        return article.cleaned_text
    except:
        return ""

# Get news from keyword and term


def getNews(keyword, term):
    g = Goose({'browser_user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2)',
              'parser_class': 'soup', 'strict': False})
    links = getDateAndLinks(keyword, term)
    articles = []
    for link in links:
        # print(link)
        article = getArticle(link[1], g)
        if article == "" or article == None or article == "Please enable JS and disable any ad blocker" or article == "Keep me logged in from this computer." or article.startswith("Access to this page has been denied because we believe you are using automation tools to browse the website."):
            continue
        articles.append(str(link[0]) + "\n\n" + article)
    return articles

def getNews(keyword, t1, t2):
    g = Goose({'browser_user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2)',
              'parser_class': 'soup', 'strict': False})
    links = getDateAndLinks(keyword, t1, t2)
    articles = []
    for link in links:
        # print(link)
        article = getArticle(link[1], g)
        if article == "" or article == None or article == "Please enable JS and disable any ad blocker" or article == "Keep me logged in from this computer." or article.startswith("Access to this page has been denied because we believe you are using automation tools to browse the website."):
            continue
        articles.append(str(link[0]) + "\n\n" + article)
    return articles

# Parse time from string
#   ex) "Tue, 02 Apr 2024 13:07:16 GMT"
#         -> time.struct_time(tm_year=2024, tm_mon=4, tm_mday=2, tm_hour=13, tm_min=7, tm_sec=16, tm_wday=1, tm_yday=93, tm_isdst=-1)
#   ref) https://docs.python.org/ko/3/library/datetime.html#strftime-strptime-behavior


def parseTime(timeStr):
    timeObj = time.strptime(timeStr, "%a, %d %b %Y %H:%M:%S %Z")
    return timeObj

# # ---------------
# # | 파일로 저장하기 |
# # ---------------
# # getDateAndLinks 로 구글 뉴스에서 키워드와 기간을 입력받아 해당 기사들의 링크를 반환
# g = Goose({'browser_user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2)', 'parser_class':'soup', 'strict': False})
# links = getDateAndLinks('TSLA', 1)

# # getArticle 로 링크를 입력받아 해당 기사의 본문을 ./articles/ 폴더 안에 파일로 저장
# for index in range(len(links)):
#   with open('./articles/' + str(index+1) + '.txt', 'w') as outfile:
#     outfile.write(getArticle(links[index][1], g))

# # ----------------
# # | 리스트로 저장하기 |
# # ----------------
# articles = getNews('TSLA', 1)
# # print(articles)
# print(len(articles))


# -----------------
# | 파일로 저장하기 2 |
# -----------------
articles = getNews('TSLA', "2024-04-07", "2024-04-08")
for index in range(len(articles)):
    with open('./test/' + str(index+1) + '.txt', 'w') as outfile:
        outfile.write(articles[index])
