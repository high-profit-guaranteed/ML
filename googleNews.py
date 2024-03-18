import requests
import feedparser
from goose3 import Goose
from bs4 import BeautifulSoup
from selenium import webdriver

# Get links from google news
def getLinks(keyword, term):
  url = 'https://news.google.com/rss/search?q=' + keyword + '+when:' + str(term) + 'd&hl=en-US&gl=US&ceid=US:en'
  text = getData(url)
  datas = feedparser.parse(text).entries
  links = []
  driver = webdriver.Chrome()
  for data in datas:
    driver.get(data.link)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    link = soup.find('a').text
    # print(link)
    links.append(link)
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
  g = Goose({'browser_user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2)', 'parser_class':'soup', 'strict': False})
  links = getLinks(keyword, term)
  articles = []
  for link in links:
    articles.append(getArticle(link, g))
  return articles

# # ---------------
# # | 파일로 저장하기 |
# # ---------------
# # getLinks 로 구글 뉴스에서 키워드와 기간을 입력받아 해당 기사들의 링크를 반환
# g = Goose({'browser_user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2)', 'parser_class':'soup', 'strict': False})
# links = getLinks('TSLA', 1)

# # getArticle 로 링크를 입력받아 해당 기사의 본문을 ./articles/ 폴더 안에 파일로 저장
# for index in range(len(links)):
#   with open('./articles/' + str(index+1) + '.txt', 'w') as outfile:
#     outfile.write(getArticle(links[index], g))

# ----------------
# | 리스트로 저장하기 |
# ----------------
articles = getNews('TSLA', 1)
# print(articles)
print(len(articles))