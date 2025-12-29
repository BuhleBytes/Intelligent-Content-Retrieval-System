
#                                                            NEWS AND MEDIA OUTLET
# import requests
# from bs4 import BeautifulSoup

# url = "https://www.bbc.com/future/article/20251218-dian-fossey-the-woman-who-gave-her-life-to-save-the-gorillas"
# response = requests.get(url, headers={"Accept":"text/html"})
# parsed_response = BeautifulSoup(response.text, "html.parser")
# article  = parsed_response.find("article")
# print(article.text)




#                                                                Education(wikis)

# import requests
# from bs4 import BeautifulSoup
# url = "https://en.wikipedia.org/wiki/Machine_learning"
# headers = {
#     "User-Agent": "StudentBot/1.0 (University Assignment; mlnhon001@myuct.ac.za)"
# }
# response = requests.get(url, headers=headers)
# parsed_response = BeautifulSoup(response.text, "html.parser")
# content = parsed_response.find("main", id="content", class_="mw-body")
# with open("scraped_content.txt", "w", encoding="utf-8") as file:
#     file.write(content.text)
# print("Content saved to scraped_content.txt - check the file!")




# #                                          <<<<<<<<<Technical Documentation>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# import requests
# from bs4 import BeautifulSoup

# url = "https://www.tensorflow.org/guide/intro_to_graphs"
# headers = {
#      "User-Agent": "StudentBot/1.0 (University Assignment; mlnhon001@myuct.ac.za)"
#  }

# response = requests.get(url, headers=headers)

# parsed_response = BeautifulSoup(response.text, "html.parser")

# content = parsed_response.find("article", class_="devsite-article")

# with open("scraped_content2.txt", "w", encoding="utf-8") as file:
#     file.write(content.text)

# print("Content has been successfully written to scraped_content2.txt")



# #                                                           Research publication 
# import requests
# from bs4 import BeautifulSoup
# url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4165831/"
# headers = {
#      "User-Agent": "StudentBot/1.0 (University Assignment; mlnhon001@myuct.ac.za)"
#  }
# response = requests.get(url, headers=headers)
# parsed_response  = BeautifulSoup(response.text, "html.parser")
# content = parsed_response.find("main", id="main-content")
# with open("scaped_content3.txt", "w", encoding="utf-8") as file:
#     file.write(content.text)
# print("Content has been successfully written to scraped_content3.txt")



import requests
from bs4 import BeautifulSoup
import time


urls = {"News":"https://www.bbc.com/future/article/20251218-dian-fossey-the-woman-who-gave-her-life-to-save-the-gorillas", "Educational":"https://en.wikipedia.org/wiki/Machine_learning", "Technical Documentation":"https://www.tensorflow.org/guide/intro_to_graphs", "Reasearch Publication":"https://pmc.ncbi.nlm.nih.gov/articles/PMC4165831/"}
headers = {
     "User-Agent": "StudentBot/1.0 (University Assignment; mlnhon001@myuct.ac.za)"
 }

for category, url in urls.items():
    response =  requests.get(url, headers=headers)
    parsed_response = BeautifulSoup(response.text, "html.parser")
    content = None
    if category == "News":
        content = parsed_response.find("article")
    elif category == "Educational":
        content = parsed_response.find("main", id="content", class_="mw-body")
    elif category == "Technical Documentation":
        content = parsed_response.find("article", class_="devsite-article")
    else:
        content = parsed_response.find("main", id="main-content")
    content_text = content.get_text(separator="\n\n", strip=True)

    
