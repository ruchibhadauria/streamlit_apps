from selenium import webdriver
import csv
import re
from bs4 import BeautifulSoup

BROWSER = webdriver.Firefox()

def write_csv(ads):
	with open('jaipur_reviews.csv', 'a', encoding="utf-8") as f:
		fields = ['review', 'date', 'rating']
		writer = csv.DictWriter(f, fieldnames=fields)

		for ad in ads:
			writer.writerow(ad)

def get_html(url):
	BROWSER.get(url)
	return BROWSER.page_source

def scrape_data(card):

	review = card.find("q", class_="XllAv H4 _a").text.strip()
	
	try:
		date = card.find("span", class_="euPKI _R Me S4 H3").text.strip()
	except:
		date = ""

	rating = card.find("span", {"class": re.compile("ui_bubble_rating bubble_\d*")})
    
	data = {"review": review, "date": date, "rating": rating}

	return data


def main():
	reviews_data = []

	for i in range(0, 900, 5):
		url = "https://www.tripadvisor.in/Hotel_Review-g304555-d6540893-Reviews-or{}-Hilton_Jaipur-Jaipur_Jaipur_District_Rajasthan.html#REVIEWS"
		html = get_html(url.format(i))
		soup = BeautifulSoup(html, 'lxml')
		cards = soup.find_all('div', {'class': "cqoFv _T"})
		
		for card in cards:
			data = scrape_data(card)
			reviews_data.append(data)
	
	write_csv(reviews_data)



if __name__ == '__main__':
	main()

