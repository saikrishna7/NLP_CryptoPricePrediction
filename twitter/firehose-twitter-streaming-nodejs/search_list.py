#import the library to query a website
import requests
import pandas as pd
from bs4 import BeautifulSoup

# specify the url
def get_search_list(url="https://coinmarketcap.com/all/views/all/"):
	# url = "https://coinmarketcap.com/all/views/all/"
	
	# launch the URL and save the html response in a variable called 'response'
	response = requests.get(url)
	
	# import Beautiful soup library to access functions to parse the data returned from the website
	# Parse the html in the 'response' variable, and store it in Beautiful Soup format
	soup = BeautifulSoup(response.text, "lxml")

	# Extract the table which has list of all crypto currencies. This table should be present in one of the html tags. Work with the tags to extract data present in them.
	table=soup.find_all('table')

	right_table=soup.find('table', _id='currencies-all')

	#Generate lists
	Name=[]
	Symbol=[]


	# skip first iteration as we dont need headers
	for row in table[0].findAll("tr")[1:]:
	    cells = row.findAll('td') # To store all other details
	    links = row.findAll('a')
	    Name.append(links[1].find(text=True))
	    Symbol.append(cells[2].find(text=True))


	df=pd.DataFrame(Name,columns=['Name'])
	df['Symbol']=Symbol
	# print(df.head())

	word_list=[]
	for c1,c2 in zip(Name,Symbol):
	    word_list.append(c1)
	    word_list.append(c2)
	word_list
	# print(word_list[0:10])

	red_list = word_list[0:40]
	for word in ['crypto','cryptocurrency','cointelegraph','blockchain','cryptocurrencies','coindesk','iamjosephyoung','btctn','ICO','cryptoexchange']:
	    red_list.append(word)
	red_list
	
	print(red_list)
	
	return(red_list)


get_search_list()
