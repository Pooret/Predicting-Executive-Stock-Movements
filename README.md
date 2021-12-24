# Predicting-Executive-Stock-Movements

Problem Statement: Can we predict daily stock changes from twitter sentiment?

Background:
The accessibility and ubiquity of social media has created a new problem for investors, in which influential individuals can cause dramatic changes in the market. For example, on January 28th, shares for the CD Projekt developers of Cyberpunk 2077 surged over 12% after Elon Musk tweeted that he liked the game the night before. In another example, Tesla shares fell over 10% when Musk tweeted that the company’s valuation was too high. The rapid market changes from these tweets has caused investors to be fixated on monitoring social media for similar occurrences. 

Andy Swan writes in Forbes that there are 4 types of tweets to look for that influence stocks.

Influential people or groups tweets about a company’s stock
Company tweets about its own stock or by influential investors
Policy events that are tweeted by government insiders or observers
Influencer tweets an opinion about a particular brand or product

Given these four tweet types, we can assess twitter feeds based off of stock, company, and product mentions. 

Proposal:
To assess which twitter feeds are most influential, Twitter feeds will be tokenized using Python’s nltk’s TweetTokenizer to perform named entity recognition analysis. Sentiment analysis will then be done on tweets that contain mentions of stocks in the S&P 500. Stock changes will be assessed a day after the tweets, and a regression analysis will be done on the stocks changes based on the tweet sentiment. 
