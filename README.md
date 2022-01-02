# Predicting-Executive-Stock-Movements


## Background:
The accessibility and ubiquity of social media has created a new problem for investors, in which influential individuals can cause dramatic changes in the market. For example, on January 28th, shares for the CD Projekt developers of Cyberpunk 2077 surged over 12% after Elon Musk tweeted that he liked the game the night before. In another example, Tesla shares fell over 10% when Musk tweeted that the company’s valuation was too high. The rapid market changes from these tweets has caused investors to be fixated on monitoring social media for similar occurrences. 
<figure>
<img src = "https://github.com/Pooret/Predicting-Executive-Stock-Movements/blob/main/figures/report/Fig1.png">
</figure>
Andy Swan writes in Forbes that there are 4 types of tweets to look for that influence stocks.

Influential people or groups tweets about a company’s stock
Company tweets about its own stock or by influential investors
Policy events that are tweeted by government insiders or observers
Influencer tweets an opinion about a particular brand or product

Given these four tweet types, we can assess twitter feeds based off of stock, company, and product mentions. 

## Business Proposal
The accessibility and ubiquity of social media has created a new problem for investors, in which influential individuals can cause dramatic changes in the market. On the other hand, it creates a potentially new and much faster means of assessing stock movements. For example, on January 28th, shares for the CD Projekt developers of Cyberpunk 2077 surged over 12% after Elon Musk tweeted that he liked the game the night before.

The Scenario:
It is hypothesized that an executive's tweets will impact their company's stock prices. Assuming this to be true, an investor could analyze the executive's posts to make a prediction as to whether the stock will rise, drop, or remain the same as indicated by a percent change threshold. To simplify this problem and open it up for future implementation strategies, we will examine if twitter data can predict whether or not the stock price changes.

An investor wants to know as early as possible if company's stock is going to change by a specified amount before the next opening or closing price. As the periods of time between opening and closing stock prices are relatively short, getting a sense of stock movement using only the twitter data would be a huge benefit. If stocks are predicted to change beyond a certain threshold (e.g. +/− 2%), an investor can look at the trend of the stock and make a decision as to purchase shares of the stock or to sell existing share.


## To Do##
*My project is never fully completed, and as I learn more I will apply what I've learned to the original problem set. I am always open to suggested edits as well, so if there are any I will include them here.*  

1. NMF component factorization to create topics that can be used to filter raw twitter data.
2. Look for any autocorrelations and partial autocorrelations for stock price detrending, for features over time
3. Predict up vs down (sorry this was my first project and didn't know a lot, I am correcting this first)
4. Get more twitter data if possible (Twint is a no go still)
5. vord2vec
6. BERT 

