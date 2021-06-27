# DataScientist Capstone StarBucks Coffee
Udacity DataScientist Capstone on StarBucks Coffee

# Predicting Starbucks customer spendings
## Optimizing app offers with Starbucks

---
### Medium article
The whole project is explained and examined in this [Medium Article](https://keanu-forthmann.medium.com/predicting-starbucks-customer-spendings-cca39b7533c2).

### The repository includes the following folder and files
- data - including three .jsons with all data used  
  - profile.json - contains people demographic data
  - transcript.json - transcript of all customer actions
  - portfolio.json - overview about all offers
- images - chart images from Jupyter Notebook
- Starbucks_Capstone_notebook.ipynb - Jupyter Notebook with all

---

This project is about Starbucks - so here is some information about Starbucks in brief. The Starbucks Corporation is an American multinational chain of coffeehouses and roastery reserves headquartered in Seattle, Washington. It was founded in 1971 and as of 2020 they operated more than 30,000 places in over 70 countries. In 2019 they made a total revenue of US$ 26.50 billion.

# Part 1 - Project Definition
Within the following three segments the problem domain, problem project origin and given data set is explained.

## 1.1/ Project Overview
Starbucks runs a rewards mobile app which is a way for customers to pay in store or skip the line and order ahead. Rewards are built right in, so they'll collect stars and start earning free drinks and food with every purchase.
The given data set for this project was provided by Starbucks via Udacity as part of my Data Science Nano Degree. The data set contains simulated data that mimics customer behavior on the rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or a buy one get one free offer (BOGO). Some users might not receive any offer during certain weeks.

This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products. The provided simulated data is stated as for the sake of testing the algorithms and not in order to mimic real people. Furthermore we have to understand that different individuals will respond in different ways on the various given offers. Regarding peoples behavior we should also consider that some people may not want to receive any type of offer, in which case we better do not offer them offers at all.

Example of Starbucks rewards mobile app, Source: Starbucks.comEvery offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. We can see in the data set that informational offers have a validity period even though these ads are merely providing information about a product. That means, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

We're given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.

The data is contained in three files:
- portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
- profile.json - demographic data for each customer
- transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:
- portfolio.json
- id (string) - offer id
- offer_type (string) - type of offer ie BOGO, discount, informational
- difficulty (int) - minimum required spend to complete an offer
- reward (int) - reward given for completing an offer
- duration (int) - time for offer to be open, in days
- channels (list of strings)

profile.json
- age (int) - age of the customer
- became_member_on (int) - date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

transcript.json
- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since start of test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record

## 1.2/ Problem Statement
So the main purpose of this project is to discover how we can predict customer spendings - the amount of money spent, to be precise. Not just on a population level but on an individual personalized level. Finding a good way to understand what makes people spend money based on different individual and common traits, would enhance Starbucks to provide customers just with the right offers and to not blindly send offers. This overall goal leads to the following questions:
How can we best predict how much money a customers will spend? 
What offers seem to work best regarding customer spendings?
How can customers be characterized regarding spending characteristics?


## 1.3 / Metrics
Once we’ve implemented all strategies explained above, we need a way to tell how accurately we are able to answer the stated questions. So how do we measure the performance of our models?

The current Starbucks logo, Source: Starbucks.com
Once we’ve applied a regression model we are able to conclude the amount of variance we can explain with given features — the respective key number here is called R-Squared (e.g. an R-Squared of 0.5 would mean that 50 percent of variance in money spend can be explained with given features).

# Part 2 — Data Analysis
## 2.1/ The offer portfolio
The portfolio.json file contains 10 different types of offers. We have 4 BOGOs, 2 informational offers and 4 discount offers. We obviously have no missing values.

Jupyter Notebook excerpt of profile.json

## 2.2/ Peoples profiles
The profile.json data contains 17'000 entries of registered individuals in total, with only missing values in columns gender and income (Fig. A.1). There are 2'175 missing values in those columns each. Considering the respective columns, this seems to be likely, since gender and income are information that one might not want to provide.

Figure A.1: Visualized overview of missing values

Figure A.2
Out of 17'000 customers, 49.9% of the people, being the majority, stated to be male (8'484 people). Only 36.1% (6'129 people) stated to be female. There were 12.8% (2'175 people) didn’t provide any information. And 1.25% (212 people) stated their gender to be other (Fig. A.2).
When looking at the distribution of age of customers (Fig. A.3), we can clearly see some strangely high occurrence of a specific age far over 100 (118 to be precise). Even though we have no missing data within age column, the data clearly seems to be captured in a way that’s telling us those information is missing — which is common for numerical data. The mean age is 54.4 years, with a standard 17.4 years as standard deviation and customer ages ranging from 18 to 101, after cleaning the data.


Figure A.3: left is uncleaned data, right is cleaned data
When looking at the 14'825 customers that provided information about income (Fig. A.4), we can’t see any abnormalities. Mean income is at 65'405 US-Dollars, with a standard deviation of 21'598 US-Dollars. Incomes are ranging from 30'000 to 120'000 US-Dollars.

Figure A.4
The became_member_on column was recorded a numerical data and is reflecting the date each customer became a member on. In order to plot and properly work with the data, the whole column was transformed into Pandas datetime format with the pd.to_datetime function. Then we are able to plot the timeline of member registrations (Fig. A.5). Here we clearly see a strong increase of members in the mid of 2015 and even more in mid of 2017.


Figure A.5
## 2.3/ Transaction data
The transcript.json data contains 306'534 entries of different transactions (and other events), with no missing values on the first sight. When looking at the distribution of different events, we can see most of the events are transactions (138'953) that contain information about the amount of money spend (next to the persons id within the corresponding dictionary of the value column). Then in decreasing order the events are offer received, offer viewed and offer completed. And this makes sense for logical reasons, since an offer can only be viewed when received before and there can not be more completed, than viewed offers. Within those rows, the data includes only the persons id in the value column (Fig. A.6).

The time column contains information about time in hours since start of test. Since time is formatted in hours, we divide every value by 24 in order to be able to interpret the data in days. The mean of days is 15.27, with a standard deviation of 8.35 days and ranging from 0 to 29.75 days. Looking at the histogram of time values (Fig. A.7) we see peaks at specific number of days — that is at 0, 7, 14, 17, 21 and 24 days.



# Part 3 — Predicting customer spendings with linear regression
## 3.1/ Data preprocessing
In order to apply ML models and answer our question how we can predict customer spendings we need to create our target variable that we want to predict. This information is kept in the dictionary within the value column of the transcript table. So the transcript table was filtered for transaction events and the amount spent was extracted and then the data was grouped by peoples id and the amount of money was summed up. Now we have a table that contains how much each person has spent since they were registered. Since there were no missing values, nothing had to be cleaned.
Next we need to gather our features that we want to use for our predictions. Therefore our new table was merged with the profile data to get demographic data for each person. We now have a data set that contains our target variable and possible features with no missing values (Fig. B.1). Another column was added that measures the number of days a customer is a member until the date of 20th of September 2020 (randomly chosen), named member_days. A later date would just add a constant amount of days to every row, statistically not changing anything (excluding scaling considerations at this point) in the way to predict spendings.

Figure B.1
In order to also take gender into consideration for regression analysis, we have to do some data wrangling, since gender is a categorical variable. So what we can do is creating dummy variables with pandas get_dummies() function. In that way, for each value in the gender column a new column is created and if a person had the specific value, the corresponding column is coded as 1. The other columns are coded as 0 (Fig. B.2). This gives us numerical values that are interpreted dichotomously and can be included in the regression analysis. Also this even includes missing data, which in this case is not a problem, since it’s imaginable that people that didn’t want to provide information about gender spend in a specific manner.


## 3.2/ Implementation
Then a function for applying the sklearn LinearRegression model was defined, that takes a pandas dataframe, with features and our target variable data. Rows with missing values in the target variable are dropped (not necessary in this case but needed for applying the regression model). Missing values within the features are imputed and filled with the mean. This is just one way to cope with missing data, but for now we do not have any missing data. Then, data is splitted into train and test data, sklearn LinearRegression model is trained and data is normalized by the models functionality. Test scores are calculated and everything of interest is put out.
R-Squared is 0.30545 for training data and 0.28487 for test data, meaning we were able to explain 28.49% of variance in average spendings by income, age, member days, time and gender (on unseen data). Since the test score is a little smaller than the train score, we seem to have slight overfitting, meaning our ML model is explaining this particular (training) data set and fails a bit to generalize on unseen data. More information about overfitting and underfitting can be found here.
To better understand the the impact of different features four plots are given below, each showing a scatter plot of average money spend over one of the features, both being data randomly taken as training set (Fig. B.4). The y-axis is again set between 0 and 35. OLS Regression plots are visualized for different gender groups with plotlys build-in regression function. It has to be mentioned that this is not visualizing the regression model, which is a multiple regression model and those shown below are different bivariate regression models.

Figure B.4
Looking at the plots and regression graphs, it seems that only income is able to explain variance in average customer spendings. The member days plot also shows us a different density of values. Considering Fig. A.5 this makes sense, because there was a strong increase in member registrations, so we have less data for longtime members. Also we see in all plots, that there seems to be a group of customers that is not spending more than around 5 Dollars in average, regardless of features we’ve taken into consideration.
Feature Engineering

Figure B.5
Looking at the distribution of average customer spendings, we can see a bimodal distribution (Fig. B.5). There’s a high peak for customers that just spend a few dollars on average and a moderate peak for the rest of the distribution. The mean (average per person) spend is 13.68 dollars, with a standard deviation of 16.06 dollars. The maximum average spend is at around 451.47 dollars.
To understand impact of features, we take a look at the coefficients of our final model (Fig B.6). Since the initial features were of different natural scales (with gender being categorical) we need to standardize the model coefficients by their corresponding standard deviations (find more about this here). Now we can validate our first impression that income has the biggest impact on average spendings. It’s followed by missing gender values (negative influence) and being female. Also confirming that people who didn’t provide gender information spend less than those who did.

Figure B.6
Next, to improve our model it might be of value to add predictions based on offer type or even the specific offer. Because so far we have almost only included demographic information. Therefore the following measures are added:
Offer viewing rate → defined as ratio of viewed offers by received offers concluding in a value between 0 and 1, representing the percentage of viewed offers, because a hypothesis might be that people who open more offers are more engaged, hence spend more money. The distribution of viewing rates are shown in Fig. B.7. Mean viewing rate is 0.759, with a standard deviation of 0.235. A lot of the people seem to view all of the offers.

Figure B.7
The number of transactions → defined as number of times a customer bought something. The right-skewed distribution of amount of transactions are shown in Fig. B.8. Mean number of transactions is 8.38 transactions, with a standard deviation of 5.01 transactions.

Figure B.8
To evaluate the impact of any offer we need to find out for every customer, which of the ten offers was (received,) viewed and then completed. Here we again have to keep in mind that an offer can be received and completed, but was never viewed. This is defined as an offer with influence on a customer. Therefore transcript and profile data sets were merged again, all events were filtered out of the transcript data set and offer id information is extracted out of the dictionary in the value column. Here a problem occurred — some dictionary keys were called “offer id” and some “offer_id”. To solve this a lambda function was used to either get value out of one key or the other for all rows. The resulting column was called offer_id (see excerpt of table in Fig. B.9).

Figure B.9
Now we can filter out those completed offers that directly have a viewed entry in the row before (viewing includes receiving for logical reasons). A user-item-matrix (Fig. B.10) is created with dummy variables for each offer and if an offer influenced a customer 1 is added to the specific offer id column. So the matrix reflects how often a customer was influenced by which offer.

Figure B.10
Finally another feature was added: the influenced offer count → it’s defined as the number of times a customer was influenced by an offer (customer received, viewed and completed any offer). You can see this as well in Fig. B.10. The distribution of influenced offers is shown in Fig. B.11. On average people are influenced by 1.28 offers, with a standard deviation of 1.19 offers.

Figure B.11
Multicollinearity
We now created a bunch of other features to hopefully improve our predictions. One thing we now definitely have to have a look at is the problem of multicollinearity. The problem occurs when two or more of the independent variables (our features) are highly correlated with each other in a multiple linear regression (find further explanation here). In those cases we have a problem distinguishing between the individual effects of features.
One technique to detect multicollinearity is the Variance Inflation Factor (VIF). Other ways would be correlation matrices and scatter plot but VIF is preferred in general as explained by Analytics Vidhya:
Although correlation matrix and scatter plots can also be used to find multicollinearity, their findings only show the bivariate relationship between the independent variables. VIF is preferred as it can show the correlation of a variable with a group of other variables.
The VIF table for our independent variables looks like the following:

Here we can see some interesting numbers. Generally high values mean the features can be predicted by other independent variables in the dataset. If the VIF is between 5 and 10, multicollinearity is likely present and you should consider dropping the variable. We have missing VIF values for the two informational offers and the not provided gender information. They probably do not explain anything at all so we drop them. If the VIF is (like in some of our cases) infinity, this means some of the independent variables are perfectly correlated. That probably is the case because the influence count is just the sum of each influenced offer count, hence doesn’t contain any additional information, so we drop the influence count. Afterward we get the following VIF table:

Now the only VIF values over 2 are female and male gender, this also makes sense because we created dummy variables and if someone is of male gender, the person can not be female as well, so we drop male gender feature. This process was done two more times until no more values above 5 were left. So finally member days, viewing rate and age were dropped as well. The final VIF table looks like this:

Those are the cleaned features, that we can use for prediction without any relevant collinearity.
Coefficient rankings
Now, taking the cleaned feature set we modeled into our linear regression model, we get coefficients that look like the following (Fig. B.12).

Figure B.12
Here we see, that the 6 features with most impact are income, female gender, transaction count, the two BOGOs with reward and difficulty 10 and the BOGO with reward and difficulty 5. The R-Squared for linear regression we now have is 0.27796. So we lost a little of predictability. Other model parameters can be seen in Fig. B.13. We see that all p-values of our model are significantly low.

Figure B.13

## 3.3/ Refinement
### Normalization
When comparing normalized and not normalized data in linear regression we so no difference in training and testing scores. Another way to fine tune our model is regularization.

### Regularization
A linear regression model works by minimizing the loss function. A coefficient for each feature is chosen and large coefficients can lead to overfitting. A way to refine the model we have so far regarding complexity is regularizing the model — e.g. using the Ridge Regression. This model is keeping the model complexity at a moderate level with giving penalties on high coefficients. More about the Ridge Regression can be found here.
So the Ridge Regression has two parameters that can be fine tuned — alpha and normalization. To find the best parameters a 5-crossfold-validation (5-CV) with GridSearchCV was implemented. Parameter space for normalization was set to True and False and for alpha a linear space with 50 samples was set. Evaluating best parameters of 5-CV tells us that we get best prediction scores (R-Squared 0.29721 as best score) for alpha being 14.69 and not normalizing.

# Part 4— Conclusion
## 4.1/ Reflection
So let’s look back what we’ve done in this project. We initially looked at the data and specific characteristics of different columns. We cleaned the data, engineered various features, applied a Linear Regression model and checked for multicollinearity. Then we threw out a bunch of features. Finally we fine-tuned our model with normalization and regularization, speaking of Ridge and Lasso Regressions, which we fine-tuned with cross-fold-validation.
Since we just have implemented linear ML models there are some things to consider regarding other ML models, e.g. Decision Trees.
Decision trees also support non linearity, where Linear Regression models only support linear solutions. When there are large number of features with less data-sets (with low noise), linear regressions may outperform Decision trees/random forests. In general cases, Decision trees will be having better average accuracy. For categorical independent variables, decision trees are better than linear regression. Decision trees handles collinearity better than LR. You can read more about different ML models in comparison here.
After cleaning, wrangling and taking a closer look at the data with all of our analysis we can come back to our initial project questions.
How can we best predict how much money a customers will spend?
We have learned that income, being female, the number of transactions and BOGOs are able to predict the average customer spendings the quite well. For logical implications this seems plausible. People with high income, women, people that do not provide gender information, people that (in the past) were already influenced often by offers and people that buy often are the groups that spend more on average.


## 4.2/ Limitations and improvement
Coming to an end, this project is far from being finished. It should be viewed as a starting point and first indication for further analysis. Although our models performed quite well, one could try to get even more statistically validated results. The following limitations should be considered:
The difference in Lasso Model selection and coefficient rankings should be further explored and explained, because features like income were deselected by Lasso Regression.


