import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("./Advertising.csv", index_col=0)
data.columns = ["TV", "Radio","Newspaper","Sales"]
data.head(5)

data.describe()

data.shape

fig, axs = plt.subplots(1, 3, sharex=True)
data.plot(kind="scatter", x = "TV", y="Sales", ax=axs[0], figsize=(16, 8))
data.plot(kind="scatter", x = "Radio", y="Sales", ax=axs[1])
data.plot(kind="scatter", x = "Newspaper", y="Sales", ax=axs[2])


# # Questions About the Advertising Data¶
# ## On the basis of this data,how should we spend our advertising money in the future? These general questions might lead you to more specific questions:
# ### Is there a relationship between ads and sales?¶
# ### How strong is that relationship?
# ### Which ad types contribute to sales?
# ### What is the effect of each ad type of sales?
# ### Given ad spending, can sales be predicted?
# ### Let us explore these questions below!

#we are taking only one variable for now
feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

print("intercept", lm.intercept_)

print("coff", lm.coef_)

#so in the above model we find A "unit" increase in TV ad spending is associated with a 0.047537 "unit" increase in Sales.

#create a DataFrame with the minimum and maximum values of TV
x_new = pd.DataFrame({"TV": [data.TV.min(), data.TV.max()]})
x_new

# make predictions for those x values and store them
preds = lm.predict(x_new)
preds

# first, plot the observed data
data.plot(kind="scatter", x = "TV", y = "Sales")
# then, plot the least squares line
plt.plot(x_new, preds, c='red', linewidth=2)

#Let's do some statistics on data
import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales ~ TV', data=data).fit()
lm.conf_int()
#Keep in mind that we only have a single sample of data, and not the entire population of data. The "true" coefficient is either within this interval or it isn't, but there's no way to actually know. We estimate the coefficient with the data we do have, and we show uncertainty about that estimate by giving a range that the coefficient is probably within.
#Note that using 95% confidence intervals is just a convention. You can create 90% confidence intervals (which will be more narrow), 99% confidence intervals (which will be wider), or whatever intervals you like.
# Hypothesis Testing and p-values¶
# Closely related to confidence intervals is hypothesis testing. Generally speaking, you start with a null hypothesis and an alternative hypothesis (that is opposite the null). Then, you check whether the data supports rejecting the null hypothesis or failing to reject the null hypothesis.
# (Note that "failing to reject" the null is not the same as "accepting" the null hypothesis. The alternative hypothesis may indeed be true, except that you just don't have enough data to show that.)
# As it relates to model coefficients, here is the conventional hypothesis test:
# null hypothesis: There is no relationship between TV ads and Sales (and thus $\beta_1$ equals zero)
# alternative hypothesis: There is a relationship between TV ads and Sales (and thus $\beta_1$ is not equal to zero)
# How do we test this hypothesis? Intuitively, we reject the null (and thus believe the alternative) if the 95% confidence interval does not include zero. Conversely, the p-value represents the probability that the coefficient is actually zero:
lm.summary()

lm.pvalues

#so on the above values we can see that the values is far less than 0.05% so, we failed to reject the null hypothesis. hence there is a relationship between TV and Sales

#will check for Rsquar values to see wether our modal is good or not
print(lm.rsquared)

#so the above r2 value is ~61% it's look like good . but it's depends on our thresold 

#Multi linear regrations
feature_cols = ['TV', 'Radio', 'Newspaper']
x = data[feature_cols]
y = data.Sales

#spliting data into train and test : 70 : 30 ratio
from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3,random_state=42)


#preparing for modal
lm = LinearRegression()
lm.fit(x_train, y_train)



print("intercept", lm.intercept_)

print("cofficient", lm.coef_)

#prediction
pred = lm.predict(x_test)


print("rsquar value", sqrt(mean_squared_error(y_test,pred)))


#lets do the statical summary
lm = smf.ols(formula="Sales ~ TV+Newspaper+Radio", data=data).fit()
lm.conf_int()
lm.summary()


#p value
print(lm.pvalues)


#so there are many points we can note in the above summary
# 1. The p value of TV, Radio is less than 0.05% , it means we failed to reject null hypothesis where is Newspaper have more than 0.05% value so will reject the null hypothesis
# 2. Sales are highly dependent on TV and Radio rather than Newspaper
# 3. The rsquar value is ~90% which far better than only predecting for TV which has ~61%


#feature selection - which variable need to keep and which we should remove
#some points we can keep in mind - 
# 
# b. keep the features who have p values is less than 0.05%
# c. check rsquar values is closer to original modal with all feature, after removing that particular feature if rsquar values is closer to original , will remove that


# # Feature Selection¶
# ### How do I decide what features has to be included in a linear model? Here's one idea:
# ##### a. remove features who have more than 50% data missing values(NA, NAN)
# ##### b.Try different models, and only keep predictors in the model if they have small p-values.
# ##### c. Check whether the R-squared value goes up when you add new predictors.
# # What are the drawbacks in this approach?
# ##### a.Linear models rely upon a lot of assumptions (such as the features being independent), and if those assumptions are violated (which they usually are), R-squared and p-values are less reliable.
# ##### b.Using a p-value cutoff of 0.05 means that if you add 100 predictors to a model that are pure noise, 5 of them (on average) will still be counted as significant.
# ##### c.R-squared is susceptible to overfitting, and thus there is no guarantee that a model with a high R-squared value will generalize. Below is an example:
#


lm = smf.ols(formula="Sales~TV+Radio", data=data).fit()
lm.rsquared


#we can see the above rsquar values (TV+Radio) is approx same as (TV+Radio+Newspaper), so we can remove Newspaper from the data
#note - R-squared will always increase as you add more features to the model


# #### Handle categorical variable


import numpy as np


np.random.seed(123) #Setting random seed to not change values always


num = np.random.rand(len(data))
mask_large = num > 0.5
data["Size"] = "Small"
data.loc[mask_large, "Size"] = "Large"
data.head()


#creating dummy binary variable
data["isLarge"] = data.Size.map({"Small":0, "Large":1})
data.head(




