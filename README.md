# <a name="top"></a>ZILLOW HOME VALUE PREDICTION - REGRESSION PROJECT
![]()

by: Vincent Banuelos

***
[[Project Description/Goals](#project_description_goals)]
[[Initial Questions](#initial_questions)]
[[Planning](#planning)]
[[Data Dictionary](#dictionary)]
[[Reproduction Requirements](#reproduce)]
[[Conclusion](#conclusion)]

___

## <a name="project_description_goals"></a>Project Description/Goals:
The purpose of this project is to construct a model to predict assessed home value for single family properties using regression techniques.

Goal 1: Find the key drivers of property value for single family properties.

Goal 2: Construct an ML Regression model that predict propery tax assessed values of Single Family Properties using attributes of the properties.

Goal 3: Deliver a report that a data science team can read through and replicate, understand what steps were taken, why and what the outcome was.

[[Back to top](#top)]


## <a name="initial_questions"></a>Initial Questions:

1. How does the size of a property affect the price?
2. Which county has the most expensive properties on average?
3. Do newer homes sell for more than older homes?
4. What affects house prices more bathrooms or bedrooms?

[[Back to top](#top)]


## <a name="planning"></a>Planning:

- Create README.md with data dictionary, project and business goals, and come up with initial hypotheses.
- Acquire data from the Codeup Database and create a function to automate this process. 
- Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process. 
- Store the acquisition and preparation functions in a wrangle.py module function, and prepare data in Final Report Notebook by importing and using the function.
- Clearly define at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Establish a baseline accuracy and document well.
- Train 3 different regression models.
- Evaluate models on train and validate datasets.
- Choose the model that performs the best and evaluate that single model on the test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.

[[Back to top](#top)]

## <a name="dictionary"></a>Data Dictionary  

| Target Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|tax_value|The total tax assessed value of the parcel|float|
---
| Feature | Definition | Data Type |
| ----- | ----- | ----- |
| parcelid |  Unique identifier for parcels (lots)  | int |
| bedrooms |  Number of bedrooms in home | float |
| bathrooms|  Number of bathrooms in home including fractional bathrooms | float |
| sqft |  Calculated total finished living area of the home | float |
| county | Name of county property is located in| object |
| fireplacecnt|  Number of fireplaces in a home (if any)| float|
| garagecnt | Total number of garages on the lot including an attached garage | float |
| lotsizesquarefeet |  Area of the lot in square feet | float |
| yearbuilt |  The Year the principal residence was built| float |
| poolcnt |  Number of pools on the lot (if any) | float |
| transactiondate | Date home was sold | datetime |
| zip | Zipcode of property | float |
| latitude |  Latitude of the middle of the parcel multiplied by 10e6 | float |
| longitude |  Longitude of the middle of the parcel multiplied by 10e6 | float |
| los_angeles| 1 fi the house is located within Los Angeles County|int|
| orange| 1 fi the house is located within Orange County|int|
| ventura| 1 fi the house is located within Ventura County|int|

---

## <a name="reproduce"></a>Reproduction Requirements:

You will need your own env.py file with database credentials then follow the steps below:

  - Download the wrangle.py, model.py, explore.py, and final_report.ipynb files
  - Add your own env.py file to the directory (user, host, password)
  - Run the final_report.ipynb notebook

[[Back to top](#top)]


## <a name="conclusion"></a>Conclusion and Next Steps:

- Exploration of Data showed that the following items showed the following features have at least some relationship to churn.
    - Square footage of the property
    - Number of bedrooms and bathrooms
    - Location of the properties
    - Property age
- Takeaways
    - Out of all three counties, Los Angeles has the cheapest average price for housing. 
    - Los angeles however has a lot more houses that are higher than the average price yet relatively smaller in terms of square footage. 
    - Los Angeles has more of an abundant of older homes.
    - No matter the county it seems homes built after 1960 are above the average Tax Value.
    - Homes with above the median amonut of bathrooms and below the median amount of bedrooms are more expensive then the opposite. 

- Model
  - The Tweedie Regressor regression model performed best with a $205,837 RMSE and a .246 R2 value for the train dataset and a $200,559 RMSE, .257 R2 value for the validate dataset.
  - For the test dataset the model performs a little worse, with a drop off of $3,000.

- Next Steps/Reccomendations
    - Following this initial project, I would like to create a more refined model. Whether it is through running different groups of features through my current model, tuning the hyperparameters or both.
    - As well as creating more python files that store functions so as to make the final report less filled by code and easier to read.
    - Create a feature that clusters locations in the dataset, most home prices fluctuate by location within cities. With certain city locations being higher in crime, less wealthy, under developed, etc.
    - Include previous assesments of the porperties to see if previous pricing of properties affect sell price.
    
    
[[Back to top](#top)]
