# <a name="top"></a>ZILLOW LOG ERROR PREDICTION - CLUSTERING PROJECT
![]()

by: Patrick Amwayi & Vincent Banuelos

***
[[Project Description/Goals](#project_description_goals)]
[[Initial Questions](#initial_questions)]
[[Planning](#planning)]
[[Data Dictionary](#dictionary)]
[[Reproduction Requirements](#reproduce)]
[[Pipeline Takeaways](#pipeline)]
[[Conclusion](#conclusion)]

___

## <a name="project_description_goals"></a>Project Description/Goals:
- The goal of this project is to identify key drivers of logerror in the Zillow property value estimates and come up with a model for predicting the logerror.

- To identify key attributes that drive the logerror in Zestimates. This will help us come up with better prediction models so that Zillow can remain competitive in the housing market. Compare all models and evaluated by how well they performs over the baseline. To give recommendations that can help reduce the overall average logerror.

[[Back to top](#top)]


## <a name="initial_questions"></a>Initial Questions:

- Does the county a property is located in affect it's log error?
- Does the tax variables of a house affect the logerror?
- Does the ratio of home sqft to lot sqft affect logerror?
- Does the year a house was built affect logerror?

[[Back to top](#top)]


## <a name="planning"></a>Planning:

- Create README.md with data dictionary, project and business goals, and come up with initial hypotheses.
- Acquire data from the Codeup Database and create a function to automate this process. 
- Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process. 
- Store the acquisition and preparation functions in a wrangle.py module function, and prepare data in Final Report Notebook by importing and using the function.
- Clearly define at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Establish a baseline accuracy and document well.
- Train at least 3 different regression models.
- Evaluate models on train and validate datasets.
- Choose the model that performs the best and evaluate that single model on the test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.

[[Back to top](#top)]

## <a name="dictionary"></a>Data Dictionary  

| Target Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|logerror|The difference betwen the log of the Zestimate and the log of the sale price|float|
---
| Feature | Definition | Data Type |
| ----- | ----- | ----- |
| county | Name of county property is located in| object |
| yearbuilt |  The Year the principal residence was built| float |
| tax_value |  The total tax assessed value of the parcel | float |
| structuretaxvaluedollarcnt | The assessed value of the built structure on the parcel| float |
| landtaxvaluedollarcnt | The assessed value of the land area of the parcel | float |
| latitude |  Latitude of the middle of the parcel multiplied by 10e6 | float |
| longitude |  Longitude of the middle of the parcel multiplied by 10e6 | float |
| los_angeles| 1 if the house is located within Los Angeles County|int|
| orange| 1 if the house is located within Orange County|int|
| ventura| 1 if the house is located within Ventura County|int|
| house_lotsize_ratio| Gives the percentage of land a house takes up out of the lotsize| float |

---

## <a name="reproduce"></a>Reproduction Requirements:

You will need your own env.py file with database credentials then follow the steps below:

  - Download the wrangle.py, model.py, explore.py, and final_report.ipynb files
  - Add your own env.py file to the directory (user, host, password)
  - Run the final_report.ipynb notebook

[[Back to top](#top)]


## <a name="pipeline"></a>Pipeline Conclusions and Takeaways:

###  Wrangling Takeaways
- We started off by pulling a SQL query for Single Family homes sold in 2017. 
- Approximately 52,000 observations were recieved from the CodeUP database using SQL.
- Following the Data Acquisition the following preparation work was done to the acquired data:
   - Removed columns and rows that were missing more than 50% of their data so as to ensure observations were suitable for this project.
   - Following data prepartion we were left with a dataframe consisting of 43,628 observations.
   - Split data into 3 datasets, train, validate and test.

### Exploration Summary
- We chose features to ivestigate, and created clusters for those selected features to see if they could assist in finding drivers for logerror. Of all the features selected only tax_value showed a relationship towards logerror.

- Whatever the case we will take these features into modeling and see if they assist in improving logerror prediction.

### Modeling takeaways
- The Polynomial Features Degrees = 3 model peformed the best out of all 4 models tested for both train and validate datasets. 

- However did not outperform baseline on the test dataset. Further research will need to be done to improve these models.

[[Back to top](#top)]


## <a name="conclusion"></a>Conclusion, Reccomendations and Next Steps:

- Of the features we investigated, the tax variable features and location features showed a relationshi[p owards logerror.

- With the housing market being so volatile and prone to being affected by outside forces it can be hard to predict both pricing and improve logerror.

- We believe that this dataset is simply too large and perhaps focusing in on smaller areas may provide some benefits.

- With that said our final conclusion is that the features elected for our model are not ones to be utilized and further research will need to be done to improve the logerror.

- Next steps after this project may be to choose different features and focus in on smaller areas with the 3 counties.    
    
[[Back to top](#top)]
