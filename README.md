## Project Title:  Predicting Depression in Rural Areas

- by CT

<img width="800" alt="image" src="https://github.com/CarolTeo11/Capstone-24.1/assets/130137674/8f0dfbf1-e0d9-4f36-9170-aabc44b94e77">

* Note: There are not many changes from Capstone-20.1.  Only made changes to plots to make it visually more informative in 3D.

### Executive summary
 
This project uses routine survey data from Kenya to predict depression.  In rural areas where psychiatric help is limited, it is important to be able to accurately predict the illness so that valuable resources can be allocated to assess and treat these cases.   

#### Background

A few years back, the World Health Organization estimated that 1.3 million Kenyans suffer from untreated major depressive disorder (MDD; commonly known as depression) every year, and that sub-Saharan Africa has the highest prevalence of the illness of any region in the world. Yet mental health treatment in Kenya suffers from a lack of resources and stigmatization. There are only two certified psychiatrists per million people in Kenya. Few facilities exist outside of urban areas and people are unlikely to know about or access them.

With this in mind, the Busara Center for Behavioral Economics has decided to challenge the data science community in Nairobi, across Africa, and around the world to predict depression cases from routine survey data.

Even though this challenge was done a while ago, the data remains relevant for data scientists to predict depression so that valuable resources can be targeted at more vulnerable persons to help assess their mental health conditions and if needed, help them seek treatment as soon as possible.  


### Rationale

Depression, if left untreated, can become an issue for the individual and their family members.  The issue can be exercerbated if the person is the sole breadwinner of the family.  However, in a country faced with poverty and limited medical resources, there is a need to prioritise such resources. 

Hence, in this assessment, i hope to be able to predict true positive with high accuracy while minimising false positive so that we strike a balance between good predictions and resource allocation.   


### Research Question

Classify, with high accuracy, the persons with depression.  

Measures of Effectiveness:  

1.  Minimize False Negative ==> Metrics: Maximise Recall = True Positive/ (True Positive + False Negative)

Secondary Measures of Effectiveness:

2.  Maximize accuracy 

3.  Minimize False Postive (to minimise unnecessary resources allocated) ==> Metrics: Maximize Precision = True Positive/ (True Positive + False Positive) 

4.  Maximize f1 scores

### Data Sources

I obtained a set of data from Kaggl for this analysis. The web address is https://www.kaggle.com/datasets/diegobabativa/depression

I have also uploaded the dataset for easy reference.  

Most of the entries are self-explanatory.

Survey_id

Ville_id

sex

Age

Married

Number_children

education_level

total_members (in the family)

gained_asset

durable_asset

save_asset

living_expenses

other_expenses

incoming_salary

incoming_own_farm

incoming_business

incoming_no_business

incoming_agricultural

farm_expenses

labor_primary

lasting_investment

no_lasting_investmen

depressed: target data, with [Zero: No depressed] and [One: depressed] (Binary for target class)

The data was split 70-30 into the train and test sets but "stratify = y" to ensure the proportion of depressed does not change despite the split.

### Methodology

In this example, I used a few methodologies for 'binary' classification models, namely, KNN, Logistic Regression, SVM, and Decision Tree.  

Thereafter, I used GridSearch CV to optimise the results on Logistic Regression and KNN.


### Results

#### Statistical Analysis on the dataset
First, it was observed that the dataset is an imbalanced dataset with 16.68% being diagnosed as depressed.  This implies that if we deploy a very simple model that categorises all individuals as being no-depressed, then our accuracy would already have been 83.32%.  

-<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/f271bb90-b0b1-4508-835c-6b55e6d635de">

A study of the correlation matrix shoes that none of the features stands out in the correlation table.  

<img width="450" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/52cc13eb-b38c-42f5-9381-3ab1413e8eba">

#### Model 1: Applying 4 classification models, Logistic Regression, Decision Tree, KNN and SVM

Nonetheless, 4 different classification models with default settings were used to make predictions on the depressed group:
- Logistic Regression
- Decision Tree
- kNearestNeighbors
- Support Vector Machine

Based on the accuracy, precision, recall and f1 scores, the initial findings show that none of the 4 models were good enough for predicting depression.  I would like to highlight that even though accuracy appears to be around 72 - 83%, it was no better than simply doing nothing.  (If we predict everyone is not depressed and simply do nothing, we would be correct 83% of the time.) 

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/f72390e0-2fcb-486a-9248-ffcb861144c0">

The following confusion matrix also demonstrates that most times, the models did not pick out any truely depressed cases.  

<img width="350" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/612f85cd-44c2-4fb4-bd45-3359b1cd33df">
<img width="350" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/2b23901e-03d2-4799-992b-4fdb078b09ed">
<img width="350" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/ef3e611e-0a4c-41a0-b6b3-60f1a6a3fda2">
<img width="350" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/04f4b987-9055-4881-9155-574de46e3698">

Hence, the initial 4 models were disregarded and are not good enough for deployment. 

#### Model 2:  Applying GridSearchCV on Logistic Regression and KNN

Since the initial models were trained on the default settings, I assumed that running GridSearchCV with scoring = 'recall' will be the solution to making the models run better.  I proceeded to apply GridSearchCV on both KNN and Logistic Regression models.  The results of the KNN-GridSearch was 

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/74e1ff3d-f245-4cf3-9f1a-b72d643861e7">

and the results from the Logistic Regression-GridSearch was 

<img width="580" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/c879b790-c113-44d4-a34b-883a39d6526c">

Based on the above, the Logistic Regression-GridSearch was no better than the KNN-GridSearch in optimising the recall and f1 scoring.  
Hence, I decided to vary the code to collate the proba and vary the threshold for classification.  This was applied to both the KNN and the Logistic Regression models.  

Again, these 2 models run on GridSearchCV were disregarded and considered not good enough for deployment.  

####  Model 3: Optimising the Logistic Regression by varying probability threshold 

Next, the probability measures were collated by varying the probability threshold and determine if the recall value increases by varying its probabilities. In this instance, if the model's probability prediction > probability threshold, then depression = 1, else 0.  The following chart rightly depicts that as probability threshold decreases, accuracy drops while recall increases.  

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/3a006267-ae30-4e94-8ae6-dbc814478fa4">

Based on the chart above, when threshold is 13%, recall exceeds 70% while accuracy was short of 40%.  Also, a recall of greater than 0.7 seems like a good value for the study.  The following shows the confusion matrix using probability threshold = 13%.  

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/c2eab411-4d96-4928-892b-d4a90bfa0c03">

I added workload measure here to understand the percentage of personnels needed to be screened for depressed.  The overall results of this model is 

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/030b6ef3-b7a9-4582-9d38-8ce11ec16f52">


####  Model 4: Optimising the KNN by varying probability threshold 

Similarly, the same methodology of varying probability threshold was repeated on KNN model.  The model showed 3 distinct clusters of results.  This is likely due to the n_neighbors factors. Again, the following chart rightly depicts that as probability threshold decreases, accuracy drops while recall increases.  

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/35c7a56d-0f41-4bd8-b349-fdf1620d69d9">

Based on the chart above, when threshold is 19%, the recall exceeds 0.9 which a very good value for the study.  The confusion matrix using probability threshold = 19% is shown here.  

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/617e9558-0857-49de-be1f-4aa4fdf53d6f">

The results of KNN with probability threshold = 19% are shown below. 

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/8bc73183-2395-4239-95ef-2be0a3af5099">


#### Best model - Model 4 is winner!!!!

As models 3 and 4 perform significantly better than the models 1 and 2, a quick comparision of the latter 2 models were made by plotting the recall-accuracy on the same chart and displaying the results in the same table.  

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/d7426ec1-1a35-4d7e-bc96-98683b967d0e">

<img width="550" alt="image" src="https://github.com/CarolTeo11/Capstone-20.1/assets/130137674/28951cf8-7ff6-47b9-b8b5-a946783c8060">


Based on a comparison of the different models and the objective set out in the problem statement, I would have deployed Model 4: KNN Model that categorised an individual as high risk requiring further medical assessment at the 19% probability level.  This is so far the best model as ~75% of the depressed will be called for further assessment and the workload is not extreme, i.e. ~61% of population.


### Next steps

Next Steps and recommendations:

1. It is recommended that the KNN model be deployed for the next campaign and new data be collated to further refine the model

2. Even though this is a classification model, a time series model can be deployed if there is sufficient data collated over a few years to study the effectiveness of this deployed model.  Also, it may be good for future data to include personnels at risk of depression so that a person can seek treatment before he/ she becomes clinically depressed.

3.  Other than endogeneous factors, exogeneous factors such as weather and economic status of the country can have an impact on a person's wellbeing.  Hence, other external data may be populated to make the study more complete. 


### Outline of project

- [Link to data](https://github.com/CarolTeo11/Capstone-20.1/blob/main/depressed.csv)
- [Link to python code](https://github.com/CarolTeo11/Capstone-20.1/blob/main/Capstone%20-%20Depression.ipynb)

