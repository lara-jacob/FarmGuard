# FarmGuard


### Problem Statement : Knowledge representation and insight generation from structured datasets.





### Objectives


>To implement precision agriculture, modern farming technique that uses research data of soil characteristics, soil types, crop yield data collection and suggests the farmers the right crop based on their site specific parameters to reduce the wrong choice on a crop and increase in productivity.

>To solve the problem by proposing a recommendation system through an ensemble model with majority voting technique crop for the site specific parameters with high accuracy and efficiency.

>To recommend fertilizer on the basis of N, P, K values and crop.

>To detect the disease.

### Unique Idea 

##### Precision Crop Recommendation: 
FarmGuard uses sophisticated algorithms to analyze various data points such as soil quality, climate conditions, and historical crop performance. This analysis enables it to recommend the most suitable crops for a particular farm, optimizing yields and promoting sustainable farming practices.

##### Optimized Fertilizer Management: 
By leveraging real-time data and advanced analytics, FarmGuard recommends precise types and quantities of fertilizers based on the specific nutrient needs of the soil. This approach minimizes over-application, reduces costs, and decreases environmental impact, while still maximizing crop productivity.

##### Early Disease Detection and Management:
Using AI-driven image recognition and sensor data, FarmGuard can identify potential diseases and pest infestations early. This early detection allows farmers to take timely action, reducing crop losses and minimizing the use of pesticides

### Features Offered

##### Crop recommendation:
Get personalized crop recommendations based on your soil type and weather conditions.

##### Fertilizer recommendation:
Discover the best fertilizers to use for the crops to ensure optimal growth and yield.

##### Disease detection:
Identify potential diseases in the crops early to prevent spread and minimize damage.

### Implementation

##### Crop Recommendation

1.Dataset Acquisition
2.Data preprocessing
  > Normalization: Ensured uniform scale across different features.

  >Handling Missing Data: Inputed missing values using technique

3.ML Model Training & creating .pkl file

   > Machine Learning Algorithms Used

        * Logistic Regression: accuracy achieved 95.23
        * Support Vector Machine (SVM): accuracy achieved 97.73
        * K-Nearest Neighbors (KNN): accuracy achieved 97.50
        * Decision Tree: accuracy achieved 97.73
        * Random Forest: accuracy achieved 99.55
 
  > Loaded .pkl model file to recommend crop based on input data
4.Choose the model K-Nearest Neighbors (KNN)
5.Made a user interface that reads input values N,P,K,temperature,humidity,pH,rainfall and recommends the output based on trained model.


##### Fertilizer Recommendation

1.Dataset Acquisition
2.Data preprocessing
  > Normalization: Ensured uniform scale across different features.

  >Handling Missing Data: Inputed missing values using technique

3.ML Model Training & creating .pkl file

   > Machine Learning Algorithms Used

        * Logistic Regression: accuracy achieved 96.66
        * Support Vector Machine (SVM): accuracy achieved 83.33
        * DecisionTreeClassifier model: accuracy achieved 93.33
        * RandomForestClassifier model: accuracy achieved 99.95
        * GradientBoostingClassifier model: accuracy achieved 93.33
 
  > Loaded .pkl model file to recommend fertilizer based on input data
4.Choose the model RandomForestClassifier
5.Made a user interface that reads input values N,P,K,temperature,humidity,moisture,soil type,crop type and recommends the output based on trained model.

##### Disease Detection

1.Data Acquisition

   >Image Scraping from Kaggle.

   >Providing disease labels.
   
2.Data Cleaning & Data Augmentation

   > Data is cleaned to remove not useful content.

   > The dataset is augmented so as to increase the variability.
   
3.DL model creation- CNN model

4.Disease Detection 

5.Made a user interface that reads image as input value and predict the output based on trained model.

### Conclusion

India’s farmers are hard at work. They help to feed a nation whose population is nearly 1.4 billion. However their productivity is threatened by some natural factors that can ruin their crops and their livelihoods. So, the solution will benefit farmers to maximize productivity in agriculture, reduce soil degradation in cultivated fields, and have informed advice on organic fertilizers/ other fertilizers and also know about the right crop by considering various attributes. This would provide a comprehensive prediction and hence benefit both farmers and environment.
