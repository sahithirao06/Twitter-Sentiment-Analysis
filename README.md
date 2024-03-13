# TwitterSentimentAnalysis
##### Problem Statement:

The goal of this project is to perform sentiment analysis on tweets to determine whether each tweet expresses a positive, negative, or neutral sentiment. Sentiment analysis will help us understand the overall sentiment of the Twitter users towards a particular topic, brand, or event.

##### Data Collection: We gathered a diverse dataset from kaggle which consists of labeled tweets encompassing positive, negative, and neutral sentiments.
##### Data Preprocessing: 
We cleaned and tokenized the text, removed stopwords, and applied techniques like stemming or lemmatization to normalize the words. Following are the steps for Preprocessing:
**Raw Data Cleaning:**
Convert Text to Lowercase: Standardize text case to reduce vocabulary size and variation.
Remove Mentions: Exclude '@mentions' to eliminate irrelevant user references.
Eliminate Special Characters: Strip punctuation and special characters for uniformity.
Filter Out Stopwords: Discard common, non-informative words ('the,' 'is,' 'and,' etc.).
Hyperlink Removal: Omit URLs, which don't contribute to sentiment analysis.
**Tokenization**:
Tokenization Breaks text into individual words (tokens) for analysis.
Tokens serve as input features for machine learning algorithms.
**Stemming**:
Stemming reduces words to their base form (stem) to capture core meanings.
For instance, 'satisfying,' 'satisfaction,' and 'satisfied' all stem to 'satisfy.'
**Feature Extraction:** We converted the text into numerical features suitable for machine learning algorithms. TF-IDF and word embeddings were commonly used techniques.

###### Splitting into Training and Testing Sets: 
To assess the effciancy of sentiment analysis models, a division of the dataset into training and testing sets waas made. The training set was the crucible for molding and refining our models, while the testing set, for evaluating the models' capacity for accurate prediction and generalization.

###### Exploratory Data Analysis (EDA): 
This phase unveiled insights and patterns residing within the dataset. EDA provided a window into the distribution of sentiment labels, unveiled any potential class imbalances, and furnished preliminary insight into the linguistic tapestry weaving through the various sentiment categories.

##### Model Training and Evaluation: 
We split the dataset into training and testing sets, trained the models, and evaluated their performance using metrics like accuracy, precision, recall, and F1-score.
After preprocessing the data to ensure its quality and suitability for analysis, Proceeded to fit the sentiment analysis model using the selected algorithms: **Logistic Regression**, **Decision Tree Classifier**, **Random Forest Classifier**, and **Naive Bayes**. Each algorithm underwent a series of steps, including feature extraction, model training, and hyperparameter tuning, to optimize their performance. Here's an overview of how we fit the models and our conclusion on which algorithm is best suited for this project:

##### Logistic Regression:
Began by encoding the preprocessed tweet text into numerical features using techniques such as TF-IDF (Term Frequency-Inverse Document Frequency). These features were then fed into the Logistic Regression model. We performed a grid search to find the optimal regularization parameter (C) through cross-validation. The resulting model was trained on the training set and evaluated on the testing set using metrics:
Training Accuracy: 78.09%
Validation Accuracy: 73.11%
F1 Score: 0.7573
Confusion Matrix: [[3963 1883]
 [1539 5340]]
##### Decision Tree Classifier:
For the Decision Tree Classifier, Used the preprocessed features as inputs and trained the model on the training set. To avoid overfitting, we explored different tree depths and minimum samples per leaf during the tuning process. The performance of the model was assessed using the same set of evaluation metrics:
Training Accuracy: 75.30%
Validation Accuracy: 70.33%
F1 Score: 0.7428
Confusion Matrix:
[[3500 2346]
 [1429 5450]]

 ##### Random Forest Classifier:
The Random Forest Classifier was trained by aggregating the predictions of multiple decision trees. Adjusted hyperparameters such as the number of trees and maximum features per split. This approach aimed to enhance the model's generalization capabilities while avoiding overfitting.
Training Accuracy: 77.82%
Validation Accuracy: 73.12%
F1 Score: 0.7616
Confusion Matrix:
[[3843 2003]
 [1417 5462]]
##### Naive Bayes:
The Naive Bayes model, which is well-suited for text classification, was fitted using the preprocessed features. Then, applied techniques like TF-IDF to transform the text into numerical values. The model's performance was evaluated using the same evaluation metrics as the other algorithms.
Training Accuracy: 70.28%
Validation Accuracy: 65.34%
F1 Score: 0.6435
Confusion Matrix:
[[4335 1511]
 [2899 3980]]

##### Results and Analysis:
Among the four algorithms tested, the Random Forest Classifier stands out as the most effective choice for your sentiment analysis project. It demonstrates the highest validation accuracy and F1 score, indicating its ability to accurately classify sentiment categories across the board. While Logistic Regression and Decision Tree also show promise, Random Forest's ensemble nature enables it to capture intricate relationships within the data, resulting in a more balanced and reliable sentiment prediction.

Naive Bayes, although providing insights, falls behind in terms of accuracy and F1 score, suggesting that it might not be the most suitable option for this specific sentiment analysis task.


In summary, based on the provided metrics, the Random Forest Classifier appears to be the optimal algorithm for your Twitter Sentiment Analysis project due to its consistent and robust performance across different sentiment categories.


