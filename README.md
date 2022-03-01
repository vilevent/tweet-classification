# Text Classification with Weka
> Group project from Data Mining course

> **SemEval-2019 Task 5 - Multilingual detection of hate speech against immigrants and women in Twitter** 
> (TASK A - Hate Speech Detection against Immigrants and Women)


### High-level problem
Identification of multilingual hate speech (English and Spanish) aimed at women and immigrants in Twitter posts. This issue regarding online hate speech directly relates to misogyny and xenophobia. Here, we focus on working with **real-life large-scale text data** for English tweets.

The problem at hand is considered to be a **binary classification problem** as each instance in the data contains one of two class labels: 
- 1 (hate speech is present in tweet; HS) 
- 0 (hate speech is *not* present in tweet; NonHS)


### High-level solution (as a black box)
An algorithm that evaluates keywords in tweets about immigrants and women to detect if the tweet is hateful or non-hateful. 

The inputs to the algorithm are the tweets. The outputs of the algorithm are the binary labels indicating hate speech presence or absence in the tweets.

##

### Tuning classifiers with the Weka Experimenter

Multiple classifier combinations were evaluated on `train.arff`. The validation technique was **10-fold cross-validation with 10 repetitions**. The primary performance metric was **F-measure**, which is the same metric used by the organizers of SemEval-2019.

#### Setup: 
With *AttributeSelectedClassifier* wrapped up inside *FiteredClassifier*, we investigated various options for: 
- ***StringToWordVector* filter** (e.g., outputWordCounts, lowerCaseTokens, wordsToKeep, Stemmer)
- **Attribute selection** (e.g., WrapperSubsetEval, CfsSubsetEval, and GainRatioAttributeEval).

We can break up each tweet into tokens by using the *StringToWordVector* filter.

**NaiveBayesMultinomial**, **ZeroR**, **OneR**, **SMO** (support vector machine - SVM), and **RandomForest** were our chosen classifiers inside *AttributeSelectedClassifier*. We had ZeroR and OneR as baselines.


### Preparing the datasets
`train+test.arff` is where we appended the testing data to the training data.

This file (of English tweets) contained 75% training instances and 25% test instances.
- **Training Instances:** 9,000
- **Test Instances:** 3,000


### Evaluating the best models on `train+test.arff`

We used F-measure for the performance metric and **Train/Test Percentage Split 75% (order preserved)** for the validation technique.

We picked the best model for each of the classifiers, and they are the following:

- **Model 1** - *NaiveBayesMultinomial*: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute Selection: CfsSubsetEval.

- **Model 2** - *ZeroR*: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute Selection: SymmetricUncertAttributeEval.

- **Model 3** - *OneR*: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute Selection: OneRAttributeEval.

- **Model 4** - *SVM*: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute selection: InfoGainAttributeEval, Ranker method, numToSelect=15.

- **Model 5** - *RandomForest*: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute Selection: SymmetricUncertAttributeEval, Ranker method, numToSelect=10. 

For the evaluation, we gathered the F-measure value for each model, as well as the number of true positives (**TPs**), true negatives (**TNs**), false positives (**FPs**), and false negatives (**FNs**).

##

### Results from the evaluation
Comparison flags: &nbsp;v for **significantly better** performance and * for **significantly worse** performance compared to *NaiveBayesMultinomial* classifier.

![Table Results](https://user-images.githubusercontent.com/96803412/148715103-ba924ceb-1943-4bb2-8fd0-3024c8d35b93.png)

*The F-measure for ZeroR could not be obtained due to a division by zero error.*

![Performance Plot](https://user-images.githubusercontent.com/96803412/148705423-b63169e0-a14e-4d7d-8d4f-0ee2e80c55b6.png)

### Interpreting the results
NaiveBayesMultinomial is an algorithm best suited for text data, so it is understandable that it was in the top three contenders (with SVM and RandomForest) for best F-measure. 

For both the train and test datasets, SVM and RandomForest performed **significantly better** than NaiveBayesMultinomial.

There were more **false positives** than **true positives** for the four models referenced in the plot, which indicates that the algorithms tended to classify tweets as hate speech when they were actually non-hate speech.

Our intuition gained from the results is that Random Forests (*an ensemble learning method*) is a powerful technique that **exhibits good predictive performance** on independent test data compared to our remaining models. Model 5 is our top model.


### Discussing the top model results

The [SemEval-2019 task paper](https://aclanthology.org/S19-2007/) reported that the state-of-the-art model was from the Fermi team; their model achieved an F-measure of 0.651 on the English tweets. In comparison with our top model, we obtained an F-measure of 0.56.

These numbers are directly comparable since the same train set and test set (with the same class labels of HS and NonHS) were utilized during the experimental setup.
