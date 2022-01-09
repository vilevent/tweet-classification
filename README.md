# Text Classification with Weka
> Group project from Data Mining course

> **SemEval-2019 Task 5 - Multilingual detection of hate speech against immigrants and women in Twitter** 
> (TASK A - Hate Speech Detection against Immigrants and Women)


### High-level problem
Identification of multilingual hate speech (English and Spanish) aimed at women and immigrants in Twitter posts. This issue regarding online hate speech directly relates to misogyny and xenophobia. Here, we focus on working with **real-life large-scale text data** for English tweets.

The problem at hand is considered to be a **binary classification problem** as each instance in the data contains one of two class labels: 
- 1 (hate speech is present in tweet; HS) 
- 0 (hate speech is not present in tweet; NonHS)


### High-level solution
An algorithm that evaluates keywords in tweets about immigrants and women to detect if the tweet is hateful or non-hateful. 

The inputs to the algorithm are the tweets. The outputs of the algorithm are the binary labels indicating hate speech presence or absence in the tweets.

##

### Tuning classifiers with the Weka Experimenter

Multiple classifier combinations were evaluated on `train.arff`. The validation technique was **10-fold cross-validation with 10 repetitions**. The primary performance metric was **F-measure**, which is the same metric used by the organizers of SemEval-2019.

#### Setup: 
With *AttributeSelectedClassifier* wrapped up inside *FiteredClassifier*, we investigated various options for: 
- ***StringToWordVector* filter** (e.g., outputWordCounts, lowerCaseTokens, wordsToKeep, Stemmer)
- **Attribute selection techniques** (e.g., WrapperSubsetEval, CfsSubsetEval, and GainRatioAttributeEval).

**NaiveBayesMultinomial**, **ZeroR**, **OneR**, **SMO** (i.e., support vector machines), and **RandomForest** were our chosen classifiers inside *AttributeSelectedClassifier*. We had ZeroR and OneR as baselines.


### Preparing the datasets
`train+test.arff` is where we appended the testing data to the training data.

This file (of English tweets) contained 75% training instances and 25% test instances.
- **Training Instances:** 9,000
- **Test Instances:** 3,000


### Evaluating the "best" models on `train+test.arff`

We used F-measure for the performance metric and Train/Test Percentage Split 75% (order preserved) for the validation technique. 

For the evaluation, we gathered the F-measure value for each model, as well as the number of true positives, true negatives, false positives, and false negatives.


**Model 1** - NaiveBayesMultinomial: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute Selection: CfsSubsetEval.

**Model 2** - ZeroR: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute Selection: SymmetricUncertAttributeEval.

**Model 3** - OneR: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute Selection: OneRAttributeEval.

**Model 4** - SVM: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute selection: InfoGainAttributeEval, Ranker method, numToSelect=15.

**Model 5** - RandomForest: lowerCaseTokens=True, Stemmer as LovinsStemmer, default for remaining StringToWordVector options. Attribute Selection: SymmetricUncertAttributeEval, Ranker method, numToSelect=10.

##

### Results from the evaluation
Comparison flags: v for **significantly better** performance and * for **significantly worse** performance compared to classifier *NaiveBayesMultinomial*.

![Table Results](https://user-images.githubusercontent.com/96803412/148700898-f3438345-9aca-4810-9245-0d79c2710c63.png)

*The F-measure for ZeroR could not be obtained due to a division by zero error*

![Performance Plot](https://user-images.githubusercontent.com/96803412/148701471-7341862e-a078-4018-a19d-3aee4355415e.png)

### Interpretation of the results
- NaiveBayesMultinomial is an algorithm best suited for text data, so it is understandable that it was in the top three contenders (with SVM and RandomForest) for best F-measure. 
- For both the train and test datasets, SVM and RandomForest performed **significantly better** than NaiveBayesMultinomial.
- There were more **false positives** than **true positives** for the four models referenced in the plot, which indicates that the algorithms tended to classify tweets as hate speech when they were actually non-hate speech.
- Our intuition gained from the results is that Random Forests (*an ensemble learning method*) is a powerful technique that **exhibits good predictive performance** on independent test data compared to our remaining models. 
