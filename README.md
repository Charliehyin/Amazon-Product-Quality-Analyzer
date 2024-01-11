# Amazon Product Quality Analyzer

## Project Overview
The Amazon Product Quality Analyzer is an AI-powered tool designed to assess product quality using Amazon reviews. It leverages advanced machine learning algorithms and natural language processing (NLP) methods to analyze textual data and predict product awesomeness.

## Key Features
- **Classifier Development**: Utilizes Naive Bayes, Logistic Regression, and Random Forest algorithms for predictive analysis
- **NLP and Sentiment Analysis**: Employs TF-IDF for text extraction and analysis, and utilizes the Hugging Face platform for sentiment assessment
- **Performance Metrics**: Achieves an F1-score of 0.74, indicating a robust model performance.

## Development Process
1. **Data Collection**: The dataset comprises CDs and Vinyl reviews from Amazon.
2. **Preprocessing and Analysis**:
   - Implemented TF-IDF vectorization for NLP on review text.
   - Extracted the following features from the review text:
     - TF-IDF vectorization.
     - Conducted sentiment analysis using nltk Vader and converted scores into a histogram.
   - Extracted the following features from the non-reviewtext datapoints:
     - % of reviews with no votes, highest # of votes, average # of votes, stdev of all votes.
     - Earliest review time, latest review time, stdev of all review times.
     - % of verified reviews
   - Standardized all data points
   - Performed feature selection by analyzing Fischer scores and running recursive feature selection algorithms.
     <div align="center">
        <img src="https://i.imgur.com/ykHVyYy.png" alt="Recursive Feature Selection">
        <br>
        <p>Figure 1: Example Recursive Feature Selection Flowchart</p>
    </div>
4. **Model Training and Selection**:
   - Employed Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, Random Forest, Decision Tree, and Adaboost on the sentiment analysis histogram and non-reviewtext datapoints.
   - Used Multinomial Naive Bayes, Bernoulli Naive Bayes, and Logarithmic Regression on TF-IDF data.
   - Hyperparameter tuned each model, and performed individual feature selection for each model.
     
     <div align="center">
        <img src="https://i.imgur.com/cQmNj6z.png" alt="Hyperparameter Tuning">
        <br>
        <p>Figure 2: Example Hyperparameter Training for a Decision Tree Classifier</p>
    </div>
    
    <div align="center">
        <br>
        <img src="https://i.imgur.com/lqNItax.png" alt="Feature Selection">
        <br>
        <p>Figure 3: Graph of F1 Score by # of Features Used for Logistic Regression</p>
    </div>
   - Created a custom voting classifier to compile prediction probabilities from all models.
5. **Ensemble Learning**:
   - Integrated ensemble learning techniques for improved accuracy.
   - Experimented with neural network models for sentiment analysis, but failed to produce better results.

## Results and Learnings
  - Achieved a F1-score of .66 using a Keras deep learning model on the sentiment analysis data; was unable to outperform traditional machine learning techniques
   <div align="center">
      <img src="https://i.imgur.com/Rk5AKoF.png" alt="Deep Learning Confusion Matrix">
      <br>
      <p>Figure 4: Confusion Matrix for Deep Learning Results</p>
  </div>
  - Achieved a final F1-score of 0.74, demonstrating the effectiveness of the ensemble approach and custom algorithm development.
   <div align="center">
      <br>
      <img src="https://i.imgur.com/cxveWPa.png" alt="Final Results">
      <br>
      <p>Figure 5: Confusion Matrix of Final Results</p>
  </div>
  
## Conclusion
Though there remains much room for improvement, this project showcases the potential of machine learning in enhancing product quality assessment using customer reviews. 
