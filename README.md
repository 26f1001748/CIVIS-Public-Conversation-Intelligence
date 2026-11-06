# CIVIS — Public Conversation Intelligence

An exploratory sentiment analysis and text analytics project using the TweetEval dataset, combining machine learning, AI-assisted analysis and human interpretation.

### Methodology
- Conducted data auditing, cleaning, duplicate inspection and text preprocessing.
- Applied TF-IDF feature extraction and trained traditional machine learning classifiers.
- Evaluated Logistic Regression, LinearSVC and a pretrained RoBERTa model.
- Performed error analysis and exploratory KMeans clustering.
- Examined ambiguous, sarcastic and context-dependent text through human inference rather than treating model predictions as definitive interpretations.
- Developed an interactive Tableau Public dashboard to explore sentiment distributions, classification performance, confusion patterns and tweet clusters.

### Results
- **LinearSVC:** 68.9% validation accuracy; 0.663 macro-F1.
- **RoBERTa benchmark:** 77.25% validation accuracy; 0.762 macro-F1.

### Technology
Python · Pandas · Scikit-learn · TF-IDF · PyTorch · Hugging Face Transformers · Tableau · SQL

### Strategic Use of AI
AI tools were used strategically to support debugging, understand unfamiliar concepts, explore implementation approaches and improve development efficiency. Established statistical and machine learning concepts guided methodological decisions, while human judgement remained important in evaluating ambiguous language and interpreting analytical findings.

### Tableau Public Dashboard

- [Sentiment Distribution](YOUR_LINK/Sheet1)
- [Confusion Matrix](YOUR_LINK/Sheet2)
- [Classification Performance](YOUR_LINK/Sheet3)
- [Prediction Accuracy](YOUR_LINK/Sheet4)
- [Tweet Cluster Explorer](YOUR_LINK/Sheet5)
  
### Limitations
TweetEval contains no timestamps, so temporal trends are not analysed. Clustering is exploratory, and sarcasm and context-dependent sentiment remain challenging.
