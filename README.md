<<<<<<< HEAD
# sentiment-analysis-project
This is a project to get a pretrained AI model from Hugging Face working for modern movie review sentiment analysis 
=======
Movie Review Sentiment Analysis with DistilBERT

--------------------------------------------------


An AI-powered sentiment analysis tool that classifies movie reviews as positive or negative. This project uses DistilBERT as a base model, fine-tuned on custom data to understand modern slang and contemporary language patterns.


Features

----------



Pre-trained DistilBERT Model: Leverages the power of transformer-based language models


Fine-tuned for Modern Language: Custom training on movie reviews including modern slang and colloquialisms
Confidence Scoring: Provides probability scores for prediction confidence
Dual Interface: Available as both CLI and web-based Streamlit application
Comprehensive Evaluation: Includes accuracy metrics, classification reports, and confusion matrix visualizations



Project Structure
--------------------



sentiment-analysis-project/
├── data/                      # Training and test datasets
├── models/                    # Saved model files
├── notebooks/                 # Jupyter notebooks for exploration
├── sentiment\_cli.py           # Command-line interface version
├── sentiment\_app.py           # Streamlit web application
├── train\_model.py             # Model training and fine-tuning script
├── evaluate\_model.py          # Model evaluation script
├── requirements.txt           # Project dependencies
└── README.md                  # This file


Installation

--------------



1. Clone the repository

---------

git clone https://github.com/your-username/sentiment-analysis-project.git
cd sentiment-analysis-project



2\. Create a virtual environment

----------

python -m venv venv



Windows:

venv\\Scripts\\activate



Mac/ Linux:
source venv/bin/activate



3\. Install dependencies



pip install -r requirements.txt


Usage
----------



Streamlit Web Application (Recommended)


1. Run the interactive web interface:

streamlit run sentiment\_app.py


This will open a browser window where you can:



-Enter your movie review in the text field
-Click "Analyze Sentiment"
-View the sentiment prediction (Positive/Negative)
-See the confidence score for the prediction



Command-Line Interface
------------------------



For quick predictions from the terminal:
python sentiment\_cli.py "This movie was absolutely amazing! Best film I've seen all year."




Model Development
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

Base Model

--------------

Started with pre-trained DistilBERT (distilbert-base-uncased)
Evaluated baseline performance on IMDB movie review dataset from Hugging Face



Initial Evaluation (Imbalanced Dataset)
Dataset: First 1000 reviews from IMDB dataset (100% negative reviews)

Accuracy: 90.8%
Classification Report:

Negative: Precision 1.00, Recall 0.91, F1-Score 0.95
Positive: Precision 0.00, Recall 0.00, F1-Score 0.00



Note: High accuracy was misleading due to class imbalance - model predicted everything as negative.


Balanced Dataset Evaluation
Dataset: 25,000 reviews (50% positive, 50% negative) from IMDB Hugging Face dataset

Accuracy: 89.07% (88.1% in Google Colab environment)
Classification Report:

Negative: Precision 0.87, Recall 0.92, F1-Score 0.89
Positive: Precision 0.91, Recall 0.86, F1-Score 0.89



Confusion Matrix Results:

--------------------------

True Negatives: 461
False Positives: 51
False Negatives: 68
True Positives: 420



Fine-Tuning Process

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

-Custom dataset including modern slang and contemporary language
-Fine-tuned on movie reviews to improve accuracy with informal language
-Optimized for understanding context-specific sentiment
-Training completed over 3 epochs with decreasing validation loss



Fine-Tuned Model Performance
--------------------------------

After Fine-Tuning:

Final Accuracy: Improved performance with better balance
Training Progress:

Epoch 1: Validation Loss 0.518, Accuracy 0.909
Epoch 2: Validation Loss 0.476, Accuracy 0.909
Epoch 3: Validation Loss 0.457, Accuracy 0.909



Evaluation Metrics

---------------------
-Both the base model and fine-tuned version were evaluated using:

-Accuracy Score: Overall prediction accuracy
-Classification Report: Precision, recall, and F1-score for each class
-Confusion Matrix Heatmap: Visual representation of prediction performance across true/predicted labels



Model Performance Summary
--------------------------

Metric

Base DistilBERT

Fine-Tuned Model

Accuracy 89.07%90.9%

Precision (Negative)0.87



Improved

Precision (Positive)0.91

Improved Recall (Negative)0.92

Improved Recall (Positive)0.86

ImprovedF1-Score0.890.909


Dependencies

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
Key libraries used:



transformers - Hugging Face transformers for DistilBERT
torch - PyTorch for model training
scikit-learn - Evaluation metrics and preprocessing
streamlit - Web application framework
pandas - Data manipulation
numpy - Numerical operations
matplotlib / seaborn - Visualization (confusion matrix)

See requirements.txt for complete list with versions.


Examples

----------------
Positive Review:
Input: "This movie absolutely slaps! The plot was fire and the acting was chef's kiss."
Output: Positive (Confidence: 94.2%)


Negative Review:
Input: "Total waste of time. The story was mid at best and the ending was trash."
Output: Negative (Confidence: 89.7%)


Known Limitations

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*


Token Truncation Issue


During development, we identified a significant limitation with the BERT-based architecture:



-512 Token Limit: DistilBERT truncates input text to a maximum of 512 tokens
-Impact on Accuracy: Longer movie reviews lose important sentiment information when truncated, reducing prediction accuracy
-Lost Context: Reviews often build sentiment throughout the entire text, and truncation can miss crucial concluding statements or context

-Example: A review that starts negative but concludes positively (or vice versa) may be misclassified if the conclusion is truncated.


Potential Solutions

------------------------
-We're exploring several approaches to address this limitation:

-Sliding Window with Aggregation

-Process the review in overlapping chunks of 512 tokens
-Aggregate predictions from each chunk (voting or averaging confidence scores)
-Provides more comprehensive sentiment analysis for long reviews



Alternative Model Architectures

----------------------------------

Longformer: Supports up to 4,096 tokens
BigBird: Efficient attention for longer sequences
Hierarchical models: Process paragraph-level sentiments then aggregate



Smart Truncation Strategies

-----------------------------

Keep first and last N tokens (capture introduction and conclusion)
Attention-based selection of most relevant sentences
Summary-based preprocessing before classification



Future Improvements

-----------------------

Implement sliding window approach for long reviews
Add neutral sentiment classification
Support for multi-language reviews
Aspect-based sentiment analysis (acting, plot, cinematography)
API endpoint for integration with other applications
Mobile-responsive design improvements
Evaluate alternative models (Longformer, BigBird) for longer text support



License
\[Specify your license here - e.g., MIT, Apache 2.0]
Contact
Your Name - your-email@example.com
Project Link: https://github.com/your-username/sentiment-analysis-project


Acknowledgments

Hugging Face for the DistilBERT model
\[Any datasets used]
\[Any other resources or inspirations]

>>>>>>> 519dc38 (Initial commit o our AI Sentiment Analysis program)
