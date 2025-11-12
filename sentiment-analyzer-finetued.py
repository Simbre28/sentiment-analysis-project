from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Path to your extracted fine-tuned model
MODEL_PATH = MODEL_PATH = "C:/Users/User/Documents/sentiment-analysis/fine_tuned_sentiment_model/fine_tuned_sentiment_model"  # Update this path!

# Load the model
print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Create pipeline with your fine-tuned model
classifier = pipeline("sentiment-analysis", 
                     model=model,
                     tokenizer=tokenizer)
print("✅ Fine-tuned model loaded successfully!\n")

# Interactive testing
def analyse_custom_text():
    print("="*50)
    print("Sentiment Analysis - Interactive Mode")
    print("(Using Fine-Tuned Model)")
    print("="*50)
    print("Type your movie reviews below.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("Enter review: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nThanks for using the sentiment analyzer!")
            break
        
        if user_input:
            result = classifier(user_input)[0]
            # Map LABEL_0/LABEL_1 to readable names
            sentiment = "POSITIVE" if result['label'] == 'LABEL_1' else "NEGATIVE"
            print(f"→ Sentiment: {sentiment}")
            print(f"→ Confidence: {result['score']:.2%}\n")
        else:
            print("Please enter some text.\n")  
            
# Run interactive mode
analyse_custom_text()