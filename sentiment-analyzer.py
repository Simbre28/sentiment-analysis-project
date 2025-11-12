from transformers import pipeline

# Load the model
print("Loading model...")
classifier = pipeline("sentiment-analysis", 
                     model="distilbert-base-uncased-finetuned-sst-2-english")
print("Model loaded successfully!\n")

#Interactive testing
def analyse_custom_text():
    print("="*50)
    print("Sentiment Analysis - Interactive Mode")
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
            print(f"→ Sentiment: {result['label']}")
            print(f"→ Confidence: {result['score']:.2%}\n")
        else:
            print("Please enter some text.\n")  
            
# Run interactive mode
analyse_custom_text() 









