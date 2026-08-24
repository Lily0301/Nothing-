
#My First AI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Greetings

trainingSentence = ["hi", "hello", "hey", "good morning", "good evening", "how are you", "what's up", "i am fine", "i am good", "i am ok", "i am not ok", "feeling bad", "i am not good", "i am not fine"]
    
trainingLevel = ["greeting", "greeting", "greeting", "greeting", "greeting", "wellbeing", "wellbeing", "userwellbeing_positive", "userwellbeing_positive", "userwellbeing_positive", "userwellbeing_negative", "userwellbeing_negative", "userwellbeing_negative", "userwellbeing_negative"]

# Convert text to Numbers (Vectorization)
vectorizer = CountVectorizer(lowercase=True)
X_train = vectorizer.fit_transform(trainingSentence)

# Train up Machine Learning Model
model = MultinomialNB()
model.fit(X_train, trainingLevel)

# Chatbot Loop
print("AI is online !! (Type 'exit' to stop)")

import random

responses = {
    "greeting": [
        "AI: Hlw !! How can i help you today?? (^_^).",
        "AI: Hi there!",
        "AI: Hey! :)"
    ],
    
    "wellbeing": [
        "AI: I'm good !! Tnx for asking ! wbu?? (≧◡≦) ..",
        "AI: I'm doing well!",
        "AI: I'm great, thanks!",
        "AI: Pretty good! 😊"
    ], 
      
    "userwellbeing_positive": [
        "AI: Glad to hear that ! How can I help you? :)",
        "AI: Nice !! ",
        "AI: That's great to hear! 😊",
        "AI: Glad you're doing well!",
        "AI: That's awesome!"
    ],
    
    "userwellbeing_negative": [
        "AI: I'm sorry to hear that.",
        "AI: I hope things get better ˘･_･˘.",
        "AI: Take care of yourself."
    ]
}

while True:
    userInput = input("You: ").strip().lower()
    
    if userInput == "exit":
        print("AI : Goodbye !! ^_~")
        break
    else:
        userVector = vectorizer.transform([userInput])
        prediction = model.predict(userVector)[0]
              
        if prediction in responses:
            response = random.choice(responses[prediction])
            print(response)
        else:
            print("AI: I don't understand! (⊙_⊙)")