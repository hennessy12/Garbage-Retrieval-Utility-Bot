import speech_recognition as sr

# Initialize the recognizer
text = ""

recognizer = sr.Recognizer()

# Use the microphone as the audio source
with sr.Microphone() as source:
    print("Adjusting for ambient noise... Please wait.")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Listening... Speak now!")

    try:
        # Capture the audio
        audio = recognizer.listen(source)
        print("Recognizing speech...")
        
        # Convert speech to text using Google Web Speech API
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

print(text.split(' ')[-1])
