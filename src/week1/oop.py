# oop.py
# Object-Oriented Programming practice for Week 1

class Chatbot:
    def __init__(self, name):
        # Encapsulation (protected attribute)
        self._name = name
        
    def greet(self):
        return f"Hi, I'm {self._name}!"

class TutorBot(Chatbot):  # Inheritance
    def teach(self, topic):
        return f"{self._name} is teaching {topic}."

if __name__ == "__main__":
    bot = TutorBot("TutorGrok")
    print(bot.greet())  # Output: Hi, I'm TutorGrok!
    print(bot.teach("Python"))  # Output: TutorGrok is teaching Python.
