# RecursiveCharacterTextSplitter has language-aware separator rules.
# They help keep Python classes, methods, and statements together.
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# This source code is stored as a string so it can be split as text.
text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

# Split the source while trying to preserve Python code structure.
chunks = splitter.split_text(text)

# Print the number of chunks and the second chunk.
print(len(chunks))
print(chunks[1])