# Language-aware recursive splitting is useful for structured documents.
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Markdown headings, paragraphs, lists, and code blocks are kept together
# when possible.
text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0
)

# The result is a list of Markdown text chunks.
chunks = splitter.split_text(text)

# Print the number of chunks and the first chunk.
print(len(chunks))
print(chunks[0])