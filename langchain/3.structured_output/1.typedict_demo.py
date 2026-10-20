# TypedDict documents the expected keys and value types for a dictionary.
from typing import TypedDict

# Define a lightweight schema without runtime validation.
class Person(TypedDict):
    
    name: str
    age: int
    
# Values can be checked by type checkers against the declared schema.
new_person: Person = {
    'name': 'Subhamy',
    'age': 35
}

print(new_person)