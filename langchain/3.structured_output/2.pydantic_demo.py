from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Subhamoy'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(
        ge=0, le=10, default=5, description='A decimal value representing the cgpa of the student'
    )
    
new_student = {
    'age':'32', 'email':'abc@gmail.com'
}

student = Student(**new_student)

student_dict = dict(student)
print(student_dict['age'])

# In Pydantic v2, model_dump_json() serializes a model instance into a JSON string.
student_json = student.model_dump_json()