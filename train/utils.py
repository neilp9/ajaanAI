import openai
import re
from django.conf import settings
from .constants import OpenAIConstants
from openai.types.beta import Assistant
from openai.types import FileObject

class OpenAIUtils:
    _client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    @classmethod
    def create_assistant(cls, name:str, instructions:str, model:str=OpenAIConstants.DEFAULT_MODEL) -> Assistant:
        assistant = cls._client.beta.assistants.create(
            instructions=instructions,
            name=name,
            tools=[{"type": "retrieval"}],
            model=model,
        )
        print(f"DEBUG: created assistant: {assistant}")
        return assistant
    
    @classmethod
    def get_assistant(cls, id:str=None, name:str=None) -> Assistant:
        result = None
        if id:
            try:
                result = cls._client.beta.assistants.retrieve(id)
            except Exception as e:
                print (f"Error {e}")
                pass
        if name:            
            assistants = cls._client.beta.assistants.list()
            for a in assistants.data:
                if name.lower() in a.name.lower():
                    result = a
        
        # print(f"DEBUG: found assistant: {result}")
        return result
    
    @classmethod
    def update_assistant(cls, assistant:Assistant, *args, **options) -> Assistant:
        cls._client.beta.assistants.update(assistant.id, **options)
        return assistant
    
    @classmethod
    def create_file(cls, filepath) -> FileObject:
        f = cls._client.files.create(
            file=open(filepath, "rb"),
            purpose="assistants"
            )
        print(f"DEBUG: created file: {f}")
        return f