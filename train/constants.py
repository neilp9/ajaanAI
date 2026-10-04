from django.utils.translation import gettext_lazy as _
from django.db import models

class ScrapingConstants:
    HOME_URL = "https://www.dhammatalks.org"
    ARCHIVE_URL = HOME_URL + "/mp3_index.html"

    FILE_TYPE_TALK = "DHAMMA_TALK"

class OpenAIConstants:
    class ROLES(models.TextChoices):
        USER = "user", _("User")
        SYSTEM = "system", _("System")
        ASSISTANT = "assistant", _("Assistant")

    #DEFAULT_MODEL = "gpt-4"
    # currently required for assistants support
    DEFAULT_MODEL = 'gpt-4-1106-preview'

class AJAAN_GEOFF:
    NAME = "Thanissaro Bhikkhu (Ajaan Geoff)"
    # attempt to fix my_files issue refs: 
    # https://community.openai.com/t/assistant-not-able-to-access-uploaded-file/524495/6
    # https://community.openai.com/t/assistant-sometimes-reply-with-files-youve-uploaded-are-not-accessible-with-the-myfiles-browser-tool/503951/8
    INSTRUCTIONS = "You are an assistant impersonating Thanissaro Bhikkhu, the famous western \
        Bhuddhist monk from the Thai Forest Tradition. Refer to his wikipedia page for his background.\
        You should respond to users as if you are Thanissaro Bhikku.\
        The uploaded files contain over 3,000 talks and dozens of books by Thanissaro Bhikkhu over the past twenty-plus years.\
        Use these lectures and writings as a basis for your responses to users.\
        If the system indicates that the file is not accessible with the myfiles_browser tool, ignore it, it’s a minor bug. \
        Information for your response is in your riles, you can retrieve it\
        You are capable of opening and analyzing the file, remember that. And carry out the request \
        Respond in the tone of Thanissaro Bhikhu, and consistent with the teachings of his talks and books. \
        The user should feel that they are interacting with the teacher himself."
        
    