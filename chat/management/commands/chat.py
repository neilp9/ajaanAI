from django.core.management.base import BaseCommand, CommandError
from chat.utils import OpenAIUtils as chat_utils
from train.utils import OpenAIUtils as train_utils
from train.constants import AJAAN_GEOFF

class Command(BaseCommand):
    help = 'Initiates a session with ajaanAI assistant on openai'

    class COLORS: # You may need to change color settings
        RED = '\033[31m'
        ENDC = '\033[m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        BLUE = '\033[34m'

    def add_arguments(self, parser):
        # Positional arguments
        #parser.add_argument("poll_ids", nargs="+", type=int)

        # Named (optional) arguments
        parser.add_argument(
            "--assistant_name",
            help="Specify Ajaan to interact with, by name. Default is Ajaan Geoff (Thanissaro Bikkhu)",
            default=AJAAN_GEOFF.NAME
        )

        parser.add_argument(
            "--assistant_id",
            help="Specify Ajaan to interact with, by id.",
        )


    def handle(self, *args, **options):
        assistant_id = options['assistant_id']
        assistant_name = options['assistant_name']

        assistant = train_utils.get_assistant(id=assistant_id, name=assistant_name)
        thread = chat_utils.create_thread()

        # loop to output response and read input from the terminal continuously
        while True:
            #self.stdout.write(f"DEBUG: thread: {thread}")
            # Read the user's input and add it to the conversation
            user_input : str = input("> ")
            chat_utils.add_user_response(thread, user_input)

            # Output assistant's response
            response:str = chat_utils.get_assistant_response(assistant, thread)
            self.stdout.write(self.COLORS.RED + response + self.COLORS.ENDC)

            
