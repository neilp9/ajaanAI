from django.core.management.base import BaseCommand, CommandError
from train.views import _merge_pdfs

class Command(BaseCommand):
    help = 'Merge PDF and HTML documents in specified folder and output to specified desitnation'

    def add_arguments(self, parser):
        # Positional arguments
        parser.add_argument("source")

        parser.add_argument(
            "--max_size",
            help="Specify max merged file size, in MB",
            default=200,
            type=int
        )

        # Named (optional) arguments
        parser.add_argument(
            "--destination",
            help="Specify a destination path for the merged documents. Default is /tmp/ajaanai/merged",
            default="/tmp/ajaanai/merged"
        )

        parser.add_argument(
            "--merge_to",
            help="Specify an existing file to append source files to",
        )

    def handle(self, *args, **options):
        source = options['source']
        max_size = options['max_size']
        dest = options['destination']
        merge_to = options['merge_to']

        print(f"merging files in {source} to {dest} with files up to {max_size}MB, merge_to specified: {merge_to}")
        return _merge_pdfs(source, max_size, dest, merge_to=merge_to)
        
