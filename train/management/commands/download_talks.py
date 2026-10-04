from django.core.management.base import BaseCommand, CommandError
from train.views import _scrape_talks, _download_docs
from train.constants import ScrapingConstants

class Command(BaseCommand):
    help = 'Downloads talks from Dhamma Talks website'

    def add_arguments(self, parser):
        # Positional arguments
        #parser.add_argument("poll_ids", nargs="+", type=int)

        # Named (optional) arguments
        parser.add_argument(
            "--destination",
            help="Specify a destination path for the downloaded documents. Default is /tmp/ajaanai",
            default="/tmp/ajaanai/"
        )

    def handle(self, *args, **options):
        dest = options['destination']

        print(f"scraping from {ScrapingConstants.ARCHIVE_URL}")
        docs = _scrape_talks()
        
        print(f"found {len(docs)} talk documents, downloading to {dest} now...")
        _download_docs(docs, dest)
        
