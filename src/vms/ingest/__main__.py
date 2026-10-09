"""`python -m vms.ingest` (make ingest): run the ingest service with uvicorn."""

import sys

from vms.ingest.main import main

sys.exit(main())
