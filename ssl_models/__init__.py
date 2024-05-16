"""Init package."""

from logging import getLogger
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
LOGGER = getLogger(__name__)

env_file = ROOT_DIR / ".env"
if env_file.is_file():
    load_dotenv(env_file)
