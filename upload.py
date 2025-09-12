"""Upload ClearML logs."""

import argparse
import json
import logging
import zipfile
from pathlib import Path

from clearml import Task

import ssl_models

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """Create a new argument parser."""
    parser = argparse.ArgumentParser("Upload offline logs")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to log dir")
    parser.add_argument("--upload_duplicates", type=bool, default=False)
    return parser


def main() -> None:
    """Entry point."""
    parser = get_parser()
    args = parser.parse_args()
    for path_zip in Path(args.log_dir).glob("*.zip"):
        try:
            json_data = zipfile.ZipFile(path_zip, "r").read("task.json")
            cfg = json.loads(json_data)
            task_name = cfg["name"]
            task_names = [
                task.name
                for task in Task.get_tasks(
                    project_name=cfg["project_name"],
                    allow_archived=False,
                )
            ]
        except Exception as e:
            msg = f"Encountered {e}"
            logger.warning(msg)
            continue
        if args.upload_duplicates or task_name not in task_names:
            Task.import_offline_session(session_folder_zip=path_zip)
        else:
            msg = f"Do not upload {task_name}"
            logger.warning(msg)


if __name__ == "__main__":
    main()
