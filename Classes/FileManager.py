"""Helpers for locating telemetry CSV files."""
import glob
import os
import re


def extract_num(path: str) -> int:
    """Extract the numeric suffix from a file name like d_12.csv.

    Args:
        path (str): Full or relative path to the file.

    Returns:
        int: The extracted number, or -1 if no match is found.
    """
    name = os.path.basename(path)
    match = re.search(r"d_(\d+)", name, re.IGNORECASE)
    return int(match.group(1)) if match else -1

def get_latest_file(folder: str) -> str | None:
    """Return the CSV file with the highest numeric suffix in a folder.

    Args:
        folder (str): Path to the directory to search.

    Returns:
        str | None: Path to the most recent CSV file, or None if no
            valid files are found.
    """
    files = glob.glob(os.path.join(folder, "*.csv"))
    valid_files = [f for f in files if extract_num(f) != -1]
    if not valid_files:
        return None
    return max(valid_files, key=extract_num)