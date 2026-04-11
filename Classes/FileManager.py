import os


def get_latest_file(folder: str) -> str | None:
    entries = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".csv")
        and os.path.isfile(os.path.join(folder, name))
    ]
    if not entries:
        return None
    return max(entries, key=os.path.getmtime)