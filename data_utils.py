def download_url(url):
    import requests
    import hashlib
    from pathlib import Path

    # create a temp file based on the hash of the url. if the file already exists, return the path
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = url.split(".")[-1].split("?")[0]
    cache_path = Path(f"/tmp/{url_hash}.{ext}")
    if not cache_path.exists():
        # download the file
        response = requests.get(url)
        with open(cache_path, "wb") as f:
            f.write(response.content)
    return cache_path
