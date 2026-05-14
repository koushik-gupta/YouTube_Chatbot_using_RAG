from urllib.parse import parse_qs, urlparse


def get_video_id(video_url):
    if "youtube.com" not in video_url and "youtu.be" not in video_url:
        return video_url

    parsed_url = urlparse(video_url)

    if parsed_url.hostname in ("youtu.be", "www.youtu.be"):
        return parsed_url.path.lstrip("/")

    query_params = parse_qs(parsed_url.query)
    return query_params.get("v", [""])[0]
