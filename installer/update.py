"""Tells the user when a newer release exists. Never blocks or fails an install: any problem returns None.

Off with --no-update-check (CLI) or DKB_NO_UPDATE_CHECK=1 (CLI and window).
"""
import json
import os
import re
import urllib.request

API = 'https://api.github.com/repos/Lucas-blip-png/dkb-pc-recomp/releases/latest'
TIMEOUT = 3


def disabled(flag=False):
    return flag or os.environ.get('DKB_NO_UPDATE_CHECK') == '1'


def fetch_latest():
    """The latest release JSON from GitHub (unauthenticated), or None."""
    try:
        req = urllib.request.Request(API, headers={'User-Agent': 'dkb-installer', 'Accept': 'application/vnd.github+json'})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)
    except Exception:
        return None


def numbers(version):
    """(0, 2, 0) from '0.2.0+33f8bbc', 'v0.2.0' or '0.2.0-dev+abc'; None for anything else (a pre-release tag too)."""
    m = re.fullmatch(r'v?(\d+(?:\.\d+)*)(?:-dev)?(?:\+.*)?', str(version).strip())
    return tuple(int(n) for n in m.group(1).split('.')) if m else None


def newer_release(current, fetch=fetch_latest):
    """(tag, url) of the latest release when it is newer than `current`, else None. `fetch` returns the API JSON."""
    try:
        info = fetch()
        tag, url = info['tag_name'], info['html_url']
        if info.get('prerelease') or info.get('draft') or '-' in tag:   # v0.3.0-rc1 is not a release to offer
            return None
        latest, mine = numbers(tag), numbers(current)
        return (tag, url) if latest and mine and latest > mine else None
    except Exception:
        return None
