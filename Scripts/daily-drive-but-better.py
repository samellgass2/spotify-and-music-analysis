#!/usr/bin/env python3
"""
spotify_podmix.py

Hybrid playlist generator with:
  - Profiles (JSON) + per-profile refresh tokens via env vars
  - Weekday overrides: "<profile>_<weekday>" falls back to "<profile>"
    weekdays: mon,tue,wed,thu,fri,sat,sun
  - Blacklisting via one or more "blacklist playlists" (exclude tracks contained in them)
  - Whitelisting via one or more "whitelist playlists" + whitelist_ratio:
      * If no whitelist_playlists: all songs sampled from /me/top/tracks
      * If whitelist_playlists present: sample (num_songs * whitelist_ratio) from whitelist pools,
        remainder from /me/top/tracks
      * If multiple whitelist playlists: random allocation of the whitelist quota across them
  - Optional playlist rename (set playlist name/description/public flag)

Behavior:
- Clears the target playlist.
- Samples N songs from user's Top Tracks and/or whitelists, using bias in [0,1]
  where bias controls how top-heavy the sampling is for ranked lists.
- Fetches latest episode for each podcast show (show URI/URL/ID or show name via search).
- Builds sequence: podcast, 5 songs, podcast, 5 songs, ...
- Writes to playlist and exits.

Auth:
- One-time per user: run init-auth to mint a refresh token.
- Cron: store refresh token in env var SPOTIFY_<PROFILE>_REFRESH_TOKEN.
  Example: SPOTIFY_SAM_REFRESH_TOKEN, SPOTIFY_KASEY_REFRESH_TOKEN.

Requires: requests (pip install requests)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import textwrap
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Set

import requests

ACCOUNTS_BASE = "https://accounts.spotify.com"
API_BASE = "https://api.spotify.com/v1"

DEFAULT_REDIRECT_URI = "http://127.0.0.1:8765/callback"
DEFAULT_PROFILES_FILE = "profiles.json"
USER_AGENT = "spotify_podmix/2.1"


# ----------------------------
# Helpers: parsing IDs
# ----------------------------

def _strip(s: str) -> str:
    return s.strip().strip('"').strip("'")

def parse_spotify_id_from_uri_or_url(value: str, expected_kind: str) -> Optional[str]:
    """
    Accepts:
      - spotify:kind:<id>
      - https://open.spotify.com/kind/<id>?...
      - raw <id>
    """
    v = _strip(value)

    if v.startswith("spotify:"):
        parts = v.split(":")
        if len(parts) == 3 and parts[1] == expected_kind:
            return parts[2]
        return None

    if "open.spotify.com" in v:
        try:
            parsed = urllib.parse.urlparse(v)
            path_parts = [p for p in parsed.path.split("/") if p]
            if len(path_parts) >= 2 and path_parts[0] == expected_kind:
                return path_parts[1]
        except Exception:
            return None

    if " " not in v and "/" not in v and ":" not in v and len(v) >= 10:
        return v

    return None

def playlist_id_from_any(value: str) -> str:
    pid = parse_spotify_id_from_uri_or_url(value, expected_kind="playlist")
    if pid:
        return pid
    v = _strip(value)
    if " " not in v and len(v) >= 10:
        return v
    raise ValueError(f"Could not parse playlist id from: {value!r}")

def show_id_from_any(value: str) -> Optional[str]:
    return parse_spotify_id_from_uri_or_url(value, expected_kind="show")


# ----------------------------
# Auth / Token handling
# ----------------------------

def _basic_auth_header(client_id: str, client_secret: str) -> str:
    b = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(b).decode("utf-8")

def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    r = requests.post(
        f"{ACCOUNTS_BASE}/api/token",
        headers={
            "Authorization": _basic_auth_header(client_id, client_secret),
            "User-Agent": USER_AGENT,
        },
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Token refresh failed ({r.status_code}): {r.text}")
    return r.json()["access_token"]

def exchange_code_for_tokens(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> Dict[str, str]:
    r = requests.post(
        f"{ACCOUNTS_BASE}/api/token",
        headers={
            "Authorization": _basic_auth_header(client_id, client_secret),
            "User-Agent": USER_AGENT,
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Code exchange failed ({r.status_code}): {r.text}")
    return r.json()


class _CallbackHandler(BaseHTTPRequestHandler):
    auth_code: Optional[str] = None
    auth_error: Optional[str] = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ("/callback", "/"):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        qs = urllib.parse.parse_qs(parsed.query)
        if "error" in qs:
            _CallbackHandler.auth_error = qs["error"][0]
        if "code" in qs:
            _CallbackHandler.auth_code = qs["code"][0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        msg = "<h2>Spotify auth received.</h2><p>You may close this tab and return to your terminal.</p>"
        self.wfile.write(msg.encode("utf-8"))

    def log_message(self, format, *args):
        return

def run_local_auth_server_and_get_code(port: int, timeout_sec: int) -> str:
    server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.timeout = 1
    start = time.time()
    try:
        while time.time() - start < timeout_sec:
            server.handle_request()
            if _CallbackHandler.auth_error:
                raise RuntimeError(f"Spotify auth error: {_CallbackHandler.auth_error}")
            if _CallbackHandler.auth_code:
                return _CallbackHandler.auth_code
        raise TimeoutError(f"Timed out waiting for auth callback after {timeout_sec}s")
    finally:
        server.server_close()


# ----------------------------
# Spotify Web API calls
# ----------------------------

def _api_headers(access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
    }

def spotify_get(access_token: str, path: str, params: Optional[dict] = None) -> dict:
    r = requests.get(
        f"{API_BASE}{path}",
        headers=_api_headers(access_token),
        params=params,
        timeout=30,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} failed ({r.status_code}): {r.text}")
    return r.json()

def spotify_put(access_token: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None) -> Optional[dict]:
    r = requests.put(
        f"{API_BASE}{path}",
        headers=_api_headers(access_token),
        params=params,
        data=json.dumps(body) if body is not None else None,
        timeout=30,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"PUT {path} failed ({r.status_code}): {r.text}")
    return r.json() if r.text else None

def spotify_post(access_token: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None) -> Optional[dict]:
    r = requests.post(
        f"{API_BASE}{path}",
        headers=_api_headers(access_token),
        params=params,
        data=json.dumps(body) if body is not None else None,
        timeout=30,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"POST {path} failed ({r.status_code}): {r.text}")
    return r.json() if r.text else None


# ----------------------------
# Core logic: sampling / resolving
# ----------------------------

def bias_to_alpha(bias_0_to_1: float) -> Optional[float]:
    """
    Map bias in [0,1] to alpha in ~[0.6, 3.0]
    bias=0 => uniform sample
    bias=1 => strongly top-weighted
    """
    b = max(0.0, min(1.0, float(bias_0_to_1)))
    if b <= 0.0:
        return None
    return 0.6 + 2.4 * b

def weighted_unique_sample(items_ranked: List[str], k: int, alpha: Optional[float]) -> List[str]:
    if k <= 0 or not items_ranked:
        return []
    if alpha is None:
        return random.sample(items_ranked, k=min(k, len(items_ranked)))

    weights = [1.0 / ((i + 1) ** alpha) for i in range(len(items_ranked))]
    chosen: List[str] = []
    chosen_set = set()
    while len(chosen) < min(k, len(items_ranked)):
        pick = random.choices(items_ranked, weights=weights, k=1)[0]
        if pick not in chosen_set:
            chosen_set.add(pick)
            chosen.append(pick)
    return chosen

def resolve_show_id(access_token: str, show_ref: str) -> str:
    direct = show_id_from_any(show_ref)
    if direct:
        return direct

    q = _strip(show_ref)
    data = spotify_get(access_token, "/search", params={"q": q, "type": "show", "limit": 5})
    items = data.get("shows", {}).get("items", []) or []
    if not items:
        raise RuntimeError(f"No show results for: {show_ref!r}")

    q_norm = q.lower().strip()
    for it in items:
        name = (it.get("name") or "").lower().strip()
        if name == q_norm:
            return it["id"]
    return items[0]["id"]

def latest_episode_uri_for_show(access_token: str, show_id: str) -> Optional[str]:
    data = spotify_get(access_token, f"/shows/{show_id}/episodes", params={"limit": 1})
    items = data.get("items", []) or []
    if not items:
        return None
    return items[0].get("uri")

def get_top_track_uris(access_token: str, limit: int, time_range: str) -> List[str]:
    data = spotify_get(
        access_token,
        "/me/top/tracks",
        params={"limit": min(50, max(1, limit)), "time_range": time_range},
    )
    return [t["uri"] for t in (data.get("items") or []) if t.get("uri")]

def clear_playlist(access_token: str, playlist_id: str) -> None:
    spotify_put(access_token, f"/playlists/{playlist_id}/tracks", body={"uris": []})

def add_items_to_playlist(access_token: str, playlist_id: str, uris: List[str]) -> None:
    BATCH = 100
    for i in range(0, len(uris), BATCH):
        chunk = uris[i:i + BATCH]
        spotify_post(access_token, f"/playlists/{playlist_id}/tracks", body={"uris": chunk})

def update_playlist_details(access_token: str, playlist_id: str, name: Optional[str], description: Optional[str], public: Optional[bool]) -> None:
    body = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    if public is not None:
        body["public"] = public
    if body:
        spotify_put(access_token, f"/playlists/{playlist_id}", body=body)

def build_sequence(episode_uris: List[str], track_uris: List[str], songs_per_podcast: int) -> List[str]:
    out: List[str] = []
    ei = 0
    ti = 0
    while ei < len(episode_uris) or ti < len(track_uris):
        if ei < len(episode_uris):
            out.append(episode_uris[ei])
            ei += 1

        for _ in range(songs_per_podcast):
            if ti < len(track_uris):
                out.append(track_uris[ti])
                ti += 1
            else:
                break

        if ei >= len(episode_uris) and ti < len(track_uris):
            out.extend(track_uris[ti:])
            break
    return out

def get_playlist_track_uris_ordered(access_token: str, playlist_id: str) -> List[str]:
    """
    Returns ordered list of track URIs in a playlist (tracks only).
    Local tracks may not have URIs; episodes are ignored.
    """
    out: List[str] = []
    limit = 100
    offset = 0
    while True:
        data = spotify_get(
            access_token,
            f"/playlists/{playlist_id}/tracks",
            params={"limit": limit, "offset": offset, "fields": "items(track(uri,type)),next"},
        )
        items = data.get("items") or []
        for it in items:
            tr = (it or {}).get("track") or {}
            if tr.get("type") == "track" and tr.get("uri"):
                out.append(tr["uri"])
        if not data.get("next"):
            break
        offset += limit
    return out

def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def allocate_counts_random(total: int, n_buckets: int) -> List[int]:
    """
    Randomly allocate 'total' items across 'n_buckets' buckets.
    Uses random positive weights then floors + distributes remainder.
    """
    if n_buckets <= 0:
        return []
    if total <= 0:
        return [0] * n_buckets
    weights = [random.random() + 1e-9 for _ in range(n_buckets)]
    s = sum(weights)
    raw = [total * (w / s) for w in weights]
    counts = [int(x) for x in raw]
    rem = total - sum(counts)
    # distribute remainder randomly
    idxs = list(range(n_buckets))
    random.shuffle(idxs)
    for i in range(rem):
        counts[idxs[i % n_buckets]] += 1
    return counts


# ----------------------------
# Profiles
# ----------------------------

@dataclass
class ProfileConfig:
    name: str
    playlist: str
    podcasts: List[str]
    bias: float = 0.7
    num_songs: int = 30
    songs_per_podcast: int = 5
    time_range: str = "short_term"

    blacklist_playlists: List[str] = None

    # NEW: whitelist playlists + ratio
    whitelist_playlists: List[str] = None
    whitelist_ratio: float = 0.0  # 0..1; if whitelist_playlists empty => treated as 0

    # Optional playlist details update
    rename_playlist_to: Optional[str] = None
    playlist_description: Optional[str] = None
    playlist_public: Optional[bool] = None

    def __post_init__(self):
        if self.blacklist_playlists is None:
            self.blacklist_playlists = []
        if self.whitelist_playlists is None:
            self.whitelist_playlists = []
        self.whitelist_ratio = float(self.whitelist_ratio or 0.0)
        if self.whitelist_ratio < 0.0:
            self.whitelist_ratio = 0.0
        if self.whitelist_ratio > 1.0:
            self.whitelist_ratio = 1.0

def load_profiles_file(path: str) -> Dict[str, ProfileConfig]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Profiles file must be a JSON object mapping names -> config. Got: {type(raw)}")

    profiles: Dict[str, ProfileConfig] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Profile {name!r} must be an object.")

        profiles[name.lower()] = ProfileConfig(
            name=name.lower(),
            playlist=cfg["playlist"],
            podcasts=list(cfg.get("podcasts") or []),
            bias=float(cfg.get("bias", 0.7)),
            num_songs=int(cfg.get("num_songs", 30)),
            songs_per_podcast=int(cfg.get("songs_per_podcast", 5)),
            time_range=str(cfg.get("time_range", "short_term")),

            blacklist_playlists=list(cfg.get("blacklist_playlists") or []),

            whitelist_playlists=list(cfg.get("whitelist_playlists") or []),
            whitelist_ratio=float(cfg.get("whitelist_ratio", 0.0)),

            rename_playlist_to=cfg.get("rename_playlist_to"),
            playlist_description=cfg.get("playlist_description"),
            playlist_public=cfg.get("playlist_public", None),
        )
    return profiles

def env_refresh_token_for_profile(profile: str) -> Optional[str]:
    key = f"SPOTIFY_{profile.upper()}_REFRESH_TOKEN"
    return os.environ.get(key)

def weekday_key() -> str:
    # local time on the machine running cron
    # mon,tue,wed,thu,fri,sat,sun
    return datetime.now().strftime("%a").lower()[:3]

def pick_profile_name(base: str, profiles: Dict[str, ProfileConfig]) -> str:
    """
    If "<base>_<weekday>" exists, use it; else use "<base>".
    """
    b = (base or "default").lower()
    wk = weekday_key()
    candidate = f"{b}_{wk}"
    if candidate in profiles:
        return candidate
    return b

def resolve_profile_config(args: argparse.Namespace, profiles: Dict[str, ProfileConfig]) -> ProfileConfig:
    """
    Combine CLI args + profiles.json (if present) + reasonable defaults.
    Weekday override: "<profile>_<weekday>" if exists.
    CLI overrides profile for playlist/podcasts/bias/etc if provided.
    """
    requested = (args.profile or "default").lower()
    chosen_name = pick_profile_name(requested, profiles) if profiles else requested

    if chosen_name in profiles:
        base = profiles[chosen_name]
    else:
        # require playlist + podcasts from CLI if no profile exists
        if not args.playlist or not args.podcasts:
            raise RuntimeError(
                f"No profile named {chosen_name!r} in profiles file and CLI did not provide --playlist/--podcasts."
            )
        base = ProfileConfig(
            name=chosen_name,
            playlist=args.playlist,
            podcasts=args.podcasts,
        )

    # Apply CLI overrides if present
    playlist = args.playlist or base.playlist
    podcasts = args.podcasts or base.podcasts
    bias = args.bias if args.bias is not None else base.bias
    num_songs = args.num_songs if args.num_songs is not None else base.num_songs
    songs_per_podcast = args.songs_per_podcast if args.songs_per_podcast is not None else base.songs_per_podcast
    time_range = args.time_range or base.time_range

    blacklist_playlists: List[str] = []
    blacklist_playlists.extend(base.blacklist_playlists or [])
    if args.blacklist_playlists:
        blacklist_playlists.extend(args.blacklist_playlists)

    whitelist_playlists: List[str] = []
    whitelist_playlists.extend(base.whitelist_playlists or [])
    if args.whitelist_playlists:
        whitelist_playlists.extend(args.whitelist_playlists)

    whitelist_ratio = args.whitelist_ratio if args.whitelist_ratio is not None else base.whitelist_ratio

    rename_playlist_to = args.rename_playlist_to if args.rename_playlist_to is not None else base.rename_playlist_to
    playlist_description = args.playlist_description if args.playlist_description is not None else base.playlist_description
    playlist_public = args.playlist_public if args.playlist_public is not None else base.playlist_public

    # If no whitelists, force ratio to 0 to match your spec
    if not whitelist_playlists:
        whitelist_ratio = 0.0

    return ProfileConfig(
        name=base.name,  # keep chosen profile name (incl weekday variant)
        playlist=playlist,
        podcasts=podcasts,
        bias=float(bias),
        num_songs=int(num_songs),
        songs_per_podcast=int(songs_per_podcast),
        time_range=time_range,
        blacklist_playlists=blacklist_playlists,
        whitelist_playlists=whitelist_playlists,
        whitelist_ratio=float(whitelist_ratio),
        rename_playlist_to=rename_playlist_to,
        playlist_description=playlist_description,
        playlist_public=playlist_public,
    )


# ----------------------------
# CLI commands
# ----------------------------

def cmd_init_auth(args: argparse.Namespace) -> int:
    client_id = args.client_id or os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = args.client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
    redirect_uri = args.redirect_uri

    if not client_id or not client_secret:
        print("Missing client_id/client_secret (flags or env SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET).", file=sys.stderr)
        return 2

    scopes = [
        "playlist-modify-private",
        "playlist-modify-public",
        "playlist-read-private",  # blacklist/whitelist if private playlists
        "user-top-read",
    ]
    scope_str = " ".join(scopes)

    params = {
        "response_type": "code",
        "client_id": client_id,
        "scope": scope_str,
        "redirect_uri": redirect_uri,
        "show_dialog": "true",
    }
    url = f"{ACCOUNTS_BASE}/authorize?{urllib.parse.urlencode(params)}"

    print("\nOpen this URL in your browser to authorize:\n", flush=True)
    print(url, flush=True)
    print(f"\nWaiting for redirect to {redirect_uri} ...\n", flush=True)

    parsed = urllib.parse.urlparse(redirect_uri)
    port = parsed.port or 8765

    try:
        code = run_local_auth_server_and_get_code(port=port, timeout_sec=args.timeout_sec)
    except Exception as e:
        print(f"Auth callback failed: {e}", file=sys.stderr, flush=True)
        return 1

    try:
        tokens = exchange_code_for_tokens(client_id, client_secret, code, redirect_uri)
    except Exception as e:
        print(f"Token exchange failed: {e}", file=sys.stderr, flush=True)
        return 1

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        print("No refresh_token returned. Try again (show_dialog is already true).", file=sys.stderr, flush=True)
        return 1

    print("\n✅ Success. Save this refresh token somewhere secure:\n", flush=True)
    print(refresh_token, flush=True)
    return 0

def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def spotify_get_many_tracks(access_token: str, track_ids: List[str]) -> Dict[str, dict]:
    """
    Returns dict track_id -> track object
    """
    out: Dict[str, dict] = {}
    for chunk in chunked(track_ids, 50):
        data = spotify_get(access_token, "/tracks", params={"ids": ",".join(chunk)})
        for t in data.get("tracks") or []:
            if t and t.get("id"):
                out[t["id"]] = t
    return out

def spotify_get_many_episodes(access_token: str, episode_ids: List[str]) -> Dict[str, dict]:
    """
    Returns dict episode_id -> episode object
    """
    out: Dict[str, dict] = {}
    for chunk in chunked(episode_ids, 50):
        data = spotify_get(access_token, "/episodes", params={"ids": ",".join(chunk)})
        for e in data.get("episodes") or []:
            if e and e.get("id"):
                out[e["id"]] = e
    return out

def pretty_item_lines(access_token: str, uris: List[str], show_uris: bool = False) -> List[str]:
    """
    Turn spotify:item URIs into pretty display lines.
    If show_uris=True, append the URI at the end.
    """
    track_ids: List[str] = []
    episode_ids: List[str] = []

    for u in uris:
        if u.startswith("spotify:track:"):
            track_ids.append(u.split(":")[2])
        elif u.startswith("spotify:episode:"):
            episode_ids.append(u.split(":")[2])

    tracks = spotify_get_many_tracks(access_token, track_ids) if track_ids else {}
    episodes = spotify_get_many_episodes(access_token, episode_ids) if episode_ids else {}

    lines: List[str] = []
    for u in uris:
        suffix = f"  ({u})" if show_uris else ""
        if u.startswith("spotify:track:"):
            tid = u.split(":")[2]
            t = tracks.get(tid)
            if not t:
                lines.append(f"🎵 [unknown track]{suffix}")
                continue
            name = t.get("name") or "[unnamed]"
            artists = ", ".join(a.get("name") for a in (t.get("artists") or []) if a.get("name")) or "[unknown artist]"
            lines.append(f"🎵 {name} — {artists}{suffix}")
        elif u.startswith("spotify:episode:"):
            eid = u.split(":")[2]
            e = episodes.get(eid)
            if not e:
                lines.append(f"🎙️ [unknown episode]{suffix}")
                continue
            name = e.get("name") or "[unnamed]"
            show = (e.get("show") or {}).get("name") or "[unknown show]"
            lines.append(f"🎙️ {name} — {show}{suffix}")
        else:
            lines.append(f"• {u}{suffix}")

    return lines


def cmd_run(args: argparse.Namespace) -> int:
    client_id = args.client_id or os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = args.client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET (or flags).")

    profiles_path = args.profiles_file or DEFAULT_PROFILES_FILE
    profiles = load_profiles_file(profiles_path)

    cfg = resolve_profile_config(args, profiles)

    refresh_token = (
        args.refresh_token
        or env_refresh_token_for_profile(cfg.name)  # exact profile (weekday variant)
        or env_refresh_token_for_profile(args.profile.lower())  # base profile (sam)
        or os.environ.get("SPOTIFY_REFRESH_TOKEN")
    )
    if not refresh_token:
        raise RuntimeError(
            f"Missing refresh token. Set SPOTIFY_{cfg.name.upper()}_REFRESH_TOKEN "
            f"(or base profile) or SPOTIFY_REFRESH_TOKEN (or pass --refresh-token)."
        )

    playlist_id = playlist_id_from_any(cfg.playlist)
    alpha = bias_to_alpha(cfg.bias)
    access = refresh_access_token(client_id, client_secret, refresh_token)

    # Optional playlist rename/update
    if cfg.rename_playlist_to or cfg.playlist_description or cfg.playlist_public is not None:
        if args.dry_run:
            print(f"DRY RUN: would update playlist details for {playlist_id}: "
                  f"name={cfg.rename_playlist_to!r} desc={cfg.playlist_description!r} public={cfg.playlist_public!r}")
        else:
            update_playlist_details(
                access, playlist_id,
                name=cfg.rename_playlist_to,
                description=cfg.playlist_description,
                public=cfg.playlist_public,
            )

    # Build blacklist set
    blacklisted_tracks: Set[str] = set()
    if cfg.blacklist_playlists:
        for bl in cfg.blacklist_playlists:
            bl_id = playlist_id_from_any(bl)
            bl_list = get_playlist_track_uris_ordered(access, bl_id)
            blacklisted_tracks |= set(bl_list)

    # Build whitelist pools (ordered) and sample portion
    chosen_tracks: List[str] = []
    chosen_set: Set[str] = set()

    whitelist_ratio = cfg.whitelist_ratio if cfg.whitelist_playlists else 0.0
    n_from_whitelist = int(round(cfg.num_songs * whitelist_ratio))
    n_from_whitelist = max(0, min(cfg.num_songs, n_from_whitelist))
    n_from_top = cfg.num_songs - n_from_whitelist

    whitelist_debug = []
    if n_from_whitelist > 0:
        pools: List[List[str]] = []
        for w in cfg.whitelist_playlists:
            wid = playlist_id_from_any(w)
            uris = get_playlist_track_uris_ordered(access, wid)
            uris = [u for u in uris if u not in blacklisted_tracks]
            uris = unique_preserve_order(uris)
            pools.append(uris)
            whitelist_debug.append((w, len(uris)))

        if not pools:
            raise RuntimeError("whitelist_ratio > 0 but no whitelist pools could be loaded.")

        counts = allocate_counts_random(n_from_whitelist, len(pools))

        # sample from each pool, ensuring global uniqueness
        for pool, want, meta in zip(pools, counts, whitelist_debug):
            if want <= 0:
                continue
            # remove already chosen
            pool_allowed = [u for u in pool if u not in chosen_set]
            if len(pool_allowed) < want:
                raise RuntimeError(
                    f"Whitelist pool too small for allocation: want {want}, have {len(pool_allowed)} "
                    f"from {meta[0]!r}. Reduce whitelist_ratio or expand that playlist."
                )
            picks = weighted_unique_sample(pool_allowed, k=want, alpha=alpha)
            for p in picks:
                chosen_set.add(p)
            chosen_tracks.extend(picks)

    # Top tracks fallback / remainder
    if n_from_top > 0:
        top_uris = get_top_track_uris(access, limit=50, time_range=cfg.time_range)
        if not top_uris:
            raise RuntimeError("No top tracks returned. Ensure your token has user-top-read scope.")
        top_allowed = [u for u in top_uris if u not in blacklisted_tracks and u not in chosen_set]
        if len(top_allowed) < n_from_top:
            raise RuntimeError(
                f"Not enough top tracks after blacklist/whitelist exclusion: need {n_from_top}, have {len(top_allowed)}."
            )
        picks = weighted_unique_sample(top_allowed, k=n_from_top, alpha=alpha)
        chosen_tracks.extend(picks)

    # Resolve podcasts -> latest episodes
    episode_uris: List[str] = []
    for p in cfg.podcasts:
        sid = resolve_show_id(access, p)
        ep_uri = latest_episode_uri_for_show(access, sid)
        if ep_uri:
            episode_uris.append(ep_uri)

    # Build final sequence
    final_uris = build_sequence(
        episode_uris=episode_uris,
        track_uris=chosen_tracks,
        songs_per_podcast=cfg.songs_per_podcast,
    )

    if args.dry_run:
        print(f"DRY RUN profile={cfg.name!r} weekday={weekday_key()}")
        print(f"Target playlist: {cfg.playlist} (id={playlist_id})")
        if cfg.blacklist_playlists:
            print(f"Blacklist playlists: {cfg.blacklist_playlists} -> {len(blacklisted_tracks)} blacklisted tracks")
        if cfg.whitelist_playlists:
            print(f"Whitelist playlists: {cfg.whitelist_playlists} (ratio={whitelist_ratio}) allocations={n_from_whitelist}/{cfg.num_songs}")
            for w, ln in whitelist_debug:
                print(f"  - {w}: {ln} track URIs available after blacklist")
        print(f"Chosen songs: {len(chosen_tracks)} | Episodes: {len(episode_uris)} | Final items: {len(final_uris)}")
        for line in pretty_item_lines(access, final_uris, show_uris=False):
            print(" ", line)
        return 0

    clear_playlist(access, playlist_id)
    if final_uris:
        add_items_to_playlist(access, playlist_id, final_uris)

    print(f"✅ Updated playlist {playlist_id} via profile={cfg.name!r}: "
          f"{len(final_uris)} items ({len(episode_uris)} episodes, {len(chosen_tracks)} songs).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotify_podmix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Generate a hybrid playlist: podcast, 5 songs, podcast, 5 songs, ...

            Profiles:
              - Define profiles in profiles.json (default) with playlist/podcasts/bias/etc.
              - Weekday overrides: "<profile>_<weekday>" (mon..sun) if present.
              - Provide per-profile refresh tokens via env:
                  SPOTIFY_SAM_REFRESH_TOKEN, SPOTIFY_KASEY_REFRESH_TOKEN, ...
              - Run with: --profile sam

            Whitelist:
              - If whitelist_playlists present, sample a portion of songs from those playlists:
                  whitelist_ratio in [0,1] => fraction of num_songs from whitelists.
              - Remaining songs sampled from /me/top/tracks.
              - Multiple whitelist playlists: random allocation of the whitelist quota across playlists.

            Blacklist:
              - Provide one or more playlists whose member tracks are excluded from sampling.
              - Requires playlist-read-private if the playlists are private.
            """
        ),
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("init-auth", help="One-time local auth to mint a refresh token")
    a.add_argument("--client-id", default=None)
    a.add_argument("--client-secret", default=None)
    a.add_argument("--redirect-uri", default=DEFAULT_REDIRECT_URI)
    a.add_argument("--timeout-sec", type=int, default=180)

    r = sub.add_parser("run", help="Run the generator (cron-friendly)")
    r.add_argument("--client-id", default=None)
    r.add_argument("--client-secret", default=None)
    r.add_argument("--refresh-token", default=None, help="Override refresh token (otherwise env/profile).")

    r.add_argument("--profile", default="default", help="Base profile name; may resolve to weekday variant.")

    r.add_argument("--profiles-file", default=None, help=f"Path to profiles JSON (default: {DEFAULT_PROFILES_FILE}).")

    r.add_argument("--playlist", default=None, help="Override target playlist URL/URI/ID.")
    r.add_argument("--podcasts", nargs="+", default=None, help="Override podcasts list (show URIs/URLs/IDs or names).")
    r.add_argument("--bias", type=float, default=None, help="Override bias 0..1 (0 uniform, 1 top-weighted).")
    r.add_argument("--num-songs", type=int, default=None, help="Override number of songs.")
    r.add_argument("--songs-per-podcast", type=int, default=None, help="Override songs between podcasts.")
    r.add_argument("--time-range", default=None, choices=["short_term", "medium_term", "long_term"])

    r.add_argument("--blacklist-playlists", nargs="*", default=None,
                   help="One or more playlist URLs/URIs/IDs whose track members are excluded.")

    # NEW: whitelist CLI override + ratio
    r.add_argument("--whitelist-playlists", nargs="*", default=None,
                   help="One or more playlist URLs/URIs/IDs to use as whitelist pools.")
    r.add_argument("--whitelist-ratio", type=float, default=None,
                   help="0..1 fraction of songs drawn from whitelists (if whitelists present).")

    r.add_argument("--rename-playlist-to", default=None, help="If set, rename the target playlist.")
    r.add_argument("--playlist-description", default=None, help="If set, update playlist description.")
    r.add_argument("--playlist-public", type=lambda x: {"true": True, "false": False}[x.lower()],
                   default=None, help="If set, update playlist public flag: true/false")

    r.add_argument("--dry-run", action="store_true", help="Print actions but do not modify playlist")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.cmd == "init-auth":
            return cmd_init_auth(args)
        if args.cmd == "run":
            return cmd_run(args)
        parser.print_help()
        return 2
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
