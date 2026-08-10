"""Upload research reports to Google Drive. Primitives adapted from patent-translation-app/gdrive.py.

Unlike that app (which relies on load_dotenv() populating os.environ), this
repo's config.py uses pydantic-settings, which parses .env into its own
Settings object without touching os.environ — so credentials are threaded
through explicitly rather than read from the environment here.
"""
from pathlib import Path

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

MIME_FOLDER = "application/vnd.google-apps.folder"


def _credentials(refresh_token: str, client_id: str, client_secret: str) -> Credentials:
    return Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri="https://oauth2.googleapis.com/token",
    )


def _get_or_create_folder(service, name: str, parent_id: str) -> str:
    q = (
        f"name='{name}' and mimeType='{MIME_FOLDER}'"
        f" and '{parent_id}' in parents and trashed=false"
    )
    results = service.files().list(q=q, fields="files(id)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": name, "mimeType": MIME_FOLDER, "parents": [parent_id]}
    return service.files().create(body=meta, fields="id").execute()["id"]


def _resolve_path(service, gdrive_path: str) -> str:
    """Walk a slash-separated path from Drive root, creating folders as needed."""
    parent_id = "root"
    for part in gdrive_path.strip("/").split("/"):
        parent_id = _get_or_create_folder(service, part, parent_id)
    return parent_id


def _upload_file(service, local_path: Path, parent_id: str) -> None:
    name = local_path.name
    media = MediaFileUpload(str(local_path), resumable=True)
    q = f"name='{name}' and '{parent_id}' in parents and trashed=false"
    existing = service.files().list(q=q, fields="files(id)").execute().get("files", [])
    if existing:
        service.files().update(fileId=existing[0]["id"], media_body=media).execute()
    else:
        service.files().create(
            body={"name": name, "parents": [parent_id]}, media_body=media
        ).execute()


def export_report(
    local_path: Path,
    gdrive_base_path: str,
    *,
    refresh_token: str,
    client_id: str,
    client_secret: str,
) -> None:
    """Upload a single report file to gdrive_base_path/ on Drive. Sync — call via run_in_threadpool."""
    creds = _credentials(refresh_token, client_id, client_secret)
    creds.refresh(GoogleAuthRequest())
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    folder_id = _resolve_path(service, gdrive_base_path)
    _upload_file(service, local_path, folder_id)
