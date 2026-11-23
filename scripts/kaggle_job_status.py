"""Read one hosted job status, refreshing cached OAuth before token expiry."""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone


def needs_refresh(expiration, now=None):
    """Use a five-minute safety margin; never treat an expired token as valid."""
    if not expiration:
        return True
    expiry=datetime.fromisoformat(expiration) if isinstance(expiration,str) else expiration
    current=now or datetime.now(timezone.utc)
    if expiry.tzinfo is None or current.tzinfo is None:
        raise ValueError("OAuth expiration must be timezone-aware")
    return expiry <= current+timedelta(minutes=5)


def refresh_cached_oauth(owner):
    """Use official SDK refresh; never print or copy tokens into project files."""
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials

    with KaggleClient() as client:
        credentials=KaggleCredentials.load(client)
        if credentials is None or credentials.get_username()!=owner:
            raise RuntimeError("Cached OAuth account does not match the job owner")
        # Inspect cached expiry rather than relying on the SDK's delayed expiry check.
        if needs_refresh(credentials._access_token_expiration):
            credentials.refresh_access_token()
            print("Cached OAuth refreshed; no credentials displayed.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel",help="Existing owner/kernel-slug; this command never starts a job")
    args=parser.parse_args()
    pieces=args.kernel.split("/")
    if len(pieces)!=2 or any(not part or not all(c.isalnum() or c in "-_" for c in part) for part in pieces):
        parser.error("Expected owner/kernel-slug")
    try:
        refresh_cached_oauth(pieces[0])
    except Exception:
        # SDK failures must not accidentally expose credential-bearing response objects.
        print("Could not refresh cached Kaggle OAuth. Check the account connection; job state is unknown.",file=sys.stderr)
        return 1
    return subprocess.run([sys.executable,"-m","kaggle.cli","kernels","status",args.kernel],check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
