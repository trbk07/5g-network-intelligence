"""Expiry arithmetic only; no credentials, network, training or SDK mutation."""

from datetime import datetime, timedelta, timezone

import pytest

from scripts.kaggle_job_status import needs_refresh


def test_expired_and_nearly_expired_tokens_refresh_before_polling():
    now=datetime(2026,11,23,14,0,tzinfo=timezone.utc)
    assert needs_refresh(now-timedelta(minutes=1),now)
    assert needs_refresh((now+timedelta(minutes=5)).isoformat(),now)
    assert needs_refresh(None,now)
    assert not needs_refresh(now+timedelta(minutes=6),now)


def test_naive_expiry_cannot_be_assumed_valid():
    with pytest.raises(ValueError,match="timezone"):
        needs_refresh(datetime(2026,11,23))
