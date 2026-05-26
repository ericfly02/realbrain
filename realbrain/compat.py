from __future__ import annotations

try:  # Python 3.11+
    from datetime import UTC as UTC
except ImportError:  # Python 3.10
    from datetime import timezone

    UTC = timezone.utc
