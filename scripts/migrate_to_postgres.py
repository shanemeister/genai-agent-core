"""Migration script: SQLite → PostgreSQL

Migrates memory cards, chat sessions, and mind file entries from SQLite to PostgreSQL.
Safe to re-run — uses ON CONFLICT DO NOTHING.

Usage:
    cd /home/exx/myCode/genai-agent-core
    python scripts/migrate_to_postgres.py
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import aiosqlite
from core.db.postgres import get_pool, init_database


async def migrate_memory_cards():
    """Migrate memory cards from SQLite to PostgreSQL."""
    print("\n[1/3] Migrating memory cards...")

    sqlite_path = Path("data/memory/memory_cards.sqlite3")
    if not sqlite_path.exists():
        print(f"  ⚠️  SQLite database not found: {sqlite_path}")
        return

    pool = await get_pool()
    count = 0

    async with aiosqlite.connect(sqlite_path.as_posix()) as db:
        async with db.execute("SELECT json FROM memory_cards") as cur:
            async for (json_str,) in cur:
                data = json.loads(json_str)

                # Convert datetime strings/objects to timezone-naive datetime objects
                # (PostgreSQL columns are TIMESTAMP WITHOUT TIME ZONE)
                def to_naive_datetime(dt_value):
                    if not dt_value:
                        return None
                    if isinstance(dt_value, str):
                        dt = datetime.fromisoformat(dt_value)
                    else:
                        dt = dt_value
                    # Remove timezone info if present
                    if dt.tzinfo is not None:
                        dt = dt.replace(tzinfo=None)
                    return dt

                created_at = to_naive_datetime(data["created_at"])
                approved_at = to_naive_datetime(data.get("approved_at"))
                rejected_at = to_naive_datetime(data.get("rejected_at"))

                async with pool.acquire() as conn:
                    await conn.execute(
                        """INSERT INTO memory_cards(
                            id, created_at, category, scope, text, approval,
                            approved_at, rejected_at, provenance
                        ) VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT(id) DO NOTHING
                        """,
                        data["id"],
                        created_at,
                        data["category"],
                        data["scope"],
                        data["text"],
                        data["approval"],
                        approved_at,
                        rejected_at,
                        json.dumps(data["provenance"]),
                    )
                    count += 1

    print(f"  ✅ Migrated {count} memory cards")


async def migrate_chat_sessions():
    """Migrate chat sessions from SQLite to PostgreSQL."""
    print("\n[2/3] Migrating chat sessions...")

    sqlite_path = Path("data/memory/memory_cards.sqlite3")  # Sessions stored in same DB
    if not sqlite_path.exists():
        print(f"  ⚠️  SQLite database not found: {sqlite_path}")
        return

    pool = await get_pool()
    count = 0

    async with aiosqlite.connect(sqlite_path.as_posix()) as db:
        async with db.execute("SELECT id, title, created_at, updated_at, messages FROM chat_sessions") as cur:
            async for row in cur:
                session_id, title, created_at_str, updated_at_str, messages_json = row

                # Convert datetime strings/objects to timezone-naive datetime objects
                # (PostgreSQL columns are TIMESTAMP WITHOUT TIME ZONE)
                def to_naive_datetime(dt_value):
                    if not dt_value:
                        return None
                    if isinstance(dt_value, str):
                        dt = datetime.fromisoformat(dt_value)
                    else:
                        dt = dt_value
                    # Remove timezone info if present
                    if dt.tzinfo is not None:
                        dt = dt.replace(tzinfo=None)
                    return dt

                created_at = to_naive_datetime(created_at_str)
                updated_at = to_naive_datetime(updated_at_str)

                # messages_json is already a JSON string from SQLite
                async with pool.acquire() as conn:
                    await conn.execute(
                        """INSERT INTO chat_sessions(
                            id, title, created_at, updated_at, messages
                        ) VALUES($1, $2, $3, $4, $5::jsonb)
                        ON CONFLICT(id) DO NOTHING
                        """,
                        session_id,
                        title,
                        created_at,
                        updated_at,
                        messages_json,  # Pass JSON string directly, cast to jsonb
                    )
                    count += 1

    print(f"  ✅ Migrated {count} chat sessions")


async def migrate_mindfile_entries():
    """Migrate mind file entries from SQLite to PostgreSQL."""
    print("\n[3/3] Migrating mind file entries...")

    sqlite_path = Path("data/memory/memory_cards.sqlite3")
    if not sqlite_path.exists():
        print(f"  ⚠️  SQLite database not found: {sqlite_path}")
        return

    pool = await get_pool()
    count = 0

    async with aiosqlite.connect(sqlite_path.as_posix()) as db:
        # Check if mindfile_entries table exists in SQLite
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mindfile_entries'"
        )
        if not await cursor.fetchone():
            print("  ⚠️  No mindfile_entries table in SQLite — nothing to migrate")
            return

        async with db.execute("SELECT json FROM mindfile_entries") as cur:
            async for (json_str,) in cur:
                data = json.loads(json_str)

                def to_naive_datetime(dt_value):
                    if not dt_value:
                        return None
                    if isinstance(dt_value, str):
                        dt = datetime.fromisoformat(dt_value)
                    else:
                        dt = dt_value
                    if dt.tzinfo is not None:
                        dt = dt.replace(tzinfo=None)
                    return dt

                created_at = to_naive_datetime(data["created_at"])
                updated_at = to_naive_datetime(data.get("updated_at", data["created_at"]))

                async with pool.acquire() as conn:
                    await conn.execute(
                        """INSERT INTO mindfile_entries(
                            id, created_at, updated_at, category, text,
                            source_memory_card_id, note
                        ) VALUES($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT(id) DO NOTHING
                        """,
                        data["id"],
                        created_at,
                        updated_at,
                        data["category"],
                        data["text"],
                        data["source_memory_card_id"],
                        data.get("note") or "",
                    )
                    count += 1

    print(f"  ✅ Migrated {count} mind file entries")


async def verify_migration():
    """Verify migration by counting records in PostgreSQL."""
    print("\n[Verification]")

    pool = await get_pool()

    async with pool.acquire() as conn:
        cards_count = await conn.fetchval("SELECT COUNT(*) FROM memory_cards")
        sessions_count = await conn.fetchval("SELECT COUNT(*) FROM chat_sessions")
        mindfile_count = await conn.fetchval("SELECT COUNT(*) FROM mindfile_entries")

        print(f"  PostgreSQL memory_cards: {cards_count}")
        print(f"  PostgreSQL chat_sessions: {sessions_count}")
        print(f"  PostgreSQL mindfile_entries: {mindfile_count}")


async def main():
    """Run migration."""
    print("=" * 60)
    print("SQLite → PostgreSQL Migration")
    print("=" * 60)

    # Initialize PostgreSQL schema
    print("\nInitializing PostgreSQL schema...")
    await init_database()

    # Migrate data
    await migrate_memory_cards()
    await migrate_chat_sessions()
    await migrate_mindfile_entries()

    # Verify
    await verify_migration()

    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print("=" * 60)
    print("\nAll data now in PostgreSQL. SQLite files kept as backup.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
