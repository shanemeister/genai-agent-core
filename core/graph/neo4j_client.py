from __future__ import annotations

from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver

from core.config import settings

_driver: AsyncDriver | None = None


async def init_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        await _driver.verify_connectivity()
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


@asynccontextmanager
async def get_session():
    driver = await init_driver()
    session = driver.session(database=settings.neo4j_database)
    try:
        yield session
    finally:
        await session.close()
