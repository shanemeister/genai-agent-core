from __future__ import annotations

import os
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "ZGRkXDGr9wcRbIs6yLg9yPwpPnDNYKzp")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "noesis")

_driver: AsyncDriver | None = None


async def init_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)
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
    session = driver.session(database=NEO4J_DATABASE)
    try:
        yield session
    finally:
        await session.close()
