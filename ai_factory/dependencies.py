from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ai_factory.database import get_session


async def get_db(session: AsyncSession = Depends(get_session)):
    yield session
