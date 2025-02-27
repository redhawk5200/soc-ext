from datetime import datetime
import logging
from typing import Optional, Sequence, Tuple

import shortuuid
from sqlalchemy import Engine
import sqlalchemy.exc
from sqlmodel import Column
from sqlmodel import desc
from sqlmodel import Field
from sqlmodel import func
from sqlmodel import Relationship
from sqlmodel import select
from sqlmodel import Session
from sqlmodel import SQLModel

from database import get_db_session

from sqlalchemy.dialects.postgresql import ARRAY, TIMESTAMP, JSONB


logger = logging.getLogger("cape")


class ControlException(SQLModel, table=True):
    id: str = Field(default_factory=lambda: f"exception-{shortuuid.uuid()}", primary_key=True)
    
    control_id: str 
    objective_id: str 
    soc_id: str 
    controls_specified: str
    number: str
    tests: str
    results: str
    servicer_response: str
    deficiency: str
    type: str
    mitigating_controls: str

    created_by: str
    document_metadata: dict = Field(
        default={
            "Document Title": "",
            "Time Period": "",
            "Auditor Name": "",
            "Company Name": "",
            "Page Number": ""
        },
        sa_column=Column(JSONB, nullable=False)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False))
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ControlException):
            return NotImplemented
        return self.id == other.id


class ControlExceptionService:
    def __init__(self) -> None:
        self._engine = get_db_session()


    def add(self, exception: ControlException) -> ControlException:
        with get_db_session() as session:  # ✅ Use session from database.py
            session.add(exception)
            session.commit()
            session.refresh(exception)
        return exception            

    # def __init__(self, engine: Engine) -> None:
    #     self._engine = engine

    # def add(self, exception: ControlException) -> ControlException:
    #     with Session(self._engine) as session:
    #         try:
    #             session.add(exception)
    #             session.commit()
    #             session.refresh(exception)

    #             return exception

    #         except sqlalchemy.exc.IntegrityError as e:
    #             logger.error(f"IntegrityError {e}")
    #             raise api_service.ControlExceptionAlreadyExistsError

    # def get(self, id: str) -> ControlException:
    #     with Session(self._engine) as session:
    #         exception = session.exec(select(ControlException).where(ControlException.id == id)).first()
    #         if exception is None:
    #             raise api_service.ControlExceptionNotFoundError
    #     return exception

    # def update(
    #     self,
    #     id: str,
    #     number: Optional[str] = None,
    #     tests: Optional[str] = None,
    #     results: Optional[str] = None,
    #     servicer_response: Optional[str] = None,
    #     deficiency: Optional[str] = None,
    #     type: Optional[str] = None,
    #     mitigating_controls: Optional[str] = None,
    # ) -> ControlException:
    #     with Session(self._engine) as session:
    #         exception = session.exec(select(ControlException).where(ControlException.id == id)).first()
    #         if exception is None:
    #             raise api_service.ControlNotFoundError

    #         if number is not None:
    #             exception.number = number

    #         if tests is not None:
    #             exception.tests = tests

    #         if results is not None:
    #             exception.results = results

    #         if servicer_response is not None:
    #             exception.servicer_response = servicer_response

    #         if deficiency is not None:
    #             exception.deficiency = deficiency

    #         if type is not None:
    #             exception.type = type

    #         if mitigating_controls is not None:
    #             exception.mitigating_controls = mitigating_controls

    #         session.commit()
    #         session.refresh(exception)

    #     return exception

    # def delete(self, id: str) -> None:
    #     with Session(self._engine) as session:
    #         exception = session.exec(select(ControlException).where(ControlException.id == id)).first()
    #         if exception is None:
    #             raise api_service.ControlExceptionNotFoundError

    #         session.delete(exception)
    #         session.commit()

    # def list(
    #     self, user_id: str, soc_id: str, limit: Optional[int] = 20, offset: int = 0
    # ) -> Tuple[Sequence[ControlException], int]:
    #     with Session(self._engine) as session:
    #         stmt = (
    #             select(ControlException)
    #             .where(ControlException.created_by == user_id)
    #             .where(ControlException.soc_id == soc_id)
    #             .order_by(desc(ControlException.created_at))
    #             .offset(offset)
    #             .limit(limit)
    #         )
    #         exceptions = session.exec(stmt).all()
    #         count = session.exec(
    #             select(func.count()).select_from(ControlException).where(ControlException.created_by == user_id)
    #         ).first()
    #     return exceptions, count or 0
