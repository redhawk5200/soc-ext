from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import shortuuid
from sqlalchemy import Engine
from sqlmodel import asc
from sqlmodel import case
from sqlalchemy import Column, func, Index, String, text
from sqlmodel import Field
from sqlmodel import func
from sqlmodel import Relationship
from sqlmodel import select
from sqlmodel import Session
from sqlmodel import SQLModel

from database import get_db_session
from sqlalchemy.dialects.postgresql import ARRAY, TIMESTAMP, JSONB


# class CUECObjectivesLink(SQLModel, table=True):
#     __tablename__ = "cuec_objectives_link"

#     cuec_id: str = Field(foreign_key="cuec.id", primary_key=True, ondelete="CASCADE")
#     objective_id: str = Field(foreign_key="controlobjective.id", primary_key=True, ondelete="CASCADE")


class CUEC(SQLModel, table=True):
    id: str = Field(default_factory=lambda: f"cuec-{shortuuid.uuid()}", primary_key=True)
    number: str
    description: str
    kb_doc_id: str
    related_criteria: List[str] = Field(sa_column=Column(ARRAY(String), nullable=True))
    # applicable: bool = Field(default=False)
    # process_owner: str
    # control_owner: str
    # control_owner_2: str
    # internal_control: str
    created_by: str

    # Metadata JSONB field with indexing enabled
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

    @classmethod
    def create_index(cls, session):
        session.execute(text("DROP INDEX IF EXISTS idx_metadata_gin"))
        session.execute(text("CREATE INDEX idx_metadata_gin ON cuec USING gin (document_metadata)"))
        session.commit()

    


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CUEC):
            return NotImplemented
        return self.id == other.id


class CUECService:
    def __init__(self) -> None:
        self._engine = get_db_session()

    def add(self, cuec: CUEC) -> CUEC:
        with get_db_session() as session:  # ✅ Use session from database.py
            session.add(cuec)
            session.commit()
            session.refresh(cuec)
        return cuec

#     def get(self, id: str) -> CUEC:
#         with Session(self._engine) as session:
#             cuec = session.exec(select(CUEC).where(CUEC.id == id)).first()
#             if cuec is None:
#                 raise api_service.CUECNotFoundError
#         return cuec

#     def update(
#         self,
#         id: str,
#         number: Optional[str] = None,
#         description: Optional[str] = None,
#         applicable: Optional[bool] = None,
#         process_owner: Optional[str] = None,
#         control_owner: Optional[str] = None,
#         control_owner_2: Optional[str] = None,
#         internal_control: Optional[str] = None,
#     ) -> CUEC:
#         with Session(self._engine) as session:
#             cuec = session.exec(select(CUEC).where(CUEC.id == id)).first()
#             if cuec is None:
#                 raise api_service.CUECNotFoundError

#             if number is not None:
#                 cuec.number = number

#             if description is not None:
#                 cuec.description = description

#             if applicable is not None:
#                 cuec.applicable = applicable

#             if process_owner is not None:
#                 cuec.process_owner = process_owner

#             if control_owner is not None:
#                 cuec.control_owner = control_owner

#             if control_owner_2 is not None:
#                 cuec.control_owner_2 = control_owner_2

#             if internal_control is not None:
#                 cuec.internal_control = internal_control

#             session.commit()
#             session.refresh(cuec)

#         return cuec

#     def update_objectives(self, cuec_id: str, objective_ids: List[str]) -> None:
#         with Session(self._engine) as session:
#             links = session.exec(select(CUECObjectivesLink).where(CUECObjectivesLink.cuec_id == cuec_id)).all()

#             old_ids = {link.objective_id for link in links}
#             new_ids = set(objective_ids)

#             to_add = new_ids.difference(old_ids)
#             to_delete = old_ids.difference(new_ids)
#             for add_id in to_add:
#                 session.add(CUECObjectivesLink(cuec_id=cuec_id, objective_id=add_id))

#             for delete_id in to_delete:
#                 link = session.exec(
#                     select(CUECObjectivesLink)
#                     .where(CUECObjectivesLink.cuec_id == cuec_id)
#                     .where(CUECObjectivesLink.objective_id == delete_id)
#                 ).one()
#                 if link is None:
#                     continue

#                 session.delete(link)
#             session.commit()

#     def delete(self, id: str) -> None:
#         with Session(self._engine) as session:
#             cuec = session.exec(select(CUEC).where(CUEC.id == id)).first()
#             if cuec is None:
#                 raise api_service.CUECNotFoundError

#             session.delete(cuec)
#             session.commit()

#     def list(self, user_id: str, doc_id: str, limit: Optional[int] = 20, offset: int = 0) -> Tuple[Sequence[CUEC], int]:
#         with Session(self._engine) as session:
#             sort_key = case((CUEC.number == "", 1), else_=0).label("sort_key")

#             stmt = (
#                 select(CUEC)
#                 .where(CUEC.created_by == user_id)
#                 .where(CUEC.kb_doc_id == doc_id)
#                 .order_by(sort_key.asc(), asc(CUEC.number))
#                 .offset(offset)
#                 .limit(limit)
#             )
#             cuecs = session.exec(stmt).unique().all()
#             count = session.exec(
#                 select(func.count()).select_from(CUEC).where(CUEC.created_by == user_id).where(CUEC.kb_doc_id == doc_id)
#             ).first()
#         return cuecs, count or 0

#     def add_link(self, cuec_id: str, objective_id: str) -> None:
#         with Session(self._engine) as session:
#             session.add(CUECObjectivesLink(cuec_id=cuec_id, objective_id=objective_id))
#             session.commit()
