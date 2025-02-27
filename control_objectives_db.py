from datetime import datetime
from typing import Optional, Sequence, Tuple, List
from database import get_db_session  # ✅ Import session manager from database.py
import shortuuid
from sqlalchemy import asc
from sqlalchemy import Engine
from sqlmodel import case
from sqlalchemy import Column, func, Index, String, text
from sqlmodel import Field
from sqlmodel import func
from sqlmodel import Relationship
from sqlmodel import select
from sqlmodel import Session
from sqlmodel import SQLModel
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB, ARRAY




class ControlObjective(SQLModel, table=True):
    id: str = Field(default_factory=lambda: f"objective-{shortuuid.uuid()}", primary_key=True)
    kb_doc_id: str
    number: int
    description: str
    supporting_control_activities: list[str] = Field(sa_column=Column(ARRAY(String), nullable=True))
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
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False))
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False))
    # testandtrial : str = Field(default="test", sa_column=Column(String, nullable=True))

    @classmethod
    def create_index(cls, session):
        session.execute(text("DROP INDEX IF EXISTS idx_metadata_gin"))
        session.execute(text("CREATE INDEX idx_metadata_gin ON controlobjective USING gin (document_metadata)"))
        session.commit()

    

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ControlObjective):
            return NotImplemented
        return self.id == other.id



class ControlObjectiveService:
    def __init__(self) -> None:
        self._engine = get_db_session()

    def add(self, objective: ControlObjective) -> ControlObjective:
        """
        Adds a new ControlObjective to the database.
        Uses `get_db_session()` to ensure session management is consistent.
        """
        with get_db_session() as session:  # ✅ Use session from database.py
            session.add(objective)
            session.commit()
            session.refresh(objective)
        return objective
    
    # def add(self, objective: ControlObjective) -> ControlObjective:
    #     with Session(self._engine) as session:
    #         session.add(objective)
    #         session.commit()
    #         session.refresh(objective)
    #     return objective


    # def get(self, id: str) -> ControlObjective:
    #     with Session(self._engine) as session:
    #         objective = session.exec(select(ControlObjective).where(ControlObjective.id == id)).first()
    #         if objective is None:
    #             raise api_service.ControlObjectiveNotFoundError
    #     return objective

    # def update(
    #     self,
    #     id: str,
    #     number: Optional[str] = None,
    #     relevant: Optional[bool] = None,
    #     notes: Optional[str] = None,
    #     objective_txt: Optional[str] = None,
    # ) -> ControlObjective:
    #     with Session(self._engine) as session:
    #         objective = session.exec(select(ControlObjective).where(ControlObjective.id == id)).first()
    #         if objective is None:
    #             raise api_service.ControlObjectiveNotFoundError

    #         if number is not None:
    #             objective.number = number

    #         if relevant is not None:
    #             objective.relevant = relevant

    #         if notes is not None:
    #             objective.notes = notes

    #         if objective_txt is not None:
    #             objective.objective = objective_txt

    #         session.commit()
    #         session.refresh(objective)

    #     return objective

    # def delete(self, id: str) -> None:
    #     with Session(self._engine) as session:
    #         objective = session.exec(select(ControlObjective).where(ControlObjective.id == id)).first()
    #         if objective is None:
    #             raise api_service.ControlObjectiveNotFoundError

    #         session.delete(objective)
    #         session.commit()

    # def list(
    #     self, user_id: str, doc_id: str, limit: Optional[int] = 20, offset: int = 0
    # ) -> Tuple[Sequence[ControlObjective], int]:
    #     with Session(self._engine) as session:
    #         sort_key = case((ControlObjective.number == "", 1), else_=0).label("sort_key")

    #         stmt = (
    #             select(ControlObjective)
    #             .where(ControlObjective.created_by == user_id)
    #             .where(ControlObjective.kb_doc_id == doc_id)
    #             .order_by(sort_key.asc(), asc(ControlObjective.number))
    #             .offset(offset)
    #             .limit(limit)
    #         )
    #         objectives = session.exec(stmt).all()
    #         count = session.exec(
    #             select(func.count())
    #             .select_from(ControlObjective)
    #             .where(ControlObjective.created_by == user_id)
    #             .where(ControlObjective.kb_doc_id == doc_id)
    #         ).first()
    #     return objectives, count or 0
