from typing import Optional, Tuple

import api_service.db.controls as control_db
from api_service.services.users import User

Control = control_db.Control


class ControlService:
    def __init__(self, control_db: control_db.ControlService):
        self._control_db = control_db

    def add(self, user: User, control: Control) -> Control:
        return self._control_db.add(control)

    def get(self, user: User, control_id: str) -> Control:
        return self._control_db.get(control_id)

    def delete(self, user: User, control_id: str) -> None:
        return self._control_db.delete(control_id)

    def update(
        self,
        user: User,
        control_id: str,
        number: Optional[str] = None,
        description: Optional[str] = None,
        test: Optional[str] = None,
    ) -> Control:
        return self._control_db.update(control_id, number, description, test)

    def list(
        self,
        user: User,
        user_id: str,
        control_objective_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[list[Control], int]:
        db_controls, count = self._control_db.list(user_id, control_objective_id, limit=limit, offset=offset)
        return [
            Control(
                id=control.id,
                number=control.number,
                description=control.description,
                test=control.test,
                created_at=control.created_at,
                updated_at=control.updated_at,
                created_by=control.created_by,
            )
            for control in db_controls
        ], count
