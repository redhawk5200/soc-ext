import json
from typing import Any, Dict

import openai

import api_service
from api_service.services.control_objectives import ControlObjective
from api_service.services.control_objectives import ControlObjectiveService
from api_service.services.cuec import CUEC
from api_service.services.cuec import CUECService
from api_service.services.llms import LLM
from api_service.services.llms import LLMService
from api_service.services.users import UserService
from core import Message
from core import Role
from core.completions import chat_completions
from core.completions import LLMError


class DocumentProcessError(Exception):
    pass


class ProcessSOC:
    def __init__(
        self,
        llm_service: LLMService,
        cuec_service: CUECService,
        objective_service: ControlObjectiveService,
        user_service: UserService,
    ) -> None:
        self._cuec_service = cuec_service
        self._objective_service = objective_service
        self._llm_service = llm_service
        self._user_service = user_service

    def __call__(self, user_id: str, kb_doc_id: str, content: str) -> None:
        try:
            llm = self._llm_service.system_get_by_name("gpt-4o", with_token=True)
        except api_service.LLMNotFoundError:
            raise DocumentProcessError("could not find llm gpt-4o")

        user = self._user_service.get(user_id)

        results = self._call_llm(llm, content)

        # TODO add functions to add these in bulk
        for objective in results["objectives"]:
            self._objective_service.add(
                user,
                ControlObjective(
                    number=objective["criteria"],
                    objective=objective["description"],
                    kb_doc_id=kb_doc_id,
                    notes="",
                    created_by=user.id,
                ),
            )

        for cuec in results["cuecs"]:
            self._cuec_service.add(
                user,
                CUEC(
                    number=cuec["number"],
                    description=cuec["description"],
                    kb_doc_id=kb_doc_id,
                    process_owner="",
                    control_owner="",
                    control_owner_2="",
                    internal_control="",
                    created_by=user.id,
                ),
            )

        # TODO add cuec to objectives link

    def _call_llm(self, llm: LLM, content: str) -> Dict[str, Any]:
        try:
            completion = chat_completions(
                llm.client,
                llm.name,
                messages=[
                    Message(
                        role=Role.system,
                        content="""Can you extract CUECs and Control Objectives from the following SOC document.
                        Control Objectives might also be known as Control Criteria. Control Objectives may be preceded by CC and
                        then the number. Such as CC1.1. It may also be preceeded by Criteria: COSO Principle and you can infer
                        the criteria number from the cells below.

                        Use this format in json:

                        {
                          "cuecs": [{"number": 1, "description": "this is a description", "related_criteria": ["CC1.1", "CC1.2"]}]
                          "objectives": [{"criteria": "CC1.1", "description": ""this is a description"}]
                        }

                        Respond in JSON format only
                        """,
                    ),
                    Message(role=Role.user, content=content),
                ],
                response_format={"type": "json_object"},
            ).message()

        except openai.OpenAIError as e:
            raise LLMError(e)

        return json.loads(completion)  # type: ignore
