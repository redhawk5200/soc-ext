import json
# import os
import re
import pdfplumber
import pandas as pd
from typing import Any, Dict, List
from io import BytesIO
import asyncio
from openai import AsyncOpenAI
# from core import Message, Role
# from core.config import config
from control_objectives_db import ControlObjective, ControlObjectiveService
from cuec_db import CUEC, CUECService
from control_exceptions_db import ControlException, ControlExceptionService
# from users_db import UserService
from datetime import datetime
from completions import get_openai_async_client
from IPython.display import display

metadata_tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_metadata",
            "description": "Extracts metadata information from the provided text of the SOC report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The complete title of the SOC Report.",
                    },
                    "time_period": {
                        "type": "string",
                        "description": "The time period of the SOC Report e.g 'Oct 1 2022 to Sept 30 2023' or '2022-2023'.",
                    },
                    "auditor_name": {
                        "type": "string",
                        "description": "The particular name of the auditor who is responsible for the whole S. The name of the auditor might be a name of an individual, institution or an organization. If no name is present in the provided text, return an empty string.",
                    },
                    "auditee_name": {
                        "type": "string",
                        "description": "The particular name of the auditee organization that is being audited. If no name is present in the provided text, return an empty string.",
                    },
                },
                "required": ["title", "time_period", "auditor_name", "auditee_name"],
            },
        },
    },
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_cuec",
            "description": "Extracts Complementary User Entity Controls (CUEC) information from the provided text of table if present in the provided text",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The number of the CUEC for example 1.1 or XYZ-2.1 etc, only one number is to be added per row",
                    },
                    "description": {
                        "type": "string",
                        "description": "The description is the text provided against a CUEC number in the provided text, if the text is in the form of a table then the description might be in nearby rows please be aware while extracting.",
                    },
                    "related_criteria": {
                        "type": "array",
                        "description": "An array of the related criteria against each CUEC number. for example [CC2.1, CC2.2] or [1.1, 3.5] etc. The related criteria are the Control Criteria numbers, and not text descriptions, that are associated with the CUEC.",
                        "items": {"type": "string"},
                    }
                },
                "required": ["number", "description", "related_criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_control_objective",
            "description": "Extracts Control Objective (CO) information from the provided text or tables, if the text is in the form of a table then the description might be in nearby rows please be aware while extracting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "integer",
                        "description": "The number of the Control Objective CO for example X or X.X etc, where X is a number.",
                    },
                    "supporting_control_activity": {
                        "type": "array",
                        "description": "The list of associated control activities for the Control Objective. The are the activities that support the implementation of the Control Objective referenced by a number and not the text description. Must be a correlating number not a text.",
                        "items": {"type": "string"},
                    },
                    "description": {
                        "type": "string",
                        "description": "The description is the text provided against a Control Objective number in the provided text or table, the dataframe might have description one above or below the Control Objective number.",
                    },
                },
                "required": ["criteria", "supporting_control_activity", "description"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "extract_csoc",
            "description": "Extracts Complementary Subservice Organization Controls (CSOC) information from the provided text or tables, if the text is in the form of a table then the description might be in nearby rows please be aware while extracting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csoc_number": {
                        "type": "string",
                        "description": "The number of the Complementary Subservice Organization Controls (CSOC) for example X or X.X etc, where X is a number.",
                    },
                    "applicable_control_objective": {
                        "type": "array",
                        "description": "The list of associated Control Objectives for the CSOC. The are the activities that support the implementation of the Control Objective referenced by a number and not the text description. Must be a correlating number not a text.",
                        "items": {"type": "string"},
                    },
                    "description": {
                        "type": "string",
                        "description": "The description is the text provided against a Complementary Subservice Organization Controls (CSOC) number in the provided text or table, the dataframe might have description one above or below the CSOC number.",
                    },
                },
                "required": ["csoc_number", "applicable_control_objective", "description"],
            },
        },
    },
   {
        "type": "function",
        "function": {
            "name": "extract_exception",
            "description": "Extracts exception information from the exceptions noted in the provided text of table or text if an only if an exception is mentioned against that control number, if the text is in the form of a table then the description might be in nearby rows please be aware while extracting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "control_number": {
                        "type": "string",
                        "description": "The control number for which the exception is being reported, for example 13.2, 18.2 or XX.X where X is a number etc.",
                    },
                    "controls_specified": {
                        "type": "string",
                        "description": "Controls specified by the service organization is the text describing the implementation of those controls in the organization.",
                    },
                    "tests_of_controls": {
                        "type": "string",
                        "description": "Tests of controls is the text mentioning the test(s) that were conducted by the auditor to validate the controls specified by the service organization.",    
                    },
                    "result_of_tests": {
                        "type": "string",
                        "description": "Result of tests is the text mentioning if no exceptions were noted, or if exceptions were noted and their description is given.",
                    },
                },
                "required": ["control_number", "controls_specified", "tests_of_controls", "result_of_tests"],
            },
        },
    },
]

system_prompt = """You are an expert text analyzer and extractor specializing in System and Organization Controls (SOC) reports and documents. You are required to extract the relevant Complementary User Entity Controls (CUEC) or Control Objective and its related information from the provided text. 
A "Control Objective" in a SOC report refers to a statement that outlines the intended purpose or aim of a specific control within a service organization's system, essentially defining the desired outcome of the control and the risks it is meant to mitigate.
Control Objective, also known as a Control Criteria specific goal or desired outcome that a control is designed to achieve within the SOC framework. It has a number in the document for example 1.11 or CO2.3 etc. It also has a description that discusses that implementations that an organization must implement.
A Complementary User Entity Controls refers to specific controls that a user organization (the customer) needs to implement on their end to fully utilize the services provided by a service organization, ensuring the overall security and compliance of the system.
CUEC is identified by a number and has a description. It is also related to a Control Criteria, which is identified by a number.

You have four tools available to extract the information of CO, CUEC, CSOC and exception respectively:
extract_control_objective: extracts Control Objective (CO) information from the provided text or tables.
extract_cuec: extracts Complementary User Entity Controls (CUEC) information from the provided text of table if present in the provided text.
extract_csoc: extracts Complementary Subservice Organization Controls (CSOC) information from the provided text or tables.
extract_exception: extracts exception information from the provided text of table or text if an only if an exception is mentioned against that control number.

IMPORTANT INSTRUCTIONS:
- Read the provided text thoroughly and make decide whether there is Control Objective information in the text or CUEC information or both.
- Use the appropriate tool based on the information present in the text or use both.
- Return empty list if no relevant information is found in the text either for Control Objective or CUEC.
- Only extract the Control Objective or CUEC information if the definition is explicitly mentioned in the text. Do not extract if the reference of any Control Objective or its sub COntrol Objective is mentioned. 
- Only extract CUEC if the definition of the CUEC, with the keyword Complementary User Entity Controls or CUEC is explicitly mentioned in the text. Do not extract if the reference of any CUEC is mentioned.
- Analyze the provided text very carefully so that there are no repetitions or incorrect extractions.
- Never extract an reference of a Control Objective or CUEC, only extract the definition that is explicitly mentioned in the text.
- If a control objective number is mentioned as COX.X, COXX.X then it is a reference to a control objective and should not be extracted. This is very important to check never extract a reference like COX.X, COXX.X or consider it a control objective.
- Make sure to extract the supporting control activities for the Control Objective very carefully. There should not be any mismatch / repetition of information. This can be empty but only and only in case of no supporting control activities mentioned in the text.
- Make sure to output the Control Objective or CUEC number correctly do not add or append any prefix / suffix like CO or 'Control Order Number' with the Control order number number.
- Make sure to output the number (cuec number) carefully and do not add or append any prefix / suffix with it like CUEC-1 or XXXX-2, just mention the number X and nothing else.
- Make sure to extract the related criteria for the CUEC very carefully. There should not be any mismatch / repetition of information. This can be empty but only and only in case of no related criteria mentioned in the text.
- If no number is present for a Control Objective or CUEC then return an empty list.
- Exception are mentioned against the control number and the control number is mentioned in the text.
- Extract the exception information only if is mentioned "Exception noted." or "No exceptions noted." in the text.
- When noting exceptions, write either "No exception noted" or "Exception noted" and nothing else, also provide a brief description of the exception if and only if there is a description written after "Exception noted."
- Do not mix management response with the result of tests, they are two different things and should be extracted separately. Be careful when extracting info, make sure it is not repeated or mixed in result of tests and management response.
- Do not make up any Control Objective or CUEC information on your own and only extract verbatim text from the provided text.
"""

metadata_prompt = """You are an expert document metadata extractor expert in working on System and Organization Controls (SOC) reports. Given the first few pages of an SOC Report, You are required to extract the following metadata information:
- Title: The title of the document.
- Time Period: The time period covered by the document.
- Auditor Name: The name of the auditor firm that performed the audit.
- Auditee Name: The name of the organization that is being audited.

IMPORTANT INSTRUCTIONS:
- Do not make up any information and only extract the information that is explicitly mentioned in the text.
- The title and time period are usually found at the beginning of the document.
- The auditor name is typically mentioned in the audit report or the cover page.
- If the information is not found, return an empty string.
- Use the extract_metadata tool to output the extracted metadata information.
"""

class DocumentProcessError(Exception):
    pass

class ProcessSOC:
    def __init__(self):
        """Initializes the required services."""
        self._cuec_service = CUECService()
        self._objective_service = ControlObjectiveService()
        self._exception_service = ControlExceptionService()
        # self._user_service = UserService()
        # self._user_service = user_service

        self._client = get_openai_async_client(base_url="https://openrouter.ai/api/v1", api_key="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")

        
    async def __call__(self, user_id: str, kb_doc_id: str, pdf_bytes: bytes) -> None:
        """Main execution method to process a PDF in bytes format."""
        # user = "12345"  # This line is not needed
        
        filtered_page_list = self._find_keyword_pages(pdf_bytes)
        extracted_data = self._extract_pdf_data(pdf_bytes, filtered_page_list)

        structured_data_co, structured_data_cuec, structured_data_exception = asyncio.run(self._process_responses_async(
            extracted_data, filtered_page_list))
        df_cleaned_co, df_cleaned_cuec, df_cleaned_exception = self._clean_and_sort_data(
            structured_data_co, structured_data_cuec, structured_data_exception
        )

        display(df_cleaned_exception)

        metadata = asyncio.run(self._get_metadata(pdf_bytes))
        # await self._get_metadata(pdf_bytes)


        for index, row in df_cleaned_co.iterrows():
            print("Inserting Control Objective with Metadata:", metadata)  # Debugging statement
            self._objective_service.add(
                ControlObjective(
                    kb_doc_id=kb_doc_id,
                    number=row["Control Objective Number"],
                    description=row["Control Objective Description"],
                    supporting_control_activities=row["Supporting Control Activity"].split(", "),
                    document_metadata={
                        "Document Title": metadata["title"],
                        "Time Period": metadata["time_period"],
                        "Auditor Name": metadata["auditor_name"],
                        "Company Name": metadata["auditee_name"],
                        "Page Number": row["Page Number"]
                    },
                    created_by=user_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
            )
        
        for index, row in df_cleaned_cuec.iterrows():
            self._cuec_service.add(
                CUEC(
                    number=row["CUEC Number"],
                    description=row["CUEC Description"],
                    related_criteria=row["Related Criteria"].split(", "),
                    kb_doc_id=kb_doc_id,
                    document_metadata={
                        "Document Title": metadata["title"],
                        "Time Period": metadata["time_period"],
                        "Auditor Name": metadata["auditor_name"],
                        "Company Name": metadata["auditee_name"],
                        "Page Number": row["Page Number"]
                    },
                    created_by=user_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
            )

        for index, row in df_cleaned_exception.iterrows():
            self._exception_service.add(
                ControlException(
                    objective_id="1",
                    control_id=row["Control Number"],
                    controls_specified=row["Controls Specified"],
                    tests=row["Tests of Controls"],
                    results=row["Result of Tests"],
                    servicer_response="",
                    number="",
                    soc_id="",
                    mitigating_controls="",
                    deficiency="",
                    type="",
                    kb_doc_id=kb_doc_id,
                    document_metadata={
                        "Document Title": metadata["title"],
                        "Time Period": metadata["time_period"],
                        "Auditor Name": metadata["auditor_name"],
                        "Company Name": metadata["auditee_name"],
                        "Page Number": row["Page Number"]
                    },
                    created_by=user_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
            )

    def _complete_keyword_search(self, text, keywords):
        found_keywords = set()
        
        for keyword in keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b')  # Ensure exact word match
            if pattern.search(text):
                found_keywords.add(keyword)
        
        return found_keywords

    def _find_keyword_pages(self, pdf_bytes: bytes) -> List[int]:
        """Finds pages containing relevant keywords in the PDF."""
        keywords = ["Control Objective", "Complementary User Entity Controls", "CUEC"]
        filtered_pages = []

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for index, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text :
                    exact_found_keywords = self._complete_keyword_search(text, keywords)
                    if exact_found_keywords:
                        filtered_pages.append(index + 1) 
 
        return filtered_pages

    def _extract_pdf_data(self, pdf_bytes: bytes, filtered_page_list: List[int]) -> List[str]:
        """Extracts text & tables only from the relevant pages of the PDF."""
        extracted_data = []

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                # Only process pages that are in filtered_page_list
                if page_number not in filtered_page_list:
                    continue

                table_regions = [table.bbox for table in page.find_tables()]

                def is_outside_table(obj):
                    """Check if a text object is outside any detected table."""
                    return all(
                        not (region[0] <= obj["x0"] <= region[2] and region[1] <= obj["top"] <= region[3])
                        for region in table_regions
                    )

                text_content = " ".join(
                    [word["text"] for word in page.extract_words() if is_outside_table(word)]
                )

                tables = page.extract_tables()
                table_strings = [
                    pd.DataFrame(table).dropna(how="all").replace({None: ""}).to_string(index=False, header=False)
                    for table in tables
                ]

                formatted_text = f"Page {page_number}:\n\n{text_content.strip()}\n\n"
                if table_strings:
                    formatted_text += "\n\n".join(table_strings) + "\n\n"

                extracted_data.append(formatted_text)

        return extracted_data


    async def _process_responses_async(self, data_list: List[str], filtered_page_list: List[int]) -> tuple:
        """Asynchronously sends extracted text to OpenAI and processes structured responses."""
        structured_data_co = []
        structured_data_cuec = []
        structured_data_exception = []



        tasks = [
            self._call_llm_async(page_data, system_prompt, tools) for page_data in data_list
        ]

        responses = await asyncio.gather(*tasks)

        for page_number, response in zip(filtered_page_list, responses):  
        
            for item in response:
                if "criteria" in item:
                    structured_data_co.append({
                        "Page Number": page_number,  
                        "Control Objective Number": item.get("criteria", ""),
                        "Supporting Control Activity": ", ".join(item.get("supporting_control_activity", [])),
                        "Control Objective Description": item.get("description", "")
                    })
                elif "number" in item:
                    structured_data_cuec.append({
                        "Page Number": page_number,  
                        "CUEC Number": item.get("number", ""),
                        "CUEC Description": item.get("description", ""),
                        "Related Criteria": item.get("related_criteria", [])
                    })
                elif "control_number" in item:
                    structured_data_exception.append({
                        "Page Number": page_number,  
                        "Control Number": item.get("control_number", ""),
                        "Controls Specified": item.get("controls_specified", ""),
                        "Tests of Controls": item.get("tests_of_controls", ""),
                        "Result of Tests": item.get("result_of_tests", ""),
                    })

        return structured_data_co, structured_data_cuec, structured_data_exception

    def _clean_and_sort_data(self, structured_data_co, structured_data_cuec, structured_data_exception):
            """Cleans and sorts the extracted data."""
            df_co = pd.DataFrame(structured_data_co)
            df_cuec = pd.DataFrame(structured_data_cuec)
            df_exception = pd.DataFrame(structured_data_exception)

            df_cleaned_co = df_co.groupby("Control Objective Number", as_index=False).agg({
                "Page Number": "first",
                "Supporting Control Activity": lambda x: ", ".join(sorted(set(", ".join(x).split(", ")) if any(x) else [])),
                "Control Objective Description": "first"
            }).sort_values(by="Page Number").reset_index(drop=True)


            df_cleaned_cuec = df_cuec.groupby("CUEC Number", as_index=False).agg({
                "Page Number": "first",
                "CUEC Description": "first",
                "Related Criteria": lambda x: ", ".join(sorted(set(item for sublist in x.dropna() if isinstance(sublist, list) for item in sublist))) if any(x.dropna()) else ""
            }).sort_values(by="CUEC Number", key=lambda x: x.astype(int)).reset_index(drop=True)

            df_filtered = df_exception[df_exception["Result of Tests"].str.contains("Exception noted", na=False)].copy()

            df_filtered[['Exception Note', 'Description']] = df_filtered["Result of Tests"].str.split("\n", n=1, expand=True)

            df_cleaned_exception = df_filtered.groupby("Control Number", as_index=False).agg({
                "Page Number": "first",
                "Controls Specified": "first",
                "Tests of Controls": "first",
                "Result of Tests": "first",
            }).sort_values(by="Control Number").reset_index(drop=True)


            return df_cleaned_co, df_cleaned_cuec, df_cleaned_exception

    

    async def _get_metadata(self, pdf_bytes: bytes) -> dict:
        """Extracts metadata information from the first few pages of the PDF."""
        combined_text = ""
        
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:5]: 
                text = page.extract_text()
                if text:  
                    combined_text += text + "\n"  
        
        metadata_response = await self._call_llm_async(input_text=combined_text, system_prompt=metadata_prompt, tools=metadata_tools)

        return metadata_response[0] if metadata_response else {}


    async def _call_llm_async(self, input_text: str, system_prompt: str, tools: List) -> List[Dict[str, Any]]:
        """Asynchronously calls OpenAI GPT-4o-mini for processing extracted pages."""
        try:
            response_text = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                ],
                tools=tools,
                tool_choice="required",
            )
            if response_text:
                return [
                json.loads(item.function.arguments) for item in response_text.choices[0].message.tool_calls
            ]
            else:
                return []
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return []
        
    


