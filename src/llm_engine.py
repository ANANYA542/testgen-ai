import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

ERP_CONTEXTS = {
    "Generic": "general enterprise software",
    "SAP S/4HANA": "SAP S/4HANA ERP system with modules like FI, MM, SD, and PP",
    "Oracle Fusion Cloud": "Oracle Fusion Cloud ERP with modules like Financials, SCM, and HCM",
    "Workday": "Workday HCM and Financial Management platform",
    "Salesforce": "Salesforce CRM platform with Sales Cloud, Service Cloud, and custom objects"
}

PROMPT_STYLES = {
    "Detailed": """
You are a senior QA engineer.

Use ONLY the provided context to generate test cases.

ERP Context: {erp_context}

Context:
{context}

Feature: {feature}

Instructions:
- Do NOT invent information outside the context
- Keep steps clear, numbered, and actionable
- Cover positive, negative, and edge cases
- Keep test cases realistic and relevant

Output format (STRICT):

Test ID: TC001
Scenario: [What is being tested]
Steps:
1. [Action 1]
2. [Action 2]
Expected Result: [Outcome]
Type: [Positive / Negative / Edge Case]

Test ID: TC002
...

Generate at least 4 thorough test cases. Do not include any other markdown or introductory text.
""",
    "Concise": """
You are a senior QA engineer.

Use ONLY the provided context to generate test cases.

ERP Context: {erp_context}

Context:
{context}

Feature: {feature}

Instructions:
- Keep everything brief and to the point
- Focus on critical paths

Output format (STRICT):

Test ID: TC001
Scenario: [Brief description]
Steps:
1. [Action]
Expected Result: [Outcome]
Type: [Positive / Negative / Edge Case]

Test ID: TC002
...

Generate 4 brief test cases. Do not include any other markdown or introductory text.
""",
    "Gherkin": """
You are a senior QA engineer.

Use ONLY the provided context to generate test cases.

ERP Context: {erp_context}

Context:
{context}

Feature: {feature}

Output format (STRICT):

Test ID: TC001
Scenario: [Scenario Name]
Steps:
Given [Precondition]
When [Action]
Then [Outcome]
Expected Result: [Outcome]
Type: [Positive / Negative / Edge Case]

Test ID: TC002
...

Generate 4 test cases in Given/When/Then format. Do not include any other markdown or introductory text.
"""
}

def generate_test_cases(feature: str, context: str, erp_system: str = "Generic", prompt_style: str = "Detailed") -> str:
    # Use the model exacted by user. Note: llama3-8b-8192 is an active Groq model
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    erp_context = ERP_CONTEXTS.get(erp_system, ERP_CONTEXTS["Generic"])
    template_str = PROMPT_STYLES.get(prompt_style, PROMPT_STYLES["Detailed"])
    
    prompt = PromptTemplate(
        input_variables=["feature", "context", "erp_context"],
        template=template_str
    )
    
    chain = prompt | llm
    
    response = chain.invoke({
        "feature": feature,
        "context": context,
        "erp_context": erp_context
    })
    
    return response.content.strip()