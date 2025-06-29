from langchain.tools import Tool

def collect_mba_application(data: str) -> str:
    with open("mba_applications.txt", "a", encoding="utf-8") as f:
        f.write(data + "\n\n")
    return "Your MBA application details have been saved. Our team will reach out to you soon."

mba_application_tool = Tool(
    name="ApplyMBA",
    func=collect_mba_application,
    description=(
        "Use this tool when the user wants to apply for the MBA program. "
        "Ask for and collect their name, email, phone number, address, qualification, and age. "
        "Once collected, pass all the data as a single string to this tool."
    )
)
