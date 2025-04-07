from pydantic import BaseModel
from ollama import chat

class Response(BaseModel):
    short_reason: str
    result: str

def compose_prompt(prompt_template, input_text):
    """Format the prompt using the provided input text."""
    return prompt_template.format(input_text=input_text)

def compose_messages(prompt_template, input_text):
    return [{
                "role": "user",
                "content": compose_prompt(prompt_template, input_text)
            },
            # {
            #     "role": "user",
            #     "content": f"Output results in the following format \n {Response.model_json_schema()}"
            # }
        ]

def ask_llm(clients, client_type, model, prompt_template, input_text):

    client = clients[client_type]

    if client_type == "openai":
        response = client.beta.chat.completions.parse(
            model=model,
            messages=compose_messages(prompt_template, input_text),
            # response_format=Response,
            temperature=0
        )
        # response_content = response.choices[0].message.parsed.result
        response_content = response.choices[0].message.content

    elif client_type == "ollama":
        response = chat(
            model=model,
            messages=compose_messages(prompt_template, input_text),
            # format=Response.model_json_schema(),
            options={'temperature': 0}, 
        )
        # response_content = Response.model_validate_json(response.message.content).result
        response_content = response.message.content
    return response_content

