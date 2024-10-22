from typing import Dict
import random
import time
import json


def generate_text(prompt: str, credentials: Dict, with_code=False, model="gpt-4-0613") -> str:
    if model == 'gemini-pro':
        assert with_code == False, 'Gemini does not support code interpreter'
        return generate_text(prompt, credentials['gemini_project_id'], credentials['gemini_project_location'])
    return generate_text_openai(prompt, model, credentials['openai_api_key'], with_code)




def generate_text_openai(prompt: str, model: str, api_key: str, with_code=False) -> str:
    from openai import OpenAI
    client = OpenAI(api_key = api_key)
    if with_code == False:
        if model == 'o1-preview':
            # does not support system message
            messages = [
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        llm_answer = client.chat.completions.create(
            model=model,
            messages=messages,
        ).choices[0].message.content
    else:
        assert model!='o1-preview', 'OpenAI API for O1 Preview does not support code interpreter'
        gpt_assistant = client.beta.assistants.create(
            name="InvestigateGPT4" + str(random.randint(0, 1000000)),
            description="You are a helpful assistant using code to solve your tasks.",
            model=model,
            tools=[{"type": "code_interpreter"}]
        )
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=gpt_assistant.id,
        )
        while run.status == "queued" or run.status == "in_progress":
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            time.sleep(1)
        def pretty_print(messages):
            print("# Messages")
            for m in messages:
                print(f"{m.role}: {m.content[0].text.value}")
            print()
        response = client.beta.threads.messages.list(thread_id=thread.id, order="asc").data
        run_steps = client.beta.threads.runs.steps.list(
            thread_id=thread.id, run_id=run.id, order="asc"
        )
        def show_json(obj):
            print(json.loads(obj.model_dump_json()))
        for step in run_steps.data:
            step_details = step.step_details
            print(json.dumps(show_json(step_details), indent=4))
        client.beta.threads.delete(thread.id)
        pretty_print(response)
        llm_answer = []
        for m in response:
            llm_answer.append(m.content[0].text.value)
        llm_answer = '\nMESSAGE\n'.join(llm_answer)
        def retrieve_code_calls(run_steps):
            run_data = run_steps.data
            code_inputs = []
            for step in run_data:
                step_type = step.type
                if step_type == "tool_calls":
                    code_input = step.step_details.tool_calls[0].code_interpreter.input
                    code_inputs.append('\n\nCODE INTERPRETER CALL:\n' + code_input + '\nCODE INTERPRETER CALL END\n\n')
            return code_inputs
        code_calls = retrieve_code_calls(run_steps)
        llm_answer = ''.join(code_calls) + llm_answer
        def cleanup():
            try:
                gpt_assistant_id = gpt_assistant.id
                client.beta.assistants.delete(gpt_assistant_id)
            except Exception as e:
                print(f'Error: {e}')
        cleanup()
    return llm_answer


    
def generate_text_gemini(prompt: str, project_id: str, location: str) -> str: # type: ignore
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
    time.sleep(15) # to avoid heavy rate limiting
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    # Load the model
    model = GenerativeModel("gemini-pro")
    # Query the model
    response = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 1, #32
        },
        stream=False,
    )
    # return response.text # type: ignore
    parts = response.to_dict()['candidates'][0]['content']['parts']
    assert len(parts) == 1
    return parts[0]['text']
