def generate_answer_forcing_prompt(num_options: int = 2):
    prompt = "Answer with"

    for i in range(num_options - 1):
        prompt += f" ({chr(65+i)}),"

    prompt += f" or ({chr(65+num_options-1)})."

    return prompt
