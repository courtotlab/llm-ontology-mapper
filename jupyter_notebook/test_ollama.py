# import os
# from ollama import Client

# client = Client(
#     host='https://ollama.com',
#     headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
# )

# messages = [
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ]

# for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
#   print(part.message.content, end='', flush=True)


from llm_ontology_mapper import OntologyMapper

# Ollama cloud
mapper = OntologyMapper(
    provider="ollama",
    model="gpt-oss:120b",
    base_url="https://ollama.com",   # or your remote host
    api_key="bf5b6a1b104b4eb4bdb378e56a33fe78.dtxOcwlRzusfvONAU7m5d2nw",
)

result = mapper.map_term(
    source_term="cough",
    source_label="Does the patient have a cough?",
    entity_type="phenotype",
)

print(result.target_code)   # HP:0012735
print(result.target_term)   # Cough
print(result.confidence)    # 0.94
print(result.notes)         # LLM reasoning