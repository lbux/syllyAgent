import json
from typing import Optional

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import PDFMinerToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.dataclasses import ChatMessage
from haystack.utils import ComponentDevice
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from pydantic import BaseModel

from prompts import reasoning_prompt, splitter_prompt, system_prompt
from utils import (
    generate_final_reasoning_pipeline,
    process_all_questions_per_chunk,
    serialize_chunks_individually,
)


# Define schema
class Question(BaseModel):
    question: str
    answer: Optional[str] = None


class Questions(BaseModel):
    questions: list[Question]


schema_json = Questions.model_json_schema()

# Resolve the best device (currently this is unused since Ollama server handles GPU selection)
device = ComponentDevice.resolve_device()
print(f"Selected device: {device.to_dict()}")

# Step 1: Chunk syllabus into smaller documents
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", PDFMinerToDocument())
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component(
    "splitter",
    DocumentSplitter(split_by="word", split_length=1500, split_overlap=200),
)
indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")

split_documents = indexing_pipeline.run({"converter": {"sources": ["syllabus.pdf"]}})
serialize_chunks_individually(split_documents, "chunks")

# Step 2: Decompose query into sub-questions
system_message = ChatMessage.from_system(system_prompt)
user_message = ChatMessage.from_user(splitter_prompt)

decomposition_pipeline = Pipeline()
decomposition_pipeline.add_component(
    "decomposition_prompt", ChatPromptBuilder([system_message, user_message])
)
decomposition_pipeline.add_component(
    "decomposition_llm",
    OllamaChatGenerator(model="llama3.2:3b", structured_format=schema_json),
)
decomposition_pipeline.connect("decomposition_prompt", "decomposition_llm")

query = "My students often say that my syllabus is too long. However, I feel like there is a lot of important information that I may be missing if I shorten it. Analyze my syllabus to see if I have all that is necessary."
decomposed_result = decomposition_pipeline.run(
    data={"decomposition_prompt": {"query": query}}
)


# Extract decomposed questions
decomposed_replies = decomposed_result["decomposition_llm"]["replies"]
if decomposed_replies:
    content = decomposed_replies[0].content
    decomposed_questions = json.loads(content)["questions"]
else:
    raise ValueError("No decomposed replies found in the response.")

# Step 3: Process each question against syllabus chunks
generator = OllamaChatGenerator(
    model="llama3.2:3b", generation_kwargs={"temperature": 0.7}
)
question_answers = process_all_questions_per_chunk(
    decomposed_questions, "chunks", generator, system_prompt, "logs"
)

# Step 4: Generate final reasoning using ChatPromptBuilder
reasoning_pipeline = generate_final_reasoning_pipeline(
    generator, system_prompt, reasoning_prompt
)

# Prepare data for reasoning
reasoning_data = {"reasoning_prompt": {"question_answers": question_answers}}

# Run the reasoning pipeline
final_recommendation = reasoning_pipeline.run(
    data=reasoning_data, include_outputs_from="reasoning_prompt"
)["llm"]["replies"][0]

print("Final Recommendation:")
print(final_recommendation.content)
