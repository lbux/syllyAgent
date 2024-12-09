import json
import os

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage


def serialize_chunks_individually(output: dict, directory: str):
    """
    Save each chunk as a separate JSON file.
    """
    documents = output.get("splitter", {}).get("documents", [])
    if not documents:
        raise ValueError("No documents found in the pipeline output.")

    if not os.path.exists(directory):
        os.makedirs(directory)

    for doc in documents:
        file_path = os.path.join(directory, f"chunk_{doc.id}.json")
        with open(file_path, "w") as f:
            json.dump(doc.to_dict(), f)


def process_question_per_chunk(
    question: str, directory: str, llm, system_prompt: str, log_file: str
):
    """
    Process a single question by iterating over individual chunk files and log the output to a file.
    """
    results = []

    # Prepare a dictionary to save structured logs
    log_data = {"question": question, "chunks_processed": []}

    # Iterate over all chunk files
    for chunk_file in sorted(os.listdir(directory)):
        if not chunk_file.endswith(".json"):
            continue

        file_path = os.path.join(directory, chunk_file)

        with open(file_path, "r") as f:
            chunk = json.load(f)

        # Prepare prompt for the LLM
        prompt = f"Question: {question}\n\nContext: {chunk['content']}\n\nAnswer:"

        # Call the LLM
        response = llm.run(
            messages=[
                ChatMessage.from_system(system_prompt),
                ChatMessage.from_user(prompt),
            ]
        )["replies"][0]

        result = {
            "chunk_id": chunk["id"],
            "chunk_content_preview": chunk["content"][
                :100
            ],  # First 100 characters for context
            "prompt_preview": prompt[:200],  # First 200 characters of the prompt
            "response": response.content,
        }
        results.append({"chunk_id": chunk["id"], "answer": response.content})
        log_data["chunks_processed"].append(result)

    # Write the log to a file for debugging
    with open(log_file, "w") as log:
        json.dump(log_data, log, indent=4)

    return results


def process_all_questions_per_chunk(
    decomposed_questions, directory, llm, system_prompt: str, log_dir: str
):
    """
    Process all questions against chunks stored as files, logging each question's results.
    """
    question_answers = {}

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for idx, question in enumerate(decomposed_questions):
        log_file = os.path.join(log_dir, f"question_{idx + 1}.json")
        chunk_results = process_question_per_chunk(
            question["question"], directory, llm, system_prompt, log_file
        )
        aggregated_answer = "\n".join(
            [f"Chunk {res['chunk_id']}: {res['answer']}" for res in chunk_results]
        )
        question_answers[question["question"]] = aggregated_answer

    return question_answers


def generate_final_reasoning_pipeline(llm, system_prompt: str, reasoning_prompt: str):
    """
    Create a pipeline to generate a final reasoning using ChatPromptBuilder.
    """
    reasoning_pipeline = Pipeline()
    system_message = ChatMessage.from_system(system_prompt)
    user_message = ChatMessage.from_user(reasoning_prompt)
    prompt_builder = ChatPromptBuilder([system_message, user_message])
    reasoning_pipeline.add_component("reasoning_prompt", prompt_builder)
    reasoning_pipeline.add_component("llm", llm)
    reasoning_pipeline.connect("reasoning_prompt", "llm")
    return reasoning_pipeline
