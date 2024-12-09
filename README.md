SyllyAgent is a tool designed to help instructors analyze and improve their course syllabi. It processes syllabi in PDF format, breaks them into manageable chunks, and evaluates them using custom queries to generate actionable recommendations.

## Features
- **Chunking**: Processes large syllabi into smaller, manageable sections.
- **Query Decomposition**: Breaks down complex queries into simpler sub-questions for focused analysis.
- **Recommendation Generation**: Provides comprehensive recommendations for improving syllabi. [WIP]
- **Customizable AI Models**: Supports Ollama and can be adapted to other models on an Ollama server.

## Requirements
- Python 3.12 or higher
- [UV](https://astral.sh/uv)
- [Ollama](https://ollama.com/download) installed and running

## Installation

### Install UV
On macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Clone the Repository
```bash
git clone https://github.com/lbux/syllyAgent
cd syllyAgent
```

### Set Up the Environment
UV will handle dependencies defined in the `pyproject.toml`:
```bash
uv sync
```

## Usage

### Step 1: Specify the PDF File
Update the source directory in `main.py`:
```python
split_documents = indexing_pipeline.run({"converter": {"sources": ["path/to/your/syllabus.pdf"]}})
```

### Step 2: Ensure Ollama Is Running
Download and install Ollama from [here](https://ollama.com/download) and start the server. By default, SyllyAgent uses the `llama3.2:3b` model.

Download `llama3.2:3b` by running the following command in your terminal
```bash
ollama run llama3.2
```

To modify the model, update the following line in `main.py`:
```python
decomposition_pipeline.add_component(
    "decomposition_llm",
    OllamaChatGenerator(model="your-model-name", structured_format=schema_json),
)
```

### Step 3: Run the Project
Once everything is set up, run the project:
```bash
python main.py
```

The output will include a detailed analysis of your syllabus and actionable recommendations. [WIP]
