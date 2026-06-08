# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-genai",
#   "google-cloud-storage",
# ]
# ///
from pathlib import Path
from google import genai
from google.genai.types import Part, Content
from google.cloud import storage

PROMPT_FILE = "prompt.txt"
VIDEO_FILE = "sample-intolerance-video.mp4"
MODEL = "gemini-3.5-flash"
GOOGLE_PROJECT = "css-lehrbereich-schwemmer"
GOOGLE_FOLDER = "css-temp-bucket-for-vertex"

prompt = Path(PROMPT_FILE).read_text()

# Upload video to the cloud (skip if already there)
client = storage.Client(project=GOOGLE_PROJECT)
cloud_file = client.bucket(GOOGLE_FOLDER).blob(VIDEO_FILE)
if not cloud_file.exists():
    cloud_file.upload_from_filename(VIDEO_FILE)

# Assemble LLM request
client = genai.Client(vertexai=True, project=GOOGLE_PROJECT, location="global")

prompt_content = Content(role="user", parts=[Part.from_text(text=prompt)])
video_content = Part.from_uri(
    file_uri=f"gs://{GOOGLE_FOLDER}/{VIDEO_FILE}",
    mime_type="video/mp4",
)

# Call the LLM
answer = client.models.generate_content_stream(
    model=MODEL,
    contents=[video_content, prompt_content],
)

# Show prompt and answer
print("=== PROMPT ===")
print(prompt)

print("=== ANSWER ===")
for chunk in answer:
    print(chunk.text, end="", flush=True)
print()
