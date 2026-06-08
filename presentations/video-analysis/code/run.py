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
GOOGLE_BUCKET = "css-temp-bucket-for-vertex"

prompt = Path(PROMPT_FILE).read_text()

# Upload video to the cloud (skip if already there)
client = storage.Client(project=GOOGLE_PROJECT)
cloud_file = client.bucket(GOOGLE_BUCKET).blob(VIDEO_FILE)
if not cloud_file.exists():
    cloud_file.upload_from_filename(VIDEO_FILE)

# Assemble LLM request
client = genai.Client(vertexai=True, project=GOOGLE_PROJECT, location="global")

video_content = Part.from_uri(
    file_uri=f"gs://{GOOGLE_BUCKET}/{VIDEO_FILE}",
    mime_type="video/mp4",
)
text_content = Content(role="user", parts=[Part.from_text(text=prompt)])

# Execute request
print("=== PROMPT ===")
print(prompt)

print("=== ANSWER ===")
for chunk in client.models.generate_content_stream(
    model=MODEL,
    contents=[video_content, text_content],
):
    print(chunk.text, end="", flush=True)
print()
