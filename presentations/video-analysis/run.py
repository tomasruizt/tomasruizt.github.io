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

print("=== PROMPT ===")
print(prompt)

# Upload video to the cloud
sclient = storage.Client(project=GOOGLE_PROJECT)
blob = sclient.bucket(GOOGLE_BUCKET).blob(VIDEO_FILE)
blob.upload_from_filename(VIDEO_FILE)


video_part = Part.from_uri(
    file_uri=f"gs://{GOOGLE_BUCKET}/{VIDEO_FILE}",
    mime_type="video/mp4",
)
text_content = Content(role="user", parts=[Part.from_text(text=prompt)])

client = genai.Client(vertexai=True, project=GOOGLE_PROJECT, location="global")
response = client.models.generate_content(
    model=MODEL,
    contents=[video_part, text_content],
)

candidate = response.candidates[0]
answer = "".join(p.text for p in candidate.content.parts)

print("=== ANSWER ===")
print(answer)
