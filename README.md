# whisper-lagoon

whisper API running on lagoon

## Local
Use docker compose to build and start the API.  

```sh
docker compose build
```

```sh
docker compose up -d
```

## Curl Test API (identical to OpenAI Whisper API example)
Note: Authorization header is currently ignored.

```sh
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F model="whisper-ch" \
  -F file="@/path/to/file/openai.mp3"
```