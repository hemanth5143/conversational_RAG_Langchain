## Setup Instructions

To run this project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/hemanth5143/conversational_RAG_Langchain.git
   cd conversational_RAG_Langchain
   ```

2. Pull the Docker image:
   ```
   docker pull hemanth5143/mindbot:latest
   ```

3. Create a `.env` file in the root directory of the project with your Google API key:
   ```
   Google_api_key="YOUR_ACTUAL_API_KEY" 
   ```
   Replace `YOUR_ACTUAL_API_KEY` with your Google API key.

4. Run the Docker container:
   ```
   docker-compose up
   ```

## Important Notes

- Ensure Docker is installed and running on your system before executing these commands.
- Keep your API key confidential and do not share it publicly.
- If you encounter any issues, make sure your `.env` file is correctly formatted and placed in the project's root directory.

## Troubleshooting

If you experience any problems with the `docker-compose up` command, you can try running the container directly using:

```
docker run -p 8080:8080 --env-file .env hemanth5143/mindbot:latest
```

This command explicitly uses the `.env` file and maps port 8080.

For any further assistance, please open an issue in the GitHub repository.

Citations:
[1] https://github.com/hemanth5143/conversational_RAG_Langchain.git