import boto3
import os
import json
from botocore.exceptions import ClientError
from aws_lambda_powertools import Logger
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

logger = Logger()

# Create the S3 client once
s3_client = boto3.client('s3')

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1",
)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-image-v1",
    client=bedrock_runtime,
)

llm = ChatBedrock(
    model_id="meta.llama3-8b-instruct-v1:0",
    model_kwargs={"temperature": 0.1},
    region_name="ap-south-1"
)

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    logger.info("Lambda function invoked.")
    event_body = json.loads(event["body"])
    human_input = event_body["prompt"]
    bucket_name = 'biosarthi-genai'
    folder_path = 'embeddings/'
    file_names = ['index.faiss', 'index.pkl']

    def check_files_and_download(bucket_name, folder_path, file_names):
        for file in file_names:
            try:
                s3_client.head_object(Bucket=bucket_name, Key=f"{folder_path}{file}")
                logger.info(f"File '{folder_path}{file}' exists.")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.error(f"File '{folder_path}{file}' does not exist.")
                    return False

        # If we reach here, all files exist
        logger.info("Files are present")
        temp_dir = '/tmp'

        # Create the directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        for file in file_names:
            s3_client.download_file(bucket_name, f"{folder_path}{file}", os.path.join(temp_dir, file))

        return True

    files_present_and_downloaded = check_files_and_download(bucket_name, folder_path, file_names)
    logger.info(f"All required files are present and downloaded: {files_present_and_downloaded}")

    if not files_present_and_downloaded:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Required files are not present in the S3 bucket."})
        }

    try:
        # Load the FAISS index
        faiss_index = FAISS.load_local("/tmp", embeddings, allow_dangerous_deserialization=True)
        retriever = faiss_index.as_retriever()

        # Define the context and prompt template
        context = '''
        You are a knowledgeable assistant specializing in the field of biogas, renewable energy, and sustainability. 
        Your role is to assist users by providing accurate and insightful answers based on the embedded information 
        from PDF documents related to biogas production, renewable energy technologies, environmental impact, and 
        sustainable practices. When a user asks a question, provide detailed answers, citing relevant data or concepts 
        from the embedded documents. Ensure your responses are clear, concise, and focused on renewable energy 
        solutions, including biogas production, usage, and the future of sustainable energy.
        '''
        template = """
        {context}
        
        Human: {question}
        """

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        # Initialize the QA chain
        qa = RetrievalQA.from_chain_type(
            llm,
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        # Example user query
        user_query = human_input
        response = qa.invoke(user_query)
        logger.info(f"Response: {response}")

        return {
            "statusCode": 200,
            "body": json.dumps({"response": response})
        }

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
