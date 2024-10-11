import boto3
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader  
from langchain_aws import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
from aws_lambda_powertools import Logger

logger = Logger()

s3 = boto3.client('s3')

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1",
)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-image-v1",
    client=bedrock_runtime,
)

bucket_name = 'biosarthi-genai'
folder_name = '20241004_101612/knowledgebase/'

def download_files_from_s3(bucket, folder):
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=folder)
        for obj in response.get('Contents', []):
            file_name = obj['Key']
            local_file_path = '/tmp/' + os.path.basename(file_name)
            s3.download_file(bucket, file_name, local_file_path)
            logger.info(f"File '{file_name}' downloaded to '{local_file_path}'")
    except Exception as e:
        logger.error(f"Error downloading files from S3: {e}")

def load_knowledgebase():
    def custom_loader(file_path):
        extension = os.path.splitext(file_path)[1].lower()
    
        if extension == '.pdf':
            return PyPDFLoader(file_path)
        elif extension == '.docx':
            return UnstructuredWordDocumentLoader(file_path)
        elif extension == '.txt':
            return TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    directory_path = "/tmp"
    
    loader = DirectoryLoader(
        directory_path,
        glob="**/*",
        loader_cls=custom_loader,
        use_multithreading=True,
        silent_errors=True
    )

    docs = loader.load()
    return docs

def splitting_embedding(docs, embeddings, bucket_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    docs_split = text_splitter.split_documents(docs)
    
    vectorstore_faiss = FAISS.from_documents(
        docs_split,
        embeddings
    )
    vectorstore_faiss.save_local("/tmp/faiss_index")
    
    # s3.upload_file("/tmp/faiss_index", bucket_name, "embeddings/")
    faiss_index_folder = "/tmp/faiss_index"
    for filename in os.listdir(faiss_index_folder):
        file_path = os.path.join(faiss_index_folder, filename)
        if os.path.isfile(file_path):
            s3.upload_file(file_path, bucket_name, f'embeddings/{filename}')
            print(f'Uploaded {filename} to s3://{bucket_name}/embeddings/{filename}')
    print('Embedding and splitting is completed hoorey')

##lmanda_handler

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    logger.info("Lambda function invoked.")
    
    download_files_from_s3(bucket_name, folder_name)
    loader = load_knowledgebase()
    logger.info("Loading is completed")
    
    splitting_embedding(loader, embeddings, bucket_name)
    logger.info("Successfully embedding is completed")

    # Return the response immediately
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
        },
        # "body": json.dumps(job_id),
    }
