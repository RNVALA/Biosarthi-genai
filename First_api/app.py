import json
import boto3
import uuid
import os

ssm_client = boto3.client('ssm')
lambda_client = boto3.client('lambda')
s3 = boto3.client('s3')
secondary_lambda_arn = os.getenv('REQUEST_FUNCTION_ARN')
print("@",secondary_lambda_arn)


def upload_to_s3(file_content,file_name):
    folder_name="20241004_101612/knowledgebase/"
    bucket_name = 'biosarthi-genai'
    s3_key=f"{folder_name}/{file_name}"
    s3.put_object(Bucket=bucket_name,key=s3_key,Body=file_content)


def invoke_secondary_lambda_async(payload):
    response = lambda_client.invoke(
        FunctionName=secondary_lambda_arn,
        InvocationType='Event',  # Asynchronous invocation
        Payload=json.dumps(payload)
    )
    return response

def lambda_handler(event, context):
    print("event",event)

    
    payload = {
        "event_data": event,
    }

    print("2",payload)
    
    # Invoke the secondary Lambda function asynchronously
    invoke_secondary_lambda_async(payload)
    
    # Return the response immediately
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
        },
        "body": json.dumps("embedding generation started"),
    }