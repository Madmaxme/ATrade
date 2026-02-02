import boto3
import os
from dotenv import load_dotenv

def check_aws_permissions():
    load_dotenv()
    
    region = os.getenv("AWS_REGION", "us-east-1")
    print(f"Checking AWS credentials in region: {region}")
    
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ Success: Authenticated as {identity['Arn']}")
    except Exception as e:
        print(f"❌ Error: Could not authenticate with AWS: {e}")
        return

    try:
        bedrock = boto3.client('bedrock', region_name=region)
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        
        # Try to list foundation models to see if API access is generally working
        print("\nChecking Bedrock Model Access...")
        models = bedrock.list_foundation_models(
            byProvider='anthropic'
        )
        print(f"✅ Success: Can list foundation models.")
        
        # Specific model to test (from config.py)
        model_id = "us.anthropic.claude-opus-4-5-20251101-v1:0"
        print(f"\nTesting invocation for model: {model_id}")
        
        try:
            # Attempt a minimal invocation
            import json
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}]
            })
            
            bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            print(f"✅ Success: Model {model_id} is accessible and functional.")
        except Exception as e:
            print(f"❌ Error: Model invocation failed: {e}")
            if "INVALID_PAYMENT_INSTRUMENT" in str(e):
                print("\n🚨 CRITICAL BILLING ISSUE DETECTED 🚨")
                print("Your AWS account has a payment instrument issue blocking Marketplace subscriptions.")
                print("Resolution Steps:")
                print("1. Log into AWS Console -> Billing and Cost Management.")
                print("2. Check for overdue invoices or expired credit cards.")
                print("3. Ensure 'Anthropic' model access is granted in the Bedrock console.")
                print("4. Check if your account requires a Marketplace subscription confirmation for Anthropic.")
            
    except Exception as e:
        print(f"❌ Error during Bedrock check: {e}")

if __name__ == "__main__":
    check_aws_permissions()
