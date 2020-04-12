import json
from boto3 import Session
session = Session(profile_name='tempusdevops-nlp-team-modeltrainers', region_name="us-east-1")
client = session.client('logs')

response = client.get_log_events(
    logGroupName='/aws/sagemaker/TrainingJobs',
    logStreamName='tempus-sage-bootcamp-MRFT-HA-2020-03-20-16-58-11/algo-1-1584723597'
    # startTime=123,
    # endTime=123,
    # nextToken='string',
    # limit=123,
    # startFromHead=True|False
)

print(json.dumps(response))
