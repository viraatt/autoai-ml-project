from ibm_watson_machine_learning import APIClient

wml_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "cpd-apikey-IBMid-6970015JQV-2025-09-30T17:27:56Z"
}

client = APIClient(wml_credentials)
print("✅ Connected to IBM Watson Machine Learning!")