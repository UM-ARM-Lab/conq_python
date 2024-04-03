# Navigation Library

All code pertaining to mapping, localizaing, and navigating with Conq.

## Tool Detector Interface

The [`tool_detector_interface.py`](tool_detector_interface.py) file contains a script for navigating a semantic map to find objects by querying ChatGPT to get a probability of where one will most likely find the object in pre-defined locations in our map.

A `.env` file must be set in the [`nvaigation_lib`](../navigation_lib/) directory with the current fields set:
```bash
OPENAI_API_KEY=<KEY>
API_VERSION=YYYY-MM-DD
openai_api_base=https://api.umgpt.umich.edu/azure-openai-api
OPENAI_organization=<BILLING_SHORTCODE>
model=gpt-35-turbo
```

## Some helpful AzureOpenAI documentation:
- [Azure OpenAI Services page](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference?WT.mc_id=AZ-MVP-5004796#rest-api-versioning) goes over the different fields in the `.env` file.
- [Quick start guide from Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=bash%2Cpython-new&pivots=programming-language-python) for using the OpenAI Python API.
- [Sending chat completions](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python-new&pivots=programming-language-chat-completions)