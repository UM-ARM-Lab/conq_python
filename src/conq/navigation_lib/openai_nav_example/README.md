# OpenAI Navigation Example

This script is an emaple of how to use the OpenAI Client in Python and to go from asking for an object to getting probabilities of where that object will be found. Please make sure to have a `.env` in the [`openai`](../openai/) directory containing the fields described in the [Tool Detector README.md](../README.md).

```bash
python3 openai_nav_example.py
```

# OpenAI Vision Example

### **NOTE:** This example is a work in progress and serves primarily as an example/playground for the ChatGPT vision service.

This script allows users to test out providing prompts to ChatGPT to use it to describe what is in an image. Currently, links to two images of various garden tools are passed to ChatGPT with the user prompt in [`example_input.json`](./example_input.json). To try different prompts, either enter a prompt into the input or set the `user_prompt` in the JSON file. To use different images, add or remove links under the `image_inputs` key. More information on using images with ChatGPT can be found on the [OpenAI Documentation](https://platform.openai.com/docs/guides/vision) page.

To run the script, run:
```bash
python3 openai_vision_example.py --json-filepath example_input.json
```

### Prompts
When prompted, enter your prompt and see the results in real time.

* `q`: quit the program
* `j`: use the user prompt in the JSON file provided
* Any other input will be interpreted as the prompt to ChatGPT