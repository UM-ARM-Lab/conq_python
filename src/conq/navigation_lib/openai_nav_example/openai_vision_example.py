from conq.navigation_lib.openai_nav.openai_nav_client import OpenAINavClient

if __name__ == "__main__":
    client = OpenAINavClient(locations=[])

    # Creating while loop to debug with
    while True:
        user_prompt = input("Enter prompt, (q) to quit: ")
        if user_prompt == "q":
            break
        else:
            print("First image processing...")
            client.generate_label_for_image(prompt=user_prompt, image_url="https://www.fromhousetohome.com/garden/wp-content/uploads/sites/2/2022/05/garden-tool-storage-ideas-3.jpg")
            print("Second image processing...")
            client.generate_label_for_image(prompt=user_prompt, image_url="https://img.hobbyfarms.com/wp-content/uploads/2009/02/18104206/harvest-tools_624.jpg")