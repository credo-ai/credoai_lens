def create_title(title: str):
    output = f"\n{title.capitalize()}\n=======\n"
    return output


if __name__ == "__main__":
    with open("./metrics_v2.rst", "a") as text_file:
        text_file.write(create_title("Testing"))
        text_file.write("This is all just a test")
