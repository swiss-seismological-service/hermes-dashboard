def pprint_snake(text: str) -> str:
    """
    Pretty Print a snake_case string into a Title Case string
    """
    return " ".join([word.capitalize() for word in text.split("_")])
