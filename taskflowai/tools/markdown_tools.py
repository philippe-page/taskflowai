import os

class MarkdownTools:
    @staticmethod
    def write_markdown(file_path: str, content: str) -> str:
        """
        Write content to a markdown file.
        
        Args:
            file_path (str): The path to the file to write to.
            content (str): The content to write to the file.
        Returns:
            str: Confirmation with the path to the file that was written to.
        """
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'w') as file:
            file.write(content)
        
        # Get the absolute path
        abs_path = os.path.abspath(file_path)
        return f"Markdown file written to {abs_path}"
