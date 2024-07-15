from datetime import datetime

class Utils:

    @staticmethod
    def format_message(role, content):
        """
        Format a message with a timestamp and role.

        Args:
            role (str): The role of the message sender (either "User" or "Assistant").
            content (str): The message content.

        Returns:
            str: Formatted message with timestamp and role.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] {role}: {content}"

    @staticmethod
    def update_history(history, role, message):
        """
        Update the conversation history with the latest message.

        Args:
            history (str): The current conversation history.
            role (str): The role of the message sender (either "User" or "Assistant").
            message (str): The message content.

        Returns:
            str: Updated conversation history.
        """
        formatted_message = Utils.format_message(role, message)
        history += f"\n{formatted_message}"
        return history.strip()

