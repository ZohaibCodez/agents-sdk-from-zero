from agents.exceptions import AgentsException
from typing import Optional, Dict, Any


class LibraryAgentWorkflowException(AgentsException):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

    def __str__(self):
        return f"{super().__str__()} (Details: {self.details})"


class BookNotFoundError(LibraryAgentWorkflowException):
    def __init__(self, book_id: str):
        message = f"Book with ID '{book_id}' not found."
        details = {"book_id": book_id}
        super().__init__(message, details)


class UserNotRegisteredError(LibraryAgentWorkflowException):
    def __init__(self, user_id: str):
        message = f"User with ID '{user_id}' is not registered."
        details = {"user_id": user_id}
        super().__init__(message, details)


class OverdueBookError(LibraryAgentWorkflowException):
    def __init__(self, book_id: str, overdue_days: int):
        message = f"Book with ID '{book_id}' is overdue by {overdue_days} days."
        details = {"book_id": book_id, "overdue_days": overdue_days}
        super().__init__(message, details)


class LibraryConfigurationError(LibraryAgentWorkflowException):
    def __init__(self, config_issue: str):
        message = f"Library agent configuration error: {config_issue}"
        details = {"config_issue": config_issue}
        super().__init__(message, details)


class LibraryCommunicationError(LibraryAgentWorkflowException):
    def __init__(self, source: str, target: str, reason: str):
        message = f"Communication error from {source} to {target}: {reason}"
        details = {"source": source, "target": target, "reason": reason}
        super().__init__(message, details)


