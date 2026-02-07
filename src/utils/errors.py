"""Custom exceptions for audit system"""


class AuditSystemError(Exception):
    """Base exception for audit system errors"""
    pass


class DatabaseError(AuditSystemError):
    """Database operation errors"""
    pass


class DatabricksConnectionError(AuditSystemError):
    """Databricks connection errors"""
    pass


class LLMError(AuditSystemError):
    """LLM API errors"""
    pass


class StateManagerError(AuditSystemError):
    """State management errors"""
    pass


class ConfigurationError(AuditSystemError):
    """Configuration loading errors"""
    pass


class AgentExecutionError(AuditSystemError):
    """Agent execution errors"""
    pass
