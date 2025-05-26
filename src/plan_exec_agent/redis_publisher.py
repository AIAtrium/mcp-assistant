import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Add Redis import with optional handling
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisPublisher:
    """Utility class for publishing plan execution events to Redis streams."""
    
    def __init__(self):
        self._redis_client: Optional[redis.Redis] = None
        if self._should_publish_to_redis():
            self._init_redis_client()

    def _should_publish_to_redis(self) -> bool:
        """Check if Redis publishing is enabled via environment variable."""
        return os.getenv("PUBLISH_TO_REDIS", "false").lower() in ("true", "1", "yes")

    def _init_redis_client(self) -> None:
        """Initialize Redis client for publishing."""
        if not REDIS_AVAILABLE:
            print("Warning: Redis not available. Install redis-py to enable publishing.")
            return
        
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_db = int(os.getenv("REDIS_DB", "0"))
            
            self._redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self._redis_client.ping()
            print(f"Redis client initialized successfully (host: {redis_host}:{redis_port})")
        except Exception as e:
            print(f"Failed to initialize Redis client: {e}")
            self._redis_client = None

    def _prepare_state_for_publishing(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare state for Redis publishing by removing heavy/sensitive data.
        
        Args:
            state: The full state dictionary
            
        Returns:
            A cleaned state dictionary suitable for publishing
        """
        # Create a copy and remove heavy/sensitive keys
        cleaned_state = state.copy()
        
        # Remove heavy data that could make the message too large
        keys_to_remove = ["tool_results", "tools"]
        for key in keys_to_remove:
            cleaned_state.pop(key, None)
        
        # Convert any non-serializable objects to strings
        if "provider" in cleaned_state:
            cleaned_state["provider"] = str(cleaned_state["provider"])
        
        # Add timestamp for when this was published
        cleaned_state["published_at"] = datetime.utcnow().isoformat()
        
        return cleaned_state

    def publish_event(
        self, 
        event_type: str, 
        state: Dict[str, Any],
        stream_name: Optional[str] = None
    ) -> None:
        """
        Publish state to Redis stream.
        
        Args:
            event_type: Type of event (e.g., "initial_plan", "final_result")
            state: State dictionary to publish
            stream_name: Optional stream name override
        """
        try:
            # Use provided stream name or default from env
            stream_name = stream_name or os.getenv("REDIS_STREAM_NAME", "plan_execution")
            
            # Prepare the state for publishing
            cleaned_state = self._prepare_state_for_publishing(state)
            
            # Create the message payload
            message = {
                "event_type": event_type,
                "session_id": state.get("langfuse_session_id", "unknown"),
                "user_id": state.get("user_id", "unknown"),
                "task_id": state.get("task_id"),
                "data": json.dumps(cleaned_state)
            }
            
            # Publish to Redis stream
            message_id = self._redis_client.xadd(stream_name, message)
            print(f"Published {event_type} to Redis stream '{stream_name}' with ID: {message_id}")
            
        except Exception as e:
            print(f"Failed to publish to Redis stream: {e}")

    def is_enabled(self) -> bool:
        """Check if Redis publishing is enabled and client is available."""
        return self._should_publish_to_redis() and self._redis_client is not None