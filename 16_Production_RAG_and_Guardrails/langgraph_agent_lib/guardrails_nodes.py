"""Guardrails validation nodes for LangGraph agent integration."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from typing_extensions import TypedDict, Annotated

from guardrails.hub import (
    RestrictToTopic,
    DetectJailbreak, 
    CompetitorCheck,
    ProfanityFree,
    GuardrailsPII
)
from guardrails import Guard


logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result types."""
    PASSED = "passed"
    FAILED = "failed" 
    FIXED = "fixed"
    BLOCKED = "blocked"


class GuardrailsState(TypedDict):
    """Extended state schema with guardrails information."""
    messages: Annotated[List[BaseMessage], add_messages]
    validation_results: Dict[str, Any]
    guard_failures: List[str]
    refinement_count: int
    max_refinements: int


class GuardrailsValidator:
    """Production guardrails validator with comprehensive safety checks."""
    
    def __init__(
        self,
        allowed_topics: List[str] = None,
        competitors: List[str] = None,
        pii_entities: List[str] = None,
        max_refinements: int = 3
    ):
        """Initialize guardrails validator.
        
        Args:
            allowed_topics: List of allowed discussion topics
            competitors: List of competitor names to filter
            pii_entities: List of PII entity types to detect
            max_refinements: Maximum number of refinement attempts
        """
        self.allowed_topics = allowed_topics or ["student loans", "financial aid", "education"]
        self.competitors = competitors or ["OpenAI", "Anthropic", "Google", "Microsoft"]
        self.pii_entities = pii_entities or ["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]
        self.max_refinements = max_refinements
        
        self._setup_guards()
    
    def _setup_guards(self):
        """Set up individual guardrails guards."""
        try:
            # Input validation guards
            self.topic_guard = Guard().use(
                RestrictToTopic(
                    valid_topics=self.allowed_topics,
                    disable_classifier=True,
                    disable_llm=False,
                    on_fail="exception"
                )
            )
            logger.info("✓ Topic restriction guard configured")
            
            self.jailbreak_guard = Guard().use(
                DetectJailbreak(on_fail="exception")
            )
            logger.info("✓ Jailbreak detection guard configured")
            
            self.pii_input_guard = Guard().use(
                GuardrailsPII(
                    entities=self.pii_entities,
                    use_local=False,  # Use cloud-based to avoid tokenizer issues
                    on_fail="exception"
                )
            )
            logger.info("✓ PII input detection guard configured")
            
            # Output validation guards
            self.profanity_guard = Guard().use(
                ProfanityFree(on_fail="fix")
            )
            logger.info("✓ Profanity filter guard configured")
            
            self.competitor_guard = Guard().use(
                CompetitorCheck(
                    competitors=self.competitors,
                    on_fail="filter"
                )
            )
            logger.info("✓ Competitor check guard configured")
            
            self.pii_output_guard = Guard().use(
                GuardrailsPII(
                    entities=self.pii_entities,
                    use_local=False,
                    on_fail="fix"
                )
            )
            logger.info("✓ PII output protection guard configured")
            
            # Note: Advanced evaluators like LlmRagEvaluator require complex setup
            # For now, we rely on the other guards for content validation
            logger.info("✓ Content validation guards configured")
            
        except Exception as e:
            logger.error(f"Failed to setup guards: {e}")
            raise
    
    def validate_input(self, user_input: str) -> Tuple[ValidationResult, str, List[str]]:
        """Validate user input against safety policies.
        
        Args:
            user_input: The user's input message
            
        Returns:
            Tuple of (result, processed_input, failure_reasons)
        """
        failures = []
        processed_input = user_input
        
        try:
            # 1. Check for jailbreak attempts
            try:
                self.jailbreak_guard.validate(user_input)
                logger.info("✓ Jailbreak validation passed")
            except Exception as e:
                logger.warning(f"Jailbreak detection failed: {e}")
                failures.append(f"Jailbreak attempt detected: {str(e)}")
                return ValidationResult.BLOCKED, processed_input, failures
            
            # 2. Check topic relevance
            try:
                self.topic_guard.validate(user_input)
                logger.info("✓ Topic validation passed")
            except Exception as e:
                logger.warning(f"Topic validation failed: {e}")
                failures.append(f"Off-topic query: {str(e)}")
                return ValidationResult.BLOCKED, processed_input, failures
            
            # 3. Check for PII in input
            try:
                validated_input = self.pii_input_guard.validate(user_input)
                if hasattr(validated_input, 'validated_output'):
                    processed_input = validated_input.validated_output
                logger.info("✓ Input PII validation passed")
            except Exception as e:
                logger.warning(f"PII input validation failed: {e}")
                failures.append(f"PII detected in input: {str(e)}")
                return ValidationResult.BLOCKED, processed_input, failures
            
            return ValidationResult.PASSED, processed_input, failures
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            failures.append(f"Validation system error: {str(e)}")
            return ValidationResult.FAILED, processed_input, failures
    
    def validate_output(self, output: str, context: str = "") -> Tuple[ValidationResult, str, List[str]]:
        """Validate agent output for safety and quality.
        
        Args:
            output: The agent's output message
            context: Additional context for validation
            
        Returns:
            Tuple of (result, processed_output, failure_reasons)
        """
        failures = []
        processed_output = output
        
        try:
            # 1. Remove profanity
            try:
                validated_output = self.profanity_guard.validate(processed_output)
                if hasattr(validated_output, 'validated_output'):
                    processed_output = validated_output.validated_output
                logger.info("✓ Profanity filter applied")
            except Exception as e:
                logger.warning(f"Profanity filtering failed: {e}")
                failures.append(f"Content moderation issue: {str(e)}")
            
            # 2. Filter competitor mentions
            try:
                validated_output = self.competitor_guard.validate(processed_output)
                if hasattr(validated_output, 'validated_output'):
                    processed_output = validated_output.validated_output
                logger.info("✓ Competitor filtering applied")
            except Exception as e:
                logger.warning(f"Competitor filtering failed: {e}")
                failures.append(f"Competitor mention detected: {str(e)}")
            
            # 3. Remove PII from output
            try:
                validated_output = self.pii_output_guard.validate(processed_output)
                if hasattr(validated_output, 'validated_output'):
                    processed_output = validated_output.validated_output
                logger.info("✓ Output PII protection applied")
            except Exception as e:
                logger.warning(f"PII output protection failed: {e}")
                failures.append(f"PII detected in output: {str(e)}")
            
            # 4. Additional content validation could be added here
            # For now, we rely on the core guards above
            logger.info("✓ Output validation completed")
            
            result = ValidationResult.FIXED if failures else ValidationResult.PASSED
            return result, processed_output, failures
            
        except Exception as e:
            logger.error(f"Output validation error: {e}")
            failures.append(f"Validation system error: {str(e)}")
            return ValidationResult.FAILED, processed_output, failures


def create_input_validation_node(validator: GuardrailsValidator):
    """Create input validation node for LangGraph."""
    
    def validate_input_node(state: GuardrailsState) -> Dict[str, Any]:
        """Validate incoming user input."""
        messages = state["messages"]
        
        if not messages:
            return {"validation_results": {"input": "no_messages"}}
        
        # Get the last human message
        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return {"validation_results": {"input": "not_human_message"}}
        
        user_input = last_message.content
        logger.info(f"Validating input: {user_input[:100]}...")
        
        # Validate input
        result, processed_input, failures = validator.validate_input(user_input)
        
        validation_info = {
            "input": {
                "result": result.value,
                "original": user_input,
                "processed": processed_input,
                "failures": failures
            }
        }
        
        # Update state
        updates = {
            "validation_results": validation_info,
            "guard_failures": failures if result == ValidationResult.BLOCKED else []
        }
        
        # If input was processed/cleaned, update the message
        if processed_input != user_input:
            updated_message = HumanMessage(content=processed_input)
            messages = messages[:-1] + [updated_message]
            updates["messages"] = messages
        
        logger.info(f"Input validation result: {result.value}")
        return updates
    
    return validate_input_node


def create_output_validation_node(validator: GuardrailsValidator):
    """Create output validation node for LangGraph."""
    
    def validate_output_node(state: GuardrailsState) -> Dict[str, Any]:
        """Validate agent output before returning to user."""
        messages = state["messages"]
        
        if not messages:
            return {"validation_results": {"output": "no_messages"}}
        
        # Get the last AI message
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return {"validation_results": {"output": "not_ai_message"}}
        
        output = last_message.content
        logger.info(f"Validating output: {output[:100]}...")
        
        # Extract context from conversation history
        context = ""
        for msg in messages[:-1]:
            if isinstance(msg, HumanMessage):
                context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"Assistant: {msg.content}\n"
        
        # Validate output
        result, processed_output, failures = validator.validate_output(output, context)
        
        validation_info = state.get("validation_results", {})
        validation_info["output"] = {
            "result": result.value,
            "original": output,
            "processed": processed_output,
            "failures": failures
        }
        
        # Update state
        updates = {
            "validation_results": validation_info,
            "guard_failures": state.get("guard_failures", []) + failures
        }
        
        # If output was processed/cleaned, update the message
        if processed_output != output:
            updated_message = AIMessage(content=processed_output)
            messages = messages[:-1] + [updated_message]
            updates["messages"] = messages
        
        logger.info(f"Output validation result: {result.value}")
        return updates
    
    return validate_output_node


def create_refinement_node(validator: GuardrailsValidator):
    """Create refinement node for handling validation failures."""
    
    def refinement_node(state: GuardrailsState) -> Dict[str, Any]:
        """Handle validation failures with refinement attempts."""
        refinement_count = state.get("refinement_count", 0)
        max_refinements = state.get("max_refinements", validator.max_refinements)
        failures = state.get("guard_failures", [])
        
        logger.info(f"Refinement attempt {refinement_count + 1}/{max_refinements}")
        
        if refinement_count >= max_refinements:
            # Maximum refinements reached, return error message
            error_message = AIMessage(
                content="I apologize, but I cannot provide a safe response to your request. "
                       "Please rephrase your question or ask about a different topic."
            )
            return {
                "messages": [error_message],
                "validation_results": {"refinement": "max_attempts_reached"},
                "refinement_count": refinement_count + 1
            }
        
        # Create refinement prompt based on failures
        refinement_prompt = "Please revise your response to address these issues:\n"
        for failure in failures:
            refinement_prompt += f"- {failure}\n"
        refinement_prompt += "\nProvide a safe, appropriate, and on-topic response."
        
        refinement_message = HumanMessage(content=refinement_prompt)
        
        return {
            "messages": [refinement_message],
            "validation_results": {"refinement": "attempting"},
            "refinement_count": refinement_count + 1,
            "guard_failures": []  # Clear failures for retry
        }
    
    return refinement_node


def should_validate_input(state: GuardrailsState) -> str:
    """Route decision for input validation."""
    validation_results = state.get("validation_results", {})
    input_result = validation_results.get("input", {}).get("result")
    
    if input_result == ValidationResult.BLOCKED.value:
        return "blocked"
    elif input_result == ValidationResult.FAILED.value:
        return "error"
    else:
        return "continue"


def should_validate_output(state: GuardrailsState) -> str:
    """Route decision for output validation."""
    validation_results = state.get("validation_results", {})
    output_result = validation_results.get("output", {}).get("result")
    guard_failures = state.get("guard_failures", [])
    
    if output_result == ValidationResult.FAILED.value and guard_failures:
        return "refine"
    else:
        return "complete"


def should_refine(state: GuardrailsState) -> str:
    """Route decision for refinement."""
    refinement_count = state.get("refinement_count", 0)
    max_refinements = state.get("max_refinements", 3)
    
    if refinement_count >= max_refinements:
        return "complete"
    else:
        return "retry"