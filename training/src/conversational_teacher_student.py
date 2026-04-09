"""
Conversational Teacher-Student Training - BOSS LEVEL

This implements a conversational approach where:
1. Teacher (Qwen3.5) generates detailed explanations
2. Student asks clarifying questions
3. Teacher elaborates
4. This dialogue trains the student model
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer

from training.src.teacher_student import (
    TeacherStudentConfig, 
    TeacherModel, 
    TeacherStudentTrainer,
    create_teacher_student_trainer
)

logger = logging.getLogger(__name__)


class ConversationalTeacherStudentConfig(TeacherStudentConfig):
    """Extended config for conversational teacher-student training."""
    
    def __init__(
        self,
        teacher_model_name: str = "Qwen/Qwen3.5-32B-Instruct-AWQ",
        temperature: float = 1.5,
        alpha: float = 0.85,
        use_kl_divergence: bool = False,
        teacher_cache_dir: Optional[str] = None,
        max_teacher_sequence_length: int = 4096,
        teacher_batch_size: int = 1,
        # Conversational settings
        conversational_mode: bool = True,
        dialogue_rounds: int = 3,
        generate_explanations: bool = True,
        explanation_depth: str = "comprehensive",
        include_reasoning_steps: bool = True,
        student_feedback: bool = True,
        teacher_elaboration: bool = True,
        # Multi-phase training
        use_phased_training: bool = True,
        phases: Optional[List[Dict]] = None,
    ):
        super().__init__(
            teacher_model_name=teacher_model_name,
            temperature=temperature,
            alpha=alpha,
            use_kl_divergence=use_kl_divergence,
            teacher_cache_dir=teacher_cache_dir,
            max_teacher_sequence_length=max_teacher_sequence_length,
            teacher_batch_size=teacher_batch_size,
        )
        self.conversational_mode = conversational_mode
        self.dialogue_rounds = dialogue_rounds
        self.generate_explanations = generate_explanations
        self.explanation_depth = explanation_depth
        self.include_reasoning_steps = include_reasoning_steps
        self.student_feedback = student_feedback
        self.teacher_elaboration = teacher_elaboration
        self.use_phased_training = use_phased_training
        self.phases = phases or [
            {"name": "warmup", "epochs": 1, "alpha": 0.9, "lr_multiplier": 1.0},
            {"name": "dialogue", "epochs": 2, "alpha": 0.8, "lr_multiplier": 0.8},
            {"name": "mastery", "epochs": 2, "alpha": 0.7, "lr_multiplier": 0.5},
        ]


class ConversationalTeacherModel(TeacherModel):
    """Teacher model with conversational capabilities."""
    
    def __init__(self, config: ConversationalTeacherStudentConfig, device: str = "auto"):
        self.conversation_history: List[Dict[str, str]] = []
        super().__init__(config, device)
    
    def generate_detailed_explanation(
        self,
        problem: str,
        answer: str,
        depth: str = "comprehensive"
    ) -> str:
        """Generate a detailed explanation from the teacher."""
        
        # Construct the prompt based on depth
        if depth == "brief":
            prompt = f"""Problem: {problem}

Provide a concise explanation of the solution.
Answer: {answer}"""
        elif depth == "detailed":
            prompt = f"""Problem: {problem}

Provide a detailed step-by-step solution with:
1. Mathematical reasoning for each step
2. Key concepts used
3. Final answer verification

Answer: {answer}"""
        else:  # comprehensive
            prompt = f"""You are an expert math tutor. Given the problem below, provide:

1. **Problem Analysis**: Identify key concepts and approach
2. **Step-by-Step Solution**: Detailed mathematical reasoning
3. **Alternative Methods**: If applicable, show different approaches
4. **Edge Cases**: Discuss variations or special cases
5. **Verification**: Prove the answer is correct
6. **Summary**: Key takeaways

Problem: {problem}

Correct Answer: {answer}

Please format your response with clear sections and use LaTeX for math expressions. End with the final answer in \\boxed{{}} format."""

        # Generate the explanation
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt
        explanation = generated_text[len(prompt):].strip()
        
        return explanation
    
    def respond_to_question(self, question: str, context: str) -> str:
        """Teacher responds to student's clarifying question."""
        
        prompt = f"""Context from previous explanation:
{context}

Student asks: {question}

As the teacher, provide a clear, helpful response that deepens understanding:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def generate_conversational_training_data(
        self,
        problem: str,
        answer: str,
        num_rounds: int = 3
    ) -> List[Dict[str, str]]:
        """Generate a full conversational dialogue for training."""
        
        conversation = []
        
        # Round 1: Initial detailed explanation
        explanation = self.generate_detailed_explanation(problem, answer, "comprehensive")
        conversation.append({
            "role": "teacher",
            "content": explanation,
            "turn": 1
        })
        
        context = explanation
        
        # Round 2: Student asks follow-up
        if self.config.student_feedback and num_rounds >= 2:
            student_question = "Can you explain the key insight or trick that makes this solution work?"
            conversation.append({
                "role": "student",
                "content": student_question,
                "turn": 2
            })
            
            if self.config.teacher_elaboration:
                elaboration = self.respond_to_question(student_question, context)
                conversation.append({
                    "role": "teacher",
                    "content": elaboration,
                    "turn": 2
                })
                context += f"\n\nQ: {student_question}\nA: {elaboration}"
        
        # Round 3: Another question
        if num_rounds >= 3:
            student_question_2 = "What would happen if the problem had different parameters?"
            conversation.append({
                "role": "student",
                "content": student_question_2,
                "turn": 3
            })
            
            elaboration_2 = self.respond_to_question(student_question_2, context)
            conversation.append({
                "role": "teacher",
                "content": elaboration_2,
                "turn": 3
            })
        
        # Final summary
        conversation.append({
            "role": "teacher",
            "content": f"Final Answer: \\boxed{{{answer}}}",
            "turn": "final"
        })
        
        return conversation


class ConversationalTeacherStudentTrainer(TeacherStudentTrainer):
    """Enhanced trainer with conversational teacher-student capabilities."""
    
    def __init__(
        self,
        teacher_config: ConversationalTeacherStudentConfig,
        *args,
        **kwargs
    ):
        self.conversation_data = []
        self.current_phase = 0
        super().__init__(teacher_config, *args, **kwargs)
        self.teacher_config = teacher_config
    
    def _initialize_teacher(self):
        """Initialize the conversational teacher model."""
        self.teacher = ConversationalTeacherModel(
            self.teacher_config,
            device=self.args.device if hasattr(self.args, 'device') else "auto"
        )
        logger.info("Conversational teacher model initialized with Qwen3.5")
    
    def prepare_conversational_dataset(self, examples: List[Dict]) -> List[Dict]:
        """Prepare dataset with conversational format."""
        conversational_examples = []
        
        for example in examples:
            problem = example.get("problem", example.get("question", ""))
            answer = example.get("answer", example.get("solution", ""))
            
            if not problem or not answer:
                continue
            
            # Generate conversational training data
            if hasattr(self.teacher, 'generate_conversational_training_data'):
                conversation = self.teacher.generate_conversational_training_data(
                    problem, answer, self.teacher_config.dialogue_rounds
                )
                
                # Convert to training format
                full_text = self._conversation_to_training_text(conversation, problem, answer)
                conversational_examples.append({
                    "text": full_text,
                    "problem": problem,
                    "answer": answer,
                    "conversation": conversation,
                    "is_conversational": True
                })
            else:
                # Fallback to standard format
                conversational_examples.append(example)
        
        logger.info(f"Generated {len(conversational_examples)} conversational training examples")
        return conversational_examples
    
    def _conversation_to_training_text(
        self, 
        conversation: List[Dict[str, str]], 
        problem: str, 
        answer: str
    ) -> str:
        """Convert conversation to training text format."""
        
        text_parts = [
            f"Problem: {problem}",
            "",
            "=== Expert Tutor Discussion ===",
            ""
        ]
        
        for turn in conversation:
            if turn["role"] == "teacher":
                text_parts.append(f"Teacher: {turn['content']}")
            else:
                text_parts.append(f"Student: {turn['content']}")
            text_parts.append("")
        
        text_parts.extend([
            "=== Solution Summary ===",
            f"Final Answer: \\boxed{{{answer}}}"
        ])
        
        return "\n".join(text_parts)
    
    def get_phase_alpha(self) -> float:
        """Get alpha for current training phase."""
        if not self.teacher_config.use_phased_training:
            return self.teacher_config.alpha
        
        phases = self.teacher_config.phases
        epoch = self.state.epoch or 0
        
        # Determine which phase we're in
        cumulative_epochs = 0
        for phase in phases:
            cumulative_epochs += phase.get("epochs", 1)
            if epoch < cumulative_epochs:
                return phase.get("alpha", self.teacher_config.alpha)
        
        return phases[-1].get("alpha", self.teacher_config.alpha) if phases else self.teacher_config.alpha
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Enhanced loss computation with phase-aware alpha and conversational data.
        """
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels must be provided for teacher-student training")
        
        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Standard student loss
        student_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        student_loss = student_loss_fct(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        try:
            # Get dynamic alpha based on current phase
            current_alpha = self.get_phase_alpha()
            
            # Get teacher logits
            teacher_logits = self.teacher.get_teacher_logits(
                inputs["input_ids"],
                inputs.get("attention_mask")
            )
            
            # Ensure shapes match
            if teacher_logits.shape != student_logits.shape:
                min_seq_len = min(teacher_logits.shape[1], student_logits.shape[1])
                teacher_logits = teacher_logits[:, :min_seq_len, :student_logits.shape[2]]
                student_logits_truncated = student_logits[:, :min_seq_len, :]
            else:
                student_logits_truncated = student_logits
            
            # Knowledge distillation with temperature
            temperature = self.teacher_config.temperature
            distill_loss = F.mse_loss(
                student_logits_truncated / temperature,
                teacher_logits / temperature
            )
            
            # Combine with phase-aware alpha
            total_loss = (
                current_alpha * distill_loss +
                (1 - current_alpha) * student_loss
            )
            
        except Exception as e:
            logger.warning(f"Conversational distillation failed: {e}")
            total_loss = student_loss
            distill_loss = torch.tensor(0.0, device=student_loss.device)
            current_alpha = self.teacher_config.alpha
        
        # Enhanced logging
        if self.state.global_step % 50 == 0:  # More frequent logging
            phase_name = "default"
            if self.teacher_config.use_phased_training and self.teacher_config.phases:
                phases = self.teacher_config.phases
                epoch = self.state.epoch or 0
                cumulative = 0
                for phase in phases:
                    cumulative += phase.get("epochs", 1)
                    if epoch < cumulative:
                        phase_name = phase.get("name", "unknown")
                        break
            
            logger.info(
                f"[Phase: {phase_name}] Step {self.state.global_step} | "
                f"Student: {student_loss.item():.4f} | "
                f"Distill: {distill_loss.item():.4f} | "
                f"Total: {total_loss.item():.4f} | "
                f"Alpha: {current_alpha:.2f}"
            )
        
        return (total_loss, {"student_loss": student_loss, "distill_loss": distill_loss}) if return_outputs else total_loss


def create_conversational_trainer(
    model: nn.Module,
    args,
    train_dataset,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    teacher_config: Optional[ConversationalTeacherStudentConfig] = None,
    **kwargs
) -> ConversationalTeacherStudentTrainer:
    """Factory function to create a conversational teacher-student trainer."""
    
    if teacher_config is None:
        teacher_config = ConversationalTeacherStudentConfig()
    
    # Remove conflicting arguments
    kwargs.pop('tokenizer', None)
    kwargs.pop('data_collator', None)
    
    return ConversationalTeacherStudentTrainer(
        teacher_config=teacher_config,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        **kwargs
    )
