from dataclasses import dataclass, field
from typing import List, Set, Union, Optional, Dict, Any


@dataclass
class Question:
	text: str
	answers: Union[Set[str], List[str]]
	id: Optional[str]
	tokens: Optional[List[str]] = field(default=None)
	
	@property
	def has_answers(self) -> bool:
		return self.answers and len(self.answers) > 0
	
	@property
	def tokenized_text(self) -> Optional[str]:
		return " ".join(self.tokens) if self.tokens is not None else None
	
	def to_json(self) -> Dict[str, Any]:
		json_dict = dict(
			question=self.text,
			id=self.id,
			answers=self.answers,
		)
		
		return json_dict
	
	@classmethod
	def from_json(cls, q_dict, id):
		return Question(
			q_dict["question"],
			q_dict.get("answer", q_dict.get("answers", None)),
			q_dict.get("id", id),
		)
	
	
# # Sample Usage
# q = Question("What is the capital of India?", ["New Delhi"], "1")
# print(q.to_json())
