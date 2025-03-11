import json
from pathlib import Path
from datetime import datetime
path = './data/Ego4D/hierarchical-emv_qa.json'
path = Path(path)

# Credits: https://github.com/lbaermann/hierarchical-emv/blob/main/llm_emv/eval/ego4d_custom_qa.py
qa_data = json.loads(path.read_text())
ego4d_video_ids = set()
my_data = {}

for i, history_sample in enumerate(qa_data):
	for q, question_sample in enumerate(history_sample['questions']):
		
		question = question_sample['q']
		answer = question_sample['a']
		supports = question_sample['support']
		video_ids = [support['video_id'] for support in supports]
		# Frames which support the question
		supporting_frames = []
		for support in supports:
			if 'frame' in support and support['frame'] is not None:
				supporting_frames.append(support['frame'])
		supporting_frames = sorted(supporting_frames)
		video_ids = set(video_ids)
		try:
			# All video_ids should be the same
			assert len(video_ids) == 1
		except AssertionError:
			print("Need multiple videos to support the question. Skipping.")

		# Let's print nicely - Video ID, Question, Answer
		video_id = video_ids.pop()
		sample_id = f"{video_id}_{i}_{q}"
		ego4d_video_ids.add(video_id)
		print(f"Video ID: {video_id}")
		print(f"Question: {question}")
		print(f"Answer: {answer}")
		print()
		
		if video_id not in my_data:
			my_data[video_id] = []
		
		my_data[video_id].append({
			"sample_id": sample_id,
			"question": question,
			"answer": answer,
			"support": supporting_frames,
		})
		
print(f"Ego4D video IDs: {ego4d_video_ids}")
print()

# Let's get number of QAs for each video
for video_id, qa_list in my_data.items():
	print(f"Video ID: {video_id} has {len(qa_list)} QAs")
	print()

# Save the data
output_path = path.parent / "my_hierarchical-emv_qa.json"
output_path.write_text(json.dumps(my_data, indent=2))