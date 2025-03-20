# Description: This script is used to analyse the results of the retrieval task. (Specific to my_hierarchical-emv_qa
# dataset.) Objective: To calculate the average distance of GT frames from the predicted segment.

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Provide the path to segment predictions
# path_to_preds = './results/longvu_cambrian_llama3_baseline_top1_segment_middleRepFrame_10m_10n_retrieval//results_cambrian_llama3_baseline_top1_segment_middleRepFrame_10m_10n_retrieval.json'
# path_to_preds = './results/longvu_cambrian_llama3_baseline_top1_segment_VideoCLIP-XL_10m_10n_retrieval/results_cambrian_llama3_baseline_top1_segment_VideoCLIP-XL_10m_10n_retrieval.json'
path_to_preds = './results/longvu_cambrian_llama3_baseline_top1_segment_VideoCLIP-XL-v2_10m_10n_retrieval/results_cambrian_llama3_baseline_top1_segment_VideoCLIP-XL-v2_10m_10n_retrieval.json'
with open(path_to_preds) as f:
	preds = json.load(f)

# Read annotation
path = './data/Ego4D/my_hierarchical-emv_qa.json'
with open(path) as f:
	annotations = json.load(f)
flattened_annotations = []
for video_id in annotations:
	flattened_annotations += annotations[video_id]

distances = []
fps = 30
question_wise_distances = {
	'what': [],
	'where': [],
	'who': [],
	'which': [],
	'how': [],
	'other': []
}
for video_id in preds:
	for sample in preds[video_id]:
		sample_id = sample['sample_id']
		for sample_annotation in flattened_annotations:
			if sample_annotation['sample_id'] == sample_id:
				gt: bool = len(sample_annotation['support']) > 0
				
				if not gt:
					continue
				
				gt_support = sample_annotation['support']
				predicted_segment = sample['metadata']['top_1_segments'][0][1]
				start_frame = predicted_segment[0]
				end_frame = predicted_segment[-1]
				
				ques = sample_annotation['question']
				ques_starts_with = ques.split()[0].lower()
				# Check for each frame in support how far it is from the predicted segment
				for gt_frame in gt_support:
					if gt_frame < start_frame:
						dist = start_frame - gt_frame
					elif gt_frame > end_frame:
						dist = gt_frame - end_frame
					else:
						dist = 0
					
					# Convert distance to seconds
					dist /= fps
					distances.append(dist)
					question_wise_distances[ques_starts_with].append(
						dist) if ques_starts_with in question_wise_distances else \
						question_wise_distances['other'].append(dist)

print("Number of samples with GT Frames: ", len(distances))
print("Average Distance in Seconds: ", np.mean(distances))
print("Std Deviation: ", np.std(distances))

bins = range(0, 3000, 60)
x_ticks = np.arange(0, 3000, 500)  # For showing x-ticks
sns.set_theme(style="whitegrid")
sns.histplot(np.clip(distances, bins[0], bins[-1]), bins=bins, edgecolor='black')
# Add a special x-tick for end bin with '3000+'
plt.xticks(list(x_ticks) + [3000], [str(b) for b in x_ticks] + ['3000+'])
plt.xlabel('Distance in Seconds')
plt.ylabel('Frequency')
plt.title('Distribution of Distance of Predicted Segment from GT Frames')
plt.savefig('dist_distribution_in_secs.png')

# Question-wise distribution
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Question-wise Distribution of Distance of Predicted Segment from GT Frames')
for i, (ques_type, dist) in enumerate(question_wise_distances.items()):
	ax = axs[i // 2, i % 2]
	sns.histplot(np.clip(dist, bins[0], bins[-1]), bins=bins, edgecolor='black', ax=ax)
	ax.set_xticks(list(x_ticks) + [3000], [str(b) for b in x_ticks] + ['3000+'])
	ax.set_title(f'{ques_type.capitalize()}')
	ax.set_ylabel('Frequency')
	# Only set x-axis label for the last row
	if i // 2 == 2:
		ax.set_xlabel('Distance in secs')
plt.savefig('dist_distribution_for_each_ques_type.png')
