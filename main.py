import torch.nn as nn

from archive.mm_encoder import build_mm_vis_towers_list
import json
import torch
from argparse import Namespace

from utils.custom import is_rank_0
from utils.mm_datautils import VideoDataset
from utils.feat import filter_feature_sim, get_spatio_temporal_features_using_pooling, split_features_by_sample
from utils.xformer import load_HF_model
from utils.img_misc import unpad_image
from archive.mm_encoder.projector import LinearProjector

from torch.utils.data import RandomSampler, DataLoader
from peft_decoder import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftMmModel


def load_vision_towers(config, device_map, device):

    vision_towers = build_mm_vis_towers_list(config)
    
    for _tower in vision_towers:
        if not _tower.is_loaded:
            _tower.load_model(device_map=device_map)
        _tower.to(device=device, dtype=torch.float16)
    
    image_processors = [
        _tower.image_processor
        for _tower in vision_towers
    ]
    return vision_towers, image_processors


def load_fm(config):
    
    # Get the tokenizer
    # Get the model
    tokenizer, model_config, model = load_HF_model(
        model_type=config.fm_type,
    )
    config.model_config = model_config
    
    # Load checkpoint
    if len(config.load_base_from_path) > 0:
        # We load the model state dict on the CPU to avoid an OOM error.
        loaded_state_dict = torch.load(config.load_base_from_path, map_location="cpu")
        loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
        model.load_state_dict(loaded_state_dict, strict=True)
        
        # release memory
        del loaded_state_dict
        
        # Log the loaded checkpoint
        msg = "Loaded decoder base checkpoint from path: {}".format(config.load_base_from_path)
        if is_rank_0():
            print(msg)
    
    # Get the config
    peft_config = PromptTuningConfig(
        task_type=TaskType.MM_CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
        num_virtual_tokens=config.num_virtual_tokens,
    )
    config.peft_config = peft_config
    
    if len(config.load_adapter_from) > 0:
        # Load the model adapters - in place
        model = PeftMmModel.from_pretrained(
            model=model,
            model_id=config.load_adapter_from,  # Must be a directory containing the model files
            config=peft_config,
        )
        msg = "[INFO] Loaded the model adapters from: {}".format(config.load_adapter_from)
        if is_rank_0():
            print(msg)
    else:
        # Initialize the model adapters
        model = get_peft_model(model, peft_config)
    
    config.total_virtual_tokens = config.num_virtual_tokens
    config.word_embedding_dim = peft_config.token_dim
    return tokenizer, model


class EMQA(torch.nn.Module):
    def __init__(self, config, encoders, decoder):
        super(EMQA, self).__init__()
        self.config = config
        self.encoders = encoders
        self.projector = LinearProjector(config)
        self.image_newline = nn.Parameter(torch.empty(config.model_config.hidden_size))
        self.decoder = decoder
        
    @torch.no_grad()
    def encode(self, batch):
        siglip, dino = self.encoders
        encoders_input = batch['images']
        # Get the number of frames in each video
        split_sizes_orig = [1 if vid.ndim == 3 else vid.shape[0] for vid in encoders_input[0]]
        
        # TODO: Set this based on the input text & context length
        image_sizes = batch['image_sizes']
        
        # Flatten along the frame dimension each encoder's input
        new_vid_encoders_ip = []
        for encoder_ip in encoders_input:
            if type(encoder_ip) is list:
                encoder_ip = [
                    vid.unsqueeze(0) if vid.ndim == 3 else vid for vid in encoder_ip
                ]
            new_sample = torch.cat([vid for vid in encoder_ip], dim=0)  # Concatenating all frames across videos
            new_vid_encoders_ip.append(new_sample)
        
        dino_input = new_vid_encoders_ip[-1]
        dino_features = dino.encode(dino_input, chunk_size=self.config.frame_chunk_size_dino)  # Generates ~576 feats
        
        # ########################################################## #
        # [First] stage of frame pruning.
        # > Use frame-frame similarity.
        # > Use Dino features since it was trained with a feature similarity objective
        (dino_features, split_sizes, new_vid_encoders_ip, selected_frame_indices_all) = filter_feature_sim(
            self.config,
            dino_features,
            new_vid_encoders_ip,
            split_sizes_orig,
            batch['input_ids'],
            image_sizes,
            window_size=8,
            threshold=0.83,
        )
        
        siglip_input = new_vid_encoders_ip[0]
        siglip_features = siglip.encode(siglip_input, chunk_size=self.config.frame_chunk_size_siglip)  # Generates ~576 feats
        
        combined_features = [
            siglip_features,
            dino_features
        ]
        # Get the original image size for each frame in each video
        extended_image_sizes = []
        for i in range(len(image_sizes)):
            for j in range(split_sizes[i]):
                extended_image_sizes.append(image_sizes[i])
        image_sizes = extended_image_sizes
        
        return combined_features, image_sizes, split_sizes
    
    def aggregate_features(self, combined_features, split_sizes):
        # Use either a SVA to combine or simply concatenate. TODO: Implement SVA
        if self.config.feature_agg == 'sva':
            raise NotImplementedError("SVA not implemented yet")
        
        elif self.config.feature_agg == 'pooling':
            # Notes:-
            # 1. This method will give fixed visual_tokens = 576 (spatial) + max_length (temporal) for each encoder.
            # 2. Make sure the number of frames in each video is less than or equal to max_length
            if isinstance(combined_features, list):
                pooled_feats = []
                for feat_type in combined_features:
                    batched_feats = split_features_by_sample(feat_type, split_sizes)
                    batched_pooled_feats = []
                    for vid_feats in batched_feats:
                        pooled_feat = get_spatio_temporal_features_using_pooling(
                            vid_feats,
                            max_length=25
                        )
                        batched_pooled_feats.append(pooled_feat.unsqueeze(0))
                   
                    # Post making sure that the pooled features per vid are of the same length -> Concatenate
                    batched_pooled_feats = torch.cat(batched_pooled_feats, dim=0)  # (n_vids, n_feats, feat_dim)
                    pooled_feats.append(batched_pooled_feats)
                combined_features = torch.cat(pooled_feats, dim=-1)
            else:
                combined_features = split_features_by_sample(combined_features, split_sizes)
                batched_pooled_feats = []
                for vid_feats in combined_features:
                    pooled_feat = get_spatio_temporal_features_using_pooling(
                        vid_feats,
                        max_length=100
                    )
                    batched_pooled_feats.append(pooled_feat)
                    
                # Post making sure that the pooled features per vid are of the same length -> Concatenate
                combined_features = torch.cat(batched_pooled_feats, dim=0)
        
        else:
            raise ValueError("Invalid feature aggregation method")
            
        return combined_features
    
    def temp(self, features, image_sizes):
        bs = features.shape[0]
        dtype = features.dtype

        image_token_len = self.config.image_token_len
        final_height = final_width = int(image_token_len ** 0.5)

        # Reshape the features to (bs, final_height, final_width, -1)
        features = features.view(bs, final_height, final_width, -1)
        
        final_size = []
        image_features_unpadded = []
        for i in range(bs):
            cur_image_feature = features[i]
            image_size = image_sizes[i]

            cur_image_feature = unpad_image(
                cur_image_feature.unsqueeze(0), image_size
            )

            cur_h, cur_w = cur_image_feature.shape[1:3]
            try:  # fix bug for some invalid image
                cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
                final_size.append((cur_h, cur_w))
            except:
                # print(f"invalid after unpad {image_features[batch_i].shape}, {image_sizes[batch_i]}", flush=True)
                cur_image_feature = features[i].unsqueeze(0)
                image_size = image_sizes[i]
                cur_h, cur_w = cur_image_feature.shape[1:3]
                cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
                final_size.append((cur_h, cur_w))
        
            cur_image_feature = torch.cat(
                (
                    cur_image_feature,
                    self.image_newline.view(1, 1, 1, -1).expand(1, cur_h, 1, -1).to(cur_image_feature.device),
                ),
                dim=2,
            )
            
            cur_image_feature = cur_image_feature.flatten(1, 2)
            image_features_unpadded.append(cur_image_feature.squeeze(0))
        
        features = image_features_unpadded

        return features
        
    def forward(self, batch):
        # Encode the video to get the features
        features, image_sizes, split_sizes = self.encode(batch)
        # Aggregate the features (if multiple encoders)
        features = self.aggregate_features(features, split_sizes)
        # Project the features into the prompt space (or the linguistic embedding) of the decoder
        features = self.projector(features)
        
        # # Temp
        # features = self.temp(features, image_sizes)
        
        # Decode the features
        output = self.decoder(
            soft_v_prompt=features,
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        return output
        

def main():
    
    # Load config
    with open('./configs/mm_encoder.json') as f:
        config = json.load(f)
    config = Namespace(**config)
    config.batch_size = 2
    config.frame_chunk_size_dino = 4  # TODO: Rename to fwd_batch_size_dino
    config.frame_chunk_size_siglip = 4
    
    # Feature Aggregator
    config.feature_agg = 'pooling'
    
    device_map = None
    device = "cpu"
    vision_encoders, image_processor = load_vision_towers(config, device_map, device)
    config.mm_hidden_size = sum([_enc.hidden_size for _enc in vision_encoders])
    
    # video_path = './baselines/LongVU/examples/video1.mp4'
    # video, image_size = read_video(video_path)
    # encoder_input = process_images(video, image_processor, config)  # List of images for each encoder
    #
    # # Convert to list of tensors (n_frames, n_channels, h, w)
    # if 'cuda' in device:
    #     encoder_input = [torch.stack(imgs_list).half().cuda() for imgs_list in encoder_input]
    # else:
    #     encoder_input = [torch.stack(imgs_list).half() for imgs_list in encoder_input]
    
    tokenizer, decoder = load_fm(config)
    config.model_max_length = tokenizer.model_max_length
    config.padding_side = tokenizer.padding_side
    config.padding_token_id = tokenizer.pad_token_id
    
    emqa = EMQA(config, vision_encoders, decoder)
    
    dataset = VideoDataset(
        image_processor,
        tokenizer=tokenizer,
        max_length=128,
        max_prompt_length=128,
        debug=True
    )
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.batch_size,
        collate_fn=dataset.collate_fn
    )
    
    for batch in data_loader:
        output = emqa(batch)
        print(output.shape)
        break
        
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
