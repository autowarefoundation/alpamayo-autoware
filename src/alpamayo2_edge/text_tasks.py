# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate freeform answers from camera images and a question."""

import copy
from typing import Any

import torch
from transformers import LogitsProcessorList

from alpamayo2_edge.chat_template import CLOSED_THINKING
from alpamayo2_edge.models.alpamayo2_edge import MaskDiscreteTrajectoryLogitsProcessor


@torch.inference_mode()
def generate_vqa_answer(
    model: Any, inputs: dict[str, Any], *, max_new_tokens: int = 512
) -> str:
    """Generate one text answer and stop before any trajectory continuation."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    tokenized = inputs["tokenized_data"]
    if tokenized["input_ids"].shape[0] != 1:
        raise ValueError("VQA generation expects one sample at a time")
    generation = copy.deepcopy(model.vlm.generation_config)
    generation.do_sample = False
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    generation.num_beams = 1
    generation.num_return_sequences = 1
    generation.max_new_tokens = max_new_tokens
    generation.pad_token_id = model.tokenizer.pad_token_id
    generation.return_dict_in_generate = False
    generation.output_scores = False
    generation.output_logits = False
    # A driving answer can end at the trajectory anchor instead of text EOS.
    eos = generation.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    generation.eos_token_id = list(dict.fromkeys([*eos, model.config.traj_ids["future_start"]]))
    logits_processor = LogitsProcessorList(
        [
            MaskDiscreteTrajectoryLogitsProcessor(
                traj_token_offset=min(
                    model.config.traj_ids["history_id0"], model.config.traj_ids["future_id0"]
                ),
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )
    sequences = model.vlm.generate(
        **tokenized, generation_config=generation, logits_processor=logits_processor
    )
    completion = sequences[:, tokenized["input_ids"].shape[1] :]
    answer = model.tokenizer.batch_decode(completion, skip_special_tokens=True)[0].strip()
    return answer.removeprefix(CLOSED_THINKING).strip()
