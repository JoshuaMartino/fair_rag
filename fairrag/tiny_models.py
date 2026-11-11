"""Tiny randomly initialised models and a word-level tokenizer for tests.

Nothing is downloaded: the models are built from configs, so the tests run
offline and on CPU in a few seconds.
"""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)

WORDS = (
    "you are a classifier which of the following three professions best matches "
    "person described in biography below choose exactly one output only "
    "profession name context documents text gender answer male female "
    "nurse surgeon teacher interior designer dj poet rapper model painter "
    "she he is was born works worked at hospital school studio true false "
    "pass fair biased relevant document query despite being became an "
    "the and of for with her his : , . - { } \" ? [UNK] [PAD] [BOS] [EOS]"
).split()


def tiny_tokenizer():
    vocab = {word: index for index, word in enumerate(dict.fromkeys(WORDS))}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace()]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    return fast


def tiny_causal(seed=0):
    torch.manual_seed(seed)
    tokenizer = tiny_tokenizer()
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return GPT2LMHeadModel(config).eval(), tokenizer


def tiny_seq2seq(seed=0):
    torch.manual_seed(seed)
    tokenizer = tiny_tokenizer()
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=2,
        decoder_start_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return T5ForConditionalGeneration(config).eval(), tokenizer
